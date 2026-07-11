"""
EDINET API v2 クライアント (金融庁公式)

edinetdb.jp(第三者サービス、1日100回制限)の代替・補完として使う。
主なメリット:
- 営業利益(Operating Profit)が取得可能  ← edinetdb.jpでは取れない
- キャッシュフロー計算書が取得可能
- 大株主情報が取得可能(オーナー社長判定に使用)
- 無料・利用回数制限なし

利用登録でAPIキーを取得し、環境変数に設定して使う:
    PowerShell: $env:EDINET_API_KEY = "発行されたキー"
"""

import io
import json
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
CACHE_DIR = Path(__file__).parent / "cache" / "edinet_v2"

# 有価証券報告書の書類種別コード
DOC_TYPE_ANNUAL = "120"       # 有価証券報告書
DOC_TYPE_QUARTERLY = "140"    # 四半期報告書
DOC_TYPE_INTERIM = "130"      # 半期報告書

# 全銘柄分の有価証券報告書メタデータを1回で作る日付インデックスのキャッシュ
# (銘柄ごとにfind_company_docsで日付を遡ると全銘柄では非現実的な時間がかかるため)
ANNUAL_INDEX_CACHE = CACHE_DIR / "annual_report_index.json"
# 有報は決算後3ヶ月以内提出が原則のため、13ヶ月遡れば全社の最新期を拾えるはず
ANNUAL_INDEX_LOOKBACK_DAYS = 450

# XBRL上の財務要素名(ローカル名ベース。JGAAP/IFRSの揺れを吸収するため複数列挙)
# IFRS採用企業はサフィックス"IFRS"が付く。実データから確認済み(トヨタ等)。
XBRL_TARGETS: dict[str, list[str]] = {
    # 損益計算書
    "net_sales": [
        "NetSales", "Revenue", "NetSalesAndOperatingRevenues",
        "RevenueIFRS", "SalesRevenuesIFRS", "TotalNetRevenuesIFRS",
        "OperatingRevenuesIFRSKeyFinancialData",
    ],
    "cost_of_sales": [
        "CostOfSales", "CostOfSalesIFRS",
    ],
    "gross_profit": [
        # 直接タグがあればそれを使う(全社が開示するわけではないので、
        # 無い場合はnet_sales - cost_of_salesで計算する側でフォールバックする)
        "GrossProfit",
    ],
    "operating_profit": [
        "OperatingProfit", "OperatingIncome",
        "OperatingProfitLoss", "ProfitLossFromOperatingActivities",
        "OperatingProfitIFRS", "ProfitLossFromOperatingActivitiesIFRS",
    ],
    "ordinary_profit": [
        "OrdinaryProfit", "OrdinaryIncome",
    ],
    "net_income": [
        "ProfitAttributableToOwnersOfParent",
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitAttributableToOwnersOfParentIFRS",
        "ProfitLossAttributableToOwnersOfParentIFRS",
        "NetIncome", "ProfitLoss",
    ],
    # キャッシュフロー計算書(JGAAP版・IFRS版)
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesIFRS",
        "CashFlowsFromUsedInOperatingActivitiesIFRS",
    ],
    "investing_cf": [
        "NetCashProvidedByUsedInInvestingActivities",
        "CashFlowsFromInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesIFRS",
        "CashFlowsFromUsedInInvestingActivitiesIFRS",
    ],
    "financing_cf": [
        "NetCashProvidedByUsedInFinancingActivitiesIFRS",
        "CashFlowsFromUsedInFinancingActivitiesIFRS",
        "NetCashProvidedByUsedInFinancingActivities",
        "CashFlowsFromFinancingActivities",
    ],
    # 財務活動によるキャッシュフローの内訳
    # (「成長の質」= 増資・借入に頼らず自己資金で株主還元しながら成長できているか、の判定に使う)
    "dividends_paid_cf": [
        # JGAAPは"...FinCF"、IFRSは"...FinCFIFRS"というサフィックスの傾向がある
        # (実データ確認済み: JGAAPは1301等、IFRSはトヨタ等)
        "CashDividendsPaidFinCF",
        "DividendsPaidToOwnersOfParentFinCFIFRS",
        "CashDividendsPaid", "DividendsPaid",
    ],
    "treasury_stock_cf": [
        # IFRS採用企業は自己株買い・処分の純額を1本で開示することが多い。
        # JGAAPも同様に自己株式の増減を1本(純額)で開示するタグがある。
        # ("PurchaseOfTreasuryShares"のようなFinCFサフィックスの無いタグは
        # 株主資本等変動計算書(キャッシュフロー計算書ではない)の項目である
        # ケースが実データで確認されたため、あえて候補から除外している)
        "ReissuanceRepurchaseOfTreasuryStockFinCFIFRS",
        "DecreaseIncreaseInTreasuryStockFinCF",
    ],
    "proceeds_share_issuance_cf": [
        "ProceedsFromIssuanceOfShares",
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromShareIssuanceToNonControllingShareholdersFinCF",
        "ProceedsFromShareIssuanceToNonControllingShareholders",
    ],
    "proceeds_long_term_debt_cf": [
        "ProceedsFromLongTermLoansPayableFinCF",
        "ProceedsFromLongTermDebtFinCFIFRS",
        "ProceedsFromLongTermLoansPayable",
        "ProceedsFromLongTermBorrowings",
    ],
    "payments_long_term_debt_cf": [
        "RepaymentOfLongTermLoansPayableFinCF",
        "PaymentsOfLongTermDebtFinCFIFRS",
        "RepaymentsOfLongTermLoansPayable",
        "RepaymentsOfLongTermBorrowings",
    ],
    "short_term_debt_change_cf": [
        "NetIncreaseDecreaseInShortTermLoansPayableFinCF",
        "IncreaseDecreaseInShortTermDebtFinCFIFRS",
        "NetIncreaseDecreaseInShortTermLoansPayable",
        "ProceedsFromRepaymentsOfShortTermLoansPayableNet",
        "IncreaseDecreaseInCommercialPapersFinCF",
    ],
    # 貸借対照表
    "total_assets": ["Assets", "TotalAssets", "AssetsIFRS"],
    "equity": ["Equity", "NetAssets", "TotalNetAssets", "EquityAttributableToOwnersOfParentIFRS"],
}

# 発行済株式数(自己株式含む)の5期推移。「経営成績等の推移」(サマリー情報)タグは
# 単一の書類の中にCurrentYear/Prior1Year〜Prior4Yearのコンテキストで複数年分が
# 含まれているため、これだけで直近5期の推移(希薄化/自己株買いによる減少)を判定できる。
SHARES_ISSUED_TAG = "TotalNumberOfIssuedSharesSummaryOfBusinessResults"
TREASURY_SHARES_TAG = "TotalNumberOfSharesHeldTreasurySharesEtc"
# サマリー系コンテキストの「何年前か」を判定するための接頭辞(出現順ではなくラベルで判定する)
YEAR_CONTEXT_PREFIXES = [
    ("CurrentYear", 0),
    ("Prior1Year", 1),
    ("Prior2Year", 2),
    ("Prior3Year", 3),
    ("Prior4Year", 4),
]

# 「経営成績等の推移」(5年サマリー)から抽出する財務指標。PER・ROEが直接タグ
# 付けされているため、edinetdb.jp(第三者サービス、1日100回制限)を使わなくても
# 同じ書類(既にcache/edinet_v2/にキャッシュ済みのXBRL)からPER・ROE・増収増益の
# 連続期数を全銘柄分計算できる。
# JGAAP側タグは連結・非連結どちらのコンテキストも同じタグ名で出てくるため、
# parse_financialsと同様に非連結を除外して連結を優先する。IFRS専用タグ(末尾IFRS)
# は実データ(トヨタ等)で連結コンテキストでしか出現しないことを確認済みのため、
# 区別せずそのまま採用する(JGAAP側で連結値が取れなかった年だけ穴埋めに使う)。
BUSINESS_RESULTS_SERIES_TAGS: dict[str, dict[str, str]] = {
    "net_sales": {
        "jgaap": "NetSalesSummaryOfBusinessResults",
        "ifrs": "OperatingRevenuesIFRSKeyFinancialData",
    },
    "ordinary_income": {
        "jgaap": "OrdinaryIncomeLossSummaryOfBusinessResults",
        "ifrs": "ProfitLossBeforeTaxIFRSSummaryOfBusinessResults",
    },
    "net_income": {
        # 実データ確認済み: JGAAPは"NetIncomeLossSummaryOfBusinessResults"ではなく
        # "ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults"が
        # 連結の「親会社株主に帰属する当期純利益」。前者は非連結(提出会社)専用の
        # タグ名で、連結コンテキストが存在しないため使うと欠損してしまう。
        "jgaap": "ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults",
        "ifrs": "ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
    },
    "roe": {
        "jgaap": "RateOfReturnOnEquitySummaryOfBusinessResults",
        "ifrs": "RateOfReturnOnEquityIFRSSummaryOfBusinessResults",
    },
    "per": {
        "jgaap": "PriceEarningsRatioSummaryOfBusinessResults",
        "ifrs": "PriceEarningsRatioIFRSSummaryOfBusinessResults",
    },
    "eps": {
        "jgaap": "BasicEarningsLossPerShareSummaryOfBusinessResults",
        "ifrs": "BasicEarningsLossPerShareIFRSSummaryOfBusinessResults",
    },
    "equity_ratio": {
        "jgaap": "EquityToAssetRatioSummaryOfBusinessResults",
        # 実データ確認済み: "EquityToAssetRatioIFRSSummaryOfBusinessResults"は
        # unitRef="JPYPerShares"(1株純資産)であり自己資本比率ではなかったため、
        # unitRef="pure"(比率)である正しいタグに差し替えている。
        "ifrs": "RatioOfOwnersEquityToGrossAssetsIFRSSummaryOfBusinessResults",
    },
    "total_assets_series": {
        "jgaap": "TotalAssetsSummaryOfBusinessResults",
        "ifrs": "TotalAssetsIFRSSummaryOfBusinessResults",
    },
    "net_assets_series": {
        "jgaap": "NetAssetsSummaryOfBusinessResults",
        "ifrs": "EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
    },
    # ここから下は「経営成績等の推移」(5年サマリー)ではなく、主要な財務諸表本体
    # (損益計算書・キャッシュフロー計算書)由来のタグ。実データ確認済みで、
    # CurrentYearDuration/Prior1YearDurationの2期分しか無い(5年分ではない)が、
    # Growth Quality Score(GQS)等の「前年比」を計算するには2期分あれば十分なため、
    # 同じparse_business_results_series(年コンテキストベースの抽出)で流用する。
    "cost_of_sales_2y": {
        # "CostOfGoodsSold"は実データ確認済み(小売・サービス業などで頻出する表記違い)
        "jgaap": ["CostOfSales", "CostOfGoodsSold"],
        "ifrs": "CostOfSalesIFRS",
    },
    "operating_cf_2y": {
        "jgaap": "NetCashProvidedByUsedInOperatingActivities",
        "ifrs": "NetCashProvidedByUsedInOperatingActivitiesIFRS",
    },
    # CapEx(設備投資)。フリーキャッシュフロー(FCF = 営業CF - CapEx)の計算に使う。
    # JGAAPは「有形固定資産の取得による支出」系のタグ(実データ確認済み、会社により
    # 表記が複数あるため候補を列挙)、IFRSは有形固定資産の追加(無形資産は含まない、
    # 一般的なCapExの定義に合わせて対象外にしている)。
    "capex_2y": {
        "jgaap": ["PurchaseOfPropertyPlantAndEquipmentInvCF", "PurchaseOfNoncurrentAssetsInvCF"],
        "ifrs": "AdditionsToFixedAssetsExcludingEquipmentLeasedToOthersInvCFIFRS",
    },
}


class EdinetV2Error(RuntimeError):
    pass


class QuotaExceededError(EdinetV2Error):
    pass


class EdinetV2Client:
    """
    EDINET API v2 クライアント。
    APIキーは EDINET_API_KEY 環境変数から読む。
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("EDINET_API_KEY")
        if not self.api_key:
            raise EdinetV2Error(
                "環境変数 EDINET_API_KEY が設定されていません。"
                "EDINET API 利用者登録でキーを取得し、"
                "$env:EDINET_API_KEY='発行されたキー' のように設定してください。"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Ocp-Apim-Subscription-Key": self.api_key,
        })
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 内部リクエスト ───────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        resp = self._session.get(
            f"{BASE_URL}{path}",
            params=params or {},
            timeout=60,
        )
        # EDINET は HTTP 200 を返しつつ、body の StatusCode でエラーを示す
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            body = resp.json()
            code = body.get("StatusCode", 200)
            if code == 401:
                raise EdinetV2Error(
                    "APIキーが無効です。EDINET_API_KEY を確認してください。"
                )
            if code != 200:
                raise EdinetV2Error(f"EDINET API エラー({code}): {body.get('message', '')}")
        else:
            resp.raise_for_status()
        return resp

    # ── 書類検索 ─────────────────────────────────────────────────

    def list_documents_by_date(
        self, target_date: date, doc_type_code: str | None = None
    ) -> list[dict]:
        """
        指定日に提出された書類の一覧を返す。
        doc_type_code を指定すると種別で絞り込む(例: '120'=有価証券報告書)。
        """
        resp = self._get("/documents.json", {"date": target_date.isoformat(), "type": 2})
        results: list[dict] = resp.json().get("results") or []
        if doc_type_code:
            results = [r for r in results if r.get("docTypeCode") == doc_type_code]
        return results

    def find_company_docs(
        self,
        sec_code: str,
        doc_types: list[str] | None = None,
        max_lookback_days: int = 500,
    ) -> list[dict]:
        """
        証券コード(例: '7203', '7203.T')から提出書類を検索する。

        ※ EDINET API には「証券コードで直接検索」するエンドポイントが無いため、
          日付をさかのぼりながらリストを絞り込む。最初に見つかった日で打ち切る。
          最大 max_lookback_days 日遡るが、通常は直近1〜3日で見つかる。
        """
        code4 = sec_code.split(".")[0].zfill(4)
        types = doc_types or [DOC_TYPE_ANNUAL, DOC_TYPE_QUARTERLY]
        cache_key = f"{code4}_{'_'.join(sorted(types))}"
        cache_path = CACHE_DIR / f"search_{cache_key}.json"

        # 当日キャッシュがあれば再利用
        if cache_path.exists():
            age_days = (date.today() - date.fromtimestamp(cache_path.stat().st_mtime)).days
            if age_days < 1:
                return json.loads(cache_path.read_text(encoding="utf-8"))

        found: list[dict] = []
        target = date.today()
        for _ in range(max_lookback_days):
            docs = self.list_documents_by_date(target, doc_type_code=None)
            matched = [
                d for d in docs
                if str(d.get("secCode", "")).startswith(code4)
                and d.get("docTypeCode") in types
            ]
            found.extend(matched)
            if found:
                break  # 最初に書類が見つかった日で打ち切り
            target -= timedelta(days=1)
            time.sleep(0.05)

        cache_path.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
        return found

    def build_annual_report_index(
        self,
        lookback_days: int = ANNUAL_INDEX_LOOKBACK_DAYS,
        force_refresh: bool = False,
        sleep_seconds: float = 0.2,
        on_progress=None,
        anchor_date: date | None = None,
    ) -> dict[str, dict]:
        """
        直近lookback_days日分の「その日に提出された有価証券報告書」を一括取得し、
        証券コード(4桁)をキーにしたインデックス({code4: メタデータdict})を作る。

        find_company_docs(1銘柄ごとに日付を遡る方式)だと、全銘柄(約3,700社)を
        処理する際に銘柄数×最大500回のAPI呼び出しが発生し非現実的な時間がかかる。
        こちらは「その日に提出された全社分」をまとめて返すlist_documents_by_dateを
        lookback_days回呼ぶだけで全銘柄分をカバーできるため、桁違いに高速。

        anchor_dateを指定すると、「今日時点」ではなく「anchor_date時点で最新だった
        有報」のインデックスを作る(過去の基準時点でのGQS再現用)。過去起点の
        インデックスは日付が固定で内容が変わらないため、一度作れば期限切れなく
        再利用する(1日1回の当日中キャッシュ判定は今日起点の場合のみ適用)。
        """
        is_today = anchor_date is None
        target = anchor_date or date.today()
        cache_path = (
            ANNUAL_INDEX_CACHE if is_today
            else CACHE_DIR / f"annual_report_index_{target.isoformat()}.json"
        )

        if not force_refresh and cache_path.exists():
            if not is_today:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            age_days = (date.today() - date.fromtimestamp(cache_path.stat().st_mtime)).days
            if age_days < 1:
                return json.loads(cache_path.read_text(encoding="utf-8"))

        index: dict[str, dict] = {}
        cursor = target
        for i in range(lookback_days):
            docs = self.list_documents_by_date(cursor, doc_type_code=DOC_TYPE_ANNUAL)
            for d in docs:
                sec_code = str(d.get("secCode") or "").strip()
                if not sec_code:
                    continue
                code4 = sec_code[:4]
                # cursorを新しい日から遡っているため、最初に見つかったものが最新の提出分
                index.setdefault(code4, d)
            cursor -= timedelta(days=1)
            time.sleep(sleep_seconds)
            if on_progress and (i + 1) % 50 == 0:
                on_progress(i + 1, lookback_days, len(index))

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        return index

    def get_financials_from_index(self, sec_code: str, index: dict[str, dict]) -> dict[str, Any]:
        """
        build_annual_report_indexで作ったインデックスを使い、日付を遡らずに
        直接財務データを取得する(get_financialsの高速版)。戻り値の形式は同じ。
        """
        code4 = sec_code.split(".")[0].zfill(4)
        meta = index.get(code4)
        if not meta:
            return {"error": f"有価証券報告書が見つかりません(インデックス内): {sec_code}"}
        return self._build_financials_result(sec_code, meta)

    def get_latest_annual_report_meta(self, sec_code: str) -> dict | None:
        """証券コードから直近の有価証券報告書のメタデータを1件返す。"""
        docs = self.find_company_docs(sec_code, doc_types=[DOC_TYPE_ANNUAL])
        return docs[0] if docs else None

    # ── XBRLダウンロード ─────────────────────────────────────────

    def download_xbrl_zip(self, doc_id: str) -> bytes:
        """
        書類IDのXBRLパッケージ(zip)をダウンロードして返す。
        同じ書類は cache/edinet_v2/{doc_id}.zip にキャッシュする。
        """
        cache_path = CACHE_DIR / f"{doc_id}.zip"
        if cache_path.exists():
            return cache_path.read_bytes()
        # type=1: XBRLのみのzip
        resp = self._get(f"/documents/{doc_id}", {"type": 1})
        data = resp.content
        cache_path.write_bytes(data)
        return data

    # ── XBRL解析 ─────────────────────────────────────────────────

    @staticmethod
    def parse_financials(zip_bytes: bytes) -> dict[str, float | None]:
        """
        XBRLパッケージ(zip)から主要財務数値を抽出する。

        ローカル名(Local Name)ベースで検索するため、名前空間の
        バリエーション(JGAAP/IFRS/各社独自)に左右されにくい。
        連結を優先し、なければ個別から取る。
        """
        result: dict[str, float | None] = {k: None for k in XBRL_TARGETS}

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                xbrl_files = [n for n in zf.namelist()
                               if n.endswith(".xbrl") and "__" not in n]
                for xbrl_name in xbrl_files:
                    try:
                        root = ET.parse(io.BytesIO(zf.read(xbrl_name))).getroot()
                    except ET.ParseError:
                        continue

                    # 全要素をローカル名でマッピング
                    for elem in root.iter():
                        local = (elem.tag.split("}")[-1]
                                 if "}" in elem.tag else elem.tag)
                        ctx = elem.get("contextRef", "")
                        text = (elem.text or "").strip()
                        if not text:
                            continue

                        for field, names in XBRL_TARGETS.items():
                            if result[field] is not None:
                                continue
                            if local not in names:
                                continue
                            try:
                                value = float(text.replace(",", ""))
                            except ValueError:
                                continue
                            # 連結・当期を最優先
                            # ("NonConsolidatedMember"も文字列として"Consolidated"を
                            # 含むため、誤って個別決算を連結扱いしないよう明示的に除外する)
                            is_non_consolidated = "NonConsolidated" in ctx
                            if "Consolidated" in ctx and not is_non_consolidated and "Current" in ctx:
                                result[field] = value
                            # 連結なし → 個別でも可(まだ未設定なら)
                            elif result[field] is None and "Prior" not in ctx:
                                result[field] = value

        except zipfile.BadZipFile:
            pass  # XBRLが取れなかった場合はNone埋めのまま返す

        return result

    @staticmethod
    def parse_major_shareholders(zip_bytes: bytes) -> list[dict]:
        """
        XBRLパッケージから大株主情報を抽出する。

        戻り値: [{"name": "株主名", "ratio": 持株比率(%)}, ...]
        """
        shareholders: list[dict] = []
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                xbrl_files = [n for n in zf.namelist()
                               if n.endswith(".xbrl") and "__" not in n]
                for xbrl_name in xbrl_files:
                    try:
                        root = ET.parse(io.BytesIO(zf.read(xbrl_name))).getroot()
                    except ET.ParseError:
                        continue

                    names: list[str] = []
                    ratios: list[float | None] = []
                    for elem in root.iter():
                        local = (elem.tag.split("}")[-1]
                                 if "}" in elem.tag else elem.tag)
                        text = (elem.text or "").strip()
                        if not text:
                            continue
                        # 名前: NameMajorShareholders
                        if local == "NameMajorShareholders":
                            names.append(text)
                        # 比率: ShareholdingRatio(実データ確認済み)
                        # MajorShareholderRatioなど他の書き方も一応カバー
                        elif local == "ShareholdingRatio" or (
                            "MajorShareholder" in local and "Ratio" in local
                        ):
                            try:
                                ratios.append(float(text.replace(",", "")) * 100
                                              if float(text) < 1.0  # 小数表記→%に変換
                                              else float(text))
                            except ValueError:
                                ratios.append(None)

                    if names:
                        for i, name in enumerate(names):
                            shareholders.append({
                                "name": name,
                                "ratio": ratios[i] if i < len(ratios) else None,
                            })
                        break  # 最初のXBRLファイルで見つかれば終了
        except zipfile.BadZipFile:
            pass

        return shareholders

    @staticmethod
    def parse_share_count_series(zip_bytes: bytes) -> dict[str, dict[int, float]]:
        """
        XBRLパッケージから、発行済株式数・自己株式数の直近5期分の推移を抽出する。

        戻り値: {"shares_issued": {0: 最新期の値, 1: 1期前, ...}, "treasury_shares": {...}}
        (キーの0が最新。連結/非連結どちらのコンテキストが取れるかは会社によって
        異なるため、"NonConsolidated"であっても既存のparse_financialsと違い
        除外しない。単一指標のみで連結/非連結の区別が難しいため。)
        """
        result: dict[str, dict[int, float]] = {"shares_issued": {}, "treasury_shares": {}}
        tag_map = {
            SHARES_ISSUED_TAG: "shares_issued",
            TREASURY_SHARES_TAG: "treasury_shares",
        }

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                xbrl_files = [n for n in zf.namelist()
                               if n.endswith(".xbrl") and "__" not in n]
                for xbrl_name in xbrl_files:
                    try:
                        root = ET.parse(io.BytesIO(zf.read(xbrl_name))).getroot()
                    except ET.ParseError:
                        continue

                    for elem in root.iter():
                        local = (elem.tag.split("}")[-1]
                                 if "}" in elem.tag else elem.tag)
                        if local not in tag_map:
                            continue
                        ctx = elem.get("contextRef", "")
                        text = (elem.text or "").strip()
                        if not text:
                            continue
                        # "...Row1Member"等、大株主一覧のような明細行のコンテキストは除外し、
                        # 素の(非連結含む)CurrentYearInstant/Prior1YearInstant系のみを対象にする
                        if "Row" in ctx:
                            continue
                        for prefix, years_ago in YEAR_CONTEXT_PREFIXES:
                            if ctx.startswith(prefix):
                                try:
                                    value = float(text.replace(",", ""))
                                except ValueError:
                                    continue
                                key = tag_map[local]
                                # 同じ年で複数コンテキスト(連結/非連結)が来た場合は
                                # 最初に見つかったものを優先する
                                result[key].setdefault(years_ago, value)
                                break
        except zipfile.BadZipFile:
            pass

        return result

    @staticmethod
    def parse_business_results_series(zip_bytes: bytes) -> dict[str, dict[int, float]]:
        """
        XBRLパッケージから、「経営成績等の推移」(5年サマリー表)由来の
        売上高・経常利益・純利益・ROE・PER・EPS・自己資本比率・総資産・純資産の
        直近5期分の推移を抽出する。

        戻り値: {"net_sales": {0: 最新期, 1: 1期前, ...}, "roe": {...}, "per": {...}, ...}
        (キーの0が最新。BUSINESS_RESULTS_SERIES_TAGSの説明の通り、JGAAPタグは
        連結を優先し、その年がJGAAPタグで取れなかった場合のみIFRSタグで埋める。
        連結の子会社を持たない単独決算のみの会社は、全ての値が
        NonConsolidatedコンテキストでしか開示されないため、連結が無い年に限り
        非連結の値をフォールバックとして使う。実データ確認済み:
        これが無いと単独決算の会社は5年サマリー自体が丸ごと空になってしまう)
        """
        result: dict[str, dict[int, float]] = {
            field: {} for field in BUSINESS_RESULTS_SERIES_TAGS
        }
        consolidated_hit: dict[str, set[int]] = {field: set() for field in BUSINESS_RESULTS_SERIES_TAGS}

        # jgaap/ifrsの値は単一タグ名(str)、または複数候補(list[str])のどちらでもよい
        # (実データで会社により表記が異なることが確認されたタグ向け。1社のXBRLには
        # 通常どれか1つの表記しか出現しないため、優先順位を気にする必要はない)。
        tag_to_field: dict[str, tuple[str, str]] = {}
        for field, tags in BUSINESS_RESULTS_SERIES_TAGS.items():
            for kind in ("jgaap", "ifrs"):
                candidates = tags[kind]
                if isinstance(candidates, str):
                    candidates = [candidates]
                for tag_name in candidates:
                    tag_to_field[tag_name] = (field, kind)

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                xbrl_files = [n for n in zf.namelist()
                               if n.endswith(".xbrl") and "__" not in n]
                for xbrl_name in xbrl_files:
                    try:
                        root = ET.parse(io.BytesIO(zf.read(xbrl_name))).getroot()
                    except ET.ParseError:
                        continue

                    for elem in root.iter():
                        local = (elem.tag.split("}")[-1]
                                 if "}" in elem.tag else elem.tag)
                        if local not in tag_to_field:
                            continue
                        field, kind = tag_to_field[local]
                        ctx = elem.get("contextRef", "")
                        if "Row" in ctx:
                            continue
                        text = (elem.text or "").strip()
                        if not text:
                            continue
                        is_non_consolidated = kind == "jgaap" and "NonConsolidated" in ctx
                        for prefix, years_ago in YEAR_CONTEXT_PREFIXES:
                            if not ctx.startswith(prefix):
                                continue
                            try:
                                value = float(text.replace(",", ""))
                            except ValueError:
                                continue
                            if kind == "jgaap":
                                if is_non_consolidated:
                                    # 連結が別途取れればそちらを優先するが、連結を
                                    # 持たない会社ではこれが唯一の値なので
                                    # フォールバックとして保持する(上書きはしない)
                                    result[field].setdefault(years_ago, value)
                                else:
                                    result[field][years_ago] = value  # 連結を優先して上書き
                                    consolidated_hit[field].add(years_ago)
                            elif years_ago not in consolidated_hit[field]:
                                # 連結JGAAPタグでまだ値が無い年だけIFRSタグで穴埋めする
                                result[field].setdefault(years_ago, value)
                            break
        except zipfile.BadZipFile:
            pass

        return result

    # ── 統合 API ────────────────────────────────────────────────

    def get_financials(self, sec_code: str) -> dict[str, Any]:
        """
        証券コードから直近の有価証券報告書を取得し、
        財務数値 + 大株主情報をまとめて返す。

        戻り値のキー:
            sec_code, company_name, period_start, period_end, doc_id
            net_sales, cost_of_sales, gross_profit,
            operating_profit, ordinary_profit, net_income
            operating_cf, investing_cf, financing_cf
            dividends_paid_cf, treasury_stock_cf, proceeds_share_issuance_cf,
            proceeds_long_term_debt_cf, payments_long_term_debt_cf, short_term_debt_change_cf
            total_assets, equity
            share_count_series: {"shares_issued": {0: 最新, 1: 1期前, ...}, "treasury_shares": {...}}
            business_results_series: {"net_sales": {0: 最新, 1: 1期前, ...}, "roe": {...},
                "per": {...}, "eps": {...}, "ordinary_income": {...}, "net_income": {...},
                "equity_ratio": {...}, "total_assets_series": {...}, "net_assets_series": {...}}
                (5年サマリー表由来。edinetdb.jpを使わずPER・ROE・増収増益の連続期数を計算できる)
            major_shareholders: [{"name": str, "ratio": float|None}]
            is_owner_managed: True/False (代表者が大株主かどうかの簡易判定)
            error: エラー時のみ
        """
        meta = self.get_latest_annual_report_meta(sec_code)
        if not meta:
            return {"error": f"有価証券報告書が見つかりません: {sec_code}"}
        return self._build_financials_result(sec_code, meta)

    def _build_financials_result(self, sec_code: str, meta: dict) -> dict[str, Any]:
        """書類メタデータ(docID等)からXBRLをダウンロード・解析し、結果をまとめる共通処理。"""
        doc_id = meta["docID"]
        try:
            zip_bytes = self.download_xbrl_zip(doc_id)
        except Exception as e:
            return {"error": f"XBRLダウンロード失敗({doc_id}): {e}"}

        financials = self.parse_financials(zip_bytes)
        shareholders = self.parse_major_shareholders(zip_bytes)
        share_count_series = self.parse_share_count_series(zip_bytes)
        business_results_series = self.parse_business_results_series(zip_bytes)

        return {
            "sec_code": sec_code,
            "company_name": meta.get("filerName"),
            "period_start": meta.get("periodStart"),
            "period_end": meta.get("periodEnd"),
            "doc_id": doc_id,
            **financials,
            "share_count_series": share_count_series,
            "business_results_series": business_results_series,
            "major_shareholders": shareholders,
            "is_owner_managed": _detect_owner_managed(
                meta.get("filerName", ""), shareholders
            ),
        }


# ── ユーティリティ ──────────────────────────────────────────────

def _detect_owner_managed(company_name: str, shareholders: list[dict]) -> bool | None:
    """
    大株主リストに個人名(代表者と思われる人物)が含まれるかを簡易判定する。
    完全な精度は保証しないが、法人名・政府機関名を除外した上で判定する。
    """
    if not shareholders:
        return None
    # 法人・機関投資家っぽいキーワード
    # ㈱=株式会社略称、㈲=有限会社略称、（相）=相互会社
    INSTITUTIONAL_KEYWORDS = [
        "信託", "銀行", "証券", "保険", "年金", "基金", "投資",
        "株式会社", "合同会社", "有限会社", "組合", "協会", "機構",
        "センター", "ファンド", "Trust", "Fund", "Bank", "Group",
        "Holdings", "Capital", "Asset", "日本", "国", "政府",
        "㈱", "㈲", "（相）", "（有）",  # 法人略称
        "バンク", "キャピタル", "マスタートラスト", "カストディ",
    ]
    for s in shareholders[:10]:  # 上位10名を確認
        name = s.get("name", "")
        if not name:
            continue
        is_institutional = any(kw in name for kw in INSTITUTIONAL_KEYWORDS)
        if not is_institutional and len(name) >= 2:
            # 個人名らしい大株主がいる → オーナー経営の可能性
            return True
    return False


# ── CLIから直接テストする場合 ────────────────────────────────────

if __name__ == "__main__":
    import sys

    sec_code = sys.argv[1] if len(sys.argv) > 1 else "7203"
    print(f"=== {sec_code} の有価証券報告書を取得中 ===")
    client = EdinetV2Client()
    result = client.get_financials(sec_code)
    if "error" in result:
        print(f"エラー: {result['error']}")
    else:
        print(f"会社名: {result['company_name']}")
        print(f"対象期間: {result['period_start']} 〜 {result['period_end']}")
        print(f"営業利益: {result.get('operating_profit')}")
        print(f"純利益:   {result.get('net_income')}")
        print(f"営業CF:   {result.get('operating_cf')}")
        print(f"大株主:")
        for s in result.get("major_shareholders", [])[:5]:
            print(f"  {s['name']}  {s['ratio']}%")
        print(f"オーナー経営の可能性: {result.get('is_owner_managed')}")
