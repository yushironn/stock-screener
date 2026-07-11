"""
「グロス・プロフィタビリティ×割安さ×成長の質」複合スクリーニングのコアロジック。

edinet_v2.get_financials()が返す1社分の生データ(dict)を受け取り、
- グロス・プロフィタビリティ (売上総利益 ÷ 総資産)
- 成長の質 (増資・借入に頼らず、自己資金で株主還元しながら成長できているか)
- 発行済株式数の推移 (自社株買い/希薄化の判定)
を計算し、複数銘柄分をまとめてパーセンタイル順位ベースの複合スコアに変換する。

参考(会話で確認した学術的根拠):
- グロス・プロフィタビリティ: Novy-Marx (2013) "The Other Side of Value"。
  PBR(割安さ)と負の相関(-0.48〜-0.50)を持つため、組み合わせることで
  「割安なのに中身も良い」銘柄を選別できる。
- 成長の質: 純発行株式数の増加(希薄化)は将来リターンに対して頑健な負の
  予測力を持つ一方、自己資金(営業CF)で賄われた成長は評価が高い。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import growth_scores

CACHE_DIR = Path(__file__).parent / "cache" / "quality"
DEFAULT_MAX_AGE_DAYS = 180  # 財務データは決算ごと(年1回程度)しか更新されないため


def _cache_path(sec_code: str) -> Path:
    return CACHE_DIR / f"{sec_code}.json"


def save_quality_record(sec_code: str, record: dict) -> None:
    """1社分の生データ(edinet_v2.get_financialsの戻り値)をディスクキャッシュに保存する。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now().isoformat(), "record": record}
    _cache_path(sec_code).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def is_quality_cache_fresh(sec_code: str, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
    """キャッシュが存在し、max_age_days以内に取得されたものかを返す。"""
    path = _cache_path(sec_code)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    fetched_at = payload.get("fetched_at")
    if not fetched_at:
        return False
    age_days = (date.today() - datetime.fromisoformat(fetched_at).date()).days
    return age_days < max_age_days


def peek_quality_cached(sec_code: str) -> dict | None:
    """
    ディスクキャッシュの内容を新鮮さに関わらずそのまま返す(無ければNone)。
    APIは一切呼ばないため、全銘柄をスキャンするランキング表示などで
    意図せずAPIクォータを消費したくない場合に使う。
    """
    path = _cache_path(sec_code)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("record")


def list_quality_cached_codes() -> set[str]:
    """品質スコア用データがキャッシュ済みの証券コード(4桁)一覧を返す。"""
    if not CACHE_DIR.exists():
        return set()
    return {p.stem for p in CACHE_DIR.glob("*.json")}


# ── 過去時点(GQSバックテスト用)の有報キャッシュ ──────────────────
# 直近有報1通だけでは売上原価・営業CF・CapExが当期・前期の2年分しか無く、
# 2年以上前を基準にしたGQS(粗利成長・FCF)を再現できないため、
# 「N年前時点で最新だった有報」自体を別途キャッシュする。
# 過去の事実は時間が経っても変わらないため、鮮度チェックは行わない。
HISTORICAL_CACHE_DIR = CACHE_DIR.parent / "quality_historical"


def _historical_cache_path(sec_code: str, years_ago: int) -> Path:
    return HISTORICAL_CACHE_DIR / f"{sec_code}_{years_ago}.json"


def save_historical_record(sec_code: str, years_ago: int, record: dict) -> None:
    HISTORICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _historical_cache_path(sec_code, years_ago).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )


def is_historical_cache_present(sec_code: str, years_ago: int) -> bool:
    return _historical_cache_path(sec_code, years_ago).exists()


def peek_historical_cached(sec_code: str, years_ago: int) -> dict | None:
    path = _historical_cache_path(sec_code, years_ago)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_freshness_summary() -> dict | None:
    """
    品質データキャッシュ(cache/quality/*.json)の更新日時の要約を返す。
    ファイルの更新日時(mtime、save_quality_record実行時に書き込まれる)を使うため、
    全ファイルの中身(JSON)を読まずに済み軽量。
    戻り値: {"count": 件数, "latest": 最新の更新日時, "oldest": 最古の更新日時}
    """
    if not CACHE_DIR.exists():
        return None
    files = list(CACHE_DIR.glob("*.json"))
    if not files:
        return None
    mtimes = [f.stat().st_mtime for f in files]
    return {
        "count": len(files),
        "latest": datetime.fromtimestamp(max(mtimes)),
        "oldest": datetime.fromtimestamp(min(mtimes)),
    }


_FINANCING_CF_BREAKDOWN_FIELDS = (
    "dividends_paid_cf", "treasury_stock_cf", "proceeds_share_issuance_cf",
    "proceeds_long_term_debt_cf", "payments_long_term_debt_cf", "short_term_debt_change_cf",
)


def is_quality_record_complete(record: dict) -> bool:
    """
    キャッシュ済みレコードが「成長の質」を計算可能な内容を含んでいるかを判定する。

    XBRL_TARGETSの候補タグ名を拡張する前に取得されたレコードは、financing_cf
    (財務活動によるCF合計)はあっても内訳が全てNoneのままキャッシュされている
    ことがある。年齢だけで新鮮さを判定すると、この「スキーマが古い」レコードを
    誤って再利用してしまうため、内訳が1つも無いのに合計値だけはある場合は
    不完全とみなし、再取得の対象にする。
    """
    if "error" in record:
        return False
    if record.get("financing_cf") is not None:
        if not any(record.get(f) is not None for f in _FINANCING_CF_BREAKDOWN_FIELDS):
            return False

    # business_results_series(PER・ROE・増収増益連続期数の元データ)を追加する前に
    # 取得されたレコードにはこのキー自体が無いため、古いスキーマとして再取得対象にする。
    # net_sales/roeは有価証券報告書なら必ず開示される項目なので、これらが両方とも
    # 空の場合は「まだ対応前のスキーマ」とみなせる(会社側の未開示ではない)。
    business = record.get("business_results_series")
    if not business or not (business.get("net_sales") or business.get("roe")):
        return False

    # cost_of_sales_2y/operating_cf_2y/capex_2y(Growth Quality Score用)を追加する前に
    # 取得されたレコードには、このキー自体が business_results_series に無い
    # (parse_business_results_seriesは対象タグ全てにキーを作るため、キーの有無で
    # 新旧スキーマを判定できる。値が空dictでも「開示が無かった」だけなのでOK)。
    if "cost_of_sales_2y" not in business:
        return False

    return True


def gross_profitability(record: dict) -> float | None:
    """
    グロス・プロフィタビリティ = 売上総利益 ÷ 総資産。

    売上高・売上原価が両方取れる場合はその差分(同一連結範囲で整合しやすい)を
    優先し、片方でも欠けている場合のみ直接開示された`gross_profit`タグに
    フォールバックする(直接タグは非連結ベースなど範囲が異なる開示との
    整合性が取れないケースがあるため、優先度を一段落とす)。
    """
    total_assets = record.get("total_assets")
    if not total_assets:
        return None

    net_sales = record.get("net_sales")
    cost_of_sales = record.get("cost_of_sales")
    if net_sales is not None and cost_of_sales is not None:
        gp = net_sales - cost_of_sales
    else:
        gp = record.get("gross_profit")
        if gp is None:
            return None

    return gp / total_assets


def growth_quality_raw(record: dict) -> float | None:
    """
    「成長の質」の生スコア = (株主還元・借入返済等で出ていった資金 − 増資・
    新規借入で入ってきた資金) ÷ 総資産。

    プラスが大きいほど、増資や借入に頼らず自己資金(営業CF)で株主還元しながら
    成長できていることを示す。財務活動によるキャッシュフローの符号をそのまま
    使わないのは、財務CFのマイナスが「単に配当性向が高いだけ」なのか
    「増資に依存せず借入も返済しつつ株主還元もできている」のかを区別する
    ため、内訳(配当・自己株買い・起債・借入)を個別に見て組み立てる。
    """
    total_assets = record.get("total_assets")
    if not total_assets:
        return None

    # 株主還元・返済(通常マイナス値で開示される) = 資金の流出
    outflow_items = [
        record.get("dividends_paid_cf"),
        record.get("treasury_stock_cf"),
        record.get("payments_long_term_debt_cf"),
    ]
    # 資金調達(通常プラス値で開示される) = 資金の流入
    inflow_items = [
        record.get("proceeds_share_issuance_cf"),
        record.get("proceeds_long_term_debt_cf"),
        record.get("short_term_debt_change_cf"),
    ]

    available = [v for v in outflow_items + inflow_items if v is not None]
    if not available:
        return None

    outflow = sum(v for v in outflow_items if v is not None)
    inflow = sum(v for v in inflow_items if v is not None)

    # 出ていった資金はマイナス値、入ってきた資金はプラス値で開示されるため、
    # 「(-outflow) - inflow」が「自己資金で還元・返済した度合い」を表す。
    net_self_funded = (-outflow) - inflow
    return net_self_funded / total_assets


def _year_series(raw_series: dict | None) -> dict[int, float]:
    """
    JSON経由でキャッシュを読み書きすると辞書のキーが文字列化される
    ({0: ...} -> {"0": ...})ため、int化してから扱う共通ヘルパー。
    """
    if not raw_series:
        return {}
    return {int(k): v for k, v in raw_series.items() if v is not None}


def latest_from_series(raw_series: dict | None) -> float | None:
    """business_results_series内の1項目から、最新期(years_ago=0)の値を返す。"""
    return _year_series(raw_series).get(0)


def growth_streak_from_series(raw_series: dict | None) -> int:
    """
    直近期から遡って、前期比で増加(YoY > 0)が何期連続しているかを返す
    (analysis.growth_streakのbusiness_results_series版)。
    """
    series = _year_series(raw_series)
    streak = 0
    for years_ago in sorted(series.keys()):
        prev_years_ago = years_ago + 1
        if prev_years_ago not in series:
            break
        if series[years_ago] > series[prev_years_ago]:
            streak += 1
        else:
            break
    return streak


def shares_issued_cagr(share_count_series: dict | None) -> float | None:
    """
    発行済株式数(自己株式含む)の年平均変化率(CAGR)。マイナスであるほど
    自社株買い等で株式数が減っていることを示す(プラスは希薄化)。
    直近5期のうち、取得できた最も古い期と最新期を比較する。
    """
    if not share_count_series:
        return None
    raw_series = share_count_series.get("shares_issued")
    if not raw_series or len(raw_series) < 2:
        return None

    # JSON経由でキャッシュを読み書きすると辞書のキーが文字列化される
    # ({0: ...} -> {"0": ...})ため、int化してから扱う。
    series = {int(k): v for k, v in raw_series.items()}

    years_ago_available = sorted(series.keys())
    oldest = years_ago_available[-1]
    latest_value = series.get(0, series[years_ago_available[0]])
    oldest_value = series[oldest]
    if not latest_value or not oldest_value or oldest == 0:
        return None

    return (latest_value / oldest_value) ** (1 / oldest) - 1


def build_quality_row(record: dict) -> dict:
    """1社分の生データから、複合スコア計算に必要な行(dict)を組み立てる。"""
    business = record.get("business_results_series") or {}
    # JSON経由のキャッシュ読み込みで各シリーズの年キーが文字列化される
    # ({0: ...} -> {"0": ...})ため、_year_seriesでint化してから扱う。
    net_sales_series = _year_series(business.get("net_sales"))
    net_income_series = _year_series(business.get("net_income"))
    ordinary_income_series = _year_series(business.get("ordinary_income"))
    cost_of_sales_2y = _year_series(business.get("cost_of_sales_2y"))
    operating_cf_2y = _year_series(business.get("operating_cf_2y"))
    capex_2y = _year_series(business.get("capex_2y"))

    # Growth Quality Score(GQS)・Growth Potential Score(GPS)用の中間値。
    # ここではEDINETデータだけで計算できるものだけを算出し、市場価格が必要な
    # 項目(FCF利回り・モメンタム・トレンド)はfinder.py側で価格キャッシュと
    # 合わせて計算する。
    gp_growth_rate = growth_scores.gross_profit_growth_rate(net_sales_series, cost_of_sales_2y)
    fcf = growth_scores.free_cash_flow(operating_cf_2y, capex_2y)
    net_sales_latest = net_sales_series.get(0)
    net_sales_prior = net_sales_series.get(1)

    # 「業績補正」代理指標の一部(経常利益率のトレンド)用
    ordinary_income_latest = ordinary_income_series.get(0) if ordinary_income_series else None
    ordinary_income_prior = ordinary_income_series.get(1) if ordinary_income_series else None
    margin_latest = growth_scores.ordinary_income_margin(ordinary_income_latest, net_sales_latest)
    margin_prior = growth_scores.ordinary_income_margin(ordinary_income_prior, net_sales_prior)

    return {
        "sec_code": record.get("sec_code"),
        "company_name": record.get("company_name"),
        "gross_profitability": gross_profitability(record),
        "growth_quality_raw": growth_quality_raw(record),
        "shares_issued_cagr_pct": (
            cagr * 100 if (cagr := shares_issued_cagr(record.get("share_count_series"))) is not None else None
        ),
        "period_end": record.get("period_end"),
        # PER・ROE・増収増益連続期数(EDINET API v2の5年サマリー表由来。
        # edinetdb.jp(1日100回制限)を使わずに全銘柄分キャッシュできる)
        "latest_per": latest_from_series(business.get("per")),
        "latest_roe_official": latest_from_series(business.get("roe")),
        "latest_equity_ratio": latest_from_series(business.get("equity_ratio")),
        "revenue_streak": growth_streak_from_series(net_sales_series),
        "net_income_streak": growth_streak_from_series(net_income_series),
        # 直近期の純利益(赤字判定用。リンチレシオが計算不能な理由がデータ欠損なのか
        # 赤字によるものなのかを表示側で区別するために使う)
        "net_income_latest": net_income_series.get(0),
        # 過去5期(5年サマリー表の範囲、今期含む)のうち純利益が赤字だった期数。
        # リスク調整の減点(投資対象として避けたい赤字企業を明確に減点する)に使う。
        # データが取得できている年だけを対象にする(未取得年を赤字扱いにはしない)。
        "deficit_years_last_5y": sum(1 for v in net_income_series.values() if v <= 0),
        # Growth Quality Score(GQS)・Growth Potential Score(GPS)用の中間値
        "gp_growth_rate": gp_growth_rate,
        "fcf": fcf,
        "fcf_margin": growth_scores.fcf_margin(fcf, net_sales_latest),
        # PSR(株価売上高倍率)計算用(finder.py側で時価総額と合わせて使う)
        "net_sales_latest": net_sales_latest,
        "revenue_growth_latest": growth_scores.yoy_growth_rate(net_sales_series, 0),
        "revenue_growth_prior": growth_scores.yoy_growth_rate(net_sales_series, 1),
        "net_income_growth_latest": growth_scores.yoy_growth_rate(net_income_series, 0),
        "ordinary_income_margin_latest": margin_latest,
        "ordinary_income_margin_prior": margin_prior,
    }


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    有効な値のみを対象に0〜100のパーセンタイル順位を付ける(NaNはNaNのまま)。
    higher_is_better=Falseの場合は順位を反転する(値が低いほど高スコア)。
    """
    ranked = series.rank(pct=True, na_option="keep") * 100
    return ranked if higher_is_better else (100 - ranked)


# 業種内の該当銘柄数がこれ未満の場合、業種中央値(補正の基準)の算出根拠が薄くなる
# (例えば5銘柄しかない業種では中央値が数社の水準でほぼ決まってしまう)ため、
# 複合スコアに「※」を付けて注記する目安値。
MIN_SECTOR_SAMPLE_SIZE = 10

# PBRが極端に低い銘柄は「本当に割安な優良株」ではなく、市場が簿価の実現可能性
# (減損リスクや業績悪化)を疑っている「バリュートラップ」の可能性がある。
# このためPBR_PENALTY_THRESHOLDを下回った分は連続的に評価を下げる(PBRが低いほど
# ペナルティが強まる)。負のPBR(債務超過)は別途除外しているため、ここでの対象は
# 0<PBR<しきい値の範囲。
PBR_PENALTY_THRESHOLD = 0.5
PBR_PENALTY_MULTIPLIER = 2.0  # しきい値を下回った分(shortfall)に掛ける倍率


def _apply_pbr_penalty(pbr: pd.Series) -> pd.Series:
    """
    PBRがPBR_PENALTY_THRESHOLD(既定0.5)を下回るほど連続的に評価を下げるための変換。
    しきい値以上はそのまま。しきい値未満は「下回った分×PBR_PENALTY_MULTIPLIER」を
    足し戻すことで、実際の値より割安ではない(=順位が下がる)ものとして扱う。
    0.5からの乖離が大きいほどペナルティが強まる連続関数になっている。
    """
    shortfall = (PBR_PENALTY_THRESHOLD - pbr).clip(lower=0)
    return pbr + shortfall * PBR_PENALTY_MULTIPLIER


def _sector_adjust(series: pd.Series, sector: pd.Series, method: str) -> pd.Series:
    """
    業種(sector)の中央値を基準に値を補正する。
    method="subtract": 値-業種中央値(GP・成長の質など、総資産比%のような加法的な指標向け)
    method="divide": 値÷業種中央値(PBRなど、倍率で語られる指標向け。業種中央値が0以下だと
    比率として意味を持たないためNaN扱いにする)
    """
    sector_median = series.groupby(sector).transform("median")
    if method == "divide":
        return series / sector_median.where(sector_median > 0)
    return series - sector_median


def composite_rank(
    df: pd.DataFrame,
    sector_col: str = "sector33",
    live_pbr_col: str = "live_pbr",
    weights: dict[str, float] | None = None,
    min_sector_sample_size: int = MIN_SECTOR_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    グロス・プロフィタビリティ・PBR(低いほど良い)・成長の質の3つに、同じ業種
    (sector_col、既定は33業種区分)の中央値を基準にした業種補正をかけたうえで、
    補正後の値を全銘柄まとめてパーセンタイル順位(0-100)に変換し、
    加重平均した複合スコア列を追加する(業種内だけでの順位付けではなく、
    業種補正後の値を全銘柄で比較する)。

    業種補正の方法:
    - GP・成長の質: 業種中央値との差分(引き算)。総資産比%の指標のため。
    - PBR: 業種中央値との比率(割り算)。PBRは「同業他社の何倍か」という倍率の
      指標のため、引き算より割り算の方が自然。

    GP・PBRは業種による水準差が大きい(資産集約度や許容PBRが業種で大きく異なる)ため、
    この業種補正によって「同業種内での相対的な良し悪し」を保ったまま、
    全銘柄で横断比較できるようにしている。
    sector_colがdfに無い場合は補正をかけず、従来通り全銘柄一律のパーセンタイル順位になる。

    GP・PBR・成長の質の3要素のうちどれか1つでも値が欠損している銘柄は、
    複合スコアがNone(判定不能)になる(一部の要素だけの平均では計算しない)。
    PBRがマイナス(BPSがマイナス=債務超過)の銘柄は、割安さの判定に使えないため
    欠損扱いにする(マイナス値を「最も割安」と誤判定しないようにするため)。
    PBRがPBR_PENALTY_THRESHOLD(既定0.5)を下回る銘柄は、バリュートラップ
    (本当に優良で割安なのではなく、市場が簿価の実現可能性を疑っている状態)の
    懸念があるため、下回った分に応じて連続的にペナルティをかける
    (_apply_pbr_penalty参照)。

    業種内の該当銘柄数がmin_sector_sample_size未満の場合、"composite_score_low_sample"列が
    Trueになる(業種中央値の算出根拠が薄いことを示すフラグで、値自体は除外しない)。
    """
    weights = weights or {"gross_profitability": 1.0, "value": 1.0, "growth_quality": 1.0}
    result = df.copy()

    # PBRがマイナス(=BPSがマイナス=債務超過)の銘柄は「割安」ではなく財務的な危険信号であり、
    # 「低いほど良い」の順位付けにそのまま使うと(マイナス値が最小値になるため)誤って
    # 「最も割安」と判定されてしまう。そのためPBR算出には使わず欠損(NaN)として扱う。
    # 0<PBR<PBR_PENALTY_THRESHOLDの範囲は、バリュートラップ懸念に対する連続的な
    # ペナルティ(_apply_pbr_penalty)をかけてから順位付けに使う。
    if live_pbr_col in result.columns:
        pbr_raw = _apply_pbr_penalty(result[live_pbr_col].where(result[live_pbr_col] > 0))
    else:
        pbr_raw = None

    has_sector = sector_col in result.columns
    if has_sector:
        sector = result[sector_col]
        gp_input = _sector_adjust(result["gross_profitability"], sector, method="subtract")
        growth_input = _sector_adjust(result["growth_quality_raw"], sector, method="subtract")
        pbr_input = _sector_adjust(pbr_raw, sector, method="divide") if pbr_raw is not None else None
        sector_counts = sector.groupby(sector).transform("count")
        result["composite_score_low_sample"] = sector_counts < min_sector_sample_size
    else:
        gp_input = result["gross_profitability"]
        growth_input = result["growth_quality_raw"]
        pbr_input = pbr_raw
        result["composite_score_low_sample"] = False

    result["_rank_gp"] = _percentile_rank(gp_input, higher_is_better=True)
    result["_rank_value"] = (
        _percentile_rank(pbr_input, higher_is_better=False) if pbr_input is not None else pd.NA
    )
    result["_rank_growth"] = _percentile_rank(growth_input, higher_is_better=True)

    rank_cols = {
        "_rank_gp": weights.get("gross_profitability", 1.0),
        "_rank_value": weights.get("value", 1.0),
        "_rank_growth": weights.get("growth_quality", 1.0),
    }

    def _weighted_avg(row: pd.Series) -> float | None:
        # 3要素(GP・PBR・成長の質)のうちどれか1つでも欠落している場合は、
        # 複合スコアを「判定不能」として扱う(一部の要素だけの平均で計算しない)。
        if any(pd.isna(row[col]) for col in rank_cols):
            return None
        total = sum(row[col] * weight for col, weight in rank_cols.items())
        total_weight = sum(rank_cols.values())
        return total / total_weight if total_weight else None

    result["composite_score"] = result.apply(_weighted_avg, axis=1)
    result = result.drop(columns=["_rank_gp", "_rank_value", "_rank_growth"])
    return result.sort_values("composite_score", ascending=False, na_position="last").reset_index(drop=True)
