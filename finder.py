"""
条件検索(キーワード・市場区分・規模区分・財務条件)で全銘柄から候補を探すロジック。

キーワード・市場区分・規模区分はtickers.csv(JPXのdata_j.xls由来)に最初から
入っているため、API呼び出し無しで全銘柄に即座に適用できる。
財務条件(PER・ROE・増収増益の連続期数)は、これまでに閲覧してディスクキャッシュ
済みの銘柄だけが対象になる(financials_cache.peek_cachedのみを使い、APIは呼ばない)。
"""

import json
from pathlib import Path

import pandas as pd

import analysis
import financials_cache
import growth_scores
import quality_score
import screener

SHARES_OUTSTANDING_CACHE_FILE = Path(__file__).parent / "cache" / "shares_outstanding.json"
DIVIDEND_PER_SHARE_CACHE_FILE = Path(__file__).parent / "cache" / "dividend_per_share.json"
# クラウド環境等、cache/quality/がまるごと無い(=ローカルキャッシュが空の)場合に使う
# フォールバック。ローカルPCがdaily_refresh.pyでbuild_quality_table()の結果を
# 定期生成・GitHubにpushしたスナップショット。
QUALITY_TABLE_SNAPSHOT_FILE = Path(__file__).parent / "quality_table.csv"


def load_master_frame(master: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(master)


def _load_shares_outstanding_cache() -> dict[str, float]:
    """
    yfinance(backfill_shares_outstanding.py)由来の、現在時点の発行済株式数キャッシュを
    読み込む。EDINET由来の株数は、株式分割があった場合に次の有報が出るまで古い値の
    ままになる(分割前の株数×分割後の株価で時価総額が過小評価され、配当利回りや
    PSRなどが実際の何倍にも過大表示されてしまう)ため、こちらを優先して使う。
    """
    if not SHARES_OUTSTANDING_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(SHARES_OUTSTANDING_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_dividend_per_share_cache() -> dict[str, float]:
    """
    yfinance(backfill_dividend_per_share.py)由来の、直近12ヶ月の1株配当合計額
    キャッシュを読み込む。有価証券報告書CF計算書の「配当金の支払額」(連結ベースの
    総支払額。少数株主への配当や現金支払いタイミングのズレが混ざる)より、
    ヤフーファイナンス等が表示する1株配当利回りに近い定義になる。
    バックテスト(backtest_dividend_yield_methods.py)で、既存方式を将来リターンとの
    相関で一度も下回らなかったため、取得できている銘柄はこちらを優先して使う。
    """
    if not DIVIDEND_PER_SHARE_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(DIVIDEND_PER_SHARE_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def keyword_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    """銘柄名・33業種・17業種のいずれかにキーワードが部分一致するか。"""
    if not keyword:
        return pd.Series(True, index=df.index)
    haystack = (
        df.get("name", "").astype(str)
        + " " + df.get("sector33", "").astype(str)
        + " " + df.get("sector17", "").astype(str)
    ).str.lower()
    return haystack.str.contains(keyword.lower(), na=False)


def build_financial_table(codes: list[str]) -> pd.DataFrame:
    """
    キャッシュ済みの銘柄についてのみ財務サマリーをまとめたDataFrameを返す。
    peek_cachedのみを使うため、このスキャン自体はAPIを消費しない。
    """
    rows = []
    for code in codes:
        sec_code = code.split(".")[0]
        records = financials_cache.peek_cached(sec_code)
        if records is None:
            continue
        summary = analysis.summarize(analysis.records_to_frame(records))
        if not summary:
            continue
        summary["code"] = code
        rows.append(summary)
    return pd.DataFrame(rows)


def _outstanding_shares_from_quality_record(
    record: dict, sec_code: str | None = None, shares_outstanding_cache: dict[str, float] | None = None
) -> float | None:
    """
    発行済株式数から自己株式を控除した実質的な流通株式数を返す。
    sec_code・shares_outstanding_cacheを指定した場合、yfinance由来の現在時点の
    株式数キャッシュがあればそちらを優先する(株式分割直後はEDINET側が古いため)。
    """
    if sec_code and shares_outstanding_cache:
        fresh = shares_outstanding_cache.get(sec_code)
        if fresh:
            return fresh

    series = (record.get("share_count_series") or {})
    # JSON経由のキャッシュ読み込みで辞書キーが文字列化される({0: ...} -> {"0": ...})ため、
    # "0"/0どちらのキーでも最新期の値を拾えるようにする。
    shares_issued_series = series.get("shares_issued") or {}
    treasury_shares_series = series.get("treasury_shares") or {}
    shares_issued = shares_issued_series.get(0, shares_issued_series.get("0"))
    treasury_shares = treasury_shares_series.get(0, treasury_shares_series.get("0", 0))
    if not shares_issued:
        return None
    outstanding = shares_issued - (treasury_shares or 0)
    return outstanding if outstanding > 0 else None


def _bps_from_quality_record(
    record: dict, sec_code: str | None = None, shares_outstanding_cache: dict[str, float] | None = None
) -> float | None:
    """純資産(equity)と発行済株式数(自己株式控除後)からBPSを計算する。"""
    equity = record.get("equity")
    outstanding = _outstanding_shares_from_quality_record(record, sec_code, shares_outstanding_cache)
    if not equity or not outstanding:
        return None
    return equity / outstanding


# 時価総額の区分(表示用)。500億/2000億/5000億円を境目にする。
MARKET_CAP_BUCKETS = [
    (0, 500_0000_0000, "〜500億"),
    (500_0000_0000, 2000_0000_0000, "500億〜2000億"),
    (2000_0000_0000, 5000_0000_0000, "2000億〜5000億"),
    (5000_0000_0000, float("inf"), "5000億〜"),
]
MARKET_CAP_BUCKET_LABELS = [label for _, _, label in MARKET_CAP_BUCKETS]


def _market_cap_bucket(market_cap: float | None) -> str | None:
    if market_cap is None:
        return None
    for low, high, label in MARKET_CAP_BUCKETS:
        if low < market_cap <= high:
            return label
    return None


def _latest_close(code: str) -> float | None:
    """価格キャッシュ(cache/prices/*.parquet)から直近の終値を返す(追加取得はしない)。"""
    df = screener._load_cache(code)
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def _load_sector_map() -> dict[str, str]:
    """tickers.csv(銘柄コード→33業種区分)のマッピングを返す。"""
    df = pd.read_csv(screener.TICKERS_FILE, dtype=str)
    return dict(zip(df["code"], df["sector33"]))


def build_quality_table(codes: list[str] | None = None) -> pd.DataFrame:
    """
    品質データ(グロス・プロフィタビリティ・成長の質)がキャッシュ済みの銘柄について、
    PBR(価格キャッシュから計算)も合わせた複合スコアをまとめたDataFrameを返す。

    複合スコアの3要素(GP・PBR・成長の質)は、業種(33業種区分)による水準差が大きいため、
    全銘柄一律ではなく同業種内でのパーセンタイル順位を使う(quality_score.composite_rank参照)。

    codesを指定した場合、複合スコアはcodesの母集団内(業種ごと)での相対順位になる。
    未指定時は品質データがキャッシュ済みの全銘柄が母集団になる(キーワード等の
    絞り込みの有無によって複合スコアの意味合いが変わらないよう、finder側の
    filter_by_qualityでは常に全キャッシュ済み銘柄を母集団として呼び出す)。
    peek_quality_cachedと価格キャッシュの読み込みのみを使うため、
    このスキャン自体はAPIを消費しない。
    """
    if codes is None:
        codes = [f"{c}.T" for c in quality_score.list_quality_cached_codes()]
        if not codes and QUALITY_TABLE_SNAPSHOT_FILE.exists():
            # ローカルの品質データキャッシュが無い(クラウド環境等)場合、
            # ローカルPCが生成・pushしたスナップショットへフォールバックする。
            return pd.read_csv(QUALITY_TABLE_SNAPSHOT_FILE)

    sector_map = _load_sector_map()
    shares_outstanding_cache = _load_shares_outstanding_cache()
    dividend_per_share_cache = _load_dividend_per_share_cache()

    rows = []
    for code in codes:
        sec_code = code.split(".")[0]
        record = quality_score.peek_quality_cached(sec_code)
        if record is None:
            continue
        row = quality_score.build_quality_row(record)
        row["code"] = code
        row["sector33"] = sector_map.get(code)
        bps = _bps_from_quality_record(record, sec_code, shares_outstanding_cache)
        price = _latest_close(code)
        if bps and price:
            row["live_pbr"] = price / bps
        outstanding = _outstanding_shares_from_quality_record(record, sec_code, shares_outstanding_cache)
        market_cap = None
        if outstanding and price:
            market_cap = outstanding * price
            row["market_cap"] = market_cap
            row["market_cap_bucket"] = _market_cap_bucket(market_cap)

        # Growth Quality Score(GQS)・Growth Potential Score(GPS)
        # (既存の複合スコアとは別枠の成長株評価システム。growth_scores.py参照)
        gqs = growth_scores.growth_quality_score(
            growth_scores.gp_score(row.get("gross_profitability")),
            growth_scores.gross_profit_growth_score(row.get("gp_growth_rate")),
            growth_scores.fcf_score(row.get("fcf_margin")),
            growth_scores.stability_score(row.get("revenue_streak")),
        )
        row["gqs"] = gqs
        net_income_growth_pct = (
            row["net_income_growth_latest"] * 100
            if row.get("net_income_growth_latest") is not None else None
        )
        # 配当利回り(%)。yfinance由来の直近12ヶ月の1株配当合計(ヤフーファイナンス等が
        # 表示する定義に近く、バックテストで既存方式を一度も下回らなかった)が取れて
        # いればそちらを優先し、無ければ有価証券報告書CF計算書の「配当金の支払額」
        # (連結ベースの総支払額の近似値)にフォールバックする。
        per_share_dividend = dividend_per_share_cache.get(sec_code)
        if per_share_dividend is not None and price:
            dividend_yield_pct = per_share_dividend / price * 100
        else:
            dividend_yield_pct = growth_scores.dividend_yield(record.get("dividends_paid_cf"), market_cap)
        # リンチレシオ(PER÷(純利益成長率%+配当利回り%))。バックテストでPEG
        # (増益率のみ)より将来リターンとの相関が同等以上だったため採用している。
        row["dividend_yield_pct"] = dividend_yield_pct
        # 配当性向(%) = |配当支払額| ÷ 純利益 × 100。純利益が赤字/ゼロの場合は計算不能(None)。
        # バックテストの結果、DCR法(早復型ゴールデンクロス×配当利回り)の対象銘柄では
        # 配当性向50%未満のほうが以降のパフォーマンスが良い傾向を確認済み
        # (backtest_dcr_payout_ratio.py参照)。
        row["payout_ratio_pct"] = None
        net_income_latest = row.get("net_income_latest")
        dividends_paid_cf = record.get("dividends_paid_cf")
        if net_income_latest and net_income_latest > 0 and dividends_paid_cf is not None:
            row["payout_ratio_pct"] = abs(dividends_paid_cf) / net_income_latest * 100
        # 前年からの増配率(支払配当金の増減率。過去1年分の有報(cache/quality_historical)と比較)
        row["dividend_growth_pct"] = None
        prior_record = quality_score.peek_historical_cached(sec_code, 1)
        if prior_record is not None:
            cur_div = record.get("dividends_paid_cf")
            prior_div = prior_record.get("dividends_paid_cf")
            if cur_div and prior_div:
                row["dividend_growth_pct"] = (abs(cur_div) / abs(prior_div) - 1) * 100
        lynch = growth_scores.lynch_ratio(row.get("latest_per"), net_income_growth_pct, dividend_yield_pct)
        row["lynch_ratio"] = lynch
        # PSR成長対比(リンチレシオが赤字・大幅減益企業で計算不能な時の代替。
        # GPSがNoneになる最大要因だったこの穴を埋める)
        revenue_growth_pct = (
            row["revenue_growth_latest"] * 100
            if row.get("revenue_growth_latest") is not None else None
        )
        psr_value = growth_scores.psr(market_cap, row.get("net_sales_latest"))
        row["psr"] = psr_value
        psr_growth = growth_scores.psr_growth_ratio(psr_value, revenue_growth_pct)
        row["psr_growth_ratio"] = psr_growth
        fcf_yield_value = growth_scores.fcf_yield(row.get("fcf"), market_cap)
        row["fcf_yield"] = fcf_yield_value
        row["gps"] = growth_scores.growth_potential_score(
            growth_scores.lynch_score(lynch),
            growth_scores.fcf_yield_score(fcf_yield_value),
            growth_scores.psr_growth_score(psr_growth),
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return quality_score.composite_rank(pd.DataFrame(rows))


def add_price_relative_scores(quality_table: pd.DataFrame, screener_result: pd.DataFrame) -> pd.DataFrame:
    """
    build_quality_table()の結果(GQS・GPSまで計算済み)に、screener.screen()の
    結果(価格ベースのモメンタム・トレンド・業績補正)を合わせてPRS・GPRS・評価
    (S/A/B/C/D)を追加する(Growth & Price Relative Score、growth_scores.py参照)。
    GPRS(60点満点)=GPS+PRS+リスク調整。GQSはバックテストで将来リターンとの相関が
    無かったため合計には含めず、質のスクリーニング用の参考情報として別列(gqs)を
    残すのみにしている。
    """
    if quality_table.empty or screener_result.empty:
        return pd.DataFrame()

    # screener_result(現在選択中の銘柄)を主として左結合する。quality_tableは
    # 全銘柄分(数千件)持っていることが多く、選択中の銘柄だけに絞るため。
    price_cols = screener_result[
        ["code", "name", "close", "prior_52w_high", "above_ma200", "ma200_rising"]
    ]
    merged = price_cols.merge(quality_table, on="code", how="left", suffixes=("", "_quality"))

    def _row_scores(row: pd.Series) -> pd.Series:
        momentum = growth_scores.momentum_score(row.get("close"), row.get("prior_52w_high"))
        trend = growth_scores.trend_direction_score(row.get("above_ma200"), row.get("ma200_rising"))
        performance = growth_scores.performance_momentum_score(
            row.get("revenue_growth_latest"),
            row.get("revenue_growth_prior"),
            row.get("ordinary_income_margin_latest"),
            row.get("ordinary_income_margin_prior"),
        )
        prs = growth_scores.price_relative_score(momentum, trend, performance)
        risk_adj = growth_scores.risk_adjustment(
            row.get("latest_equity_ratio"), row.get("growth_quality_raw"), row.get("latest_per"),
            row.get("deficit_years_last_5y"),
        )
        gprs = growth_scores.growth_price_relative_score(row.get("gps"), prs, risk_adj)
        return pd.Series({
            "momentum_score": momentum,
            "trend_score": trend,
            "performance_score": performance,
            "prs": prs,
            "risk_adjustment": risk_adj,
            "gprs": gprs,
            "gprs_grade": growth_scores.gprs_grade(gprs),
        })

    scores = merged.apply(_row_scores, axis=1)
    result = pd.concat([merged, scores], axis=1)

    # GPRSは業種(資産集約度等)によって偏りが出やすいため、絶対基準による判定
    # (GPRS・評価)自体は変えずに、「同業種内で何位か」を参考情報として別列で
    # 追加する(業種は現在価格データがある銘柄の範囲内での相対比較。GPRSが
    # 判定不能(None)の銘柄は順位の対象外にする)。
    if "sector33" in result.columns:
        gprs_for_rank = result["gprs"].where(result["gprs"].notna())
        result["gprs_sector_rank"] = (
            gprs_for_rank.groupby(result["sector33"]).rank(ascending=False, method="min")
        )
        result["gprs_sector_size"] = (
            gprs_for_rank.groupby(result["sector33"]).transform("count")
        )
    else:
        result["gprs_sector_rank"] = pd.NA
        result["gprs_sector_size"] = pd.NA

    return result


def filter_by_quality(
    master: list[dict],
    keyword: str = "",
    markets: list[str] | None = None,
    size_categories: list[str] | None = None,
    min_composite_score: float | None = None,
    min_gross_profitability_pct: float | None = None,
    max_pbr: float | None = None,
    min_growth_quality_pct: float | None = None,
) -> pd.DataFrame:
    """
    キーワード・市場区分・規模区分で絞り込んだ上で、グロス・プロフィタビリティ×
    割安さ(PBR)×成長の質の複合スコアでランキングする。品質データが未キャッシュの
    銘柄は結果から除外される(バックフィル未実行の場合は空になる)。
    """
    df = load_master_frame(master)
    mask = keyword_mask(df, keyword)
    if markets:
        mask &= df["market"].isin(markets)
    if size_categories:
        mask &= df["size_category"].isin(size_categories)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        return df

    quality_table = build_quality_table()
    if quality_table.empty:
        return quality_table

    result = df.merge(quality_table, on="code", how="inner")
    if min_composite_score is not None:
        result = result[result["composite_score"] >= min_composite_score]
    if min_gross_profitability_pct is not None:
        result = result[result["gross_profitability"] * 100 >= min_gross_profitability_pct]
    if max_pbr is not None and "live_pbr" in result.columns:
        result = result[result["live_pbr"] <= max_pbr]
    if min_growth_quality_pct is not None:
        result = result[result["growth_quality_raw"] * 100 >= min_growth_quality_pct]

    return result.sort_values("composite_score", ascending=False, na_position="last").reset_index(drop=True)


def filter_candidates(
    master: list[dict],
    keyword: str = "",
    markets: list[str] | None = None,
    size_categories: list[str] | None = None,
    max_per: float | None = None,
    min_roe_pct: float | None = None,
    min_revenue_streak: int = 0,
    min_profit_streak: int = 0,
) -> pd.DataFrame:
    """
    各条件はAND結合。financial系の条件(max_per/min_roe_pct/streak)が
    どれか1つでも指定されている場合のみ、財務データキャッシュとの照合を行う
    (未キャッシュの銘柄は結果から除外される)。

    後方互換のために残しているが、財務指標・品質指標を同時にかけたい場合は
    filter_combined()を使うこと。
    """
    df = load_master_frame(master)
    mask = keyword_mask(df, keyword)
    if markets:
        mask &= df["market"].isin(markets)
    if size_categories:
        mask &= df["size_category"].isin(size_categories)
    df = df[mask].reset_index(drop=True)

    use_financial = any(
        v not in (None, 0) for v in (max_per, min_roe_pct, min_revenue_streak, min_profit_streak)
    )
    if not use_financial or df.empty:
        return df

    fin_table = build_financial_table(df["code"].tolist())
    if fin_table.empty:
        return df.iloc[0:0]

    df = df.merge(fin_table, on="code", how="inner")
    if max_per is not None and "latest_per" in df.columns:
        df = df[df["latest_per"] <= max_per]
    if min_roe_pct is not None and "latest_roe_official" in df.columns:
        df = df[df["latest_roe_official"] * 100 >= min_roe_pct]
    if min_revenue_streak:
        df = df[df.get("revenue_streak", 0) >= min_revenue_streak]
    if min_profit_streak:
        df = df[df.get("net_income_streak", 0) >= min_profit_streak]

    return df.reset_index(drop=True)


def filter_combined(
    master: list[dict],
    keyword: str = "",
    markets: list[str] | None = None,
    size_categories: list[str] | None = None,
    market_cap_buckets: list[str] | None = None,
    max_per: float | None = None,
    min_roe_pct: float | None = None,
    min_revenue_streak: int = 0,
    min_profit_streak: int = 0,
    min_composite_score: float | None = None,
    min_gross_profitability_pct: float | None = None,
    max_pbr: float | None = None,
    min_growth_quality_pct: float | None = None,
) -> pd.DataFrame:
    """
    キーワード・市場区分・規模区分(またはmarket_cap_buckets指定時は時価総額区分)で
    絞り込んだ上で、財務指標(PER・ROE・増収増益、financials_cache/edinetdb.jp由来)と
    品質・割安・成長(グロス・プロフィタビリティ・PBR・成長の質・複合スコア、
    quality_score/EDINET API v2由来)の両方を同時にAND条件でかけられるようにしたもの。
    「🔍 銘柄スクリーニング」の統合ロジック。

    どちらのテーブルも常に左結合で付与するため、フィルタを何も指定しなくても
    表示用に取得できる範囲の財務指標・品質指標が列として付いてくる
    (未キャッシュの銘柄はその列がNaN/空欄になるだけで、結果から除外はされない)。
    実際に閾値を指定したフィルタだけが、該当列がNaN(=未キャッシュ)の行を除外する。
    market_cap_bucketsは品質データキャッシュ由来(発行済株式数×直近終値)のため、
    未キャッシュの銘柄は指定時に結果から除外される。
    """
    df = load_master_frame(master)
    mask = keyword_mask(df, keyword)
    if markets:
        mask &= df["market"].isin(markets)
    if size_categories:
        mask &= df["size_category"].isin(size_categories)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        return df

    fin_table = build_financial_table(df["code"].tolist())
    if not fin_table.empty:
        df = df.merge(fin_table, on="code", how="left")

    quality_table = build_quality_table()
    if not quality_table.empty:
        df = df.merge(quality_table, on="code", how="left", suffixes=("", "_quality"))
        # PER・ROE・増収増益連続期数は、financials_cache(edinetdb.jp、閲覧した
        # 銘柄のみのオポチュニスティックキャッシュ)とquality_table(EDINET API v2、
        # 全銘柄バックフィル済み)の両方が同じ列名を持ちうる。全銘柄をカバーする
        # quality_table側の値を優先し、無ければfinancials_cache側の値で補完する。
        for col in ("latest_per", "latest_roe_official", "revenue_streak", "net_income_streak"):
            q_col = f"{col}_quality"
            if q_col not in df.columns:
                continue
            df[col] = df[q_col].combine_first(df[col]) if col in df.columns else df[q_col]
            df = df.drop(columns=[q_col])

    if market_cap_buckets:
        if "market_cap_bucket" not in df.columns:
            return df.iloc[0:0]
        df = df[df["market_cap_bucket"].isin(market_cap_buckets)]

    if max_per is not None:
        if "latest_per" not in df.columns:
            return df.iloc[0:0]
        df = df[df["latest_per"].notna() & (df["latest_per"] <= max_per)]
    if min_roe_pct is not None:
        if "latest_roe_official" not in df.columns:
            return df.iloc[0:0]
        df = df[df["latest_roe_official"].notna() & (df["latest_roe_official"] * 100 >= min_roe_pct)]
    if min_revenue_streak:
        if "revenue_streak" not in df.columns:
            return df.iloc[0:0]
        df = df[df["revenue_streak"].fillna(0) >= min_revenue_streak]
    if min_profit_streak:
        if "net_income_streak" not in df.columns:
            return df.iloc[0:0]
        df = df[df["net_income_streak"].fillna(0) >= min_profit_streak]

    if min_composite_score is not None:
        if "composite_score" not in df.columns:
            return df.iloc[0:0]
        df = df[df["composite_score"].notna() & (df["composite_score"] >= min_composite_score)]
    if min_gross_profitability_pct is not None:
        if "gross_profitability" not in df.columns:
            return df.iloc[0:0]
        df = df[df["gross_profitability"].notna() & (df["gross_profitability"] * 100 >= min_gross_profitability_pct)]
    if max_pbr is not None:
        if "live_pbr" not in df.columns:
            return df.iloc[0:0]
        df = df[df["live_pbr"].notna() & (df["live_pbr"] <= max_pbr)]
    if min_growth_quality_pct is not None:
        if "growth_quality_raw" not in df.columns:
            return df.iloc[0:0]
        df = df[df["growth_quality_raw"].notna() & (df["growth_quality_raw"] * 100 >= min_growth_quality_pct)]

    if df.empty:
        return df.reset_index(drop=True)

    if "composite_score" in df.columns and df["composite_score"].notna().any():
        df = df.sort_values("composite_score", ascending=False, na_position="last")
    else:
        df = df.sort_values("code")
    return df.reset_index(drop=True)
