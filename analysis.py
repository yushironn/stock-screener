"""
財務分析のコアロジック。

EDINET DB(edinetdb.get_financials)が返す会計年度ごとの財務レコードを受け取り、
- 売上高・経常利益・純利益の推移とYoY成長率
- 増収/増益が何期連続しているか(トレンド判定)
- ROE・自己資本比率・ROAなどの財務健全性指標
- EPS・BPS・PERなどの株価関連指標(分割調整済みの値を使い複数年比較できるようにする)
を計算する。
"""

import pandas as pd

# 複数年比較する際は分割調整済みの値を優先して使う(rawのeps/bpsは株式分割で
# 跳ねるため、そのままでは前年比較ができない)。
GROWTH_COLUMNS = {
    "revenue": "売上高",
    "ordinary_income": "経常利益",
    "net_income": "純利益",
}
HEALTH_COLUMNS = {
    "roe_official": "ROE",
    "equity_ratio_official": "自己資本比率",
    "roa": "ROA",
}
PER_SHARE_COLUMNS = {
    "adjusted_eps": "EPS(分割調整済)",
    "adjusted_bps": "BPS(分割調整済)",
}
DIVIDEND_COLUMNS = {
    "dividend_per_share": "1株配当(円)",
}

EPS_CAGR_YEARS = 5  # ピーター・リンチ流に、単年度ではなく複数年(最大5年)の平均成長率を使う


def eps_cagr(df: pd.DataFrame, years: int = EPS_CAGR_YEARS) -> float | None:
    """
    直近の分割調整済みEPSから、最大years年(取れる年数がそれより少なければその年数)の
    CAGR(年平均成長率)を計算する。開始/終了どちらかがゼロ以下(無配・赤字など)の場合はNoneを返す。
    """
    if "adjusted_eps" not in df.columns:
        return None
    eps = df["adjusted_eps"].dropna()
    if len(eps) < 2:
        return None
    n = min(years, len(eps) - 1)
    end_eps = eps.iloc[-1]
    start_eps = eps.iloc[-1 - n]
    if end_eps <= 0 or start_eps <= 0:
        return None
    return (end_eps / start_eps) ** (1 / n) - 1


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """財務レコードのリストを、fiscal_year昇順のDataFrameに変換する。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).dropna(subset=["fiscal_year"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df = df.drop_duplicates(subset="fiscal_year", keep="last")
    df = df.sort_values("fiscal_year").reset_index(drop=True)
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = df["net_income"] / df["total_assets"]
    if "ordinary_income" in df.columns and "revenue" in df.columns:
        df["ordinary_income_margin"] = df["ordinary_income"] / df["revenue"]
    return df


def yoy_growth(df: pd.DataFrame, column: str) -> pd.Series:
    """指定列の前年比成長率(%)。先頭年度はNaN。"""
    if column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].pct_change() * 100


def growth_streak(df: pd.DataFrame, column: str) -> int:
    """
    直近年度から遡って、前年比で増加(YoY > 0)が何期連続しているかを返す。
    データが無い/直近年度が減少している場合は0。
    """
    growth = yoy_growth(df, column).dropna()
    streak = 0
    for value in reversed(growth.tolist()):
        if value > 0:
            streak += 1
        else:
            break
    return streak


def summarize(df: pd.DataFrame) -> dict:
    """直近年度のスナップショットと増収増益のトレンド判定をまとめて返す。"""
    if df.empty:
        return {}

    latest = df.iloc[-1]
    summary = {
        "latest_fiscal_year": int(latest["fiscal_year"]),
        "years_available": len(df),
    }

    for col, label in {
        **GROWTH_COLUMNS,
        **HEALTH_COLUMNS,
        **PER_SHARE_COLUMNS,
        **DIVIDEND_COLUMNS,
    }.items():
        if col in df.columns and pd.notna(latest.get(col)):
            summary[f"latest_{col}"] = float(latest[col])

    cagr = eps_cagr(df)
    if cagr is not None:
        summary["eps_cagr_pct"] = float(cagr * 100)

    for col in GROWTH_COLUMNS:
        if col in df.columns:
            growth = yoy_growth(df, col)
            summary[f"{col}_yoy_pct"] = (
                float(growth.iloc[-1]) if pd.notna(growth.iloc[-1]) else None
            )
            summary[f"{col}_streak"] = growth_streak(df, col)

    summary["revenue_growing"] = summary.get("revenue_streak", 0) >= 1
    summary["profit_growing"] = summary.get("net_income_streak", 0) >= 1

    if "per" in df.columns and pd.notna(latest.get("per")):
        summary["latest_per"] = float(latest["per"])

    if "ordinary_income_margin" in df.columns:
        margin = df["ordinary_income_margin"]
        if pd.notna(margin.iloc[-1]):
            summary["latest_ordinary_income_margin"] = float(margin.iloc[-1])
        if len(margin) >= 2 and pd.notna(margin.iloc[-1]) and pd.notna(margin.iloc[-2]):
            summary["ordinary_income_margin_change_pt"] = float(
                (margin.iloc[-1] - margin.iloc[-2]) * 100
            )

    return summary


def live_valuation(latest_price: float, df: pd.DataFrame) -> dict:
    """
    現在の株価と直近期の分割調整済みEPS/BPSから、現在時点のPER・PBRを計算する。
    (EDINET DBのperフィールドは決算期末時点の値で、現在株価とは異なる)
    """
    if df.empty or latest_price is None:
        return {}
    latest = df.iloc[-1]
    result = {}
    eps = latest.get("adjusted_eps")
    bps = latest.get("adjusted_bps")
    if pd.notna(eps) and eps:
        result["live_per"] = latest_price / eps
    if pd.notna(bps) and bps:
        result["live_pbr"] = latest_price / bps
    return result
