"""
52週来高値スクリーナー(プロトタイプ) - コアロジック

yfinance(Yahoo Financeの無料データ)を使い、指定した日本株について
基準日(指定がなければ最新営業日)が52週高値(基準日より前の過去252営業日の
最高値)を更新しているかどうかを判定する。

CLIから直接実行することもできる:
    python screener.py              # 最新営業日基準
    python screener.py 2026-03-15   # 指定日基準
"""

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

TICKERS_FILE = Path(__file__).parent / "tickers.csv"
OUTPUT_FILE = Path(__file__).parent / "result.csv"
CACHE_DIR = Path(__file__).parent / "cache" / "prices"
LOOKBACK_DAYS = 252  # 52週 ≒ 252営業日
HISTORY_BUFFER_DAYS = 800  # 52週判定の土台(約1年)+ 直近更新日を探索する期間(約1年)


def load_tickers(path: Path = TICKERS_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code.replace('/', '_')}.parquet"


def _load_cache(code: str) -> pd.DataFrame | None:
    path = _cache_path(code)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    return df if not df.empty else None


def _save_cache(code: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(code))


def _approx_last_trading_day(d: date) -> date:
    """土日を除いた直近の営業日を返す(祝日は考慮しない簡易版)。"""
    while d.weekday() >= 5:  # 5=土, 6=日
        d -= timedelta(days=1)
    return d


def _fetch(codes: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """yfinanceから[start, end]を一括取得し、銘柄ごとのDataFrameに分ける。"""
    if not codes:
        return {}
    raw = yf.download(
        codes,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    result = {}
    for code in codes:
        try:
            df = raw[code].dropna(subset=["High", "Close"]) if len(codes) > 1 else raw.dropna(
                subset=["High", "Close"]
            )
        except KeyError:
            continue
        if not df.empty:
            result[code] = df
    return result


def _get_price_history(codes: list[str], basis_date: date) -> dict[str, pd.DataFrame]:
    """
    各銘柄の価格データを、ローカルキャッシュ(cache/prices/*.parquet)を使って取得する。
    キャッシュが無い/古すぎる(必要な開始日より後からしか無い)銘柄は全期間を再取得し、
    キャッシュの最終日がbasis_dateより前の銘柄は不足分だけ追加取得してマージする。
    """
    desired_start = basis_date - timedelta(days=HISTORY_BUFFER_DAYS)
    expected_last_trading_day = _approx_last_trading_day(basis_date)

    cached: dict[str, pd.DataFrame] = {}
    need_full: list[str] = []
    need_incremental: list[str] = []

    for code in codes:
        df = _load_cache(code)
        if df is None or df.index.min().date() > desired_start:
            need_full.append(code)
            continue
        cached[code] = df
        if df.index.max().date() < expected_last_trading_day:
            need_incremental.append(code)

    if need_full:
        fetched = _fetch(need_full, desired_start, basis_date)
        for code, df in fetched.items():
            _save_cache(code, df)
            cached[code] = df

    if need_incremental:
        incr_start = min(cached[c].index.max().date() for c in need_incremental) + timedelta(days=1)
        fetched = _fetch(need_incremental, incr_start, basis_date)
        for code in need_incremental:
            new_df = fetched.get(code)
            if new_df is None or new_df.empty:
                continue
            merged = pd.concat([cached[code], new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            _save_cache(code, merged)
            cached[code] = merged

    return cached


def screen(tickers: list[dict], as_of: str | date | None = None) -> pd.DataFrame:
    """
    as_of: 基準日(YYYY-MM-DD文字列 or date)。Noneなら最新営業日を基準にする。
    基準日が休日の場合は、その直前の営業日のデータが使われる。
    """
    codes = [t["code"] for t in tickers]
    names = {t["code"]: t["name"] for t in tickers}

    basis_date = pd.Timestamp(as_of).date() if as_of is not None else date.today()
    price_history = _get_price_history(codes, basis_date)

    rows = []
    for code in codes:
        df = price_history.get(code)
        if df is None:
            continue

        df = df[df.index.date <= basis_date]

        if len(df) < 2:
            continue

        # 各営業日について、その日より前の過去252営業日の最高値を更新したかを判定し、
        # 直近で更新した日(=最新52週来高値更新日)を探す。
        prior_high_series = (
            df["High"].rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).max().shift(1)
        )
        is_new_high_series = (df["High"] >= prior_high_series).fillna(False)
        new_high_positions = np.flatnonzero(is_new_high_series.to_numpy())
        if len(new_high_positions):
            latest_pos = new_high_positions[-1]
            latest_new_high_date = df.index[latest_pos]
            trading_days_since_high = (len(df) - 1) - latest_pos
        else:
            latest_new_high_date = None
            trading_days_since_high = None

        recent = df.tail(LOOKBACK_DAYS + 1)  # 基準日 + 過去分
        today = recent.iloc[-1]
        past = recent.iloc[:-1]

        prior_52w_high = past["High"].max()
        today_high = today["High"]
        today_close = today["Close"]
        is_new_high = today_high >= prior_52w_high

        rows.append(
            {
                "code": code,
                "name": names.get(code, ""),
                "date": recent.index[-1].strftime("%Y-%m-%d"),
                "close": round(float(today_close), 1),
                "today_high": round(float(today_high), 1),
                "prior_52w_high": round(float(prior_52w_high), 1),
                "new_52w_high": is_new_high,
                "pct_vs_prior_high": round(
                    float((today_high / prior_52w_high - 1) * 100), 2
                ),
                "latest_52w_high_date": (
                    latest_new_high_date.strftime("%Y-%m-%d")
                    if latest_new_high_date is not None
                    else None
                ),
                "trading_days_since_high": trading_days_since_high,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("pct_vs_prior_high", ascending=False).reset_index(drop=True)


def main():
    if not TICKERS_FILE.exists():
        print(f"tickers.csv が見つかりません: {TICKERS_FILE}", file=sys.stderr)
        sys.exit(1)

    as_of = sys.argv[1] if len(sys.argv) > 1 else None

    tickers = load_tickers(TICKERS_FILE)
    label = as_of if as_of else "最新営業日"
    print(f"{len(tickers)}銘柄のデータを取得中...(基準日: {label})")
    result = screen(tickers, as_of=as_of)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    new_highs = result[result["new_52w_high"]]
    print(f"\n=== 基準日({label})時点で52週来高値を更新した銘柄 ===")
    if new_highs.empty:
        print("該当なし")
    else:
        print(new_highs[["code", "name", "date", "close", "prior_52w_high"]].to_string(index=False))

    print(f"\n=== 各銘柄の最新52週高値更新日(直近1年の探索範囲内) ===")
    history = result.dropna(subset=["latest_52w_high_date"]).sort_values(
        "latest_52w_high_date", ascending=False
    )
    if history.empty:
        print("該当なし(データ不足の可能性があります)")
    else:
        print(
            history[["code", "name", "latest_52w_high_date", "trading_days_since_high"]].to_string(
                index=False
            )
        )

    print(f"\n全銘柄の結果を {OUTPUT_FILE} に保存しました。")


if __name__ == "__main__":
    main()
