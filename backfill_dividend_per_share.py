"""
yfinanceの配当支払い履歴(Ticker.dividends)経由で、全銘柄の直近12ヶ月分の
1株配当合計額を取得してキャッシュする。

【背景】これまでの配当利回りは、有価証券報告書CF計算書の「配当金の支払額」
(連結ベースの総支払額、少数株主への配当も混ざる) ÷ 時価総額 で計算していたが、
これはヤフーファイナンス等が表示する「1株配当利回り」とは定義が異なり、数値が
ズレる原因になっていた。バックテストの結果(backtest_dividend_yield_methods.py)、
1株配当ベースの方が将来リターンとの相関で既存方式を一度も下回らなかったため、
こちらに切り替える。

yfinanceのTicker.dividendsは実際に支払われた1株配当額の履歴(日付付き)を返す。
直近12ヶ月分を合計することで、ヤフーファイナンスに近い「年間配当金額」を算出する。

保存先: cache/dividend_per_share.json ({code4桁: 直近12ヶ月の1株配当合計} の
単一ファイル、1銘柄ずつ読み込むと遅いため全銘柄分をまとめて1ファイルにする)

実行方法:
    python backfill_dividend_per_share.py [--limit N] [--sleep 0.3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_FILE = Path(__file__).parent / "cache" / "dividend_per_share.json"
LOG_FILE = Path(__file__).parent / "backfill_dividend_per_share.log"


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(limit: int | None, sleep_seconds: float) -> None:
    df = pd.read_csv("tickers.csv", dtype=str)
    codes = df["code"].tolist()
    if limit:
        codes = codes[:limit]

    existing: dict[str, float] = {}
    if CACHE_FILE.exists():
        try:
            existing = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    total = len(codes)
    fetched = 0
    failed = 0
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=365)
    _log(f"=== 1株配当(直近12ヶ月)のバックフィル開始(対象{total}銘柄) ===")

    for i, code in enumerate(codes, start=1):
        sec_code = code.split(".")[0]
        try:
            divs = yf.Ticker(code).dividends
        except Exception as e:
            failed += 1
            _log(f"[{i}/{total}] {code}: 取得失敗: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds)
            continue

        if divs is not None and len(divs) > 0:
            divs.index = pd.to_datetime(divs.index).tz_localize(None)
            trailing = divs[divs.index > cutoff]
            total_dividend = float(trailing.sum())
            existing[sec_code] = total_dividend
            fetched += 1
        else:
            # 無配または取得不可(0を明示的に保存し、未取得と区別する)
            existing[sec_code] = 0.0
            fetched += 1

        if i % 200 == 0:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
            _log(f"[{i}/{total}] 取得済み{fetched}件、失敗{failed}件(中間保存)")
        time.sleep(sleep_seconds)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
    _log(f"=== 完了: 新規取得{fetched}件 / 失敗{failed}件 / 合計{total}件(キャッシュ総数{len(existing)}件) ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    run(limit=args.limit, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
