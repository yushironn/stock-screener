"""
東証全銘柄の価格キャッシュ(cache/prices/*.parquet)を更新するための定期実行用スクリプト。
Windowsタスクスケジューラから平日15:31(東証大引け後)に呼び出すことを想定している。

実行方法:
    python daily_refresh.py
"""

import sys
from datetime import datetime
from pathlib import Path

from screener import OUTPUT_FILE, TICKERS_FILE, load_tickers, screen

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_FILE = Path(__file__).parent / "refresh.log"


def _log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    _log("=== 定期更新開始 ===")
    try:
        tickers = load_tickers(TICKERS_FILE)
        _log(f"対象銘柄数: {len(tickers)}")
        result = screen(tickers)
        result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        new_highs = int(result["new_52w_high"].sum()) if not result.empty else 0
        _log(f"取得成功: {len(result)}/{len(tickers)}銘柄、本日の新高値: {new_highs}件")
    except Exception as e:
        _log(f"エラー: {type(e).__name__}: {e}")
    _log("=== 定期更新終了 ===")


if __name__ == "__main__":
    main()
