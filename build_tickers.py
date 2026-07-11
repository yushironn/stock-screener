"""
JPX(日本取引所グループ)が無料公開している東証上場銘柄一覧(data_j.xls)から、
tickers.csv(code,name,market,sector33,sector17,size_category)を生成し直すスクリプト。

【アプリ全体のルール】このアプリ(stock_screener)は東証プライム・スタンダード・
グロースに上場する「個別銘柄」のみを取り扱う。ETF・ETN、REIT・各種ファンド、
PRO Market、出資証券、指数そのもの、海外市場の銘柄などは対象外とする。
新機能を追加する際もこの前提を崩さないこと(tickers.csvの対象範囲を広げない)。

ETF・ETN、REIT・各種ファンド、PRO Market、出資証券は対象外とし、
プライム/スタンダード/グロース(内国株式・外国株式)の普通株式のみを抽出する。

market/sector33/sector17/size_categoryは条件検索(キーワード・市場区分・
時価総額帯)用にdata_j.xlsからそのまま引き継ぐ。size_categoryはTOPIXの
規模区分(Core30/Large70/Mid400/Small1/Small2)で、時価総額の目安として使う。

実行方法:
    python build_tickers.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_J_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
RAW_FILE = Path(__file__).parent / "data_j.xls"
TICKERS_FILE = Path(__file__).parent / "tickers.csv"

# 普通株式として扱う市場・商品区分(ETF・REIT・PRO Market・出資証券は除外)
INCLUDE_MARKET_KEYWORDS = ("プライム", "スタンダード", "グロース")


def _consolidate_market(market: str) -> str:
    """
    市場区分を「東証プライム/東証スタンダード/東証グロース」の3つに集約する。
    JPXの原データは「プライム（内国株式）」「プライム（外国株式）」のように
    内国株式/外国株式で分かれているが、本アプリはこの区別を扱わない
    (このアプリで取り扱うのは東証のみのため、内外国の表記は不要)。
    """
    if "プライム" in market:
        return "東証プライム"
    if "スタンダード" in market:
        return "東証スタンダード"
    if "グロース" in market:
        return "東証グロース"
    return market


def download_data_j() -> None:
    resp = requests.get(DATA_J_URL, timeout=60)
    resp.raise_for_status()
    RAW_FILE.write_bytes(resp.content)


def build() -> pd.DataFrame:
    if not RAW_FILE.exists():
        download_data_j()

    df = pd.read_excel(RAW_FILE)
    df.columns = [
        "日付", "コード", "銘柄名", "市場区分",
        "33業種コード", "33業種区分", "17業種コード", "17業種区分",
        "規模コード", "規模区分",
    ]

    is_stock = df["市場区分"].astype(str).str.contains("|".join(INCLUDE_MARKET_KEYWORDS))
    stocks = df[is_stock].copy()

    stocks["code"] = stocks["コード"].astype(str).str.strip() + ".T"
    stocks["name"] = stocks["銘柄名"].astype(str).str.strip()
    stocks["market"] = stocks["市場区分"].astype(str).str.strip().map(_consolidate_market)
    stocks["sector33"] = stocks["33業種区分"].astype(str).str.strip()
    stocks["sector17"] = stocks["17業種区分"].astype(str).str.strip()
    stocks["size_category"] = stocks["規模区分"].astype(str).str.strip()

    columns = ["code", "name", "market", "sector33", "sector17", "size_category"]
    return stocks[columns].drop_duplicates(subset="code").sort_values("code")


def main() -> None:
    tickers = build()
    tickers.to_csv(TICKERS_FILE, index=False, encoding="utf-8")
    print(f"{len(tickers)}銘柄を {TICKERS_FILE} に書き出しました。")


if __name__ == "__main__":
    main()
