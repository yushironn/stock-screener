"""
EDINET DB(edinetdb.jp)クライアント

EDINET DBは金融庁EDINETの有価証券報告書データを構造化して提供する
第三者サービス(Cabocia Inc.運営、金融庁とは無関係)。

利用には無料のAPIキーが必要(個人登録、Google/Microsoft/メールいずれか):
    https://edinetdb.jp/developers
取得したキーは環境変数に設定して使う(コードや会話に直接書かない):
    PowerShell: $env:EDINETDB_API_KEY = "発行されたキー"

API仕様(公式ドキュメント https://edinetdb.jp/docs/api、実データで検証済み):
- ベースURL: https://edinetdb.jp/v1
- 認証: ヘッダー "X-API-Key: <キー>"
- /companies/{code}/financials : 財務時系列(会計年度ごとのレコードのリスト)。
  主なフィールド: revenue(売上高), ordinary_income(経常利益), net_income(純利益),
  eps/adjusted_eps, bps/adjusted_bps, per, dividend_per_share, payout_ratio,
  roe_official, equity_ratio_official, total_assets, shareholders_equity,
  shares_issued, fiscal_year など。
  ※ operating_income(営業利益)は提供されない。
  ※ adjusted_eps/adjusted_bpsは株式分割調整済みで複数年比較に向く。
- 無料プランは100回/日まで
"""

import json
import os
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://edinetdb.jp/v1"

# 無料枠は100回/日。安全マージンを残して既定では90回までに自制する
# (環境変数 EDINETDB_DAILY_LIMIT で上書き可能)。
DAILY_LIMIT = int(os.environ.get("EDINETDB_DAILY_LIMIT", "90"))
_USAGE_FILE = Path(__file__).parent / ".edinetdb_usage.json"


class QuotaExceededError(RuntimeError):
    """本日の無料枠を使い切った場合に発生する。"""


def _load_usage() -> dict:
    today = date.today().isoformat()
    try:
        data = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"date": today, "count": 0}
    if data.get("date") != today:
        return {"date": today, "count": 0}
    return data


def _save_usage(data: dict) -> None:
    _USAGE_FILE.write_text(json.dumps(data), encoding="utf-8")


def get_usage_today() -> tuple[int, int]:
    """(本日のAPI呼び出し回数, 1日の上限)を返す。"""
    usage = _load_usage()
    return usage["count"], DAILY_LIMIT


def _api_key() -> str:
    key = os.environ.get("EDINETDB_API_KEY")
    if not key:
        raise RuntimeError(
            "環境変数 EDINETDB_API_KEY が設定されていません。"
            "https://edinetdb.jp/developers でAPIキーを取得し、"
            "$env:EDINETDB_API_KEY='発行されたキー' のように設定してください。"
        )
    return key


def _get(path: str, params: dict | None = None) -> dict:
    usage = _load_usage()
    if usage["count"] >= DAILY_LIMIT:
        raise QuotaExceededError(
            f"EDINET DBの1日あたりの呼び出し上限({DAILY_LIMIT}回)に達しました。"
            "日本時間の日付が変わるまでお待ちください。"
        )

    resp = requests.get(
        f"{BASE_URL}{path}",
        params=params or {},
        headers={"X-API-Key": _api_key()},
        timeout=30,
    )
    resp.raise_for_status()

    usage["count"] += 1
    _save_usage(usage)

    return resp.json()


def search_company(query: str) -> list[dict]:
    """企業検索(認証不要だが、統一的にAPIキーを付けて呼ぶ)。"""
    return _get("/search", params={"q": query}).get("data", [])


def find_edinet_code(sec_code: str) -> str | None:
    """
    証券コード(4桁、例: "7203")からEDINETコード(例: "E02144")を解決する。
    /companies/{sec_code}/financials はsec_codeでは404になり、edinet_codeでないと
    取得できないため、まず検索APIで変換する。
    """
    for company in search_company(sec_code):
        if (company.get("sec_code") or "")[:4] == sec_code:
            return company.get("edinet_code")
    return None


def get_financials(sec_code: str) -> list[dict]:
    """
    証券コード(4桁、例: "7203")の財務時系列を取得する。
    戻り値は会計年度(fiscal_year)ごとのレコードのリスト。
    増収増益判定など派生指標の算出はanalysis.pyを使うこと
    (operating_incomeはAPIで提供されないので、ordinary_income/net_incomeを使う)。
    """
    edinet_code = find_edinet_code(sec_code)
    if not edinet_code:
        return []
    data = _get(f"/companies/{edinet_code}/financials")
    return data.get("data") or []
