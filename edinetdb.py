"""
EDINET DB(edinetdb.jp)クライアント(プロトタイプ・未検証)

EDINET DBは金融庁EDINETの有価証券報告書データを構造化して提供する
第三者サービス(Cabocia Inc.運営、金融庁とは無関係)。
公式EDINET APIと違い、財務数値(売上高・営業利益など)が既に整理された
形で取得できるため、XBRL/CSVの自前パースが不要。

利用には無料のAPIキーが必要(個人登録、Google/Microsoft/メールいずれか):
    https://edinetdb.jp/developers
取得したキーは環境変数に設定して使う(コードや会話に直接書かない):
    PowerShell: $env:EDINETDB_API_KEY = "発行されたキー"

参考にしたAPI仕様(公式ドキュメント https://edinetdb.jp/docs/api より):
- ベースURL: https://edinetdb.jp/v1
- 認証: ヘッダー "X-API-Key: <キー>"
- /companies/{code}/financials : 財務時系列(revenue, operating_income, net_income, eps)
- 無料プランは100回/日まで

注意: 実際のAPIキーでまだ動作確認していないため、レスポンスの実際の構造
(日付の並び順や期間の表記など)は実データで調整が必要になる可能性がある。
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
    """
    edinet_code = find_edinet_code(sec_code)
    if not edinet_code:
        return []
    data = _get(f"/companies/{edinet_code}/financials")
    return data.get("data") or []


def judge_growth_from_records(records: list[dict], sec_code: str = "") -> dict | None:
    """
    既に取得済みの財務時系列(get_financialsの戻り値)から、直近2期分の
    売上高・営業利益を比較し、増収・増益かどうかを判定する。
    (追加のAPI呼び出しを発生させたくない場合はこちらを使う)
    """
    if len(records) < 2:
        return None

    # 期間の新しい順に並んでいるか分からないため、fiscal_yearでソートする
    records = sorted(records, key=lambda r: r.get("fiscal_year") or 0, reverse=True)
    latest, prior = records[0], records[1]

    latest_revenue = latest.get("revenue")
    prior_revenue = prior.get("revenue")
    latest_profit = latest.get("operating_income")
    prior_profit = prior.get("operating_income")

    return {
        "sec_code": sec_code,
        "latest_period": latest.get("fiscal_year"),
        "prior_period": prior.get("fiscal_year"),
        "latest_revenue": latest_revenue,
        "prior_revenue": prior_revenue,
        "latest_operating_income": latest_profit,
        "prior_operating_income": prior_profit,
        "sales_growing": (
            latest_revenue is not None and prior_revenue is not None and latest_revenue > prior_revenue
        ),
        "profit_growing": (
            latest_profit is not None and prior_profit is not None and latest_profit > prior_profit
        ),
    }


def judge_growth(sec_code: str) -> dict | None:
    """
    直近2期分の売上高・営業利益を比較し、増収・増益かどうかを判定する。
    (内部でget_financialsを呼ぶため、財務時系列を別途使う場合は
    judge_growth_from_recordsを使ってAPI呼び出しを節約すること)
    """
    records = get_financials(sec_code)
    return judge_growth_from_records(records, sec_code)
