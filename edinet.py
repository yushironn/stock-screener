"""
EDINET API クライアント(プロトタイプ・未検証)

EDINET(金融庁の開示システム)から、各銘柄の直近の有価証券報告書を探し、
売上高・営業利益が前期比で増えているか(増収増益)を判定する。

利用には無料のAPIキーが必要(個人登録):
    https://api.edinet-fsa.go.jp/api/auth/index.aspx?mode=1
取得したキーは環境変数に設定して使う(コードや会話に直接書かない):
    PowerShell: $env:EDINET_API_KEY = "発行されたキー"

参考にしたAPI仕様:
- 書類一覧API: GET https://api.edinet-fsa.go.jp/api/v2/documents.json
    params: date(YYYY-MM-DD), type=2(一覧+メタデータ), Subscription-Key
- 書類取得API: GET https://api.edinet-fsa.go.jp/api/v2/documents/{docID}
    params: type=5(主要な経営指標等のCSVをzipで取得), Subscription-Key
- docTypeCode "120" = 有価証券報告書
- CSVはUTF-16LE・タブ区切り。列に「項目名」「相対年度」「値」を含む。

注意: 実際のAPIキーでまだ動作確認していないため、CSVの列名や「相対年度」の
表記ゆれ(当期/当連結会計年度/当事業年度 等)は実データで調整が必要になる可能性がある。
"""

import io
import os
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
YUHO_DOC_TYPE_CODE = "120"  # 有価証券報告書

_CURRENT_PERIOD_HINTS = ("当期", "当連結会計年度", "当事業年度")
_PRIOR_PERIOD_HINTS = ("前期", "前連結会計年度", "前事業年度")


def _api_key() -> str:
    key = os.environ.get("EDINET_API_KEY")
    if not key:
        raise RuntimeError(
            "環境変数 EDINET_API_KEY が設定されていません。"
            "EDINET APIキーを取得し、$env:EDINET_API_KEY='発行されたキー' のように設定してください。"
        )
    return key


def _list_documents(day: date) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/documents.json",
        params={"date": day.isoformat(), "type": 2, "Subscription-Key": _api_key()},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()

    status_code = str(body.get("metadata", {}).get("status") or body.get("StatusCode") or "")
    if status_code and status_code != "200":
        message = body.get("metadata", {}).get("message") or body.get("message") or body
        raise RuntimeError(f"EDINET APIエラー(status={status_code}): {message}")

    return body.get("results") or []


def find_latest_yuho_docs(
    sec_codes: set[str], basis_date: date, max_lookback_days: int = 420
) -> dict[str, dict]:
    """
    各証券コード(4桁)について、basis_date以前で直近の有価証券報告書を探す。
    1日ずつ遡って書類一覧を取得し、見つかった銘柄から確定させる(全銘柄見つかるか
    max_lookback_daysに達したら終了)。

    戻り値: {証券コード4桁: {"docID", "filerName", "submitDateTime", "periodEnd"}}
    """
    remaining = set(sec_codes)
    found: dict[str, dict] = {}

    for offset in range(max_lookback_days):
        if not remaining:
            break
        day = basis_date - timedelta(days=offset)
        try:
            docs = _list_documents(day)
        except requests.RequestException:
            continue

        for doc in docs:
            sec_code = (doc.get("secCode") or "")[:4]
            if sec_code in remaining and doc.get("docTypeCode") == YUHO_DOC_TYPE_CODE:
                found[sec_code] = {
                    "docID": doc.get("docID"),
                    "filerName": doc.get("filerName"),
                    "submitDateTime": doc.get("submitDateTime"),
                    "periodEnd": doc.get("periodEnd"),
                }
                remaining.discard(sec_code)

    return found


def _to_float(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def fetch_financial_summary(doc_id: str) -> dict | None:
    """
    有価証券報告書のdocIDから、売上高・営業利益の当期/前期の値を抜き出し、
    増収・増益かどうかを判定する。
    """
    resp = requests.get(
        f"{BASE_URL}/documents/{doc_id}",
        params={"type": 5, "Subscription-Key": _api_key()},
        timeout=60,
    )
    resp.raise_for_status()

    values: dict[str, float | None] = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [
            n for n in zf.namelist() if n.lower().endswith(".csv") and "jpcrp" in n.lower()
        ]
        for name in csv_names:
            with zf.open(name) as f:
                df = pd.read_csv(f, encoding="utf-16le", sep="\t")

            if "項目名" not in df.columns or "相対年度" not in df.columns:
                continue

            for label, key in (("売上高", "net_sales"), ("営業利益", "operating_income")):
                rows = df[df["項目名"] == label]
                for _, row in rows.iterrows():
                    period = str(row.get("相対年度", ""))
                    value = _to_float(row.get("値"))
                    if value is None:
                        continue
                    if any(hint in period for hint in _CURRENT_PERIOD_HINTS):
                        values[f"{key}_current"] = value
                    elif any(hint in period for hint in _PRIOR_PERIOD_HINTS):
                        values[f"{key}_prior"] = value

    if not values:
        return None

    values["sales_growing"] = (
        values.get("net_sales_current") is not None
        and values.get("net_sales_prior") is not None
        and values["net_sales_current"] > values["net_sales_prior"]
    )
    values["profit_growing"] = (
        values.get("operating_income_current") is not None
        and values.get("operating_income_prior") is not None
        and values["operating_income_current"] > values["operating_income_prior"]
    )
    return values


def attach_fundamentals(codes: list[str], basis_date: date) -> dict[str, dict]:
    """
    Yahoo Finance形式のコード(例: "7203.T")のリストを受け取り、
    各銘柄の直近有価証券報告書から増収増益フラグを取得する。
    """
    sec_codes = {c.split(".")[0] for c in codes}
    docs = find_latest_yuho_docs(sec_codes, basis_date)

    result: dict[str, dict] = {}
    for code in codes:
        sec_code = code.split(".")[0]
        doc = docs.get(sec_code)
        if not doc:
            continue
        summary = fetch_financial_summary(doc["docID"])
        if summary is None:
            continue
        result[code] = {**doc, **summary}

    return result
