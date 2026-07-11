"""
EDINET DB財務データのディスクキャッシュ。

EDINET DBは無料枠100回/日(1銘柄の取得に検索+財務データで最大2回消費)しかないため、
同じ銘柄を見るたびにAPIを呼び直さないよう、取得結果をcache/financials/{code}.jsonに
永続化する。財務データは決算ごと(年に数回)しか更新されないため、ある程度古くても
実用上問題ない。

事前に全銘柄をバッチ取得することはせず、ユーザーが個別銘柄分析や条件検索で実際に
触れた銘柄だけをその場でキャッシュしていく方針(オプトイン・オポチュニスティック)。
"""

import json
from datetime import date, datetime
from pathlib import Path

import edinetdb

CACHE_DIR = Path(__file__).parent / "cache" / "financials"
DEFAULT_MAX_AGE_DAYS = 180


def _cache_path(sec_code: str) -> Path:
    return CACHE_DIR / f"{sec_code}.json"


def _load_raw(sec_code: str) -> dict | None:
    path = _cache_path(sec_code)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_raw(sec_code: str, records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now().isoformat(), "records": records}
    _cache_path(sec_code).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def get_financials_cached(sec_code: str, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> list[dict]:
    """
    証券コード(4桁)の財務時系列を返す。新鮮なキャッシュがあればAPIを呼ばずに返し、
    無い/古い場合はedinetdb.get_financialsで取得してキャッシュに保存する
    (QuotaExceededErrorはそのまま呼び出し元に伝播する)。
    """
    cached = _load_raw(sec_code)
    if cached is not None:
        fetched_at = datetime.fromisoformat(cached["fetched_at"]).date()
        if (date.today() - fetched_at).days < max_age_days:
            return cached["records"]

    try:
        records = edinetdb.get_financials(sec_code)
    except edinetdb.QuotaExceededError:
        if cached is not None:
            return cached["records"]
        raise

    _save_raw(sec_code, records)
    return records


def peek_cached(sec_code: str) -> list[dict] | None:
    """
    ディスクキャッシュの内容を新鮮さに関わらずそのまま返す(無ければNone)。
    APIは一切呼ばないため、多数の銘柄を一括でスキャンする条件検索などで
    意図せずAPIクォータを消費したくない場合に使う。
    """
    cached = _load_raw(sec_code)
    return cached["records"] if cached is not None else None


def list_cached_codes() -> set[str]:
    """財務データがキャッシュ済みの証券コード(4桁)一覧を返す。"""
    if not CACHE_DIR.exists():
        return set()
    return {p.stem for p in CACHE_DIR.glob("*.json")}
