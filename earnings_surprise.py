"""
決算上振れ分析モジュール

データソース:
  1. yfinance earnings_dates  → 四半期コンセンサスサプライズ%(過去8〜12期)
  2. 株探 annual finance page → 年次EPS実績トレンド + 次期会社予想EPS

キャッシュ: cache/earnings_surprise/{code}.json  (TTL: 7日)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "earnings_surprise"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_DAYS = 7

_KABUTAN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


# ──────────────────────────────────────────────────────────
# キャッシュ I/O
# ──────────────────────────────────────────────────────────

def _cache_path(code: str) -> Path:
    safe = code.replace(".", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(code: str) -> dict | None:
    p = _cache_path(code)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched > timedelta(days=CACHE_TTL_DAYS):
            return None
        return data
    except Exception:
        return None


def _save_cache(code: str, data: dict) -> None:
    data["fetched_at"] = datetime.now().isoformat()
    _cache_path(code).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────
# 1. yfinance: 四半期コンセンサスサプライズ履歴
# ──────────────────────────────────────────────────────────

def _fetch_yf_consensus(code: str) -> list[dict]:
    """
    yfinance earnings_dates から四半期サプライズ%(対コンセンサス)を取得。
    Returns: [{"date": "YYYY-MM-DD", "eps_estimate": float, "eps_actual": float, "surprise_pct": float}, ...]
             新しい順
    """
    try:
        tkr = yf.Ticker(code)
        ed = tkr.earnings_dates
        if ed is None or ed.empty:
            return []

        rows = []
        for dt_idx, row in ed.iterrows():
            surp = row.get("Surprise(%)")
            est  = row.get("EPS Estimate")
            act  = row.get("Reported EPS")

            if surp is None or (isinstance(surp, float) and pd.isna(surp)):
                continue

            try:
                if hasattr(dt_idx, "tzinfo") and dt_idx.tzinfo:
                    date_str = dt_idx.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d")
                else:
                    date_str = pd.Timestamp(dt_idx).strftime("%Y-%m-%d")
            except Exception:
                continue

            rows.append({
                "date":         date_str,
                "eps_estimate": float(est)  if est  is not None and not pd.isna(est)  else None,
                "eps_actual":   float(act)  if act  is not None and not pd.isna(act)  else None,
                "surprise_pct": round(float(surp), 2),
            })

        return rows[:16]  # 最大16期(4年分)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────
# 2. 株探: 年次EPS実績 + 次期会社予想EPS
# ──────────────────────────────────────────────────────────

def _parse_num(s: str) -> float | None:
    """'37,154,298' や '179.5' を float に変換。失敗は None。"""
    try:
        return float(s.replace(",", "").replace("*", "").strip())
    except Exception:
        return None


def _fetch_kabutan_annual(code_raw: str) -> dict:
    """
    株探 annual finance page を取得・解析。
    Returns: {
        "annual_eps": [{"period": "2023.03", "eps": 179.5, "is_forecast": False}, ...],
        "next_forecast_eps": float | None,   # 来期会社予想EPS
        "next_forecast_period": str | None,
    }
    """
    code = code_raw.replace(".T", "").split(".")[0]
    url  = f"https://kabutan.jp/stock/finance?code={code}"
    try:
        r = requests.get(url, headers=_KABUTAN_HEADERS, timeout=10)
        if r.status_code != 200:
            return {}

        soup   = BeautifulSoup(r.content, "html.parser")
        tables = soup.find_all("table")

        # 年次サマリーテーブルを探す(ヘッダーに「修正1株益」が含まれるもの)
        annual_table = None
        for t in tables:
            header_text = t.find("tr")
            if header_text and "修正1株益" in header_text.get_text():
                annual_table = t
                break

        if annual_table is None:
            return {}

        annual_eps     = []
        next_eps       = None
        next_period    = None

        for row in annual_table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if len(cells) < 6:
                continue
            period_raw = cells[0]
            if not period_raw or period_raw in ("決算期", "前期比", ""):
                continue

            is_forecast = "予" in period_raw
            period = period_raw.replace("I", "").replace("予", "").replace("\xa0", "").strip()
            if not period:
                continue

            eps = _parse_num(cells[5])  # 修正1株益
            if eps is None:
                continue

            entry = {"period": period, "eps": eps, "is_forecast": is_forecast}
            annual_eps.append(entry)

            if is_forecast:
                next_eps    = eps
                next_period = period

        return {
            "annual_eps":          annual_eps,
            "next_forecast_eps":   next_eps,
            "next_forecast_period": next_period,
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────
# 3. 統合: サプライズサマリーを返す
# ──────────────────────────────────────────────────────────

def get_surprise_data(code: str, force_refresh: bool = False) -> dict:
    """
    キャッシュを優先しつつ、コンセンサス履歴 + 年次EPS + 上振れ分析を返す。

    Returns:
        {
          "consensus_history":  [...],        # 四半期コンセンサスサプライズ
          "consensus_avg_pct":  float|None,   # 過去平均サプライズ%(最新除く)
          "consensus_latest":   float|None,   # 最新サプライズ%
          "consensus_accel":    bool|None,    # 今回が平均を上回るか
          "consensus_trend":    str,          # "加速" / "減速" / "横ばい" / "データなし"

          "annual_eps":         [...],        # 年次EPS(実績+会社予想)
          "next_forecast_eps":  float|None,
          "next_forecast_period": str|None,
          "annual_beat_rate":   float|None,  # 最新完了期のコンセンサス比(概算)

          "fetched_at":         str,
        }
    """
    if not force_refresh:
        cached = _load_cache(code)
        if cached:
            return cached

    # --- yfinance コンセンサス ---
    consensus_history = _fetch_yf_consensus(code)
    consensus_avg    = None
    consensus_latest = None
    consensus_accel  = None
    consensus_trend  = "データなし"

    if len(consensus_history) >= 2:
        # [0]が最新。過去平均は[1:]から計算
        consensus_latest = consensus_history[0]["surprise_pct"]
        hist_surps = [r["surprise_pct"] for r in consensus_history[1:] if r["surprise_pct"] is not None]
        if hist_surps:
            consensus_avg   = round(sum(hist_surps) / len(hist_surps), 1)
            diff = consensus_latest - consensus_avg
            consensus_accel = diff > 0
            if diff >= 3:
                consensus_trend = "加速"
            elif diff <= -3:
                consensus_trend = "減速"
            else:
                consensus_trend = "横ばい"

    # --- 株探 年次EPS ---
    time.sleep(1.5)  # 株探へのリクエスト間隔
    kabutan = _fetch_kabutan_annual(code)

    annual_eps          = kabutan.get("annual_eps", [])
    next_forecast_eps   = kabutan.get("next_forecast_eps")
    next_forecast_period= kabutan.get("next_forecast_period")

    # 最新コンセンサスサプライズをannual_beat_rateとして使用
    annual_beat_rate = consensus_latest

    result = {
        "consensus_history":    consensus_history,
        "consensus_avg_pct":    consensus_avg,
        "consensus_latest":     consensus_latest,
        "consensus_accel":      consensus_accel,
        "consensus_trend":      consensus_trend,
        "annual_eps":           annual_eps,
        "next_forecast_eps":    next_forecast_eps,
        "next_forecast_period": next_forecast_period,
        "annual_beat_rate":     annual_beat_rate,
    }

    _save_cache(code, result)
    return result


# ──────────────────────────────────────────────────────────
# 4. スクリーナー向けバルク取得(候補銘柄リスト用)
# ──────────────────────────────────────────────────────────

def get_surprise_flags(codes: list[str], sleep_sec: float = 1.5) -> pd.DataFrame:
    """
    複数銘柄のサプライズフラグをDataFrameで返す。
    スクリーナー結果テーブルへの追加用。

    Columns: code, consensus_latest, consensus_avg_pct, consensus_trend
    """
    rows = []
    for code in codes:
        try:
            data = get_surprise_data(code)
            rows.append({
                "code":              code,
                "最新サプライズ%":    data.get("consensus_latest"),
                "過去平均サプライズ%": data.get("consensus_avg_pct"),
                "上振れトレンド":     data.get("consensus_trend", "データなし"),
            })
        except Exception:
            rows.append({
                "code": code,
                "最新サプライズ%":    None,
                "過去平均サプライズ%": None,
                "上振れトレンド":     "エラー",
            })
        time.sleep(sleep_sec)

    return pd.DataFrame(rows)
