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
VOL_AVG_WINDOW = 20  # 出来高比率を計算する際の平均期間

# 「上昇基調」判定の閾値。
MA_TREND_WINDOW = 200  # 移動平均線(長期)の期間
MA_SHORT_TREND_WINDOW = 25  # 移動平均線(短期)の期間
MA_TREND_LOOKBACK = 20  # 移動平均線が上向きかどうかを判定する比較窓(直近20営業日前との比較)
MONTHLY_VOL_MIN_MONTHS = 6  # 月次ボラ算出に必要な最低月数
MONTHLY_VOL_THRESHOLD = 0.07  # 月次リターン標準偏差の上限(7%)
WORST_MONTH_THRESHOLD = -8.0  # 直近12ヶ月のうち最も悪い月次リターンの下限(%)。これより悪い月が
# 1回でもあった銘柄は、月次ボラの基準を満たしていても一時的な急落があったとみなして除外する。

# ゴールデンクロス(25日線が75日線を下から上に抜ける)の判定に使う閾値。
# 「上昇基調」(既存の5条件)は1年間ずっと上昇している銘柄向けのため、まだそこまで育っていない
# 「これから上昇が始まるかもしれない」銘柄を早めに拾うための補助シグナルとして別途用意する。
MA_GOLDEN_SHORT_WINDOW = 25   # 短期線
MA_GOLDEN_LONG_WINDOW = 75    # 長期線
GOLDEN_CROSS_RECENT_DAYS = 10        # この日数以内に交差していれば「発生して間もない」扱い
GOLDEN_CROSS_IMMINENT_LOOKBACK = 10  # 乖離が縮まってきているかを見る比較窓
GOLDEN_CROSS_IMMINENT_GAP_THRESHOLD = -0.03  # 乖離率(25日線-75日線)÷株価がこの値以上(0に近い)なら「間近」
# 「接近中」だった後、交差せずに再び乖離し始めていないかを確認する窓。バックテストの結果
# (backtest_dcr_imminent_failure.py)、この「不発」パターンはその後の株価がはっきり悪化する
# 傾向を年別に見ても確認済みのため、単に非表示にするのではなく「乖離中」として明示する。
GOLDEN_CROSS_DIVERGE_LOOKBACK = 30

# 「クイックリカバリー型」判定用: 直近のゴールデンクロスの前に、この営業日数以内で
# デッドクロス(25日線が75日線を下から上ではなく上から下に抜ける)があったかを見る。
# バックテスト(backtest_whipsaw_golden_cross.py)で5〜40営業日のどの区切りでも
# 効果に大差が無かったため、余裕を持って40営業日を採用している。
QUICK_RECOVERY_MAX_DAYS = 40

# 「6ヶ月/1年できれいに右肩上がり」の判定。対数終値に単回帰をかけ、傾き(年率換算)がプラスかつ
# 決定係数R²(0〜1、1に近いほど直線的=滑らか)が高いことを「なめらかな上昇トレンド」とみなす。
TREND_QUALITY_WINDOWS = [("3ヶ月", 63), ("6ヶ月", 126), ("1年", 252)]  # 63営業日≒3ヶ月、126営業日≒6ヶ月、252営業日≒1年
CLEAN_TREND_R2_THRESHOLD = 0.7
# R²は「全体の値動きの大きさに対する相対的ななめらかさ」しか見ないため、上昇幅自体が
# 大きい銘柄だと、期間中に大きく波打っても相対的にはR²が高く出てしまう弱点がある
# (実データ確認済み: R²≧0.7の銘柄の最大ドローダウン中央値が-16.7%、4分の1が-23.8%超と、
# 「なめらか」とは言い難い波打ちを見逃していた)。これを補うため、期間中の最大ドローダウン
# (高値からの最大下落率)が一定以内であることも合わせて要求する。
CLEAN_TREND_MAX_DRAWDOWN_THRESHOLD = 12.0  # (%) この値より大きい下落があれば「なめらか」から除外

# 自社データ(日本株3,664銘柄、2024/3〜2026/6、52週高値更新イベント21,920件)による
# 実証分析の結果。「新高値更新時の出来高が直前20日平均の何倍だったか」によって、
# その後20営業日の値動きにどんな傾向があったかを集計したもの。
# あくまで過去の傾向であり、個別銘柄の将来を保証するものではない。
VOLUME_TENDENCY_BUCKETS = [
    (0.0, 0.5, "小(0.5倍未満)", 92.0, 13.9, 474),
    (0.5, 1.0, "やや少なめ(0.5〜1.0倍)", 91.5, 15.4, 5785),
    (1.0, 1.5, "平常(1.0〜1.5倍)", 90.9, 17.6, 6454),
    (1.5, 2.0, "やや多め(1.5〜2.0倍)", 89.0, 19.5, 3042),
    (2.0, 3.0, "多い(2.0〜3.0倍)", 85.4, 22.4, 2508),
    (3.0, float("inf"), "急増(3.0倍以上)", 72.9, 37.0, 3657),
]


def _volume_tendency(vol_ratio: float | None) -> dict | None:
    """
    出来高比率から、過去データに基づく「その後20営業日の傾向」を返す。
    継続率: その後さらに高値を更新した割合。反落率: 終値が当日高値から10%以上
    切り下がった割合(いずれも自社データの過去集計、件数nが大きいほど参考になる)。
    """
    if vol_ratio is None or pd.isna(vol_ratio):
        return None
    for low, high, label, continuation_rate, pullback_rate, n in VOLUME_TENDENCY_BUCKETS:
        if low <= vol_ratio < high:
            return {
                "label": label,
                "continuation_rate": continuation_rate,
                "pullback_rate": pullback_rate,
                "n": n,
            }
    return None


def _ma_rising(close: pd.Series, window: int) -> bool | None:
    """指定期間の移動平均線が、直近MA_TREND_LOOKBACK営業日前と比べて上向きかどうか。"""
    ma = close.rolling(window=window, min_periods=window).mean()
    if len(ma) <= MA_TREND_LOOKBACK or pd.isna(ma.iloc[-1]) or pd.isna(ma.iloc[-MA_TREND_LOOKBACK - 1]):
        return None
    return bool(ma.iloc[-1] > ma.iloc[-MA_TREND_LOOKBACK - 1])


def _steady_rise_metrics(close: pd.Series) -> dict:
    """
    「上昇基調」判定に使う5指標を計算する。
    ①直近1年のリターン ②200日移動平均が上向き、かつ株価がその上 ③月次リターンの標準偏差(ボラティリティ)
    ④直近12ヶ月のワーストマンス(最も悪い月次リターン) ⑤25日移動平均が上向き
    データが足りない指標はNoneになり、steady_riseの判定からは自動的に外れる(=False)。
    """
    return_1y = None
    if len(close) > LOOKBACK_DAYS:
        return_1y = float(close.iloc[-1] / close.iloc[-LOOKBACK_DAYS - 1] - 1)

    ma200 = close.rolling(window=MA_TREND_WINDOW, min_periods=MA_TREND_WINDOW).mean()
    ma200_today = ma200.iloc[-1]
    above_ma200 = bool(close.iloc[-1] > ma200_today) if pd.notna(ma200_today) else None
    # 株価が200日線から何%離れているか(Growth & Price Relative Scoreの「トレンド方向」で使う)。
    # プラス=200日線より上、マイナス=200日線より下。
    ma200_pct_diff = (
        float(close.iloc[-1] / ma200_today - 1) if pd.notna(ma200_today) else None
    )
    ma200_rising = _ma_rising(close, MA_TREND_WINDOW)
    ma25_rising = _ma_rising(close, MA_SHORT_TREND_WINDOW)

    monthly_close = close.resample("ME").last().dropna()
    monthly_returns = monthly_close.pct_change().dropna().tail(12)
    monthly_vol = (
        float(monthly_returns.std())
        if len(monthly_returns) >= MONTHLY_VOL_MIN_MONTHS
        else None
    )
    worst_month = (
        float(monthly_returns.min() * 100)
        if len(monthly_returns) >= MONTHLY_VOL_MIN_MONTHS
        else None
    )

    steady_rise = (
        return_1y is not None
        and return_1y > 0
        and above_ma200 is True
        and ma200_rising is True
        and monthly_vol is not None
        and monthly_vol <= MONTHLY_VOL_THRESHOLD
        and worst_month is not None
        and worst_month >= WORST_MONTH_THRESHOLD
        and ma25_rising is True
    )

    return {
        "return_1y": return_1y,
        "above_ma200": above_ma200,
        "ma200_pct_diff": ma200_pct_diff,
        "ma200_rising": ma200_rising,
        "monthly_vol": monthly_vol,
        "worst_month": worst_month,
        "ma25_rising": ma25_rising,
        "steady_rise": steady_rise,
    }


def _golden_cross_status(close: pd.Series) -> dict | None:
    """
    25日線(短期)・75日線(長期)のゴールデンクロスの状態を判定する。
    - "crossed": 直近GOLDEN_CROSS_RECENT_DAYS営業日以内に25日線が75日線を下から上に抜けた
    - "imminent": まだ交差していないが、乖離(25日線-75日線を株価で正規化したもの)が
      直近で縮まってきており、かつ25日線が上向き(=交差の兆候)
    どちらでもなければNone(該当なし)。

    "crossed"の場合、そのゴールデンクロスの直前(QUICK_RECOVERY_MAX_DAYS営業日以内)に
    デッドクロスがあったか(＝一度崩れてから短期間で立て直した「クイックリカバリー型」か)
    も合わせて判定する(is_quick_recovery)。バックテストで、配当利回りが高い銘柄に限り、
    通常のゴールデンクロスよりその後の株価パフォーマンスが優れていることを確認済み
    (backtest_whipsaw_golden_cross.py参照)。
    """
    required_len = MA_GOLDEN_LONG_WINDOW + max(GOLDEN_CROSS_IMMINENT_LOOKBACK, QUICK_RECOVERY_MAX_DAYS) + GOLDEN_CROSS_RECENT_DAYS
    if len(close) < required_len:
        return None

    ma_short = close.rolling(window=MA_GOLDEN_SHORT_WINDOW, min_periods=MA_GOLDEN_SHORT_WINDOW).mean()
    ma_long = close.rolling(window=MA_GOLDEN_LONG_WINDOW, min_periods=MA_GOLDEN_LONG_WINDOW).mean()
    gap = ((ma_short - ma_long) / close).dropna()
    if len(gap) < GOLDEN_CROSS_IMMINENT_LOOKBACK + 1:
        return None

    gap_today = float(gap.iloc[-1])

    if gap_today > 0:
        crossed_up = (gap.shift(1) <= 0) & (gap > 0)
        cross_positions = np.flatnonzero(crossed_up.to_numpy())
        if len(cross_positions):
            last_cross_pos = cross_positions[-1]
            days_since = (len(gap) - 1) - last_cross_pos
            if days_since <= GOLDEN_CROSS_RECENT_DAYS:
                crossed_down = (gap.shift(1) >= 0) & (gap < 0)
                dead_positions = np.flatnonzero(crossed_down.to_numpy())
                dead_before = dead_positions[dead_positions < last_cross_pos]
                is_quick_recovery = False
                dead_cross_date = None
                if len(dead_before):
                    days_between = last_cross_pos - dead_before[-1]
                    is_quick_recovery = 0 < days_between <= QUICK_RECOVERY_MAX_DAYS
                    # 早復型かどうかに関わらず、直前のデッドクロス日は分かる限り表示する
                    # (通常型も、単に早期復帰の期間より前にデッドクロスがあっただけ)
                    dead_cross_date = gap.index[dead_before[-1]]
                return {
                    "status": "crossed", "days_since": int(days_since),
                    "is_quick_recovery": is_quick_recovery,
                    "cross_date": gap.index[last_cross_pos],
                    "dead_cross_date": dead_cross_date,
                }
        return None

    # まだ交差前 → 「間近」の兆候を確認する
    gap_prior = float(gap.iloc[-1 - GOLDEN_CROSS_IMMINENT_LOOKBACK])
    narrowing = gap_today > gap_prior  # マイナス方向で0に近づいてきているか
    short_prior = ma_short.iloc[-1 - GOLDEN_CROSS_IMMINENT_LOOKBACK]
    short_rising = bool(ma_short.iloc[-1] > short_prior) if pd.notna(short_prior) else False

    # クロス前かどうかに関わらず、直近のデッドクロス日は「接近中」「乖離中」どちらでも使う。
    crossed_down = (gap.shift(1) >= 0) & (gap < 0)
    dead_positions = np.flatnonzero(crossed_down.to_numpy())
    is_quick_recovery = False
    dead_cross_date = None
    if len(dead_positions):
        days_since_dead = (len(gap) - 1) - dead_positions[-1]
        is_quick_recovery = 0 < days_since_dead <= QUICK_RECOVERY_MAX_DAYS
        # 早復型の見込みかどうかに関わらず、直前のデッドクロス日は分かる限り表示する
        dead_cross_date = gap.index[dead_positions[-1]]

    if narrowing and short_rising and gap_today >= GOLDEN_CROSS_IMMINENT_GAP_THRESHOLD:
        # このまま交差した場合に「クイックリカバリー型」になりそうか
        # (直近QUICK_RECOVERY_MAX_DAYS営業日以内にデッドクロスがあったか)も合わせて示す。
        return {
            "status": "imminent", "gap_pct": gap_today * 100,
            "is_quick_recovery": is_quick_recovery, "dead_cross_date": dead_cross_date,
        }

    # 「接近中」ではないが、直近で交差に近づいた後、再び乖離し始めていないか確認する。
    # バックテストの結果、この「不発」パターン(接近→未交差のまま後退)はその後の株価が
    # 明確に悪化する傾向を年別に見ても確認済み(backtest_dcr_imminent_failure.py参照、
    # 転換成功時と比べて勝率が18〜36ポイント低い)。単に非表示にせず「乖離中」として明示する。
    if len(gap) >= GOLDEN_CROSS_DIVERGE_LOOKBACK + 1:
        recent_window = gap.tail(GOLDEN_CROSS_DIVERGE_LOOKBACK)
        # 交差済み(gap>=0)の点は「接近したが未交差」の対象外にする(それは別の話=一度
        # 交差してから再度デッドクロスしたケースで、ここで扱う「不発」とは区別する)。
        pre_cross_window = recent_window[recent_window < 0]
        if pre_cross_window.empty:
            return None
        peak_pos_in_window = int(np.argmax(pre_cross_window.to_numpy()))
        peak_gap = float(pre_cross_window.iloc[peak_pos_in_window])
        peak_date = pre_cross_window.index[peak_pos_in_window]
        days_since_peak = (len(gap) - 1) - gap.index.get_indexer([peak_date])[0]
        if (
            peak_gap >= GOLDEN_CROSS_IMMINENT_GAP_THRESHOLD
            and days_since_peak >= GOLDEN_CROSS_IMMINENT_LOOKBACK
            and gap_today < peak_gap
        ):
            return {
                "status": "diverging", "gap_pct": gap_today * 100,
                "peak_gap_pct": peak_gap * 100,
                "is_quick_recovery": is_quick_recovery, "dead_cross_date": dead_cross_date,
            }
    return None


def _trend_quality(close: pd.Series, window: int) -> dict | None:
    """
    直近window営業日の対数終値に単回帰をかけ、傾き(年率換算%)・決定係数R²・
    期間中の最大ドローダウン(高値からの最大下落率、%)を返す。
    R²は0〜1で、1に近いほど直線的な滑らかさを示すが、あくまで全体の値動きの大きさに
    対する相対値のため、上昇幅自体が大きい銘柄では期間中に大きく波打ってもR²が
    高く出てしまうことがある。そのため最大ドローダウンも合わせて見て、絶対的な
    波の大きさもチェックする(_clean_trend_summary参照)。データが足りない場合はNone。
    """
    if len(close) < window:
        return None
    series = close.tail(window)
    y = np.log(series.to_numpy())
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    drawdown = (series / series.cummax() - 1) * 100
    total_return_pct = float(series.iloc[-1] / series.iloc[0] - 1) * 100
    return {
        "slope_annualized_pct": float((np.exp(slope * 252) - 1) * 100),
        "r_squared": r_squared,
        "max_drawdown_pct": float(-drawdown.min()),  # 正の値(下落率の大きさ)にして返す
        "total_return_pct": total_return_pct,  # 期間の最初から最後までの実際の変化率(年率換算ではない)
    }


def _clean_trend_summary(close: pd.Series) -> dict:
    """
    TREND_QUALITY_WINDOWSの各期間について_trend_qualityを計算し、傾きプラス・
    R²がCLEAN_TREND_R2_THRESHOLD以上・最大ドローダウンがCLEAN_TREND_MAX_DRAWDOWN_THRESHOLD
    以内の期間をまとめて返す。
    """
    qualifying: list[str] = []
    detail: dict[str, dict | None] = {}
    for label, window in TREND_QUALITY_WINDOWS:
        tq = _trend_quality(close, window)
        detail[label] = tq
        if (
            tq is not None
            and tq["slope_annualized_pct"] > 0
            and tq["r_squared"] is not None
            and tq["r_squared"] >= CLEAN_TREND_R2_THRESHOLD
            and tq["max_drawdown_pct"] <= CLEAN_TREND_MAX_DRAWDOWN_THRESHOLD
        ):
            qualifying.append(label)
    return {
        "clean_trend_periods": "・".join(qualifying) if qualifying else None,
        "detail": detail,
    }


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
    """
    yfinanceから[start, end]を一括取得し、銘柄ごとのDataFrameに分ける。

    auto_adjust=False(実際に取引された生の株価。株式分割は元々自動調整されて
    返ってくるため、Falseでも分割で不連続にはならない。一時False→Trueに変更を
    検討したが、Trueは配当分も遡って差し引かれ実際の取引価格と乖離するため、
    「現実の株価に寄せたい」という方針でFalseに戻した)。
    """
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

        vol_ratio = None
        if "Volume" in df.columns and len(df) >= VOL_AVG_WINDOW + 1:
            avg_vol = df["Volume"].iloc[-(VOL_AVG_WINDOW + 1):-1].mean()
            if avg_vol and avg_vol > 0:
                vol_ratio = float(df["Volume"].iloc[-1] / avg_vol)
        tendency = _volume_tendency(vol_ratio) if is_new_high else None

        # 出来高増減傾向: 直近20日平均の出来高が、その前20日平均と比べて増えているか減っているか。
        # (vol_ratioは直近1日の出来高だけを見るスナップショットなのに対し、こちらは複数日のトレンド)
        volume_trend_ratio = None
        if "Volume" in df.columns and len(df) >= VOL_AVG_WINDOW * 2:
            recent_avg_vol = df["Volume"].iloc[-VOL_AVG_WINDOW:].mean()
            prior_avg_vol = df["Volume"].iloc[-VOL_AVG_WINDOW * 2 : -VOL_AVG_WINDOW].mean()
            if prior_avg_vol and prior_avg_vol > 0:
                volume_trend_ratio = float(recent_avg_vol / prior_avg_vol)

        steady_rise_metrics = _steady_rise_metrics(df["Close"])
        golden_cross = _golden_cross_status(df["Close"])
        clean_trend = _clean_trend_summary(df["Close"])
        trend_detail = clean_trend["detail"]

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
                "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
                "volume_trend_ratio": (
                    round(volume_trend_ratio, 2) if volume_trend_ratio is not None else None
                ),
                "vol_tendency": tendency["label"] if tendency else None,
                "tendency_continuation_rate": tendency["continuation_rate"] if tendency else None,
                "tendency_pullback_rate": tendency["pullback_rate"] if tendency else None,
                "tendency_n": tendency["n"] if tendency else None,
                "latest_52w_high_date": (
                    latest_new_high_date.strftime("%Y-%m-%d")
                    if latest_new_high_date is not None
                    else None
                ),
                "trading_days_since_high": trading_days_since_high,
                "return_1y": (
                    round(steady_rise_metrics["return_1y"] * 100, 2)
                    if steady_rise_metrics["return_1y"] is not None
                    else None
                ),
                "above_ma200": steady_rise_metrics["above_ma200"],
                "ma200_pct_diff": (
                    round(steady_rise_metrics["ma200_pct_diff"] * 100, 2)
                    if steady_rise_metrics["ma200_pct_diff"] is not None
                    else None
                ),
                "ma200_rising": steady_rise_metrics["ma200_rising"],
                "ma25_rising": steady_rise_metrics["ma25_rising"],
                "monthly_vol": (
                    round(steady_rise_metrics["monthly_vol"] * 100, 2)
                    if steady_rise_metrics["monthly_vol"] is not None
                    else None
                ),
                "worst_month": (
                    round(steady_rise_metrics["worst_month"], 2)
                    if steady_rise_metrics["worst_month"] is not None
                    else None
                ),
                "steady_rise": steady_rise_metrics["steady_rise"],
                "golden_cross_status": golden_cross["status"] if golden_cross else None,
                "golden_cross_days_since": (
                    golden_cross["days_since"]
                    if golden_cross and golden_cross["status"] == "crossed"
                    else None
                ),
                "golden_cross_gap_pct": (
                    round(golden_cross["gap_pct"], 2)
                    if golden_cross and golden_cross["status"] in ("imminent", "diverging")
                    else None
                ),
                "golden_cross_peak_gap_pct": (
                    round(golden_cross["peak_gap_pct"], 2)
                    if golden_cross and golden_cross["status"] == "diverging"
                    else None
                ),
                "golden_cross_is_quick_recovery": (
                    golden_cross.get("is_quick_recovery") if golden_cross else None
                ),
                "golden_cross_date": (
                    golden_cross["cross_date"].strftime("%Y-%m-%d")
                    if golden_cross and golden_cross.get("cross_date") is not None
                    else None
                ),
                "golden_cross_dead_cross_date": (
                    golden_cross["dead_cross_date"].strftime("%Y-%m-%d")
                    if golden_cross and golden_cross.get("dead_cross_date") is not None
                    else None
                ),
                "clean_trend_periods": clean_trend["clean_trend_periods"],
                "trend_r2_3m": (
                    round(trend_detail["3ヶ月"]["r_squared"], 2)
                    if trend_detail.get("3ヶ月") and trend_detail["3ヶ月"]["r_squared"] is not None
                    else None
                ),
                "trend_r2_6m": (
                    round(trend_detail["6ヶ月"]["r_squared"], 2)
                    if trend_detail.get("6ヶ月") and trend_detail["6ヶ月"]["r_squared"] is not None
                    else None
                ),
                "trend_r2_1y": (
                    round(trend_detail["1年"]["r_squared"], 2)
                    if trend_detail.get("1年") and trend_detail["1年"]["r_squared"] is not None
                    else None
                ),
                "trend_drawdown_3m": (
                    round(trend_detail["3ヶ月"]["max_drawdown_pct"], 2) if trend_detail.get("3ヶ月") else None
                ),
                "trend_drawdown_6m": (
                    round(trend_detail["6ヶ月"]["max_drawdown_pct"], 2) if trend_detail.get("6ヶ月") else None
                ),
                "trend_drawdown_1y": (
                    round(trend_detail["1年"]["max_drawdown_pct"], 2) if trend_detail.get("1年") else None
                ),
                "trend_return_3m": (
                    round(trend_detail["3ヶ月"]["total_return_pct"], 2) if trend_detail.get("3ヶ月") else None
                ),
                "trend_return_6m": (
                    round(trend_detail["6ヶ月"]["total_return_pct"], 2) if trend_detail.get("6ヶ月") else None
                ),
                "trend_return_1y": (
                    round(trend_detail["1年"]["total_return_pct"], 2) if trend_detail.get("1年") else None
                ),
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
