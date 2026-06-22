"""
52週来高値スクリーナー Web UI(プロトタイプ)

実行方法:
    streamlit run app.py
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

import edinetdb
from screener import OUTPUT_FILE, TICKERS_FILE, load_tickers, screen

# Streamlit Community Cloudの共有IPからはyfinanceがブロックされやすいため、
# クラウド環境ではライブ取得をせず、ローカルPCが定期更新してpushしたresult.csvを表示する。
# Streamlit CloudのSecretsに DEPLOY_ENV = "cloud" を設定して切り替える。
CLOUD_MODE = os.environ.get("DEPLOY_ENV") == "cloud"

st.set_page_config(page_title="52週来高値スクリーナー", layout="wide")

st.title("📈 52週来高値スクリーナー(プロトタイプ)")
st.caption(
    "yfinance(Yahoo Finance)の無料データを使い、基準日時点の高値が"
    "その基準日より前の過去252営業日(52週)の最高値を更新しているかを判定します。"
    "データは約15分遅延、対象はtickers.csvに記載した銘柄のみです。"
)
if CLOUD_MODE:
    st.caption(
        "クラウド版はyfinanceがブロックされることがあるため、まずライブ取得を試し、"
        "失敗した場合はローカルPCが定期取得した結果を表示します。"
    )

master = load_tickers(TICKERS_FILE)
master_by_code = {t["code"]: t["name"] for t in master}
all_labels = [f"{t['code']} - {t['name']}" for t in master]

DEFAULT_TICKERS_FILE = Path(__file__).parent / "tickers_sample30.csv.bak"

SELECTION_KEY = "selected_labels"
if SELECTION_KEY not in st.session_state:
    if DEFAULT_TICKERS_FILE.exists():
        # 全銘柄(3,700件超)をデフォルトで全選択すると重いため、当初の主要30銘柄を初期値にする
        defaults = load_tickers(DEFAULT_TICKERS_FILE)
        default_codes = {t["code"] for t in defaults}
        st.session_state[SELECTION_KEY] = [
            label for label in all_labels if label.split(" - ", 1)[0] in default_codes
        ]
    else:
        st.session_state[SELECTION_KEY] = []


def _add_picks_to_selection():
    picks = st.session_state.get("picks_to_add", [])
    st.session_state[SELECTION_KEY] = st.session_state[SELECTION_KEY] + picks
    st.session_state["picks_to_add"] = []


def _select_all():
    st.session_state[SELECTION_KEY] = all_labels.copy()


def _clear_all():
    st.session_state[SELECTION_KEY] = []


with st.sidebar:
    st.header("設定")
    as_of_date = st.date_input(
        "基準日",
        value=date.today(),
        max_value=date.today(),
        help="この日付時点で52週高値を更新していたかを判定します。休日を指定した場合は直前の営業日が使われます。",
    )

    st.subheader("銘柄を検索して追加")
    search_query = st.text_input("🔍 コードまたは銘柄名で検索", "")
    if search_query:
        q = search_query.lower()
        filtered = [
            t for t in master
            if q in t["code"].lower() or q in t["name"].lower()
        ]
    else:
        filtered = master
    current_picks = st.session_state.get("picks_to_add", [])
    addable_options = current_picks + [
        label
        for t in filtered
        if (label := f"{t['code']} - {t['name']}") not in st.session_state[SELECTION_KEY]
        and label not in current_picks
    ]
    st.multiselect("検索結果", options=addable_options, key="picks_to_add")
    st.button(
        "➕ 選択中の銘柄に追加",
        use_container_width=True,
        on_click=_add_picks_to_selection,
    )

    st.subheader("選択中の銘柄(外すと除外)")
    col_all, col_clear = st.columns(2)
    col_all.button("✅ 全選択", use_container_width=True, on_click=_select_all)
    col_clear.button("🗑️ 全解除", use_container_width=True, on_click=_clear_all)
    st.caption(f"全{len(all_labels)}銘柄中 {len(st.session_state[SELECTION_KEY])}銘柄を選択中")
    st.multiselect(
        "対象銘柄",
        options=all_labels,
        key=SELECTION_KEY,
        label_visibility="collapsed",
    )

    run_clicked = st.button("🔄 スクリーニング実行", type="primary", use_container_width=True)

    st.subheader("ファンダメンタルズ")
    used_calls, daily_limit = edinetdb.get_usage_today()
    st.caption(f"EDINET DB 本日の使用量: {used_calls} / {daily_limit} 回")
    show_fundamentals = st.checkbox(
        "📑 増収増益を取得(EDINET DB)",
        value=False,
        help="銘柄1件につき最大2回(検索+財務データ)を消費します。無料枠は1日100回(うち90回まで自制)。",
    )

selected_codes = [label.split(" - ", 1)[0] for label in st.session_state[SELECTION_KEY]]


@st.cache_data(ttl=300, show_spinner=False)
def run_screen(codes: tuple[str, ...], as_of: date) -> pd.DataFrame:
    rows = [{"code": c, "name": master_by_code.get(c, "")} for c in codes]
    return screen(rows, as_of=as_of)


@st.cache_data(ttl=300, show_spinner=False)
def load_precomputed_result() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(OUTPUT_FILE)


if "result" not in st.session_state:
    st.session_state.result = None

if not selected_codes:
    st.warning("左のサイドバーで銘柄を1つ以上選択してください。")
    st.stop()

used_fallback = False

if run_clicked or st.session_state.result is None:
    with st.spinner("データ取得中..."):
        try:
            st.session_state.result = run_screen(tuple(selected_codes), as_of_date)
        except Exception:
            st.session_state.result = pd.DataFrame()
result = st.session_state.result

if (result is None or result.empty) and CLOUD_MODE:
    full_result = load_precomputed_result()
    if not full_result.empty:
        used_fallback = True
        as_of_date = full_result["date"].max()
        result = full_result[full_result["code"].isin(selected_codes)].reset_index(drop=True)

if result is None or result.empty:
    st.warning("データを取得できませんでした。銘柄コードや基準日を確認してください。")
    st.stop()

if used_fallback:
    st.info(
        f"ライブ取得に失敗したため、ローカルPCが最後に更新した結果(基準日: {as_of_date})を表示しています。"
    )

if show_fundamentals:

    @st.cache_data(ttl=21600, show_spinner=False)
    def fetch_growth_tag(code: str) -> dict:
        sec_code = code.split(".")[0]
        summary = edinetdb.judge_growth(sec_code)
        return summary or {"error": "データなし"}

    growth_tags: dict[str, dict] = {}
    quota_hit = False
    with st.spinner("EDINET DBから財務データを取得中..."):
        for code in result["code"]:
            try:
                growth_tags[code] = fetch_growth_tag(code)
            except edinetdb.QuotaExceededError as e:
                quota_hit = True
                growth_tags[code] = {"error": str(e)}
            except Exception as e:
                growth_tags[code] = {"error": str(e)}

    if quota_hit:
        remaining = [c for c, t in growth_tags.items() if "error" in t]
        st.warning(
            f"EDINET DBの無料枠上限に達したため、{len(remaining)}件は取得できませんでした"
            "(❓表示)。日付が変わるまでお待ちください。"
        )
    else:
        errors = {c: t["error"] for c, t in growth_tags.items() if "error" in t}
        if errors and len(errors) == len(growth_tags):
            st.warning(
                "EDINET DBからのデータ取得に失敗しました。EDINETDB_API_KEYが正しく設定されているか確認してください。"
                f" (例: {next(iter(errors.values()))})"
            )

    def _mark(code: str, key: str) -> str:
        tag = growth_tags.get(code, {})
        if "error" in tag:
            return "❓"
        return "✅" if tag.get(key) else "❌"

    result = result.copy()
    result["増収"] = result["code"].map(lambda c: _mark(c, "sales_growing"))
    result["増益"] = result["code"].map(lambda c: _mark(c, "profit_growing"))

new_highs = result[result["new_52w_high"]]

st.subheader(f"🎯 {as_of_date} 時点で52週来高値を更新した銘柄({len(new_highs)}件)")
if new_highs.empty:
    st.info("該当銘柄はありません。")
else:
    columns = ["code", "name", "date", "close", "prior_52w_high", "pct_vs_prior_high"]
    if show_fundamentals:
        columns += ["増収", "増益"]
    st.dataframe(
        new_highs[columns],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("📅 選択銘柄の最新52週高値更新日(新しい順)")
history = result.dropna(subset=["latest_52w_high_date"]).sort_values(
    "latest_52w_high_date", ascending=False
)
if history.empty:
    st.info("探索範囲(約1年)内に52週高値を更新した銘柄がありません。")
else:
    st.dataframe(
        history[["code", "name", "latest_52w_high_date", "trading_days_since_high"]].rename(
            columns={
                "latest_52w_high_date": "最新更新日",
                "trading_days_since_high": "経過営業日数",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("📋 全銘柄ランキング(52週高値からの距離順)")
st.dataframe(
    result.style.apply(
        lambda row: ["background-color: #fff3b0" if row["new_52w_high"] else "" for _ in row],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("📊 個別銘柄チャート")
selected_code = st.selectbox("銘柄を選択", result["code"].tolist())
if selected_code:
    try:
        hist = yf.Ticker(selected_code).history(period="1y")
    except Exception:
        hist = pd.DataFrame()
    if hist.empty:
        st.caption("チャート用データの取得に失敗しました(yfinanceがブロックされている可能性があります)。")
    else:
        prior_high = result.loc[result["code"] == selected_code, "prior_52w_high"].iloc[0]
        chart_df = hist[["Close", "High"]].copy()
        chart_df["52週高値ライン"] = prior_high
        st.line_chart(chart_df.rename(columns={"Close": "終値", "High": "高値"}))

if used_fallback:
    st.caption(f"データ基準日: {as_of_date}(ローカルPCの定期更新による、ライブ取得失敗時のフォールバック)")
else:
    st.caption("結果はキャッシュされ、5分間は再取得しません。最新化するには左の「スクリーニング実行」を押してください。")
