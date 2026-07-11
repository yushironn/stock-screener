"""
52週来高値スクリーナー Web UI(プロトタイプ)

実行方法:
    streamlit run app.py
"""

import os
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

import analysis
import edinetdb
import earnings_surprise
import financials_cache
import growth_scores
import quality_score
import finder
from finder import filter_by_quality, filter_candidates
import screener
from screener import OUTPUT_FILE, TICKERS_FILE, load_tickers, screen

# Streamlit Community Cloudの共有IPからはyfinanceがブロックされやすいため、
# クラウド環境ではライブ取得をせず、ローカルPCが定期更新してpushしたresult.csvを表示する。
# Streamlit CloudのSecretsに DEPLOY_ENV = "cloud" を設定して切り替える。
CLOUD_MODE = os.environ.get("DEPLOY_ENV") == "cloud"

COLUMN_LABELS_JA = {
    "code": "コード",
    "name": "銘柄名",
    "date": "日付",
    "close": "終値",
    "today_high": "本日高値",
    "prior_52w_high": "直前52週高値",
    "new_52w_high": "52週高値更新",
    "pct_vs_prior_high": "高値差分(%)",
    "latest_52w_high_date": "最新更新日",
    "trading_days_since_high": "経過営業日数",
    "return_1y": "年間リターン(%)",
    "above_ma200": "200日線上",
    "ma200_rising": "200日線上向き",
    "monthly_vol": "月次ボラ(%)",
    "worst_month": "ワーストマンス(%)",
    "ma25_rising": "25日線上向き",
    "steady_rise": "上昇基調",
    "vol_ratio": "出来高比率",
    "volume_trend_ratio": "出来高増減傾向",
    "vol_tendency": "出来高傾向",
    "tendency_continuation_rate": "継続率",
    "tendency_pullback_rate": "反落率",
    "trend_r2_3m": "R²(3ヶ月)",
    "trend_r2_6m": "R²(6ヶ月)",
    "trend_r2_1y": "R²(1年)",
    "clean_trend_periods": "安定上昇型",
}

# 表示用の書式はcolumn_config(NumberColumn)に任せ、データ自体は数値型のまま保つ。
# (文字列に変換すると表のクリックソートが辞書順になってしまい、数値順に並ばなくなるため)
# アプリ内の指標・パーセンテージは小数点以下2桁に統一する(整数の期数・日数・件数は対象外)。
NUMBER_COLUMN_FORMATS = {
    "終値": "%,d",
    "本日高値": "%,d",
    "直前52週高値": "%,d",
    "高値差分(%)": "%.2f",
    "経過営業日数": "%d日",
    "年間リターン(%)": "%.2f",
    "月次ボラ(%)": "%.2f",
    "ワーストマンス(%)": "%.2f",
    "出来高比率": "%.2f",
    "出来高増減傾向": "%.2f倍",
    "継続率": "%.2f%%",
    "反落率": "%.2f%%",
    "R²(3ヶ月)": "%.2f",
    "R²(6ヶ月)": "%.2f",
    "R²(1年)": "%.2f",
    "PER": "%.2f倍",
    "ROE(%)": "%.2f%%",
    "増収増益": "%d年",
    "PEG": "%.2f",
    "配当利回り": "%.2f%%",
    "リンチレシオ": "%.2f",
    "売上高変化": "%+.2f%%",
    "経常利益率変化": "%+.2fpt",
    "営業利益率(%)": "%.2f%%",
    "営業CF(億円)": "%.2f",
}
# GP・PBR・成長の質・複合スコアは、数値ではなく整形済み文字列として扱う
# (NumberColumnではなく下記の専用関数でフォーマットする)。
# GP・成長の質・PBR(個別の要素)が取れていない場合は「ー」、複合スコア(3要素の
# 判定結果)がどれか1要素でも欠けて計算不能な場合は「判定不能」と区別して表示する。
QUALITY_INDETERMINATE_LABEL = "判定不能"
QUALITY_MISSING_LABEL = "ー"


def _combined_growth_streak(revenue_streak: pd.Series, profit_streak: pd.Series) -> pd.Series:
    """
    「増収」かつ「増益」が連続した期数 = 増収連続期数と増益連続期数のうち小さい方。
    (両方とも「直近から連続で増加している期数」なので、片方でも途切れればそこで
    「増収増益」の連続も途切れる。よってmin()がそのまま正しい計算になる)
    """
    return pd.concat([revenue_streak, profit_streak], axis=1).min(axis=1)


def _format_quality_value(
    values: pd.Series, decimals: int = 2, suffix: str = "", missing_label: str = QUALITY_MISSING_LABEL
) -> pd.Series:
    """
    複合スコア関連の数値(GP・PBR・成長の質・複合スコア)を表示用文字列に変換する。
    数値と文字列が同じ列に混在するとStreamlitの表示(pyarrow変換)がエラーになるため、
    値が取れている行も含めて列全体を文字列にする。
    missing_labelは値が取れていない場合の表示(既定は「ー」。複合スコアは
    「判定不能」を使うため呼び出し側で明示的に指定する)。
    """
    return values.map(
        lambda v: f"{v:.{decimals}f}{suffix}" if pd.notna(v) else missing_label
    )


QUALITY_DEFICIT_LABEL = "赤字"


def _format_quality_value_with_deficit(
    values: pd.Series, is_deficit: pd.Series, decimals: int = 1,
    missing_label: str = QUALITY_INDETERMINATE_LABEL, deficit_label: str = QUALITY_DEFICIT_LABEL,
) -> pd.Series:
    """
    GPS・GPRSの「判定不能」のうち、赤字(直近純利益<=0)が原因のものは
    単なるデータ欠損と区別して「赤字」と表示する(リンチレシオは赤字企業だと
    計算式の性質上そもそも定義できないため、データが足りないわけではない)。
    """
    formatted = []
    for value, deficit in zip(values, is_deficit.reindex(values.index)):
        if pd.notna(value):
            formatted.append(f"{value:.{decimals}f}")
        else:
            formatted.append(deficit_label if deficit else missing_label)
    return pd.Series(formatted, index=values.index)


QUALITY_DEBT_EXCEEDS_ASSETS_LABEL = "債務超過"


def _format_pbr(values: pd.Series, missing_label: str = QUALITY_MISSING_LABEL) -> pd.Series:
    """
    PBRを表示用文字列に変換する。PBRがマイナス(BPSがマイナス=債務超過)の場合は、
    「取得できていない(missing_label)」とは区別して「債務超過」と明示する
    (割安という意味のマイナスではなく、財務的な危険信号であるため)。
    """
    def _fmt(v: float) -> str:
        if pd.isna(v):
            return missing_label
        if v < 0:
            return QUALITY_DEBT_EXCEEDS_ASSETS_LABEL
        return f"{v:.2f}"
    return values.map(_fmt)


def _band_note(value: float | None, bands: list[tuple[float, float, float]], as_pct: bool = True) -> str:
    """
    GPRSの各サブスコア(growth_scores.pyのBANDS定数)について、実際の値が
    どのランクに該当し何点になったかを一言で説明する文字列を作る
    (「なぜこのスコアなのか」をGPRS詳細分析ダイアログで示すための補助)。
    """
    if value is None or pd.isna(value):
        return "データなし"

    def _fmt(v: float) -> str:
        return f"{v * 100:.1f}%" if as_pct else f"{v:.2f}"

    for low, high, score in bands:
        if low <= value < high:
            if low == float("-inf"):
                return f"{_fmt(high)}未満 → {score:+.0f}点"
            if high == float("inf"):
                return f"{_fmt(low)}以上 → {score:+.0f}点"
            return f"{_fmt(low)}〜{_fmt(high)} → {score:+.0f}点"
    return "該当ランクなし"


def _lynch_note(
    lynch: float | None, per: float | None, net_income_growth_pct: float | None,
    dividend_yield_pct: float | None,
) -> str:
    """
    リンチレシオスコアが「ー」になっている理由を具体的に示す。単なるデータ欠損なのか、
    赤字、または(純利益成長率+配当利回り)がマイナスで計算式上定義できないのかを区別する。
    """
    if lynch is not None:
        return _band_note(lynch, growth_scores.LYNCH_SCORE_BANDS, as_pct=False)
    if per is None:
        return "PERデータなし"
    if per <= 0:
        return "赤字のためリンチレシオ計算対象外"
    if net_income_growth_pct is None and dividend_yield_pct is None:
        return "純利益成長率・配当利回りデータなし"
    combined = (net_income_growth_pct or 0) + (dividend_yield_pct or 0)
    if combined <= 0:
        return "純利益成長率+配当利回りがマイナスのためリンチレシオ計算対象外"
    return "データなし"


def _psr_growth_note(psr_growth: float | None, lynch: float | None, psr_value: float | None,
                      revenue_growth_pct: float | None) -> str:
    """
    PSR成長対比スコアの判定根拠。リンチレシオが使える場合はGPSの計算上こちらは
    不使用(リンチレシオを優先、代用関係であって足し算ではないため)であることを明示する。
    """
    if lynch is not None:
        return "リンチレシオが使えるため今回は不使用(リンチレシオが使えない時の代替)"
    if psr_growth is not None:
        return _band_note(psr_growth, growth_scores.PSR_GROWTH_SCORE_BANDS, as_pct=False)
    if psr_value is None:
        return "PSRデータなし"
    if revenue_growth_pct is None:
        return "売上高成長率データなし"
    if revenue_growth_pct <= 0:
        return "減収のためPSR成長対比計算対象外"
    return "データなし"


def _gprs_component_table(components: list[tuple[str, str, float | None, float, str]]) -> pd.DataFrame:
    """(項目名, 実際の値の表示, 獲得点数, 満点, 判定根拠)のリストから表示用DataFrameを作る。"""
    return pd.DataFrame({
        "項目": [c[0] for c in components],
        "実際の値": [c[1] for c in components],
        "点数": [f"{c[2]:+.1f}" if c[2] is not None else QUALITY_INDETERMINATE_LABEL for c in components],
        "満点": [c[3] for c in components],
        "判定根拠": [c[4] for c in components],
    })


def _band_reference_text(bands: list[tuple[float, float, float]], as_pct: bool = True) -> str:
    """
    growth_scores.pyのBANDS定数から、全ランクの基準値一覧を「目安」として
    1行のテキストにまとめる(GPRS詳細分析でカーソルを合わせた時のツールチップ用)。
    """
    def _fmt(v: float) -> str:
        return f"{v * 100:.0f}%" if as_pct else f"{v:.1f}"

    parts = []
    for low, high, score in bands:
        if low == float("-inf"):
            parts.append(f"{_fmt(high)}未満: {score:+.0f}点")
        elif high == float("inf"):
            parts.append(f"{_fmt(low)}以上: {score:+.0f}点")
        else:
            parts.append(f"{_fmt(low)}〜{_fmt(high)}: {score:+.0f}点")
    return " / ".join(parts)


# GPRS詳細分析のツールチップに出す、各項目が「何を測っているか」の説明文
GPRS_ITEM_DESCRIPTIONS: dict[str, str] = {
    "gp_score": "総資産に対してどれだけ効率よく粗利益を稼げているかを示す収益性指標"
    "(グロス・プロフィタビリティ)。高いほど資本効率が良い。",
    "gp_growth": "売上総利益(粗利)が前期からどれだけ伸びたかを示す成長ペース。",
    "fcf_score": "営業CFから設備投資(CapEx)を引いたフリーキャッシュフローが、売上高に対して"
    "どれだけ出ているかを示す現金創出力。",
    "stability": "増収(売上高が前期比プラス)が何期連続しているかを示す、成長の安定性。",
    "lynch": "PER(株価収益率)を(純利益成長率%+配当利回り%)で割った指標(ピーター・"
    "リンチ流)。成長率のわりに株価が割安かどうかを示す(低いほど割安)。PEGに配当利回りを"
    "足すことで、増益率は低くても株主還元がしっかりした成熟企業を不当に低評価しない。"
    "赤字、または(成長率+配当利回り)がマイナスの場合は計算式の性質上定義できない。",
    "psr_growth": "PSR(株価売上高倍率)を売上高成長率(%)で割った、リンチレシオの売上高版。"
    "リンチレシオが赤字などで計算できない時の代替として使う(黒字化前のグロース株なども"
    "評価できるようにするため。リンチレシオが使える場合はそちらを優先し、足し算はしない)。",
    "fcf_yield": "フリーキャッシュフローが時価総額に対してどれだけ出ているかを示す、"
    "株価に対する現金創出力の割安さ。",
    "momentum": "現在の株価が52週高値にどれだけ近いかを示す、値動きの強さ。",
    "trend": "200日移動平均線より上に位置し、かつ上向きかどうかで中長期の株価トレンドを判定。",
    "performance": "増収率が前期より加速しているか、経常利益率が改善しているかで、"
    "株価モメンタムの裏付けとなる業績動向を判定(本来は決算修正情報を使いたいが、"
    "無料データの制約で増収加速度・利益率トレンドで代用)。",
    "equity_ratio": "総資産に対する自己資本の割合。低いほど借入への依存度が高く、財務リスクが高い。",
    "growth_quality_penalty": "株主還元・借入返済に使った資金と、増資・新規借入で調達した資金の"
    "差分を総資産で割った値。マイナスが大きいほど自己資金ではなく外部資金への依存度が高い。",
    "per_safety": "株価収益率(PER)が絶対水準として割高すぎないかをチェックする安全弁"
    "(リンチレシオスコアだけでは成長率・配当利回り次第で見かけ上割安に見えてしまう"
    "ケースがあるため)。",
    "deficit": "直近5期(今期含む、5年サマリー表の範囲)のうち純利益が赤字だった期数。"
    "赤字企業には投資しないという方針を反映し、他のリスク要素より重い減点(1期ごとに"
    "-10点)を科す。",
}


def _item_tooltip(description_key: str, band_reference: str) -> str:
    """項目の説明文と目安(バンド一覧)を1つのツールチップ用テキストにまとめる。"""
    description = GPRS_ITEM_DESCRIPTIONS.get(description_key, "")
    return f"{description}\n目安: {band_reference}" if description else f"目安: {band_reference}"


def _render_component_table(components: list[tuple[str, str, float | None, float, str, str]]) -> None:
    """
    (項目名, 実際の値, 獲得点数, 満点, 判定根拠, 目安ツールチップ)のリストから、
    項目名にカーソルを合わせると目安(各ランクの基準値一覧)がツールチップ表示される
    HTMLテーブルを描画する。st.dataframeは行ごとのツールチップに対応していないため、
    ここだけHTML(title属性)で代用する。
    """
    rows_html = ""
    for name, value_str, score, max_score, note, tooltip in components:
        score_str = f"{score:+.1f}" if score is not None else QUALITY_INDETERMINATE_LABEL
        tooltip_attr = tooltip.replace('"', "&quot;")
        rows_html += (
            "<tr>"
            f'<td title="{tooltip_attr}" style="cursor:default;border-bottom:1px solid rgba(128,128,128,0.3);'
            f'padding:4px 8px;">{name} 🛈</td>'
            f'<td style="border-bottom:1px solid rgba(128,128,128,0.3);padding:4px 8px;">{value_str}</td>'
            f'<td style="border-bottom:1px solid rgba(128,128,128,0.3);padding:4px 8px;">{score_str}</td>'
            f'<td style="border-bottom:1px solid rgba(128,128,128,0.3);padding:4px 8px;">{max_score}</td>'
            f'<td style="border-bottom:1px solid rgba(128,128,128,0.3);padding:4px 8px;">{note}</td>'
            "</tr>"
        )
    html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em;">'
        "<thead><tr>"
        '<th style="text-align:left;padding:4px 8px;">項目</th>'
        '<th style="text-align:left;padding:4px 8px;">実際の値</th>'
        '<th style="text-align:left;padding:4px 8px;">点数</th>'
        '<th style="text-align:left;padding:4px 8px;">満点</th>'
        '<th style="text-align:left;padding:4px 8px;">判定根拠</th>'
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _format_composite_score(values: pd.Series, low_sample_flags: pd.Series | None) -> pd.Series:
    """
    複合スコアを表示用文字列に変換する。業種内の該当銘柄数が少ない(順位の信頼性が
    下がる)銘柄には「※」を付けて注記する(quality_score.MIN_SECTOR_SAMPLE_SIZE参照)。
    """
    if low_sample_flags is None:
        return _format_quality_value(values)
    formatted = []
    for value, low_sample in zip(values, low_sample_flags.reindex(values.index)):
        if pd.isna(value):
            formatted.append(QUALITY_INDETERMINATE_LABEL)
        else:
            formatted.append(f"{value:.2f}※" if low_sample else f"{value:.2f}")
    return pd.Series(formatted, index=values.index)


def number_column_config(columns) -> dict:
    return {
        col: st.column_config.NumberColumn(format=fmt)
        for col, fmt in NUMBER_COLUMN_FORMATS.items()
        if col in columns
    }


# GP・PBR・成長の質・複合スコアは「判定不能」等の文字列と数値が混在するため
# 数値列(NumberColumn)ではなく文字列列(TextColumn)として扱っている。そのままだと
# 既定で左寄せになるため、右寄せに指定する。
QUALITY_TEXT_COLUMNS = ["GP(%)", "PBR", "成長の質(%)", "複合スコア"]


def quality_text_column_config(columns) -> dict:
    return {
        col: st.column_config.TextColumn(alignment="right")
        for col in QUALITY_TEXT_COLUMNS
        if col in columns
    }


def section_gate(key: str, label: str = "🔍 情報取得") -> bool:
    """
    大項目(上昇基調銘柄など)ごとに「情報取得」ボタンと、その隣の展開/縮小ボタンを
    別々に用意する。情報取得ボタンを押すまではその項目のスクリーニング・表示を
    行わない。押すとデータ取得(スクリーニング)を行い、自動的に展開状態にする。
    展開/縮小ボタンは、取得済みのデータの表示・非表示だけを切り替える
    (再取得はしない)。開閉状態・取得済みかどうかはセッション内で保持する。
    """
    ready_key = f"{key}_gate_ready"
    expanded_key = f"{key}_gate_expanded"

    col1, col2, _ = st.columns([1, 1, 3])
    if col1.button(label, key=f"{key}_gate_button"):
        st.session_state[ready_key] = True
        st.session_state[expanded_key] = True

    is_ready = bool(st.session_state.get(ready_key))
    is_expanded = bool(st.session_state.get(expanded_key))

    if is_ready:
        toggle_label = "▲ 縮小" if is_expanded else "▼ 展開"
        if col2.button(toggle_label, key=f"{key}_gate_toggle"):
            is_expanded = not is_expanded
            st.session_state[expanded_key] = is_expanded

    return is_ready and is_expanded


def section_heading(text: str) -> None:
    """
    大項目の見出しに色付きの帯を付けて表示する(st.subheaderの代わりに使う。
    セクションの区切りを視覚的に強調するため)。
    """
    st.markdown(
        f'<div style="background-color:rgba(120,170,255,0.18); '
        f'padding:0.5em 0.9em; border-radius:6px; border-left:5px solid #5b8def; '
        f'font-size:1.3rem; font-weight:600; margin:1.2em 0 0.6em 0;">{text}</div>',
        unsafe_allow_html=True,
    )


YAHOO_FINANCE_URL_TEMPLATE = "https://finance.yahoo.co.jp/quote/{code}"


def code_links(codes) -> list[str]:
    return [YAHOO_FINANCE_URL_TEMPLATE.format(code=code) for code in codes]


def abbreviate_market(market: str) -> str:
    """市場区分を一覧表示用に短縮する(東証プライム→Ｐ、スタンダード→Ｓ、グロース→Ｇ)。"""
    if not isinstance(market, str):
        return market
    if "プライム" in market:
        return "Ｐ"
    if "スタンダード" in market:
        return "Ｓ"
    if "グロース" in market:
        return "Ｇ"
    return market


def code_link_column_config() -> dict:
    """「コード」列をYahoo Financeの該当銘柄ページへのリンクにする(別タブで開く)。"""
    return {
        "コード": st.column_config.LinkColumn(
            "コード",
            display_text=r"https://finance\.yahoo\.co\.jp/quote/(.+)",
        )
    }


TENDENCY_DISPLAY_COLUMNS = ["出来高傾向", "継続率", "反落率"]


def add_tendency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    出来高比率に基づく過去データの傾向(継続率/反落率)を、列を分けて見やすく追加する。
    あくまで過去の統計的な傾向であり、個別銘柄の将来を予測するものではない。
    """
    df = df.copy()
    required = {"vol_tendency", "tendency_continuation_rate", "tendency_pullback_rate", "tendency_n"}
    if not required.issubset(df.columns):
        return df

    df["出来高傾向"] = df["vol_tendency"]
    df["継続率"] = df["tendency_continuation_rate"]
    df["反落率"] = df["tendency_pullback_rate"]
    return df


PERIOD_OPTIONS = {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "3年": "3y"}


def render_ma_chart(code: str, prior_high: float, key_prefix: str) -> None:
    """
    表示期間選択+終値+移動平均線+52週高値ラインのチャートを描画する
    (チェックボックス・ラジオボタンのkeyは重複しないようprefixで区別)。
    y軸の値幅は終値・移動平均線の実勢レンジに合わせる。52週高値が実勢から大きく離れている場合、
    そのままだとy軸が間延びして値動きが見づらくなるため、高値ラインは軸の外に出る分を切って表示する。
    """
    period_label = st.radio(
        "表示期間",
        list(PERIOD_OPTIONS.keys()),
        index=3,
        horizontal=True,
        key=f"{key_prefix}_period",
    )
    hist = fetch_chart_history(code, PERIOD_OPTIONS[period_label])
    if hist.empty:
        st.caption("チャート用データの取得に失敗しました(yfinanceがブロックされている可能性があります)。")
        return

    st.caption("移動平均線")
    ma_all_windows = [25, 75, 200]
    ma_default = {25: True, 75: True, 200: True}
    ma_cols = st.columns([1] * len(ma_all_windows) + [1, 7])
    ma_windows = [
        window
        for window, col in zip(ma_all_windows, ma_cols)
        if col.checkbox(f"{window}日", value=ma_default[window], key=f"{key_prefix}_ma_{window}")
    ]
    show_high_line = ma_cols[len(ma_all_windows)].checkbox(
        "52週高値", value=True, key=f"{key_prefix}_ma_52w_line"
    )

    ma_colors = {25: "#f4a261", 75: "#2a9d8f", 200: "#6c5ce7"}  # 窓ごとに見分けられる色にする
    chart_df = hist[["Close"]].rename(columns={"Close": "終値"})
    color_map = {"終値": "#e63946"}  # 終値を太く目立つ赤で強調
    for window in ma_windows:
        chart_df[f"{window}日移動平均"] = hist["Close"].rolling(window=window).mean()
        color_map[f"{window}日移動平均"] = ma_colors[window]

    price_values = pd.concat([chart_df[c] for c in chart_df.columns]).dropna()
    y_min, y_max = float(price_values.min()), float(price_values.max())
    pad = (y_max - y_min) * 0.05 or y_max * 0.05
    y_domain = [y_min - pad, y_max + pad]

    if show_high_line:
        chart_df["52週高値ライン"] = prior_high
        color_map["52週高値ライン"] = "#457b9d"

    chart_reset = chart_df.reset_index()
    date_col = chart_reset.columns[0]
    chart_long = chart_reset.melt(id_vars=date_col, var_name="系列", value_name="値")

    x_format = "%Y/%m" if period_label in ("1年", "3年") else "%Y/%m/%d"
    chart = (
        alt.Chart(chart_long)
        .mark_line(clip=True)
        .encode(
            x=alt.X(
                f"{date_col}:T",
                title=None,
                axis=alt.Axis(format=x_format, labelAngle=0),
            ),
            y=alt.Y("値:Q", title=None, scale=alt.Scale(domain=y_domain)),
            color=alt.Color(
                "系列:N",
                title=None,
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            ),
        )
    )
    st.altair_chart(chart, use_container_width=True)


st.set_page_config(page_title="安全成長・急成長株スクリーナー", layout="wide")

# 他のセクション(GPRS・上昇基調銘柄など)から「個別銘柄を分析」にコードを
# 引き継ぐ時は、text_input(key="analysis_query")が既にこのスクリプト内で
# 生成された後にst.session_state["analysis_query"]へ直接代入すると
# StreamlitAPIException(ウィジェット生成後はそのkeyを直接変更できない)になる。
# そのため一旦analysis_query_overrideに退避しておき、ウィジェットが生成される
# 前(スクリプトの一番最初)でここに反映する。
if "analysis_query_override" in st.session_state:
    st.session_state["analysis_query"] = st.session_state.pop("analysis_query_override")

st.title("📈 安全成長・急成長株スクリーナー")
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

_quality_freshness = quality_score.cache_freshness_summary()
if _quality_freshness:
    _latest_str = _quality_freshness["latest"].strftime("%Y-%m-%d")
    _oldest_str = _quality_freshness["oldest"].strftime("%Y-%m-%d")
    _freshness_range = (
        f"{_latest_str}"
        if _latest_str == _oldest_str
        else f"{_oldest_str}〜{_latest_str}"
    )
    st.caption(
        f"📅 財務指標(PER・ROE・GP・成長の質・時価総額等)のキャッシュ更新日: "
        f"{_freshness_range}({_quality_freshness['count']}銘柄、EDINET API v2)"
    )

master = load_tickers(TICKERS_FILE)
master_by_code = {t["code"]: t["name"] for t in master}
master_by_code_market = {t["code"]: t.get("market") for t in master}
master_by_code_sector = {t["code"]: t.get("sector33") for t in master}
all_labels = [f"{t['code']} - {t['name']}" for t in master]

SELECTION_KEY = "selected_labels"
if SELECTION_KEY not in st.session_state:
    # 全銘柄をデフォルト選択にする(初回読み込みは数十秒かかるが、結果はCACHE_TTL_SECONDS
    # 経過するまでキャッシュされるので、毎回は待たない)。
    st.session_state[SELECTION_KEY] = all_labels.copy()


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
    # 実際に表示中のデータがいつ時点のものかは、この後の計算(result)が終わるまで
    # 分からないため、st.emptyでプレースホルダーを確保しておき、計算が終わった時点で
    # 中身を書き込む(サイドバーのこの位置に、同じスクリプト実行内で反映される)。
    data_date_placeholder = st.empty()
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
        width="stretch",
        on_click=_add_picks_to_selection,
    )

    st.subheader("選択中の銘柄(外すと除外)")
    col_all, col_clear = st.columns(2)
    col_all.button("✅ 全選択", width="stretch", on_click=_select_all)
    col_clear.button("🗑️ 全解除", width="stretch", on_click=_clear_all)
    st.caption(f"全{len(all_labels)}銘柄中 {len(st.session_state[SELECTION_KEY])}銘柄を選択中")
    st.multiselect(
        "対象銘柄",
        options=all_labels,
        key=SELECTION_KEY,
        label_visibility="collapsed",
    )

    run_clicked = st.button("🔄 スクリーニング実行", type="primary", width="stretch")

    st.subheader("ファンダメンタルズ")
    used_calls, daily_limit = edinetdb.get_usage_today()
    st.caption(f"EDINET DB 本日の使用量: {used_calls} / {daily_limit} 回")

selected_codes = [label.split(" - ", 1)[0] for label in st.session_state[SELECTION_KEY]]


RUN_SCREEN_CACHE_TTL = 6 * 60 * 60  # 価格データはdaily_refresh.pyが1日1回更新するだけなので、長めに保つ


@st.cache_data(ttl=RUN_SCREEN_CACHE_TTL, show_spinner=False)
def run_screen(codes: tuple[str, ...], as_of: date) -> pd.DataFrame:
    rows = [{"code": c, "name": master_by_code.get(c, "")} for c in codes]
    return screen(rows, as_of=as_of)


@st.cache_data(ttl=300, show_spinner=False)
def load_precomputed_result() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(OUTPUT_FILE)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chart_history(code: str, period: str) -> pd.DataFrame:
    try:
        return yf.Ticker(code).history(period=period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_margin_tag(code: str) -> dict:
    sec_code = code.split(".")[0]
    fin_df = analysis.records_to_frame(financials_cache.get_financials_cached(sec_code))
    return analysis.summarize(fin_df) or {"error": "データなし"}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_edinet_v2_tag(code: str) -> dict:
    """
    EDINET公式API v2 から財務データを取得する(無料・制限なし)。
    edinetdb.jp(第三者API)では取れなかった営業利益・CF・大株主情報を補う。
    APIキーは環境変数 EDINET_API_KEY に設定済みであること。
    """
    sec_code = code.split(".")[0]
    try:
        from edinet_v2 import EdinetV2Client
        result = EdinetV2Client().get_financials(sec_code)
    except Exception as e:
        return {"error": str(e)}
    if "error" in result:
        return result
    # 営業利益率(%)を算出 — net_salesとoperating_profitが両方取れた場合のみ
    ns = result.get("net_sales")
    op = result.get("operating_profit")
    if isinstance(ns, (int, float)) and isinstance(op, (int, float)) and ns > 0:
        result["operating_margin_pct"] = op / ns * 100
    return result


fundamentals_fetch_results: dict[str, dict] = st.session_state.setdefault(
    "fundamentals_fetch_results", {}
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_quality_table_cached() -> pd.DataFrame:
    """
    品質データキャッシュ(cache/quality/*.json、EDINET API v2で全銘柄分バックフィル済み)から
    PER・ROE・増収増益・GP・成長の質・複合スコア・PBRをまとめて読み込む。
    ディスクを毎回スキャンしないよう1時間キャッシュする(バックフィルを再実行した直後に
    最新化したい場合は、アプリを再起動するかキャッシュのTTL切れを待つ)。
    """
    return finder.build_quality_table()


def _numeric_or_none(tag: dict, key: str) -> float | None:
    v = tag.get(key)
    return v if isinstance(v, (int, float)) else None


DEFAULT_LEGEND_LINES = [
    "増収/増益: ✅=直近決算が前年より増加 ❌=減少 ❓=取得エラー 未取得=まだ取得していない",
]
# 品質データキャッシュ(EDINET API v2、全銘柄バックフィル済み)から常時表示できる列の説明。
# クリックしての個別取得が不要なため、show_margin_detailに関わらず常に表示する。
QUALITY_ALWAYS_LEGEND_LINES = [
    "PER: 決算期末時点の値(EDINET API v2の5年サマリー表より)。現在株価ベースではない",
    "ROE(%)・PBR・増収増益・GP・成長の質・複合スコア: 品質データキャッシュ"
    "(全銘柄分バックフィル済み)より。PBRは価格キャッシュの最新終値ベース",
    "増収増益: 「増収」かつ「増益」が連続した期数(増収連続期数と増益連続期数の"
    "うち小さい方)。各銘柄の最新の有価証券報告書時点から遡って、最大5年分"
    "(4期分の前年比較)のデータに基づく最大値(最長4期)",
    "複合スコア: GP・成長の質は業種(33業種区分)中央値との差分、PBRは業種中央値との比率で"
    "業種補正したうえで、全銘柄でのパーセンタイル順位に変換して平均したもの。"
    f"「※」は同業種内の該当銘柄数が{quality_score.MIN_SECTOR_SAMPLE_SIZE}未満で"
    "業種中央値(補正の基準)の算出根拠が薄いことを示す(値自体は参考として表示)",
    "PBRが「債務超過」と表示される銘柄: BPS(1株純資産)がマイナス、つまり負債が資産を"
    "上回っている状態。「割安」という意味のマイナスではなく財務的な危険信号のため、"
    "複合スコアの計算からは除外している(判定不能扱い)",
    f"複合スコア(PBR): PBRが{quality_score.PBR_PENALTY_THRESHOLD}を下回るほど連続的に"
    "評価を下げている(本当に割安な優良株ではなく、市場が簿価の実現可能性を疑う"
    "「バリュートラップ」の可能性があるため)",
]
MARGIN_DETAIL_LEGEND_LINES = [
    "PEG: (取得時点の)PER÷直近1年の純利益成長率。1.0未満で割安とされることが多いが、単年度の成長率なので一時的な増減に振れやすい",
    "配当利回り: 1株配当÷株価。高いほど配当による株主還元が大きい",
    "リンチレシオ: (EPSの5年平均成長率% + 配当利回り%)÷(取得時点の)PER(ピーター・リンチ流)。1.0以上で割安、0.5未満で割高とされる目安(絶対基準ではない)",
    "営業利益率(%): 営業利益÷売上高。EDINET公式API(有価証券報告書)から取得。経常利益率との差が大きい場合は金融収支・特別損益の影響が強い",
    "営業CF(億円): 本業で稼いだキャッシュ。プラスが大きいほど資金繰りが安定している",
    "オーナー経営: ✅=創業者・大株主が経営に関与している可能性あり(上位株主に個人名)。確認は有価証券報告書の株主リストで",
]


def render_fundamentals_picker(
    df: pd.DataFrame,
    columns: list[str],
    key_prefix: str,
    show_margin_detail: bool = False,
    legend_lines: list[str] | None = None,
    render_legend: bool = True,
) -> None:
    """
    df[columns]を表示し、末尾にPER・ROE(%)・PBR・増収/増益連続期数・GP・成長の質・複合スコア
    (品質データキャッシュより、クリック不要で常時表示)と「増収」「増益」
    (show_margin_detail時は売上高変化・経常利益率変化・営業利益率変化・PEG・配当利回り・
    リンチレシオも)列を追加する。
    行をクリックするとその銘柄のチャートに切り替わり、「🔍 (銘柄名)の増収増益を取得」ボタンで
    その銘柄だけEDINET DB(+営業利益率変化のみYahoo Finance)から取得する。結果は
    fundamentals_fetch_results(銘柄コード共有)に保存するので、他の表で同じ
    銘柄を取得済みならそのまま再利用される。
    PEG・リンチレシオの分母のPERは、取得時点のEPS(分割調整済)と表示中の終値から計算する
    ライブ値(常時表示のPER列は決算期末時点の値のため、現在の評価とはズレることがある。
    それぞれ用途が異なるため別々に扱う)。
    PEGはPER÷純利益成長率(前年比%)。単年度の成長率なので一時的な増減に振れやすい。
    リンチレシオは(EPSの5年CAGR% + 配当利回り%)÷PER(ピーター・リンチ流)。複数年平均の
    成長率を使うことでPEGより単年度のブレに強く、配当を出す低成長の成熟企業も公平に評価できる。
    数値列は表のクリックソートが正しく数値順になるよう生の数値型のまま持たせ、書式は
    column_config(NumberColumn)側で付与する。算出に使う値がマイナス/ゼロ/未取得の場合は
    Noneにして空欄表示にする(取得状況自体は「増収」「増益」列の表示で分かる)。
    legend_linesを指定すると、表下部の注意書きを呼び出し元のカスタム内容に差し替えられる
    (未指定時はDEFAULT_LEGEND_LINES、show_margin_detail時はMARGIN_DETAIL_LEGEND_LINESも追加)。
    render_legend=Falseにすると、この関数内では注意書きを表示しない(呼び出し元で
    別途(expanderなどに)まとめて表示したい場合に使う)。
    """
    codes_in_order = df["code"].tolist()
    names_in_order = df["name"].tolist()
    closes_in_order = df["close"].tolist()
    revenue_col, profit_col = [], []
    revenue_change_col, margin_change_col = [], []
    op_margin_col, operating_cf_col, owner_col = [], [], []
    peg_col: list[float | None] = []
    div_yield_col, lynch_col = [], []
    for code, close_price in zip(codes_in_order, closes_in_order):
        tag = fundamentals_fetch_results.get(code)
        if tag is None:
            revenue_col.append("未取得")
            profit_col.append("未取得")
            revenue_change_col.append(None)
            margin_change_col.append(None)
            op_margin_col.append(None)
            operating_cf_col.append(None)
            owner_col.append("未取得")
            peg_col.append(None)
            div_yield_col.append(None)
            lynch_col.append(None)
        elif "error" in tag:
            revenue_col.append("❓")
            profit_col.append("❓")
            revenue_change_col.append(None)
            margin_change_col.append(None)
            op_margin_col.append(None)
            operating_cf_col.append(None)
            owner_col.append("❓")
            peg_col.append(None)
            div_yield_col.append(None)
            lynch_col.append(None)
        else:
            revenue_col.append("✅" if tag.get("revenue_growing") else "❌")
            profit_col.append("✅" if tag.get("profit_growing") else "❌")
            revenue_change_col.append(_numeric_or_none(tag, "revenue_yoy_pct"))
            margin_change_col.append(_numeric_or_none(tag, "ordinary_income_margin_change_pt"))
            # 営業利益率(%) — EDINET公式APIから取得(yfinanceより信頼性高い)
            op_margin_col.append(_numeric_or_none(tag, "operating_margin_pct"))
            # 営業CF(億円単位に変換して表示)
            op_cf = tag.get("operating_cf")
            operating_cf_col.append(
                round(op_cf / 1e8, 1) if isinstance(op_cf, (int, float)) else None
            )
            # オーナー経営
            is_owner = tag.get("is_owner_managed")
            owner_col.append(
                "✅" if is_owner is True else "❌" if is_owner is False else "―"
            )
            eps = tag.get("latest_adjusted_eps")
            dividend = tag.get("latest_dividend_per_share")
            # PEG・リンチレシオの分母専用のライブPER(取得時点のEPS÷表示中の終値)。
            # 常時表示の「PER」列(品質データキャッシュ、決算期末時点)とは別物として扱う。
            per_value = close_price / eps if isinstance(eps, (int, float)) and eps > 0 else None
            div_yield_value = (
                dividend / close_price * 100
                if isinstance(dividend, (int, float)) and dividend > 0 and close_price > 0
                else None
            )
            div_yield_col.append(div_yield_value)

            profit_growth = tag.get("net_income_yoy_pct")
            if per_value is not None and isinstance(profit_growth, (int, float)) and profit_growth > 0:
                peg_col.append(per_value / profit_growth)
            else:
                peg_col.append(None)

            eps_cagr_pct = tag.get("eps_cagr_pct")
            lynch_numerator = (eps_cagr_pct or 0) + (div_yield_value or 0)
            if per_value is not None and lynch_numerator > 0 and (eps_cagr_pct is not None or div_yield_value is not None):
                lynch_col.append(lynch_numerator / per_value)
            else:
                lynch_col.append(None)

    display_df = df[columns].rename(columns=COLUMN_LABELS_JA)
    if "コード" in display_df.columns:
        display_df["コード"] = code_links(codes_in_order)

    # PER・ROE・PBR・増収増益・GP・成長の質・複合スコアは、品質データキャッシュ
    # (EDINET API v2、全銘柄バックフィル済み)から取得済みのためクリック不要で常時表示する。
    quality_table = load_quality_table_cached()
    if not quality_table.empty:
        quality_by_code = quality_table.set_index("code")
        quality_slice = quality_by_code.reindex(codes_in_order)
        display_df["PER"] = quality_slice["latest_per"].to_numpy()
        display_df["ROE(%)"] = (quality_slice["latest_roe_official"] * 100).to_numpy()
        display_df["増収増益"] = _combined_growth_streak(
            quality_slice["revenue_streak"], quality_slice["net_income_streak"]
        ).to_numpy()
        display_df["PBR"] = (
            _format_pbr(quality_slice["live_pbr"], missing_label=QUALITY_INDETERMINATE_LABEL)
            if "live_pbr" in quality_slice.columns
            else QUALITY_INDETERMINATE_LABEL
        )
        display_df["GP(%)"] = _format_quality_value(quality_slice["gross_profitability"] * 100)
        display_df["成長の質(%)"] = _format_quality_value(quality_slice["growth_quality_raw"] * 100)
        display_df["複合スコア"] = _format_composite_score(
            quality_slice["composite_score"], quality_slice.get("composite_score_low_sample")
        )

    if show_margin_detail:
        display_df["売上高変化"] = revenue_change_col
        display_df["経常利益率変化"] = margin_change_col
        display_df["営業利益率(%)"] = op_margin_col
        display_df["営業CF(億円)"] = operating_cf_col
        display_df["オーナー経営"] = owner_col
        display_df["PEG"] = peg_col
        display_df["配当利回り"] = div_yield_col
        display_df["リンチレシオ"] = lynch_col
    display_df["増収"] = revenue_col
    display_df["増益"] = profit_col

    event = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            **number_column_config(display_df.columns),
            **quality_text_column_config(display_df.columns),
            **code_link_column_config(),
        },
        key=f"{key_prefix}_table",
    )

    if legend_lines is None:
        legend_lines = list(DEFAULT_LEGEND_LINES) + list(QUALITY_ALWAYS_LEGEND_LINES)
        if show_margin_detail:
            legend_lines += MARGIN_DETAIL_LEGEND_LINES
    if render_legend:
        for line in legend_lines:
            st.caption(line)

    rows = event.selection["rows"]
    if not rows:
        st.caption("行をクリックすると、その銘柄のチャートを表示し、増収増益を取得できます。")
        return

    selected_code = codes_in_order[rows[0]]
    selected_name = names_in_order[rows[0]]
    st.session_state["chart_select"] = f"{selected_code} - {selected_name}"

    if st.button(f"🔍 {selected_name}の増収増益を取得", key=f"{key_prefix}_fetch_button"):
        try:
            tag = fetch_margin_tag(selected_code)
        except edinetdb.QuotaExceededError as e:
            tag = {"error": str(e)}
            st.warning(str(e))
        except Exception as e:
            tag = {"error": str(e)}
        if "error" not in tag:
            # EDINET公式APIから営業利益・CF・大株主情報を補完
            edinet_v2 = fetch_edinet_v2_tag(selected_code)
            if "error" not in edinet_v2:
                tag.update({k: edinet_v2[k] for k in (
                    "operating_profit", "operating_margin_pct",
                    "operating_cf", "investing_cf", "financing_cf",
                    "is_owner_managed", "major_shareholders",
                ) if k in edinet_v2})
        fundamentals_fetch_results[selected_code] = tag
        st.rerun()


section_heading("🔍 銘柄スクリーニング")
st.caption(
    "キーワード・市場・時価総額・財務指標(PER・ROE・増収増益)・"
    "品質・割安・成長(グロス・プロフィタビリティ・PBR・成長の質・複合スコア)を"
    "組み合わせて条件検索できます(何も入力しなければ絞り込みなし、全銘柄が対象)。"
    "条件を入力したら「🔍 スクリーニング実行」ボタンを押してください。"
)

sc1, sc2, sc3 = st.columns(3)
screen_keyword = sc1.text_input(
    "キーワード(銘柄名・業種)", key="screen_keyword", placeholder="例: 半導体 / 銀行 / 食品"
)
market_options = sorted({t["market"] for t in master if t.get("market")})
screen_markets = sc2.multiselect("市場", market_options, key="screen_markets")
screen_market_caps = sc3.multiselect(
    "時価総額", finder.MARKET_CAP_BUCKET_LABELS, key="screen_market_caps",
    help="品質データキャッシュ(発行済株式数×直近終値)から算出。未キャッシュの銘柄は対象外になります。",
)

quality_cached_count = len(quality_score.list_quality_cached_codes())
if quality_cached_count == 0:
    st.info(
        "品質スコア用データがまだキャッシュされていません。"
        "先にターミナルで `python backfill_quality_universe.py` を実行してください"
        "(公式EDINET APIで全銘柄分を取得します。数時間規模の処理です)。"
    )

st.caption(f"📑 財務指標(PER・ROE・増収増益。財務データキャッシュ済み: {quality_cached_count}銘柄が対象)")
ff1, ff2, ff3 = st.columns(3)
per_input = ff1.number_input(
    "PER以下", min_value=0.0, value=0.0, step=1.0, key="screen_max_per",
    help="株価収益率。株価÷EPS(1株利益)。決算期末時点の値(EDINET API v2の5年サマリー表より、"
    "現在株価ベースではない)。低いほど割安とされるが業種により目安が異なる。"
    "0を指定すると絞り込みなし",
)
roe_input = ff2.number_input(
    "ROE(%) 以上", min_value=0.0, value=0.0, step=1.0, key="screen_min_roe",
    help="自己資本利益率。純利益÷自己資本(株主資本)。高いほど資本効率が良いとされる"
    "(品質データキャッシュより)。0を指定すると絞り込みなし",
)
screen_max_per = per_input or None
screen_min_roe = roe_input or None
screen_min_growth_streak = int(
    ff3.number_input(
        "増収増益 ◯期以上連続", min_value=0, value=1, step=1, key="screen_min_growth_streak",
        help="増収(売上高が前年比プラス)・増益(純利益が前年比プラス)の両方が指定期数以上"
        "連続している銘柄に絞り込む",
    )
)
screen_min_rev_streak = screen_min_profit_streak = screen_min_growth_streak

st.caption(f"💎 品質・割安・成長(GP・PBR・成長の質・複合スコア。品質データキャッシュ済み: {quality_cached_count}銘柄が対象)")
qf1, qf2, qf3, qf4 = st.columns(4)
q_min_composite_raw = qf1.slider(
    "複合スコア 以上", min_value=0, max_value=100, value=0, step=5, key="screen_min_composite",
)
screen_min_composite = q_min_composite_raw or None
q_min_gp_raw = qf2.number_input(
    "GP(%) 以上", min_value=0.0, value=0.0, step=1.0, key="screen_min_gp",
    help="GP(グロス・プロフィタビリティ)：資産をどれだけ効率よく使って粗利益を稼げているかを"
    "示す指標。中央値は約26%。30%超えで平均並み以上、40%超えで上位25%程度、"
    "60%超えで上位10%程度の水準。",
)
screen_min_gp = q_min_gp_raw or None
q_max_pbr_raw = qf3.number_input(
    "PBR以下", min_value=0.0, value=20.0, step=0.1, key="screen_max_pbr",
    help="0を指定すると絞り込みなし。価格キャッシュが無い銘柄はPBR未計算のため対象外になります。",
)
screen_max_pbr = q_max_pbr_raw or None
q_min_growth_raw = qf4.number_input(
    "成長の質(総資産比%) 以上", min_value=-100.0, value=0.0, step=1.0, key="screen_min_growth",
    help="成長の質：(配当・自己株買い・借入返済等の支出 − 増資・新規借入等の収入)÷総資産。"
    "プラスが大きいほど、増資や借入に頼らず自己資金(営業キャッシュフロー)で株主還元しながら"
    "成長できていることを示す。中央値は約+1.9%。0%超え(プラス)であれば自己資金型の成長・"
    "還元ができている水準、+3%超えで上位25%程度、+6%超えで上位10%程度の水準。マイナスは"
    "増資・借入への依存度が高いことを示す。",
)
screen_min_growth = q_min_growth_raw if q_min_growth_raw > -100.0 else None

if st.button("🔍 スクリーニング実行", key="screen_run_button", type="primary"):
    st.session_state["screen_candidates"] = finder.filter_combined(
        master,
        keyword=screen_keyword,
        markets=screen_markets,
        market_cap_buckets=screen_market_caps,
        max_per=screen_max_per,
        min_roe_pct=screen_min_roe,
        min_revenue_streak=screen_min_rev_streak,
        min_profit_streak=screen_min_profit_streak,
        min_composite_score=screen_min_composite,
        min_gross_profitability_pct=screen_min_gp,
        max_pbr=screen_max_pbr,
        min_growth_quality_pct=screen_min_growth,
    )

screen_candidates = st.session_state.get("screen_candidates")

if screen_candidates is None:
    st.caption("条件を入力して「🔍 スクリーニング実行」ボタンを押してください。")
elif screen_candidates.empty:
    st.caption(f"該当: {len(screen_candidates)}件")
    st.info("該当する銘柄がありません。財務・品質条件を使っている場合、キャッシュ済み銘柄の中に該当がない可能性があります。")
else:
    st.caption(f"該当: {len(screen_candidates)}件")
    s_display = screen_candidates.copy()
    rename_map = {
        "code": "コード", "name": "銘柄名", "market": "市場", "sector33": "業種",
        "latest_per": "PER",
    }
    if "market" in s_display.columns:
        s_display["market"] = s_display["market"].map(abbreviate_market)
    if "latest_roe_official" in s_display.columns:
        s_display["ROE(%)"] = (s_display["latest_roe_official"] * 100).round(2)
    if "revenue_streak" in s_display.columns and "net_income_streak" in s_display.columns:
        s_display["増収増益"] = _combined_growth_streak(
            s_display["revenue_streak"], s_display["net_income_streak"]
        )
    if "gross_profitability" in s_display.columns:
        s_display["GP(%)"] = _format_quality_value(s_display["gross_profitability"] * 100)
    if "growth_quality_raw" in s_display.columns:
        s_display["成長の質(%)"] = _format_quality_value(s_display["growth_quality_raw"] * 100)
    if "composite_score" in s_display.columns:
        s_display["複合スコア"] = _format_composite_score(
            s_display["composite_score"], s_display.get("composite_score_low_sample")
        )
    if "live_pbr" in s_display.columns:
        s_display["PBR"] = _format_pbr(s_display["live_pbr"], missing_label=QUALITY_INDETERMINATE_LABEL)
    if "market_cap_bucket" in s_display.columns:
        s_display["時価総額"] = s_display["market_cap_bucket"]
    display_cols = [
        c for c in [
            "code", "name", "market", "sector33", "時価総額",
            "latest_per", "ROE(%)", "増収増益",
            "GP(%)", "PBR", "成長の質(%)", "複合スコア",
        ]
        if c in s_display.columns
    ]
    screen_display = s_display[display_cols].rename(columns=rename_map)
    screen_display["コード"] = code_links(s_display["code"].tolist())
    screen_event = st.dataframe(
        screen_display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            **number_column_config(screen_display.columns),
            **quality_text_column_config(screen_display.columns),
            **code_link_column_config(),
        },
        key="screen_table",
    )
    screen_rows = screen_event.selection["rows"]
    if screen_rows:
        screen_selected_code = s_display["code"].iloc[screen_rows[0]]
        screen_selected_name = s_display["name"].iloc[screen_rows[0]]
        if st.button(f"🔍 {screen_selected_name}を「個別銘柄を分析」で開く", key="screen_open_analysis"):
            st.session_state["analysis_query_override"] = screen_selected_code.split(".")[0]
            st.session_state["analysis_ready"] = True
            st.rerun()

    # ── 決算サプライズ一括取得 ──────────────────────────────
    with st.expander("📈 決算サプライズ分析(候補銘柄一覧)", expanded=False):
        st.caption(
            "候補銘柄のEPSサプライズ履歴(対コンセンサス)を取得します。"
            "銘柄数が多いと取得に時間がかかります(1銘柄あたり約2〜3秒)。"
        )
        max_fetch = st.slider("取得上限銘柄数", 5, 50, 20, key="screen_surprise_max")
        if st.button("📥 サプライズデータを取得", key="screen_surprise_fetch"):
            codes = screen_candidates["code"].tolist()[:max_fetch]
            prog = st.progress(0, text="取得中...")
            rows_surp = []
            for i, code in enumerate(codes):
                try:
                    d = earnings_surprise.get_surprise_data(code)
                    rows_surp.append({
                        "コード":          code,
                        "最新サプライズ%":  d.get("consensus_latest"),
                        "過去平均%":        d.get("consensus_avg_pct"),
                        "上振れトレンド":   d.get("consensus_trend", "─"),
                        "来期予想EPS":      d.get("next_forecast_eps"),
                    })
                except Exception:
                    rows_surp.append({"コード": code, "最新サプライズ%": None,
                                      "過去平均%": None, "上振れトレンド": "エラー",
                                      "来期予想EPS": None})
                prog.progress((i + 1) / len(codes), text=f"{code} 取得中... ({i+1}/{len(codes)})")
            prog.empty()
            st.session_state["screen_surprise_df"] = pd.DataFrame(rows_surp)

        if "screen_surprise_df" in st.session_state and not st.session_state["screen_surprise_df"].empty:
            surp_df = st.session_state["screen_surprise_df"]

            def _trend_icon(t: str) -> str:
                return {"加速": "◎ 加速", "横ばい": "△ 横ばい", "減速": "▽ 減速"}.get(t, t)
            surp_df["上振れトレンド"] = surp_df["上振れトレンド"].apply(_trend_icon)

            st.dataframe(
                surp_df,
                hide_index=True,
                column_config={
                    "最新サプライズ%":  st.column_config.NumberColumn(format="%.1f%%"),
                    "過去平均%":        st.column_config.NumberColumn(format="%.1f%%"),
                    "来期予想EPS":      st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                "最新サプライズ% / 過去平均%：対コンセンサス(yfinance)。"
                "◎加速 = 今回が過去平均を+3pt以上上回る。"
                "来期予想EPS：株探の会社予想値。"
            )

    if "market_cap_bucket" in screen_candidates.columns:
        st.caption(
            "時価総額：品質データキャッシュの発行済株式数(自己株式控除後)×直近終値で計算。"
            "500億円以下/500億〜2000億円/2000億〜5000億円/5000億円超、の4区分。"
            "品質データが無い銘柄は空欄になります。"
        )

    if "gross_profitability" in screen_candidates.columns or "growth_quality_raw" in screen_candidates.columns:
        st.caption(
            "GP(グロス・プロフィタビリティ)：資産をどれだけ効率よく使って粗利益を稼げているかを"
            "示す指標。中央値は約26%。30%超えで平均並み以上、40%超えで上位25%程度、"
            "60%超えで上位10%程度の水準。"
        )
        st.caption(
            "成長の質：(配当・自己株買い・借入返済等の支出 − 増資・新規借入等の収入)÷総資産。"
            "プラスが大きいほど、増資や借入に頼らず自己資金(営業キャッシュフロー)で株主還元しながら"
            "成長できていることを示す。中央値は約+1.9%。0%超え(プラス)であれば自己資金型の成長・"
            "還元ができている水準、+3%超えで上位25%程度、+6%超えで上位10%程度の水準。マイナスは"
            "増資・借入への依存度が高いことを示す。"
        )
        st.caption(
            "上記の目安は、この一覧の元になっている品質データキャッシュの現時点の"
            "分布から算出した参考値であり、固定の合格ラインではありません。"
            "バックフィルが進む(全銘柄のデータが揃う)につれて数値は変動します。"
        )

        with st.expander("📊 業種別のグロス・プロフィタビリティ水準(中央値)", expanded=False):
            st.caption(
                "業種(33業種区分)による資産集約度の違いが大きいため、絶対水準ではなく"
                "同業種内での相対比較の参考にしてください。品質データキャッシュ済みの"
                "全銘柄(現在のキーワード等の絞り込みには依存しません)を対象に算出しています。"
            )
            industry_table = finder.build_quality_table()
            if industry_table.empty:
                st.caption("業種別集計に使えるデータがまだありません。")
            else:
                sector_stats = (
                    industry_table.dropna(subset=["gross_profitability", "sector33"])
                    .groupby("sector33")["gross_profitability"]
                    .agg(median="median", count="count")
                    .sort_values("median", ascending=False)
                )
                sector_stats["グロス・プロフィタビリティ 中央値(%)"] = (sector_stats["median"] * 100).round(1)
                sector_stats["銘柄数"] = sector_stats["count"].astype(int)
                sector_display = sector_stats[["グロス・プロフィタビリティ 中央値(%)", "銘柄数"]].reset_index()
                sector_display = sector_display.rename(columns={"sector33": "業種(33業種区分)"})
                st.dataframe(sector_display, width="stretch", hide_index=True)

st.divider()

def _close_analysis_dialog() -> None:
    st.session_state["analysis_ready"] = False


@st.dialog("🔍 個別銘柄を分析", width="large", on_dismiss=_close_analysis_dialog)
def _show_stock_analysis_dialog(analysis_code: str, analysis_name: str) -> None:
    with st.spinner(f"{analysis_name}を分析中..."):
        try:
            single_result = run_screen((analysis_code,), as_of_date)
        except Exception:
            single_result = pd.DataFrame()
        single_fallback = False
        single_as_of = as_of_date
        if (single_result is None or single_result.empty) and CLOUD_MODE:
            full_result = load_precomputed_result()
            if not full_result.empty:
                single_fallback = True
                single_as_of = full_result["date"].max()
                single_result = full_result[full_result["code"] == analysis_code].reset_index(drop=True)

    if single_result is None or single_result.empty:
        st.warning(f"{analysis_name}({analysis_code})のデータを取得できませんでした。")
        return

    st.subheader(f"{analysis_name}({analysis_code})")
    row = single_result.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("終値", f"{int(row['close']):,}円")
    m2.metric("52週高値からの差分", f"{row['pct_vs_prior_high']:.2f}%")
    m3.metric("本日52週高値更新", "✅" if row["new_52w_high"] else "❌")
    if pd.notna(row.get("trading_days_since_high")):
        m4.metric("最新更新からの営業日数", f"{int(row['trading_days_since_high'])}日")

    if pd.notna(row.get("vol_tendency")):
        st.info(
            f"出来高傾向: {row['vol_tendency']} → 過去データで継続"
            f"{row['tendency_continuation_rate']:.2f}%/反落{row['tendency_pullback_rate']:.2f}%"
            f"(n={int(row['tendency_n'])}、個別銘柄の予測ではなく過去統計です)"
        )

    if single_fallback:
        st.caption(
            f"ライブ取得に失敗したため、ローカルPCの最終更新データ(基準日: {single_as_of})を表示しています。"
        )

    try:
        splits = yf.Ticker(analysis_code).splits
    except Exception:
        splits = None
    if splits is not None and not splits.empty:
        recent = splits.tail(5)
        split_desc = "、".join(
            f"{d.strftime('%Y-%m-%d')}({'1→' + f'{r:g}' if r >= 1 else f'{1/r:g}→1'})"
            for d, r in zip(recent.index, recent.values)
        )
        st.caption(f"📐 株式分割: 過去{len(splits)}回(直近{len(recent)}件: {split_desc})")
    else:
        st.caption("📐 株式分割: 履歴なし")

    # ── 決算サプライズ分析 ──────────────────────────────────
    if st.checkbox("📈 決算サプライズ分析(コンセンサス比 + 会社予想EPS)", key="analysis_surprise"):
        with st.spinner("決算データを取得中..."):
            try:
                surp = earnings_surprise.get_surprise_data(analysis_code)
            except Exception as e:
                surp = {}
                st.warning(f"取得失敗: {e}")

        if surp:
            latest  = surp.get("consensus_latest")
            avg_pct = surp.get("consensus_avg_pct")
            trend   = surp.get("consensus_trend", "データなし")
            trend_icon = {"加速": "◎", "横ばい": "△", "減速": "▽"}.get(trend, "─")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(
                "最新EPSサプライズ%(対コンセンサス)",
                f"{latest:+.1f}%" if latest is not None else "─",
            )
            sc2.metric(
                "過去平均サプライズ%",
                f"{avg_pct:+.1f}%" if avg_pct is not None else "─",
            )
            sc3.metric(
                "上振れトレンド",
                f"{trend_icon} {trend}",
                help="今回のサプライズ%が過去平均より+3pt以上なら「加速」",
            )

            # 四半期履歴テーブル
            hist = surp.get("consensus_history", [])
            if hist:
                st.caption("📋 四半期コンセンサスサプライズ履歴(新しい順・対コンセンサス)")
                hist_df = pd.DataFrame(hist[:12]).rename(columns={
                    "date":         "発表日",
                    "eps_estimate": "予想EPS",
                    "eps_actual":   "実績EPS",
                    "surprise_pct": "サプライズ%",
                })
                # 平均ラインを超えているかフラグ
                if avg_pct is not None:
                    hist_df["平均超え"] = hist_df["サプライズ%"].apply(
                        lambda v: "◎" if (v is not None and v > avg_pct + 3) else
                                  ("▽" if (v is not None and v < avg_pct - 3) else "─")
                    )
                st.dataframe(
                    hist_df,
                    hide_index=True,
                    column_config={
                        "サプライズ%": st.column_config.NumberColumn(format="%+.1f%%"),
                        "予想EPS":    st.column_config.NumberColumn(format="%.2f"),
                        "実績EPS":    st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            # 年次EPS推移テーブル(株探)
            annual = surp.get("annual_eps", [])
            if annual:
                st.caption("📊 年次EPS推移(株探) ─ 会社予想 vs 実績")
                ann_df = pd.DataFrame(annual).rename(columns={
                    "period":      "決算期",
                    "eps":         "修正1株益(EPS)",
                    "is_forecast": "区分",
                })
                ann_df["区分"] = ann_df["区分"].map({True: "🔮 会社予想", False: "✅ 実績"})
                st.dataframe(ann_df, hide_index=True,
                             column_config={"修正1株益(EPS)": st.column_config.NumberColumn(format="%.1f")})

            next_eps    = surp.get("next_forecast_eps")
            next_period = surp.get("next_forecast_period")
            if next_eps and next_period:
                st.info(
                    f"来期会社予想EPS ({next_period}): **{next_eps:.1f}円**　"
                    f"(株探より。コンセンサスとは異なる場合があります)"
                )

            st.caption(
                "サプライズ%はyfinance(アナリストコンセンサス比)。"
                "年次EPSは株探からスクレイピング。データ取得タイミングにより最新情報と差がある場合があります。"
            )

    if st.checkbox("📑 財務・業績分析を取得(EDINET DB)", key="analysis_fundamentals"):
        fin_records = []
        try:
            fin_records = financials_cache.get_financials_cached(analysis_code.split(".")[0])
        except edinetdb.QuotaExceededError as e:
            st.warning(str(e))
        except Exception as e:
            st.warning(f"EDINET DB取得失敗: {e}")

        fin_df = analysis.records_to_frame(fin_records)
        fg_summary = analysis.summarize(fin_df)
        if fg_summary:
            fcol1, fcol2 = st.columns(2)
            fcol1.metric("増収", "✅" if fg_summary.get("revenue_growing") else "❌")
            fcol2.metric("増益", "✅" if fg_summary.get("profit_growing") else "❌")

        if not fin_df.empty:
            fin_df = fin_df.set_index("fiscal_year")

            revenue_roe_cols = {}
            if "revenue" in fin_df.columns:
                revenue_roe_cols["売上高(百万円)"] = fin_df["revenue"] / 1e6
            if "roe_official" in fin_df.columns:
                revenue_roe_cols["ROE(%)"] = fin_df["roe_official"] * 100
            if revenue_roe_cols:
                st.caption("📊 売上高・ROEの推移")
                st.dataframe(pd.DataFrame(revenue_roe_cols).round(1), width="stretch")

            profit_cols = [c for c in ["ordinary_income", "net_income"] if c in fin_df.columns]
            if profit_cols:
                st.caption("📊 利益の推移(百万円、営業利益はEDINET DBで非提供のため経常利益・純利益のみ)")
                profit_df = (fin_df[profit_cols] / 1e6).rename(columns={
                    "ordinary_income": "経常利益",
                    "net_income": "純利益",
                })
                st.bar_chart(profit_df)

            if "equity_ratio_official" in fin_df.columns:
                st.caption("📈 自己資本比率の推移(%)")
                st.line_chart((fin_df["equity_ratio_official"] * 100).rename("自己資本比率"))
        else:
            st.caption("財務データが取得できませんでした。")

    render_ma_chart(analysis_code, row["prior_52w_high"], key_prefix="analysis")


def _close_gprs_detail_dialog() -> None:
    st.session_state["gprs_detail_ready"] = False


@st.dialog("📊 GPRS詳細分析", width="large", on_dismiss=_close_gprs_detail_dialog)
def _show_gprs_detail_dialog(row: dict, sector_medians: dict) -> None:
    """
    GPRSの合計点だけでは分からない「なぜこの点数なのか」を、GQS・GPS・PRS・
    リスク調整それぞれのサブスコアまで分解して表示する。growth_scores.pyの
    BANDS定数を直接使って、各サブスコアの実際の値がどのランクに該当したかを示す。
    同業種内での中央値との比較、株価チャートも合わせて表示する。
    """
    name = row.get("name") or ""
    code_full = str(row.get("code") or "")
    code = code_full.split(".")[0]
    st.subheader(f"{name}({code})")
    if row.get("period_end"):
        st.caption(f"財務データ基準: {row['period_end']}期(決算短信・有価証券報告書ベース)")

    gprs = row.get("gprs")
    grade = row.get("gprs_grade")
    sector_rank = row.get("gprs_sector_rank")
    sector_size = row.get("gprs_sector_size")
    m1, m2, m3 = st.columns(3)
    m1.metric("GPRS総合", f"{gprs:.1f}点" if pd.notna(gprs) else QUALITY_INDETERMINATE_LABEL)
    m2.metric("評価", grade or QUALITY_INDETERMINATE_LABEL)
    if pd.notna(sector_rank) and pd.notna(sector_size):
        m3.metric("業種内順位", f"{int(sector_rank)}位/{int(sector_size)}社中")
    if sector_medians and sector_medians.get("count"):
        st.caption(
            f"同業種({row.get('sector33') or '不明'}、{int(sector_medians['count'])}社)の中央値: "
            f"GQS {sector_medians['gqs']:.1f} ／ GPS {sector_medians['gps']:.1f} ／ "
            f"PRS {sector_medians['prs']:.1f} ／ GPRS {sector_medians['gprs']:.1f}"
        )

    gp = row.get("gross_profitability")
    gp_growth = row.get("gp_growth_rate")
    fcf_margin = row.get("fcf_margin")
    streak = row.get("revenue_streak")
    gp_s = growth_scores.gp_score(gp)
    gp_growth_s = growth_scores.gross_profit_growth_score(gp_growth)
    fcf_s = growth_scores.fcf_score(fcf_margin)
    stability_s = growth_scores.stability_score(streak)

    lynch = row.get("lynch_ratio")
    fcf_yield_value = row.get("fcf_yield")
    lynch_s = growth_scores.lynch_score(lynch)
    fcf_yield_s = growth_scores.fcf_yield_score(fcf_yield_value)
    net_income_growth_latest = row.get("net_income_growth_latest")
    net_income_growth_pct = net_income_growth_latest * 100 if net_income_growth_latest is not None else None
    dividend_yield_pct = row.get("dividend_yield_pct")
    psr_growth = row.get("psr_growth_ratio")
    psr_growth_s = growth_scores.psr_growth_score(psr_growth)
    psr_value = row.get("psr")

    momentum_s = row.get("momentum_score")
    trend_s = row.get("trend_score")
    performance_s = row.get("performance_score")

    equity_ratio = row.get("latest_equity_ratio")
    growth_quality_raw = row.get("growth_quality_raw")
    per = row.get("latest_per")
    equity_penalty = growth_scores.equity_ratio_penalty(equity_ratio)
    growth_quality_pen = growth_scores.growth_quality_penalty(growth_quality_raw)
    per_penalty = growth_scores.per_safety_penalty(per)
    deficit_years = row.get("deficit_years_last_5y")
    deficit_pen = growth_scores.deficit_penalty(deficit_years)

    def _bucket_header(label: str, max_score: int, value: float | None, median_key: str) -> str:
        base = f"{value:.1f}/{max_score}点" if pd.notna(value) else "判定不能"
        median = sector_medians.get(median_key) if sector_medians else None
        if median is not None and pd.notna(median):
            base += f"(業種中央値 {median:.1f})"
        return f"#### {label} 内訳 — {base}"

    st.markdown(_bucket_header("GQS(Growth Quality Score、参考情報・GPRS合計には含まない)", 40, row.get("gqs"), "gqs"))
    st.caption(
        "バックテストでGQSの4要素がいずれも将来リターンとほぼ無相関〜わずかにマイナスだったため、"
        "GPRSの合計には含めていない(質のスクリーニング用の目安として表示)。"
    )
    _render_component_table([
        ("GPスコア(粗利÷総資産)", f"{gp*100:.1f}%" if gp is not None else "ー", gp_s, 10,
         _band_note(gp, growth_scores.GP_SCORE_BANDS),
         _item_tooltip("gp_score", _band_reference_text(growth_scores.GP_SCORE_BANDS))),
        ("粗利成長スコア(YoY)", f"{gp_growth*100:.1f}%" if gp_growth is not None else "ー", gp_growth_s, 10,
         _band_note(gp_growth, growth_scores.GROSS_PROFIT_GROWTH_SCORE_BANDS),
         _item_tooltip("gp_growth", _band_reference_text(growth_scores.GROSS_PROFIT_GROWTH_SCORE_BANDS))),
        ("FCFスコア(FCFマージン)", f"{fcf_margin*100:.1f}%" if fcf_margin is not None else "ー", fcf_s, 10,
         _band_note(fcf_margin, growth_scores.FCF_SCORE_BANDS),
         _item_tooltip("fcf_score", _band_reference_text(growth_scores.FCF_SCORE_BANDS))),
        ("安定性係数(増収連続期数)", f"{int(streak)}期" if streak is not None else "ー", stability_s, 10,
         f"{int(streak)}期連続" if streak is not None else "データなし",
         _item_tooltip("stability", "3期以上: 10点 / 2期: 8点 / 1期: 5点 / 0期: 2点")),
    ])

    revenue_growth_pct_for_psr = (
        row.get("revenue_growth_latest") * 100 if row.get("revenue_growth_latest") is not None else None
    )
    st.markdown(_bucket_header("GPS(Growth Potential Score)", 30, row.get("gps"), "gps"))
    _render_component_table([
        ("リンチレシオスコア", f"{lynch:.2f}" if lynch is not None else "ー", lynch_s, 15,
         _lynch_note(lynch, per, net_income_growth_pct, dividend_yield_pct),
         _item_tooltip("lynch", _band_reference_text(growth_scores.LYNCH_SCORE_BANDS, as_pct=False))),
        ("PSR成長対比スコア(リンチレシオ代替)", f"{psr_growth:.2f}" if psr_growth is not None else "ー",
         None if lynch is not None else psr_growth_s, 15,
         _psr_growth_note(psr_growth, lynch, psr_value, revenue_growth_pct_for_psr),
         _item_tooltip("psr_growth", _band_reference_text(growth_scores.PSR_GROWTH_SCORE_BANDS, as_pct=False))),
        ("FCF利回りスコア", f"{fcf_yield_value*100:.1f}%" if fcf_yield_value is not None else "ー", fcf_yield_s, 10,
         _band_note(fcf_yield_value, growth_scores.FCF_YIELD_SCORE_BANDS),
         _item_tooltip("fcf_yield", _band_reference_text(growth_scores.FCF_YIELD_SCORE_BANDS))),
    ])

    close = row.get("close")
    prior_52w_high = row.get("prior_52w_high")
    momentum_ratio = (close / prior_52w_high) if close and prior_52w_high else None
    above_ma200 = row.get("above_ma200")
    ma200_rising = row.get("ma200_rising")
    trend_desc = (
        "200日線より上・上向き" if above_ma200 and ma200_rising
        else "200日線より上のみ" if above_ma200
        else "200日線が上向きのみ" if ma200_rising
        else "200日線より下・横ばい/下向き" if above_ma200 is not None and ma200_rising is not None
        else "データなし"
    )
    revenue_growth_latest = row.get("revenue_growth_latest")
    revenue_growth_prior = row.get("revenue_growth_prior")
    margin_latest = row.get("ordinary_income_margin_latest")
    margin_prior = row.get("ordinary_income_margin_prior")
    performance_desc_parts = []
    if revenue_growth_latest is not None and revenue_growth_prior is not None:
        performance_desc_parts.append(
            f"増収率 {revenue_growth_prior*100:.1f}%→{revenue_growth_latest*100:.1f}%"
        )
    if margin_latest is not None and margin_prior is not None:
        performance_desc_parts.append(
            f"経常利益率 {margin_prior*100:.1f}%→{margin_latest*100:.1f}%"
        )
    performance_desc = "、".join(performance_desc_parts) if performance_desc_parts else "データなし"

    st.markdown(_bucket_header("PRS(Price Relative Score)", 30, row.get("prs"), "prs"))
    _render_component_table([
        ("モメンタム位置(52週高値比)", f"{momentum_ratio*100:.1f}%" if momentum_ratio is not None else "ー",
         momentum_s, 10, _band_note(momentum_ratio, growth_scores.MOMENTUM_SCORE_BANDS),
         _item_tooltip("momentum", _band_reference_text(growth_scores.MOMENTUM_SCORE_BANDS))),
        ("トレンド方向(200日線)", trend_desc, trend_s, 10, trend_desc,
         _item_tooltip("trend", "上向き・上に位置: 10点 / どちらか一方のみ: 7点 / どちらも無し: 4点")),
        ("業績補正(増収加速・利益率改善)", performance_desc, performance_s, 10, performance_desc,
         _item_tooltip("performance", "増収加速(+2pt超): 10点 / 横ばい(±2pt以内): 8点 / 減速: 3点"
         "(利益率トレンドも同様の考え方、両方取得できた場合は平均)")),
    ])

    st.markdown(f"#### リスク調整(減点のみ) — {row.get('risk_adjustment', 0):+.0f}点")
    _render_component_table([
        ("自己資本比率", f"{equity_ratio*100:.1f}%" if equity_ratio is not None else "ー", equity_penalty, 0,
         _band_note(equity_ratio, growth_scores.EQUITY_RATIO_PENALTY_BANDS),
         _item_tooltip("equity_ratio", _band_reference_text(growth_scores.EQUITY_RATIO_PENALTY_BANDS))),
        ("成長の質(増資・借入への依存度)", f"{growth_quality_raw*100:.1f}%" if growth_quality_raw is not None else "ー",
         growth_quality_pen, 0, _band_note(growth_quality_raw, growth_scores.GROWTH_QUALITY_PENALTY_BANDS),
         _item_tooltip("growth_quality_penalty", _band_reference_text(growth_scores.GROWTH_QUALITY_PENALTY_BANDS))),
        ("PER(割高すぎないかの安全弁)", f"{per:.1f}倍" if per is not None else "ー", per_penalty, 0,
         f"{growth_scores.PER_SAFETY_PENALTY_THRESHOLD:.0f}倍超で{growth_scores.PER_SAFETY_PENALTY:+.0f}点" if per is not None and per > 0 else "赤字のため判定対象外",
         _item_tooltip("per_safety", f"{growth_scores.PER_SAFETY_PENALTY_THRESHOLD:.0f}倍以下: 0点 / "
         f"{growth_scores.PER_SAFETY_PENALTY_THRESHOLD:.0f}倍超: {growth_scores.PER_SAFETY_PENALTY:+.0f}点(赤字は判定対象外)")),
        ("過去5期の赤字年数", f"{int(deficit_years)}期" if deficit_years is not None else "ー", deficit_pen, 0,
         f"{int(deficit_years)}期赤字 × {growth_scores.DEFICIT_PENALTY_PER_YEAR:+.0f}点" if deficit_years is not None else "データなし",
         _item_tooltip("deficit", f"赤字1期ごとに{growth_scores.DEFICIT_PENALTY_PER_YEAR:+.0f}点"
         "(例: 2期赤字なら-20点)")),
    ])

    # 「どの要素が一番点数を落としているか」を機械的に(満点に対する獲得比率が
    # 最も低いもの)拾って一言で示す。GPRSの合計に実際に含まれるGPS・PRSの5項目のみ
    # 対象にする(GQSは参考情報でGPRSの合計に含まないため対象外、リスク調整は
    # 減点のみで満点=0のため比率で比較できない)。
    valuation_component_name = "リンチレシオスコア" if lynch is not None else "PSR成長対比スコア"
    valuation_component_score = lynch_s if lynch is not None else psr_growth_s
    positive_components = [
        (valuation_component_name, valuation_component_score, 15), ("FCF利回りスコア", fcf_yield_s, 10),
        ("モメンタム位置", momentum_s, 10), ("トレンド方向", trend_s, 10),
        ("業績補正", performance_s, 10),
    ]
    valid = [(n, s, mx) for n, s, mx in positive_components if s is not None]
    if valid:
        weakest_name, weakest_score, weakest_max = min(valid, key=lambda x: x[1] / x[2])
        st.info(f"💡 最も点数を落としている項目: **{weakest_name}**({weakest_score:.1f}/{weakest_max}点)")
    penalties = [(n, p) for n, p in [
        ("自己資本比率", equity_penalty), ("成長の質", growth_quality_pen), ("PER", per_penalty),
        ("過去5期の赤字", deficit_pen),
    ] if p is not None and p < 0]
    if penalties:
        penalty_text = "、".join(f"{n}({p:+.0f}点)" for n, p in penalties)
        st.warning(f"⚠️ リスク調整で減点されている項目: {penalty_text}")

    if code_full and prior_52w_high:
        st.markdown("#### 株価チャート(PRSの根拠確認用)")
        render_ma_chart(code_full, prior_52w_high, key_prefix="gprs_detail")


section_heading("🔍 個別銘柄を分析")
st.caption(
    "サイドバーでの選択や上のスクリーニング条件に関わらず、全銘柄から検索して分析できます。"
    "上の「🔍 銘柄スクリーニング」で候補を絞り込んでから、行を選んで「個別銘柄を分析で開く」"
    "を押すと、ここにポップアップで表示されます。"
)
analysis_query = st.text_input("コードまたは銘柄名で検索", key="analysis_query", placeholder="例: 7203 / トヨタ")
if st.button("🔍 検索実行", key="analysis_search_button"):
    st.session_state["analysis_ready"] = True

analysis_matches = []
if not st.session_state.get("analysis_ready"):
    st.caption("銘柄コードまたは名前を入力して「🔍 検索実行」を押してください。")
elif analysis_query:
    q = analysis_query.lower()
    analysis_matches = [
        t for t in master if q in t["code"].lower() or q in t["name"].lower()
    ][:50]

if st.session_state.get("analysis_ready") and analysis_query and not analysis_matches:
    st.info("該当する銘柄が見つかりませんでした。")
elif analysis_matches:
    analysis_label = st.selectbox(
        "候補から選択",
        [f"{t['code']} - {t['name']}" for t in analysis_matches],
        key="analysis_pick",
    )
    analysis_code = analysis_label.split(" - ", 1)[0]
    analysis_name = master_by_code.get(analysis_code, "")
    _show_stock_analysis_dialog(analysis_code, analysis_name)

st.divider()

if "result" not in st.session_state:
    st.session_state.result = None

if not selected_codes:
    st.warning("左のサイドバーで銘柄を1つ以上選択してください。")
    st.stop()

used_fallback = False

if run_clicked:
    run_screen.clear()  # ボタンを押した時はキャッシュ期間内でも強制的に最新化する
    with st.spinner("データ取得中..."):
        try:
            st.session_state.result = run_screen(tuple(selected_codes), as_of_date)
        except Exception:
            st.session_state.result = pd.DataFrame()
elif st.session_state.result is None:
    # 初回表示は、daily_refresh.pyが保存した前日以前の事前計算結果(result.csv)を使う。
    # 全銘柄でのライブ計算(数十秒〜1分規模)はボタンを押すまで行わず、即座に表示する。
    full_result = load_precomputed_result()
    if not full_result.empty:
        used_fallback = True
        as_of_date = full_result["date"].max()
        st.session_state.result = full_result[full_result["code"].isin(selected_codes)].reset_index(drop=True)
    else:
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

data_date_placeholder.caption(f"📅 現在のデータ基準日: {result['date'].max()}")

if used_fallback:
    st.info(
        f"事前計算結果(基準日: {as_of_date})を表示しています。最新のデータで再計算するには"
        "サイドバーの「🔄 スクリーニング実行」を押してください。"
    )

section_heading("📈 Growth & Price Relative Score(GPRS)")
with st.expander("計算方法の説明", expanded=False):
    st.caption(
        "既存の複合スコア(業種補正パーセンタイル順位ベース)とは別枠の、成長株向け"
        "評価システム(加点方式)。まだ仮定義で、バックテストや使用感により基準が"
        "変わる可能性があります。"
    )
    st.caption(
        "GPRS(60点満点) = GPS + PRS + リスク調整。54点以上→S、48〜54→A、42〜48→B、"
        "36〜42→C、36未満→D"
    )
    st.caption(
        "GPS(Growth Potential Score、配点30点) = リンチレシオスコア(15点満点、赤字などで"
        "計算できない場合はPSR成長対比スコアで代用) + FCF利回りスコア(10点満点)"
    )
    st.caption(
        "PRS(Price Relative Score、30点満点) = モメンタム位置 + トレンド方向 + 業績補正"
        "(各10点満点)"
    )
    st.caption(
        "リスク調整: 自己資本比率(借入依存度)・成長の質(増資/借入への依存度)・PER(割高すぎ"
        "ないか)・過去5期の赤字年数(1期ごとに-10点)を見て、リスクがある場合にのみ減点する"
        "安全弁(加点はしない、0が上限)。判定できない要素は減点なし(0)として扱うため、"
        "GPRS自体が判定不能になることはない。"
    )
    st.caption(
        "業績補正は本来「決算短信の業績予想修正(上方修正等)」を使うべきだが無料データでは"
        "取得できないため、増収率の加速度と経常利益率のトレンドを組み合わせた代理指標を使っている。"
    )
    st.caption(
        "GPS・PRSは、算出に必要な値が1つでも欠けている場合はいずれも「判定不能」になる"
        "(一部の要素だけで代用しない)。"
    )
    st.caption(
        "GQS(Growth Quality Score、40点満点) = GPスコア + 粗利成長スコア + FCFスコア + "
        "安定性係数(各10点満点)は別枠の参考情報として列だけ表示している(GPRSの合計には"
        "含まない)。複数時点・複数保有期間のバックテストで、GQSの4要素がいずれも将来"
        "リターンとほぼ無相関〜わずかにマイナス(特にGPスコアと安定性係数)で、GPRSに含めると"
        "GPS+PRSのみの場合より相関が悪化する結果が出たため、質のスクリーニング用の目安に"
        "とどめている。"
    )
    st.caption(
        "GPSが高い＝利益成長・配当のわりに株価(PER)が割安(リンチレシオ)で、かつ時価総額に"
        "対してFCFがしっかり出ている(FCF利回り)、「成長の中身に対して株価が過熱していない」会社。"
    )
    st.caption(
        "PRSが高い＝株価が52週高値近辺にいて(モメンタム)、200日線より上で上向き(トレンド)、"
        "しかも増収が加速・利益率も改善傾向(業績補正)、「市場もその成長を評価し始めている、"
        "勢いのある会社」。"
    )
    st.caption(
        "GPRSが高い(特にS・A評価)＝割安感を保ちながら株価もすでに上向き始めている銘柄。"
        "PRSだけ高くGPSが低ければ「勢いはあるが割安感が無い可能性(値動き先行のリスク)」、"
        "GPSだけ高くPRSが低ければ「割安だが市場にまだ評価されていない(先回り投資向き)」"
        "という読み方もできる。GQS(参考情報)が高ければ「収益性・成長の質も高いが、それ自体は"
        "株価の先行きを保証しない」点には注意。"
    )

if not section_gate("gprs"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    gprs_quality_table = load_quality_table_cached()
    gprs_table = finder.add_price_relative_scores(gprs_quality_table, result)

    if not gprs_table.empty:
        gprs_table["market"] = gprs_table["code"].map(lambda c: master_by_code_market.get(c))
        gf1, gf2, gf3, gf4 = st.columns(4)
        gprs_market_filter = gf1.multiselect(
            "市場で絞り込み", sorted(gprs_table["market"].dropna().unique().tolist()), key="gprs_market_filter"
        )
        gprs_sector_filter = gf2.multiselect(
            "業種で絞り込み", sorted(gprs_table["sector33"].dropna().unique().tolist()), key="gprs_sector_filter"
        )
        gprs_market_cap_filter = gf3.multiselect(
            "時価総額で絞り込み",
            [label for label in finder.MARKET_CAP_BUCKET_LABELS if label in gprs_table["market_cap_bucket"].values],
            key="gprs_market_cap_filter",
        )
        gprs_grade_filter = gf4.multiselect(
            "評価で絞り込み",
            [g for g in ["S", "A", "B", "C", "D"] if g in gprs_table["gprs_grade"].values],
            key="gprs_grade_filter",
        )
        if gprs_market_filter:
            gprs_table = gprs_table[gprs_table["market"].isin(gprs_market_filter)]
        if gprs_sector_filter:
            gprs_table = gprs_table[gprs_table["sector33"].isin(gprs_sector_filter)]
        if gprs_market_cap_filter:
            gprs_table = gprs_table[gprs_table["market_cap_bucket"].isin(gprs_market_cap_filter)]
        if gprs_grade_filter:
            gprs_table = gprs_table[gprs_table["gprs_grade"].isin(gprs_grade_filter)]
        gprs_table = gprs_table.reset_index(drop=True)

    if gprs_table.empty:
        grade_summary = ""
    else:
        grade_counts = gprs_table["gprs_grade"].value_counts()
        grade_summary = "(" + "　".join(
            f"{grade}:{int(grade_counts.get(grade, 0))}件" for grade in ["S", "A", "B", "C", "D"]
        ) + ")"
    st.caption(f"該当: {len(gprs_table)}件{grade_summary}")
    if gprs_table.empty:
        st.info("該当銘柄はありません。")
    else:
        gprs_table = gprs_table.sort_values("gprs", ascending=False, na_position="last").reset_index(drop=True)
        gprs_market = gprs_table["market"].map(abbreviate_market)
        # PEGは赤字(直近純利益<=0)の場合、データ欠損ではなく計算式の性質上そもそも
        # 定義できない(赤字企業にPERベースの割安判定は意味を持たない)。GPS・GPRSが
        # 「判定不能」になる原因のうち最も多いのがこれなので、「赤字」と明示して
        # 単なるデータ欠損と区別する。
        gprs_is_deficit = (
            gprs_table["net_income_latest"].notna() & (gprs_table["net_income_latest"] <= 0)
            if "net_income_latest" in gprs_table.columns else pd.Series(False, index=gprs_table.index)
        )
        gprs_display = pd.DataFrame({
            "コード": code_links(gprs_table["code"].tolist()),
            "銘柄名": gprs_table["name"],
            "市場": gprs_market,
            "業種": gprs_table["sector33"],
            "時価総額": gprs_table["market_cap_bucket"].fillna(QUALITY_MISSING_LABEL),
            "終値": gprs_table["close"],
            "GQS": _format_quality_value(gprs_table["gqs"], decimals=1, missing_label=QUALITY_INDETERMINATE_LABEL),
            "GPS": _format_quality_value_with_deficit(gprs_table["gps"], gprs_is_deficit, decimals=1),
            "PRS": _format_quality_value(gprs_table["prs"], decimals=1, missing_label=QUALITY_INDETERMINATE_LABEL),
            "リスク調整": gprs_table["risk_adjustment"].fillna(0).map(lambda v: f"{v:+.0f}" if v else "0"),
            "GPRS": _format_quality_value_with_deficit(gprs_table["gprs"], gprs_is_deficit, decimals=1),
            "評価": gprs_table["gprs_grade"].fillna(QUALITY_INDETERMINATE_LABEL),
            "業種内順位": [
                f"{int(rank)}位/{int(size)}社中" if pd.notna(rank) and pd.notna(size) else QUALITY_INDETERMINATE_LABEL
                for rank, size in zip(gprs_table["gprs_sector_rank"], gprs_table["gprs_sector_size"])
            ],
        })
        GPRS_COLUMN_HELP = {
            "GQS": "Growth Quality Score(40点満点、参考情報。GPRSの合計には含まない)＝GPスコア+"
            "粗利成長スコア+FCFスコア+安定性係数。バックテストで将来リターンとの相関がほぼ無い"
            "ことを確認したため、質のスクリーニング用の目安としてのみ表示している。",
            "GPS": "Growth Potential Score(配点30点)＝成長対比の割安さ(リンチレシオスコア。"
            "赤字などで計算できない場合はPSR成長対比スコアで代用)+FCF利回りスコア。利益・配当と"
            "株価成長のわりに割安で、時価総額に対してFCFがしっかり出ている「株価が過熱していない」度合い。",
            "PRS": "Price Relative Score(30点満点)＝モメンタム位置+トレンド方向+業績補正。株価が"
            "52週高値近辺・200日線より上で上向き・増収加速や利益率改善もある「市場が評価し始めている」度合い。",
            "リスク調整": "自己資本比率(借入依存度)・成長の質(増資/借入依存度)・PER(割高すぎないか)・"
            "過去5期の赤字年数(1期ごとに-10点)を見て、リスクがある場合にのみ減点する(加点はしない、"
            "0が上限)。判定できない要素は減点なし(0)として扱う。",
            "GPRS": "GPS+PRS+リスク調整(60点満点)。54点以上→S、48〜54→A、42〜48→B、36〜42→C、"
            "36未満→D。PRSだけ高くGPSが低ければ「勢い先行で割安感が無い可能性」、GPSだけ高ければ"
            "「割安だがまだ市場に評価されていない」。",
        }
        gprs_event = st.dataframe(
            gprs_display,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                **number_column_config(gprs_display.columns),
                **{
                    col: st.column_config.TextColumn(alignment="right", help=GPRS_COLUMN_HELP.get(col))
                    for col in ("GQS", "GPS", "PRS", "リスク調整", "GPRS", "評価", "業種内順位")
                },
                **code_link_column_config(),
            },
            key="gprs_table",
        )
        gprs_rows = gprs_event.selection["rows"]
        gprs_has_selection = bool(gprs_rows)
        gprs_selected_name = gprs_table["name"].iloc[gprs_rows[0]] if gprs_has_selection else None
        gprs_btn_col1, gprs_btn_col2 = st.columns(2)
        with gprs_btn_col1:
            gprs_analysis_label = (
                f"🔍 {gprs_selected_name}を「個別銘柄を分析」で開く" if gprs_has_selection
                else "🔍 個別銘柄を分析で開く(表の行を選択してください)"
            )
            if st.button(gprs_analysis_label, key="gprs_open_analysis", disabled=not gprs_has_selection):
                gprs_selected_code = gprs_table["code"].iloc[gprs_rows[0]]
                st.session_state["analysis_query_override"] = gprs_selected_code.split(".")[0]
                st.session_state["analysis_ready"] = True
                st.rerun()
        with gprs_btn_col2:
            gprs_detail_label = (
                f"📊 {gprs_selected_name}のGPRS詳細を見る" if gprs_has_selection
                else "📊 GPRS詳細を見る(表の行を選択してください)"
            )
            if st.button(gprs_detail_label, key="gprs_open_detail", disabled=not gprs_has_selection):
                st.session_state["gprs_detail_row"] = gprs_table.iloc[gprs_rows[0]].to_dict()
                selected_sector = gprs_table["sector33"].iloc[gprs_rows[0]]
                sector_peers = gprs_table[gprs_table["sector33"] == selected_sector]
                st.session_state["gprs_detail_sector_medians"] = {
                    "gqs": sector_peers["gqs"].median(),
                    "gps": sector_peers["gps"].median(),
                    "prs": sector_peers["prs"].median(),
                    "gprs": sector_peers["gprs"].median(),
                    "count": len(sector_peers),
                }
                st.session_state["gprs_detail_ready"] = True
                st.rerun()

        if st.session_state.get("gprs_detail_ready") and st.session_state.get("gprs_detail_row"):
            _show_gprs_detail_dialog(
                st.session_state["gprs_detail_row"],
                st.session_state.get("gprs_detail_sector_medians") or {},
            )

section_heading("🌱 上昇基調銘柄")
steady_rise_legend_lines = [
    "年間リターン: 直近1年の株価リターン。プラスであれば上昇基調の条件を満たしている",
    "月次ボラ: 直近12ヶ月の月次リターンの標準偏差。小さいほど値動きが滑らかで、急騰急落の少ない上昇基調であることを示す",
    "ワーストマンス: 直近12ヶ月で最も悪かった月次リターン。-8%以上(これより悪い月が無い)が条件",
    "出来高増減傾向: 直近20日平均の出来高÷その前20日平均。1倍超は出来高が増えている(関心が高まっている)、1倍未満は減っていることを示す",
    *QUALITY_ALWAYS_LEGEND_LINES,
    *MARGIN_DETAIL_LEGEND_LINES,
]
with st.expander("条件・用語の説明", expanded=False):
    st.caption(
        "①直近1年のリターンがプラス ②200日移動平均線が上向き、かつ株価がその上 "
        "③月次リターンの標準偏差(ボラティリティ)が7%以下 ④直近12ヶ月のワーストマンス(最悪月)が"
        "-8%以上 ⑤25日移動平均線が上向き、の5条件をすべて満たす銘柄。"
    )
    for _line in steady_rise_legend_lines:
        st.caption(_line)

if not section_gate("steady_rise"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    steady_rise = result[result["steady_rise"] == True].sort_values("monthly_vol")  # noqa: E712
    st.caption(f"該当: {len(steady_rise)}件")
    if steady_rise.empty:
        st.info("該当銘柄はありません。")
    else:
        steady_rise_columns = [
            "code", "name", "close", "return_1y", "monthly_vol", "worst_month", "volume_trend_ratio"
        ]
        render_fundamentals_picker(
            steady_rise,
            steady_rise_columns,
            key_prefix="steady_rise",
            show_margin_detail=True,
            legend_lines=steady_rise_legend_lines,
            render_legend=False,
        )

# 早復型(クイックリカバリー型)×配当利回り2%以上の銘柄で、6ヶ月後の勝率が60%を境に
# 明確に分かれた業種のリスト(サンプル数15件以上のみ対象にしたバックテスト結果より)。
GOLDEN_CROSS_STRONG_SECTORS = [
    "電気・ガス業", "パルプ・紙", "倉庫・運輸関連業", "非鉄金属", "証券、商品先物取引業",
    "建設業", "銀行業", "ガラス・土石製品", "陸運業", "金属製品", "不動産業",
    "その他金融業", "卸売業", "機械", "繊維製品", "輸送用機器", "電気機器",
    "鉄鋼", "食料品", "情報・通信業", "精密機器",
]
GOLDEN_CROSS_WEAK_SECTORS = ["小売業", "その他製品", "化学", "サービス業", "医薬品"]

section_heading("⚡ ゴールデンクロス(25日線×75日線)")
with st.expander("条件・用語の説明", expanded=False):
    st.caption(
        "25日移動平均線が75日移動平均線を下から上に抜けるのがゴールデンクロス。"
        f"直近{screener.GOLDEN_CROSS_RECENT_DAYS}営業日以内に交差していれば「発生」、"
        "まだ交差していないが乖離(25日線-75日線)が縮まってきていて25日線が上向きの場合は"
        "「兆候あり」として表示する(兆候はあくまで参考の早期シグナルで、実際には交差せず"
        "反転することもある)。「上昇基調銘柄」は1年間ずっと上昇している銘柄向けの条件のため、"
        "まだそこまで育っていない、これから上昇が始まるかもしれない銘柄を早めに拾う用途。"
    )
    st.caption(
        f"安定上昇型(3ヶ月/6ヶ月/1年): 対数終値に回帰直線を当てはめ、傾きがプラスかつ決定係数"
        f"R²が{screener.CLEAN_TREND_R2_THRESHOLD}以上(直線に近い滑らかさ=急騰急落の少ない上昇)の"
        "期間を表示する。該当する期間が無ければ「―」。"
    )
    st.caption(
        f"R²だけだと「全体の値動きの大きさに対する相対的ななめらかさ」しか見ないため、"
        "上昇幅自体が大きい銘柄は期間中に大きく波打ってもR²が高く出てしまうことがある"
        "(実データ確認済み)。そのため、期間中の最大ドローダウン(高値からの最大下落率)が"
        f"{screener.CLEAN_TREND_MAX_DRAWDOWN_THRESHOLD:.0f}%以内であることも条件に加えている。"
    )
    st.caption(
        "「1年」とだけ表示されている場合は、直近1年の期間でこの条件を満たしているが、"
        "3ヶ月・6ヶ月の期間では満たしていないことを示す(短い期間では傾きがマイナス、"
        "またはR²が低い=直近の値動きが荒いということ)。複数の期間で条件を満たす場合は"
        "「6ヶ月・1年」のようにまとめて表示する。"
    )
    st.caption(
        f"早復型(クイックリカバリー型): ゴールデンクロス(または兆候)の前に、{screener.QUICK_RECOVERY_MAX_DAYS}"
        "営業日以内にデッドクロス(25日線が75日線を上から下に抜けた)があった場合を指す。"
        "一度崩れてから短期間で立て直したパターンで、バックテストの結果、通常のゴールデンクロス"
        "より、特に配当利回りが高い(2%以上)銘柄でその後の株価パフォーマンスが優れていることを"
        "確認済み(backtest_whipsaw_golden_cross.py参照)。配当利回り2%未満ではこの効果はほぼ"
        "見られなかった。"
    )
    st.caption(
        "「DCR法(デッドクロスリバーサル投資法)」: 早復型×配当利回り2%以上の組み合わせに"
        "この名前を付けている(デッドクロスからの反転=リバーサルを狙う投資法、の意)。"
        "以下の業種別の効果もこの組み合わせでの検証結果。"
    )
    st.caption(
        "業種による効果の違い: 早復型×配当利回り2%以上の銘柄で見ると、"
        "電気・ガス業(6ヶ月勝率88.5%)・建設業(75.3%)・銀行業(73.7%)・不動産業(68.7%)・"
        "卸売業(67.0%)・機械(67.0%)のような伝統的・バリュー的な業種で特に効きが良く、"
        "医薬品(45.8%、コイン投げ以下)・サービス業(54.3%)・化学(56.7%)のような、"
        "個社要因や成長期待で株価が動きやすい業種では効きが弱い傾向がある"
        "(いずれもサンプル数が少ない業種は参考程度に)。"
    )
    for _line in QUALITY_ALWAYS_LEGEND_LINES:
        st.caption(_line)

if not section_gate("golden_cross"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    golden_candidates = result[result["golden_cross_status"].notna()].copy()

    # PER・ROE・PBR・増収増益・GP・成長の質・複合スコア・配当利回りを品質データキャッシュから付与
    golden_quality_table = load_quality_table_cached()
    if not golden_quality_table.empty:
        golden_quality_slice = golden_quality_table.set_index("code").reindex(golden_candidates["code"])
        golden_candidates["dividend_yield_pct"] = golden_quality_slice["dividend_yield_pct"].to_numpy()
        golden_candidates["dividend_growth_pct"] = golden_quality_slice["dividend_growth_pct"].to_numpy()
        golden_candidates["payout_ratio_pct"] = golden_quality_slice["payout_ratio_pct"].to_numpy()
    else:
        golden_candidates["dividend_yield_pct"] = pd.NA
        golden_candidates["dividend_growth_pct"] = pd.NA
        golden_candidates["payout_ratio_pct"] = pd.NA

    # 配当利回り10%以上は「配当罠(株価急落で利回りが跳ね上がっただけ)」の可能性が高く、
    # バックテストでも件数が少なくバラつきが大きすぎて信用できないため除外する
    # (backtest_high_div_yield.py参照。NaNは除外対象にしない)。
    golden_candidates = golden_candidates[golden_candidates["dividend_yield_pct"].fillna(0) < 10.0]

    golden_candidates["業種"] = golden_candidates["code"].map(master_by_code_sector)

    golden_recommended_only = st.checkbox(
        "🌟 DCR法(早復型×配当利回り2.0%以上)",
        value=True, key="golden_recommended_only",
    )
    gc1, gc2, gc3, gc4 = st.columns(4)
    golden_quick_recovery_only = gc1.checkbox(
        "早復型", key="golden_quick_recovery_only",
        help="デッドクロス後、短期間で立て直したパターン(クイックリカバリー型)",
    )
    golden_normal_only = gc2.checkbox(
        "通常型", key="golden_normal_only",
        help="デッドクロスを伴わない通常のゴールデンクロス。"
        "早復型と両方チェックすると両方とも表示する",
    )
    golden_crossed_only = gc3.checkbox(
        "GC発生済み", key="golden_crossed_only",
        help="すでにゴールデンクロスが発生した銘柄のみ(兆候ありは除く)",
    )
    golden_imminent_only = gc4.checkbox(
        "GC兆候あり", key="golden_imminent_only",
        help="まだ交差していないが兆候がある銘柄のみ(発生済みは除く)",
    )
    golden_strong_sector_only = st.checkbox(
        "効きが強い業種(電気・ガス業/建設業/銀行業/不動産業/卸売業/機械 等)",
        key="golden_strong_sector_only",
        help="DCR法(早復型×配当利回り2%以上)の銘柄で、"
        "6ヶ月後の勝率が60%以上だった業種のみ",
    )
    golden_weak_sector_only = st.checkbox(
        "効きが弱い業種(医薬品/サービス業/化学/その他製品/小売業)",
        key="golden_weak_sector_only",
        help="DCR法(早復型×配当利回り2%以上)の銘柄で、"
        "6ヶ月後の勝率が60%未満だった業種のみ"
        "(このパターンが機能しにくい業種、あえて避けたい場合や参考比較用)",
    )
    golden_payout_ratio_only = st.checkbox(
        "配当性向50%未満", value=True, key="golden_payout_ratio_only",
        help="DCR法の対象銘柄では、配当性向(純利益に対する配当支払額の割合)が"
        "50%未満のほうが以降のパフォーマンスが良い傾向をバックテストで確認済み"
        "(backtest_dcr_payout_ratio.py参照)。データが無い銘柄は除外しない。"
        "既定でオンだが、必要な時はチェックを外せる。",
    )
    golden_clean_trend_only = st.checkbox(
        "安定上昇型", key="golden_clean_trend_only",
        help="3ヶ月/6ヶ月/1年のいずれかの期間で、なめらかな上昇(R²・値幅制限を満たす)"
        "銘柄のみに絞り込む。既定は推奨設定。下の「詳細設定」で自分で調整することも可能。",
    )
    with st.expander("安定上昇型: 値幅制限を自分で調整する(詳細設定)", expanded=False):
        gc7, gc8 = st.columns(2)
        golden_max_drawdown = gc7.number_input(
            "許容ドローダウン(%)以内", min_value=1.0,
            value=float(screener.CLEAN_TREND_MAX_DRAWDOWN_THRESHOLD),
            step=1.0, key="golden_max_drawdown",
            help="期間中の高値からの最大下落率がこの値を超えたら「安定上昇型」から除外する。"
            "例えば「1年前1000円→半年前4000円→現在2000円」のような、結果的には上昇でも"
            "途中の値幅が大きい銘柄を弾くための調整。既定値はおすすめ設定(サーバー側の判定基準と同じ)。",
        )
        golden_min_rise = gc8.number_input(
            "最低上昇率(%)以上", min_value=0.0, value=0.0, step=5.0, key="golden_min_rise",
            help="期間の最初から最後までの上昇率がこの値未満なら「安定上昇型」から除外する"
            "(未調整だと1%の上昇でも100%の上昇でも同じ「安定上昇型」として扱われてしまうため)。",
        )

    _CLEAN_TREND_WINDOW_COLS = [
        ("3ヶ月", "trend_r2_3m", "trend_drawdown_3m", "trend_return_3m"),
        ("6ヶ月", "trend_r2_6m", "trend_drawdown_6m", "trend_return_6m"),
        ("1年", "trend_r2_1y", "trend_drawdown_1y", "trend_return_1y"),
    ]

    def _recompute_clean_trend(row: pd.Series) -> str:
        qualifying = []
        for label, r2_col, dd_col, ret_col in _CLEAN_TREND_WINDOW_COLS:
            r2 = row.get(r2_col)
            dd = row.get(dd_col)
            ret = row.get(ret_col)
            if (
                pd.notna(r2) and r2 >= screener.CLEAN_TREND_R2_THRESHOLD
                and pd.notna(dd) and dd <= golden_max_drawdown
                and pd.notna(ret) and ret >= golden_min_rise
            ):
                qualifying.append(label)
        return "・".join(qualifying) if qualifying else "―"

    golden_candidates["きれいな上昇"] = golden_candidates.apply(_recompute_clean_trend, axis=1)

    if golden_recommended_only:
        golden_candidates = golden_candidates[golden_candidates["golden_cross_is_quick_recovery"] == True]  # noqa: E712
        golden_candidates = golden_candidates[golden_candidates["dividend_yield_pct"].fillna(-1) >= 2.0]
    if golden_quick_recovery_only and not golden_normal_only:
        golden_candidates = golden_candidates[golden_candidates["golden_cross_is_quick_recovery"] == True]  # noqa: E712
    elif golden_normal_only and not golden_quick_recovery_only:
        golden_candidates = golden_candidates[golden_candidates["golden_cross_is_quick_recovery"] != True]  # noqa: E712
    if golden_crossed_only:
        golden_candidates = golden_candidates[golden_candidates["golden_cross_status"] == "crossed"]
    if golden_imminent_only:
        golden_candidates = golden_candidates[golden_candidates["golden_cross_status"] == "imminent"]
    if golden_strong_sector_only:
        golden_candidates = golden_candidates[golden_candidates["業種"].isin(GOLDEN_CROSS_STRONG_SECTORS)]
    if golden_weak_sector_only:
        golden_candidates = golden_candidates[golden_candidates["業種"].isin(GOLDEN_CROSS_WEAK_SECTORS)]
    if golden_clean_trend_only:
        golden_candidates = golden_candidates[golden_candidates["きれいな上昇"] != "―"]
    if golden_payout_ratio_only:
        golden_candidates = golden_candidates[golden_candidates["payout_ratio_pct"].fillna(0) < 50.0]
    golden_candidates = golden_candidates.reset_index(drop=True)

    st.caption(f"該当: {len(golden_candidates)}件")
    if golden_candidates.empty:
        st.info("該当銘柄はありません。")
    else:
        st.caption("「ゴールデンクロス」列: 🟢=早復型(クイックリカバリー型)　⚪=通常型")

        def _golden_cross_label(row: pd.Series) -> str | None:
            # 型式の名前は表示せず、色付きの丸(🟢=早復型/⚪=通常型)で区別する
            dot = "🟢" if row.get("golden_cross_is_quick_recovery") else "⚪"
            if row["golden_cross_status"] == "crossed":
                return f"{dot} 発生({int(row['golden_cross_days_since'])}営業日前)"
            if row["golden_cross_status"] == "imminent":
                return f"{dot} 兆候あり(乖離{row['golden_cross_gap_pct']:.2f}%)"
            return None

        golden_candidates["ゴールデンクロス"] = golden_candidates.apply(_golden_cross_label, axis=1)

        if golden_recommended_only or golden_quick_recovery_only:
            # 早復型のみ表示中は、バックテストで確認した「早復型×配当利回り」の2条件そのものを
            # 反映し、配当利回りが高い順に並べる
            # (絞り込み自体で1条件目は満たしているため、2条件目の配当利回りが並び替えの軸になる)
            golden_candidates = golden_candidates.sort_values(
                "dividend_yield_pct", ascending=False, na_position="last"
            ).reset_index(drop=True)
        else:
            status_order = {"crossed": 0, "imminent": 1}
            golden_candidates["_status_sort"] = golden_candidates["golden_cross_status"].map(status_order)
            # 早復型を同じステータス内で上位に表示する
            golden_candidates["_quick_sort"] = (~golden_candidates["golden_cross_is_quick_recovery"].fillna(False)).astype(int)
            golden_candidates = golden_candidates.sort_values(
                ["_status_sort", "_quick_sort", "golden_cross_days_since", "golden_cross_gap_pct"],
                na_position="last",
            ).reset_index(drop=True)

        # 銘柄の特定 → クロスのシグナル → 配当の健全性 → 会社の基礎体力 → 値動きの参考情報、
        # の順に並べて、上から見ていけば判断に必要な情報が揃うようにする。
        golden_display_cols = ["code", "name", "業種", "close", "ゴールデンクロス"]
        golden_display = golden_candidates[golden_display_cols].rename(columns=COLUMN_LABELS_JA)
        golden_display["コード"] = code_links(golden_candidates["code"].tolist())
        golden_display["DC発生"] = golden_candidates["golden_cross_dead_cross_date"]
        golden_display["GC発生"] = golden_candidates["golden_cross_date"]
        golden_display["配当利回り(%)"] = golden_candidates["dividend_yield_pct"].round(2)
        golden_display["増配率(%)"] = golden_candidates["dividend_growth_pct"].round(2)
        golden_display["配当性向(%)"] = golden_candidates["payout_ratio_pct"].round(2)

        if not golden_quality_table.empty:
            golden_quality_slice = golden_quality_table.set_index("code").reindex(golden_candidates["code"])
            golden_display["PER"] = golden_quality_slice["latest_per"].to_numpy()
            golden_display["ROE(%)"] = (golden_quality_slice["latest_roe_official"] * 100).to_numpy()

        golden_display["R²(1年)"] = golden_candidates["trend_r2_1y"]
        golden_display["上昇率6ヶ月(%)"] = golden_candidates["trend_return_6m"]
        golden_display["年間リターン(%)"] = golden_candidates["return_1y"]
        golden_display["月次ボラ(%)"] = golden_candidates["monthly_vol"]
        golden_display["ワーストマンス(%)"] = golden_candidates["worst_month"]
        golden_event = st.dataframe(
            golden_display,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                **number_column_config(golden_display.columns),
                **quality_text_column_config(golden_display.columns),
                **code_link_column_config(),
            },
            key="golden_cross_table",
        )
        golden_rows = golden_event.selection["rows"]
        if golden_rows:
            golden_selected_code = golden_candidates["code"].iloc[golden_rows[0]]
            golden_selected_name = golden_candidates["name"].iloc[golden_rows[0]]
            if st.button(f"🔍 {golden_selected_name}を「個別銘柄を分析」で開く", key="golden_open_analysis"):
                st.session_state["analysis_query_override"] = golden_selected_code.split(".")[0]
                st.session_state["analysis_ready"] = True
                st.rerun()

section_heading(f"🎯 52週来高値更新銘柄({as_of_date})")
if not section_gate("new_highs"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    new_highs = result[result["new_52w_high"]].sort_values("code")
    st.caption(f"該当: {len(new_highs)}件")
    if new_highs.empty:
        st.info("該当銘柄はありません。")
    else:
        new_highs_with_tendency = add_tendency_summary(new_highs)
        columns = ["code", "name", "date", "close", "prior_52w_high", "pct_vs_prior_high"]
        if "出来高傾向" in new_highs_with_tendency.columns:
            columns += TENDENCY_DISPLAY_COLUMNS
        new_highs_legend_lines = list(DEFAULT_LEGEND_LINES) + list(QUALITY_ALWAYS_LEGEND_LINES) + [
            "「出来高傾向」は自社データ(日本株3,664銘柄、52週高値更新21,920件、2024/3〜2026/6)に基づく"
            "過去の統計的傾向であり、個別銘柄の将来の値動きを予測・保証するものではありません。",
        ]
        render_fundamentals_picker(
            new_highs_with_tendency, columns, key_prefix="new_highs",
            legend_lines=new_highs_legend_lines, render_legend=False,
        )
        with st.expander("項目の説明", expanded=False):
            for _line in new_highs_legend_lines:
                st.caption(_line)

section_heading("📅 選択銘柄の最新52週高値更新日(新しい順)")
if not section_gate("history"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    HISTORY_LOOKBACK_TRADING_DAYS = 7  # 基準日から何営業日前までの52週高値更新を表示するか
    history = result.dropna(subset=["latest_52w_high_date"])
    history = history[history["trading_days_since_high"] <= HISTORY_LOOKBACK_TRADING_DAYS].sort_values(
        "latest_52w_high_date", ascending=False
    )
    st.caption(f"該当: {len(history)}件(基準日から{HISTORY_LOOKBACK_TRADING_DAYS}営業日前まで)")
    if history.empty:
        st.info(f"直近{HISTORY_LOOKBACK_TRADING_DAYS}営業日以内に52週高値を更新した銘柄がありません。")
    else:
        history_display = history[
            ["code", "name", "latest_52w_high_date", "trading_days_since_high"]
        ].rename(columns=COLUMN_LABELS_JA)
        history_display["コード"] = code_links(history["code"].tolist())
        event = st.dataframe(
            history_display,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={**number_column_config(history_display.columns), **code_link_column_config()},
            key="history_table",
        )
        rows = event.selection["rows"]
        if rows:
            clicked = history.iloc[rows[0]]
            st.session_state["chart_select"] = f"{clicked['code']} - {clicked['name']}"

section_heading("📋 全銘柄ランキング(52週高値からの距離順)")
if not section_gate("ranking"):
    st.caption("「🔍 情報取得」ボタンを押してください。")
else:
    st.caption(f"該当: {len(result)}件")
    RANKING_EXCLUDED_COLUMNS = {
        "tendency_n", "golden_cross_status", "golden_cross_days_since", "golden_cross_gap_pct",
    }
    ranking_columns = [c for c in result.columns if c not in RANKING_EXCLUDED_COLUMNS]
    render_fundamentals_picker(result, ranking_columns, key_prefix="ranking")

section_heading("📊 個別銘柄チャート")
chart_labels = (result["code"] + " - " + result["name"]).tolist()
if st.session_state.get("chart_select") not in chart_labels:
    st.session_state["chart_select"] = chart_labels[0] if chart_labels else None
selected_label = st.selectbox("銘柄を選択", chart_labels, key="chart_select")
selected_code = selected_label.split(" - ", 1)[0] if selected_label else None
if selected_code:
    prior_high = result.loc[result["code"] == selected_code, "prior_52w_high"].iloc[0]
    render_ma_chart(selected_code, prior_high, key_prefix="bulk")

if used_fallback:
    st.caption(f"データ基準日: {as_of_date}(daily_refresh.pyによる事前計算結果。初回表示を高速化するため、"
               "ボタンを押すまではライブ計算しない仕様)")
else:
    st.caption("結果はキャッシュされ、5分間は再取得しません。最新化するには左の「スクリーニング実行」を押してください。")
