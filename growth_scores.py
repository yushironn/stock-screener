"""
Growth Quality Score(GQS)・Growth Potential Score(GPS)・Price Relative Score(PRS)・
Growth & Price Relative Score(GPRS)。

既存の複合スコア(quality_score.composite_rank、業種補正パーセンタイル順位ベース)とは
別枠の、成長株向け評価システム。100点満点の加点方式。

    GQS(40点満点) = GPスコア + 粗利成長スコア + FCFスコア + 安定性係数    (各10点満点)
    GPS(理論上25点満点、"配点"としては30点/100点) = PEGスコア + FCF利回りスコア
        (PEGスコアは15点満点、FCF利回りスコアは10点満点)
    PRS(30点満点) = モメンタム位置 + トレンド方向 + 業績補正             (各10点満点)
    GPRS(100点満点) = GQS + GPS + PRS

    GPRS評価: 90点以上→S、80〜89→A、70〜79→B、60〜69→C、50以下→D

注意: ユーザーからの元の定義は「GQS＝GP×粗利成長スコア×…」のように「×」表記だが、
各要素が点数(例: 10点満点)である以上、掛け算では合計点(40点満点等)と整合しない
(10×10×10×10=10000等になってしまう)ため、ユーザー確認の上で「足し算」として
実装している。GPRS＝GQS+GPS+PRSも同様にユーザー確認済み。

評価基準(区切り値・配点)は今後バックテストや使用感により変更される前提のため、
下記の各BANDS定数だけを編集すれば基準を更新できるようにしてある。
ユーザー指定に無い区間(例: 粗利成長率0〜5%)は暫定値を補っており、
コメントで明示している。

各スコアは、算出に必要な値が1つでも欠けている場合はNone(判定不能)になる
(一部の要素だけで代用しない。quality_score.composite_rankと同じ考え方)。
"""

from __future__ import annotations


def _score_from_bands(
    value: float | None, bands: list[tuple[float, float, float]]
) -> float | None:
    """bandsは(下限, 上限, 点数)のリスト(下限<=value<上限で該当)。"""
    if value is None:
        return None
    for low, high, score in bands:
        if low <= value < high:
            return score
    return None


def _sum_if_all_present(*values: float | None) -> float | None:
    """全て値がある場合のみ合計を返す(1つでも欠けたらNone=判定不能)。"""
    if any(v is None for v in values):
        return None
    return sum(values)


def yoy_growth_rate(series: dict[int, float] | None, years_ago: int = 0) -> float | None:
    """seriesの[years_ago]と[years_ago+1]から前年比成長率を計算する(business_results_series用)。"""
    if not series:
        return None
    if years_ago not in series or (years_ago + 1) not in series:
        return None
    prior = series[years_ago + 1]
    if not prior:
        return None
    return series[years_ago] / prior - 1


# ══════════════════════════════════════════════════════════════
# Growth Quality Score(GQS) = GPスコア + 粗利成長スコア + FCFスコア + 安定性係数
# (各10点満点、合計40点満点)
# ══════════════════════════════════════════════════════════════

# ①GPスコア: GP(グロス・プロフィタビリティ) = 売上総利益 ÷ 総資産
GP_SCORE_BANDS: list[tuple[float, float, float]] = [
    (0.60, float("inf"), 10),
    (0.45, 0.60, 8),
    (0.35, 0.45, 5),
    (float("-inf"), 0.35, 2),
]


def gp_score(gross_profitability: float | None) -> float | None:
    return _score_from_bands(gross_profitability, GP_SCORE_BANDS)


# ②粗利成長スコア: 売上総利益成長率(YoY)。0〜5%の区間はユーザー指定に無かったため、
# 前後の水準(0%以下=-5点、5〜10%=4点)の中間として0点を暫定的に補っている。
GROSS_PROFIT_GROWTH_SCORE_BANDS: list[tuple[float, float, float]] = [
    (0.40, float("inf"), 5),
    (0.20, 0.40, 10),
    (0.10, 0.20, 7),
    (0.05, 0.10, 4),
    (0.0, 0.05, 0),  # 暫定値(ユーザー指定に無い区間)
    (float("-inf"), 0.0, -5),
]


def gross_profit_growth_rate(
    net_sales_series: dict[int, float] | None, cost_of_sales_series: dict[int, float] | None
) -> float | None:
    """直近期・前期の(売上高-売上原価)から売上総利益成長率(YoY)を計算する。"""
    if not net_sales_series or not cost_of_sales_series:
        return None
    if 0 not in net_sales_series or 1 not in net_sales_series:
        return None
    if 0 not in cost_of_sales_series or 1 not in cost_of_sales_series:
        return None
    gp_current = net_sales_series[0] - cost_of_sales_series[0]
    gp_prior = net_sales_series[1] - cost_of_sales_series[1]
    if not gp_prior:
        return None
    return gp_current / gp_prior - 1


def gross_profit_growth_score(growth_rate: float | None) -> float | None:
    return _score_from_bands(growth_rate, GROSS_PROFIT_GROWTH_SCORE_BANDS)


# ③FCFスコア: FCFマージン = FCF ÷ 売上高
FCF_SCORE_BANDS: list[tuple[float, float, float]] = [
    (0.15, float("inf"), 10),
    (0.05, 0.15, 8),
    (0.0, 0.05, 5),
    (float("-inf"), 0.0, -5),
]


def free_cash_flow(
    operating_cf_series: dict[int, float] | None, capex_series: dict[int, float] | None
) -> float | None:
    """
    FCF(フリーキャッシュフロー) = 営業CF - CapEx(設備投資)。
    CapExはXBRL上マイナス値(支出)で開示されるため、営業CFに足す形で引き算になる。
    """
    if not operating_cf_series or not capex_series:
        return None
    if 0 not in operating_cf_series or 0 not in capex_series:
        return None
    return operating_cf_series[0] + capex_series[0]


def fcf_margin(fcf: float | None, net_sales_latest: float | None) -> float | None:
    if fcf is None or not net_sales_latest:
        return None
    return fcf / net_sales_latest


def fcf_score(margin: float | None) -> float | None:
    return _score_from_bands(margin, FCF_SCORE_BANDS)


# ④安定性係数: 増収(売上高YoYプラス)の連続期数(quality_score.growth_streak_from_seriesで計算済み)
def stability_score(revenue_streak: int | None) -> float | None:
    if revenue_streak is None:
        return None
    if revenue_streak >= 3:
        return 10
    if revenue_streak == 2:
        return 8
    if revenue_streak == 1:
        return 5
    return 2


def growth_quality_score(
    gp: float | None, gp_growth: float | None, fcf: float | None, stability: float | None
) -> float | None:
    """GQS(40点満点) = GPスコア + 粗利成長スコア + FCFスコア + 安定性係数"""
    return _sum_if_all_present(gp, gp_growth, fcf, stability)


# ══════════════════════════════════════════════════════════════
# Growth Potential Score(GPS) = 成長対比の割安さ(リンチレシオスコア、無い場合は
# PSR成長対比スコアで代用) + FCF利回りスコア
# (配点30点/100点。成長対比の割安さ15点満点+FCF利回りスコア10点満点=理論上25点満点)
# ══════════════════════════════════════════════════════════════

# 元々はPEG(PER÷純利益成長率)を使っていたが、バックテストで比較した結果
# リンチレシオ(PER÷(純利益成長率%+配当利回り%)、ピーター・リンチの提唱する
# 「配当を出す成熟企業の株主還元を、増益率だけでは評価できないPEGの弱点を補う」
# 指標)の方が同じ区切り値のまま将来リターンとの相関が同等以上(特に1年・2年
# 保有で改善、2年保有ではPEGが逆相関気味だったのに対しプラスを維持)だったため、
# PEGを完全に置き換えた。区切り値自体はPEG時代のものをそのまま流用している。
LYNCH_SCORE_BANDS: list[tuple[float, float, float]] = [
    (float("-inf"), 0.8, 15),
    (0.8, 1.2, 15),
    (1.2, 1.5, 10),
    (1.5, float("inf"), 0),
]


def lynch_ratio(
    per: float | None, net_income_growth_pct: float | None, dividend_yield_pct: float | None = None
) -> float | None:
    """
    リンチレシオ = PER ÷ (純利益成長率(%) + 配当利回り(%))。
    配当利回りが取得できない場合は0として扱う(配当が無い/データが無いだけで
    リンチレシオ自体は純利益成長率だけでも計算できるようにするため)。
    分母(成長率+配当利回り)がマイナス/ゼロの場合は意味を持たないためNone。
    """
    if per is None or per <= 0 or net_income_growth_pct is None:
        return None
    combined_growth = net_income_growth_pct + (dividend_yield_pct or 0)
    if combined_growth <= 0:
        return None
    return per / combined_growth


def lynch_score(ratio: float | None) -> float | None:
    return _score_from_bands(ratio, LYNCH_SCORE_BANDS)


def dividend_yield(dividends_paid_cf: float | None, market_cap: float | None) -> float | None:
    """
    配当利回り(%)の近似値 = 支払配当金(キャッシュフロー計算書由来、マイナス値で
    開示される) ÷ 時価総額 × 100。1株配当ではなく総還元ベースの近似。
    """
    if dividends_paid_cf is None or not market_cap:
        return None
    return (-dividends_paid_cf / market_cap) * 100


# PSR成長対比スコア: リンチレシオ(PEGの後継)は赤字・大幅減益企業では計算式の
# 性質上定義できず、これがGPSの「判定不能」の最大要因になっている(黒字化前の
# グロース株などが軒並み除外されてしまう)。PSR(株価売上高倍率)は黒字/赤字を
# 問わず計算できるため、リンチレシオが使えない場合の代替(足し算ではなく代用、
# 同じ15点満点の重みを維持)として使う。
# 基準値はPEG/リンチレシオのような確立された経験則(PERの0.8/1.2/1.5倍)が
# 無いため、実際のユニバース(2,462銘柄、PSR>0かつ増収)の分布(25/50/75
# パーセンタイル位置)から暫定的に定めている。将来のバックテストで見直す前提。
PSR_GROWTH_SCORE_BANDS: list[tuple[float, float, float]] = [
    (float("-inf"), 0.05, 15),
    (0.05, 0.15, 10),
    (0.15, 0.30, 5),
    (0.30, float("inf"), 0),
]


def psr(market_cap: float | None, net_sales_latest: float | None) -> float | None:
    """PSR(株価売上高倍率) = 時価総額 ÷ 売上高。"""
    if not market_cap or not net_sales_latest or net_sales_latest <= 0:
        return None
    return market_cap / net_sales_latest


def psr_growth_ratio(psr_value: float | None, revenue_growth_pct: float | None) -> float | None:
    """PSR成長対比 = PSR ÷ 売上高成長率(%)。PEGと同じ考え方の売上高版。"""
    if psr_value is None or psr_value <= 0 or revenue_growth_pct is None or revenue_growth_pct <= 0:
        return None
    return psr_value / revenue_growth_pct


def psr_growth_score(ratio: float | None) -> float | None:
    return _score_from_bands(ratio, PSR_GROWTH_SCORE_BANDS)


FCF_YIELD_SCORE_BANDS: list[tuple[float, float, float]] = [
    (0.05, float("inf"), 10),
    (0.03, 0.05, 8),
    (0.0, 0.03, 5),
    (float("-inf"), 0.0, -5),
]


def fcf_yield(fcf: float | None, market_cap: float | None) -> float | None:
    if fcf is None or not market_cap:
        return None
    return fcf / market_cap


def fcf_yield_score(yield_: float | None) -> float | None:
    return _score_from_bands(yield_, FCF_YIELD_SCORE_BANDS)


def growth_potential_score(
    lynch: float | None, fcf_yield_: float | None, psr_growth: float | None = None
) -> float | None:
    """
    GPS(配点30点/100点) = 成長対比の割安さ(リンチレシオスコア。無ければPSR成長対比
    スコアで代用、どちらも15点満点で重みは同じ) + FCF利回りスコア。
    """
    valuation_component = lynch if lynch is not None else psr_growth
    return _sum_if_all_present(valuation_component, fcf_yield_)


# ══════════════════════════════════════════════════════════════
# Price Relative Score(PRS) = モメンタム位置 + トレンド方向 + 業績補正
# (各10点満点、合計30点満点)
# ══════════════════════════════════════════════════════════════

MOMENTUM_SCORE_BANDS: list[tuple[float, float, float]] = [
    (0.9, float("inf"), 10),
    (0.7, 0.9, 8),
    (0.5, 0.7, 6),
    (float("-inf"), 0.5, 4),
]


def momentum_score(close: float | None, prior_52w_high: float | None) -> float | None:
    if not close or not prior_52w_high:
        return None
    return _score_from_bands(close / prior_52w_high, MOMENTUM_SCORE_BANDS)


def trend_direction_score(above_ma200: bool | None, ma200_rising: bool | None) -> float | None:
    """
    200日移動平均線との位置関係。「移動線付近」の細かい乖離率のしきい値は
    ユーザー指定に無いため、上向き/上に位置しているかの組み合わせで近似する。
    - 上向きかつ上に位置: 10点(明確な上昇トレンド)
    - どちらか一方のみ: 7点(移動線付近＝トレンド転換中の可能性)
    - どちらも満たさない: 4点
    """
    if above_ma200 is None or ma200_rising is None:
        return None
    if above_ma200 and ma200_rising:
        return 10
    if above_ma200 or ma200_rising:
        return 7
    return 4


def _revenue_growth_acceleration_score(
    latest_growth: float | None, prior_growth: float | None
) -> float | None:
    """
    増収率が前期より加速しているか。
    - 加速(今期の増収率 - 前期の増収率 > 2pt): 10点
    - ほぼ横ばい(±2pt以内): 8点
    - 減速(2pt超のマイナス): 3点
    """
    if latest_growth is None or prior_growth is None:
        return None
    diff_pt = (latest_growth - prior_growth) * 100
    if diff_pt > 2:
        return 10
    if diff_pt >= -2:
        return 8
    return 3


def ordinary_income_margin(
    ordinary_income: float | None, net_sales: float | None
) -> float | None:
    if ordinary_income is None or not net_sales:
        return None
    return ordinary_income / net_sales


def _margin_trend_score(
    latest_margin: float | None, prior_margin: float | None
) -> float | None:
    """
    経常利益率が前期より拡大しているか(pt差で判定)。
    - 拡大(0.5pt超の改善): 10点
    - ほぼ横ばい(±0.5pt以内): 8点
    - 縮小(0.5pt超の悪化): 3点
    """
    if latest_margin is None or prior_margin is None:
        return None
    diff_pt = (latest_margin - prior_margin) * 100
    if diff_pt > 0.5:
        return 10
    if diff_pt >= -0.5:
        return 8
    return 3


def performance_momentum_score(
    latest_growth: float | None,
    prior_growth: float | None,
    latest_margin: float | None = None,
    prior_margin: float | None = None,
) -> float | None:
    """
    「業績補正」の代理指標。本来は決算短信の業績予想修正(上方修正等、TDnet由来)を
    使うべきだが、無料データでは取得できないため、①増収率の加速度 と
    ②経常利益率のトレンド(拡大/縮小) の2つを組み合わせて代用する
    (どちらも既存データのみで計算できる)。
    両方が取得できた場合は平均、片方だけの場合はその値を使う
    (どちらも欠けている場合はNone=判定不能)。
    """
    growth_score = _revenue_growth_acceleration_score(latest_growth, prior_growth)
    margin_score = _margin_trend_score(latest_margin, prior_margin)
    if growth_score is None and margin_score is None:
        return None
    if growth_score is None:
        return margin_score
    if margin_score is None:
        return growth_score
    return (growth_score + margin_score) / 2


def price_relative_score(
    momentum: float | None, trend: float | None, performance: float | None
) -> float | None:
    """PRS(30点満点) = モメンタム位置 + トレンド方向 + 業績補正"""
    return _sum_if_all_present(momentum, trend, performance)


# ══════════════════════════════════════════════════════════════
# リスク調整(安全性チェック) — GQS・GPS・PRSの高得点だけでは見えない
# 「安全性の欠如」を減点方式で反映する。加点はせず、リスクがある場合にのみ
# 減点する(0が上限)。判定できない要素は0点(減点なし)として扱い、
# GPRS自体が判定不能になることは無いようにする(GQS・GPS・PRSとは扱いが異なる、
# あくまで補助的なブレーキ役のため)。
# ══════════════════════════════════════════════════════════════

EQUITY_RATIO_PENALTY_BANDS: list[tuple[float, float, float]] = [
    (0.40, float("inf"), 0),
    (0.30, 0.40, -2),
    (0.20, 0.30, -5),
    (float("-inf"), 0.20, -10),
]


def equity_ratio_penalty(equity_ratio: float | None) -> float | None:
    """自己資本比率が低い(借入依存度が高い)ほど減点する。"""
    return _score_from_bands(equity_ratio, EQUITY_RATIO_PENALTY_BANDS)


GROWTH_QUALITY_PENALTY_BANDS: list[tuple[float, float, float]] = [
    (0.0, float("inf"), 0),
    (-0.05, 0.0, -3),
    (float("-inf"), -0.05, -7),
]


def growth_quality_penalty(growth_quality_raw: float | None) -> float | None:
    """
    「成長の質」(quality_score.growth_quality_raw、既存の複合スコアで使っている指標)が
    マイナス(増資・借入への依存度が高い)ほど減点する。
    """
    return _score_from_bands(growth_quality_raw, GROWTH_QUALITY_PENALTY_BANDS)


PER_SAFETY_PENALTY_THRESHOLD = 100.0
PER_SAFETY_PENALTY = -10


def per_safety_penalty(per: float | None) -> float | None:
    """
    PERが極端に高い(リンチレシオスコアの計算上は成長率・配当利回り次第で見かけ上
    「割安」に見えてしまうケースがあるため)場合に、絶対水準としての行き過ぎを
    別途チェックする安全弁。赤字(PERが意味を持たない、per<=0)は判定不能としてNoneを返す。
    """
    if per is None:
        return None
    if per <= 0:
        return None
    return PER_SAFETY_PENALTY if per > PER_SAFETY_PENALTY_THRESHOLD else 0


DEFICIT_PENALTY_PER_YEAR = -10


def deficit_penalty(deficit_years: int | None) -> float | None:
    """
    直近5期(今期含む、5年サマリー表の範囲)のうち純利益が赤字だった期数に応じて
    減点する(1期赤字ごとに-10点。例: 2期赤字なら-20点)。赤字企業には投資しない
    という明確な方針をスコアに直接反映するための、他のリスク要素より重い減点。
    データが無い(判定できない)場合はNone(減点なし)。
    """
    if deficit_years is None:
        return None
    return DEFICIT_PENALTY_PER_YEAR * deficit_years


def risk_adjustment(
    equity_ratio: float | None, growth_quality_raw: float | None, per: float | None,
    deficit_years: int | None = None,
) -> float:
    """
    4つのリスク要素の減点を合計する。判定できない要素は0(減点なし)として無視し、
    全て判定できない場合も0を返す(GPRS計算自体を妨げない)。
    """
    penalties = [
        equity_ratio_penalty(equity_ratio),
        growth_quality_penalty(growth_quality_raw),
        per_safety_penalty(per),
        deficit_penalty(deficit_years),
    ]
    return sum(p for p in penalties if p is not None)


# ══════════════════════════════════════════════════════════════
# Growth & Price Relative Score(GPRS) = GPS + PRS + リスク調整(60点満点、
# リスク調整は0以下なのでこの2つが満点でも60点を超えることはない)
#
# 元々はGQS(40点満点)も合計に含めていたが、複数時点・複数保有期間のバックテストで
# GQSの4サブスコア(GP・粗利成長・FCF・安定性)がいずれも将来リターンとほぼ無相関
# 〜わずかにマイナス(特にGPスコアと安定性係数)で、GQSを含めるとGPS+PRSのみの
# 場合より全期間で相関が悪化する結果が出たため、GPRSの合計からは除外した。
# GQS自体は削除せず、質のスクリーニング用の参考情報として別枠で表示を続ける
# (quality_score.build_quality_row・finder.build_quality_table参照)。
# S/A/B/C/Dの評価境界は、100点満点時代の90/80/70/60(=90%/80%/70%/60%)と
# 同じ割合を60点満点に換算した値(54/48/42/36)にしている。
# ══════════════════════════════════════════════════════════════

GPRS_GRADE_BANDS: list[tuple[float, float, str]] = [
    (54, float("inf"), "S"),
    (48, 54, "A"),
    (42, 48, "B"),
    (36, 42, "C"),
    (float("-inf"), 36, "D"),
]


def growth_price_relative_score(
    gps: float | None, prs: float | None, risk_adjustment_value: float = 0
) -> float | None:
    """
    GPRS(60点満点) = GPS + PRS + リスク調整。
    GPS・PRSはいずれか1つでも欠けている場合はNone(判定不能)。
    リスク調整は必須ではなく、既定値0(調整なし)。
    """
    base = _sum_if_all_present(gps, prs)
    if base is None:
        return None
    return base + risk_adjustment_value


def gprs_grade(gprs: float | None) -> str | None:
    """GPRSの点数からS/A/B/C/D評価を返す。"""
    if gprs is None:
        return None
    for low, high, grade in GPRS_GRADE_BANDS:
        if low <= gprs < high:
            return grade
    return None
