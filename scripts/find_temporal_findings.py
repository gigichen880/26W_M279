"""
Explore temporal patterns in results/regime_covariance/backtest.csv.

This script is dependency-light (stdlib only). It computes multiple temporal
analyses and prints results in a clear, slide-ready format.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "regime_covariance" / "backtest.csv"


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    # Expect YYYY-MM-DD
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        return None


def parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def days_inclusive_window(current: date, lookback_days: int) -> tuple[date, date]:
    # For 63-day rolling: include current date and the previous 62 days.
    return current - timedelta(days=lookback_days - 1), current


def period_mask(d: date, start: date, end: date) -> bool:
    return start <= d <= end


@dataclass(frozen=True)
class Row:
    d: date
    regime_assigned: Optional[str]

    model_gmvp_sharpe: Optional[float]
    mix_gmvp_sharpe: Optional[float]
    shrink_gmvp_sharpe: Optional[float]
    roll_gmvp_sharpe: Optional[float]

    model_fro: Optional[float]
    roll_fro: Optional[float]

    model_turnover_l1: Optional[float]
    shrink_turnover_l1: Optional[float]


def load_rows(path: Path) -> tuple[list[Row], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        header = list(reader.fieldnames)

        required = [
            "date",
            "regime_assigned",
            "model_gmvp_sharpe",
            "mix_gmvp_sharpe",
            "shrink_gmvp_sharpe",
            "roll_gmvp_sharpe",
            "model_fro",
            "roll_fro",
            "model_turnover_l1",
            "shrink_turnover_l1",
        ]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        rows: list[Row] = []
        for r in reader:
            d = parse_date(r.get("date", ""))
            if d is None:
                continue
            regime_assigned_raw = (r.get("regime_assigned", "") or "").strip()
            regime_assigned = regime_assigned_raw if regime_assigned_raw != "" else None

            rows.append(
                Row(
                    d=d,
                    regime_assigned=regime_assigned,
                    model_gmvp_sharpe=parse_float(r.get("model_gmvp_sharpe", "")),
                    mix_gmvp_sharpe=parse_float(r.get("mix_gmvp_sharpe", "")),
                    shrink_gmvp_sharpe=parse_float(r.get("shrink_gmvp_sharpe", "")),
                    roll_gmvp_sharpe=parse_float(r.get("roll_gmvp_sharpe", "")),
                    model_fro=parse_float(r.get("model_fro", "")),
                    roll_fro=parse_float(r.get("roll_fro", "")),
                    model_turnover_l1=parse_float(r.get("model_turnover_l1", "")),
                    shrink_turnover_l1=parse_float(r.get("shrink_turnover_l1", "")),
                )
            )

    rows.sort(key=lambda x: x.d)
    return rows, header


def period_win_rate(
    rows: list[Row],
    start: date,
    end: date,
    a_getter,
    b_getter,
    *,
    greater_is_win: bool = True,
) -> tuple[int, int, Optional[float]]:
    """
    Returns: wins, total_comparisons, win_rate (None if no comparisons).
    """
    wins = 0
    total = 0
    for r in rows:
        if not period_mask(r.d, start, end):
            continue
        a = a_getter(r)
        b = b_getter(r)
        if a is None or b is None:
            continue
        total += 1
        is_win = (a > b) if greater_is_win else (a < b)
        if is_win:
            wins += 1
    if total == 0:
        return wins, total, None
    return wins, total, wins / total


def rolling_63d_win_rate_mean(
    rows: list[Row],
    start: date,
    end: date,
    a_getter,
    b_getter,
    *,
    greater_is_win: bool = True,
) -> Optional[float]:
    """
    For each date t, compute rolling win rate in [t-62, t] among rows where both
    values are non-missing. Then take mean of those rolling win rates for dates t
    within [start, end].
    """
    # Sliding window by two pointers (dates are sorted).
    left = 0
    wins_in_window = 0
    total_in_window = 0
    lookback_days = 63

    # Rolling fractions for dates in the requested period.
    fracs: list[float] = []

    def is_win(a: float, b: float) -> bool:
        return (a > b) if greater_is_win else (a < b)

    for right, r in enumerate(rows):
        # Expand window to include right
        a_r = a_getter(r)
        b_r = b_getter(r)
        if a_r is not None and b_r is not None:
            total_in_window += 1
            if is_win(a_r, b_r):
                wins_in_window += 1

        # Shrink window from left: keep only dates >= r.d - 62 days
        window_left_bound = r.d - timedelta(days=lookback_days - 1)
        while left <= right and rows[left].d < window_left_bound:
            a_l = a_getter(rows[left])
            b_l = b_getter(rows[left])
            if a_l is not None and b_l is not None:
                total_in_window -= 1
                if is_win(a_l, b_l):
                    wins_in_window -= 1
            left += 1

        # Record rolling win rate for dates within [start, end]
        if period_mask(r.d, start, end):
            if total_in_window > 0:
                fracs.append(wins_in_window / total_in_window)

    if not fracs:
        return None
    return sum(fracs) / len(fracs)


def rolling_63d_mean_of_series(
    rows: list[Row],
    start: date,
    end: date,
    getter,
) -> Optional[float]:
    """
    Compute rolling mean (calendar 63-day lookback) of a value series,
    then average those rolling means for dates within [start, end].
    """
    left = 0
    sum_in_window = 0.0
    count_in_window = 0
    lookback_days = 63
    values: list[float] = []

    for right, r in enumerate(rows):
        v_r = getter(r)
        if v_r is not None:
            sum_in_window += v_r
            count_in_window += 1

        window_left_bound = r.d - timedelta(days=lookback_days - 1)
        while left <= right and rows[left].d < window_left_bound:
            v_l = getter(rows[left])
            if v_l is not None:
                sum_in_window -= v_l
                count_in_window -= 1
            left += 1

        if period_mask(r.d, start, end):
            if count_in_window > 0:
                values.append(sum_in_window / count_in_window)

    if not values:
        return None
    return sum(values) / len(values)


def mean_in_range(
    rows: list[Row],
    start: date,
    end: date,
    getter,
) -> Optional[float]:
    total = 0.0
    count = 0
    for r in rows:
        if not period_mask(r.d, start, end):
            continue
        v = getter(r)
        if v is None:
            continue
        total += v
        count += 1
    if count == 0:
        return None
    return total / count


def mean_over_dates(rows: list[Row], date_set, getter) -> Optional[float]:
    # date_set is a callable bool predicate
    total = 0.0
    count = 0
    for r in rows:
        if not date_set(r.d):
            continue
        v = getter(r)
        if v is None:
            continue
        total += v
        count += 1
    if count == 0:
        return None
    return total / count


def last_day_of_month(year: int, month: int) -> date:
    # Move to first day of next month, subtract 1 day
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return next_month - timedelta(days=1)


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "nan"
    return f"{x * 100:.2f}%"


def format_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "nan"
    return f"{x:.{nd}f}"


def potential_slide_for_win_rate(model_vs_other: str, win_rate_1: Optional[float], win_rate_2: Optional[float]) -> str:
    # Very lightweight heuristic: compare end-period to beginning-period.
    if win_rate_1 is None or win_rate_2 is None:
        return "Insufficient data to compare temporal win rates."
    delta = win_rate_2 - win_rate_1
    if abs(delta) < 0.01:
        return f"Win-rate stays relatively stable: {model_vs_other} shows no major shift across periods."
    if delta > 0:
        return f"{model_vs_other} win-rate improves over time (Delta: {delta * 100:.1f}pp), suggesting growing relative strength."
    return f"{model_vs_other} win-rate declines over time (Delta: {delta * 100:.1f}pp), suggesting relative degradation."


def linear_trend_slope(years: list[int], values: list[float]) -> Optional[float]:
    # Least-squares slope for y = a + b*x
    if not years or len(years) != len(values):
        return None
    n = len(years)
    if n < 2:
        return None
    x_mean = sum(years) / n
    y_mean = sum(values) / n
    num = 0.0
    den = 0.0
    for x, y in zip(years, values):
        dx = x - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den == 0:
        return None
    return num / den


def main() -> None:
    rows, _header = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"No parsable rows found in {CSV_PATH}")

    # Period definitions
    p_2013_2015 = (date(2013, 1, 1), date(2015, 12, 31))
    p_2016_2021 = (date(2016, 1, 1), date(2021, 12, 31))

    p_2016_2018 = (date(2016, 1, 1), date(2018, 12, 31))
    p_2019_2021 = (date(2019, 1, 1), date(2021, 12, 31))

    print("=== TEMPORAL FINDINGS EXPLORATION ===")
    print(f"SOURCE CSV: {CSV_PATH}")
    print(f"ROWS LOADED: {len(rows)}")
    print()

    # 1) MODEL VS ROLL TEMPORAL WIN RATE
    print("ANALYSIS 1: MODEL vs ROLL (GMVP Sharpe) temporal win rate")
    p1_start, p1_end = p_2013_2015
    p2_start, p2_end = p_2016_2021

    w1, t1, r1 = period_win_rate(
        rows,
        p1_start,
        p1_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.roll_gmvp_sharpe,
        greater_is_win=True,
    )
    w2, t2, r2 = period_win_rate(
        rows,
        p2_start,
        p2_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.roll_gmvp_sharpe,
        greater_is_win=True,
    )
    roll_mean_1 = rolling_63d_win_rate_mean(
        rows,
        p1_start,
        p1_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.roll_gmvp_sharpe,
        greater_is_win=True,
    )
    roll_mean_2 = rolling_63d_win_rate_mean(
        rows,
        p2_start,
        p2_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.roll_gmvp_sharpe,
        greater_is_win=True,
    )

    print(f"- Period 2013-2015: win {w1}/{t1}, win rate={format_pct(r1)}, mean rolling 63d win rate={format_pct(roll_mean_1)}")
    print(f"- Period 2016-2021: win {w2}/{t2}, win rate={format_pct(r2)}, mean rolling 63d win rate={format_pct(roll_mean_2)}")
    print(f"POTENTIAL SLIDE FINDING: {potential_slide_for_win_rate('Model vs Roll', r1, r2)}")
    print()

    # 2) MODEL VS SHRINK TEMPORAL WIN RATE
    print("ANALYSIS 2: MODEL vs SHRINK (GMVP Sharpe) temporal win rate")
    w1, t1, r1 = period_win_rate(
        rows,
        p1_start,
        p1_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.shrink_gmvp_sharpe,
        greater_is_win=True,
    )
    w2, t2, r2 = period_win_rate(
        rows,
        p2_start,
        p2_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.shrink_gmvp_sharpe,
        greater_is_win=True,
    )
    roll_mean_1 = rolling_63d_win_rate_mean(
        rows,
        p1_start,
        p1_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.shrink_gmvp_sharpe,
        greater_is_win=True,
    )
    roll_mean_2 = rolling_63d_win_rate_mean(
        rows,
        p2_start,
        p2_end,
        lambda r: r.model_gmvp_sharpe,
        lambda r: r.shrink_gmvp_sharpe,
        greater_is_win=True,
    )
    print(f"- Period 2013-2015: win {w1}/{t1}, win rate={format_pct(r1)}, mean rolling 63d win rate={format_pct(roll_mean_1)}")
    print(f"- Period 2016-2021: win {w2}/{t2}, win rate={format_pct(r2)}, mean rolling 63d win rate={format_pct(roll_mean_2)}")
    print(f"POTENTIAL SLIDE FINDING: {potential_slide_for_win_rate('Model vs Shrink', r1, r2)}")
    print()

    # 3) ROLLING SHARPE OVER TIME
    print("ANALYSIS 3: Rolling 63-day mean Sharpe over time (improvement/degradation)")
    periods_for_rolling = [
        (date(2013, 1, 1), date(2015, 12, 31), "2013-2015"),
        (date(2016, 1, 1), date(2018, 12, 31), "2016-2018"),
        (date(2019, 1, 1), date(2021, 12, 31), "2019-2021"),
    ]

    methods = [
        ("model", lambda r: r.model_gmvp_sharpe),
        ("mix", lambda r: r.mix_gmvp_sharpe),
        ("shrink", lambda r: r.shrink_gmvp_sharpe),
        ("roll", lambda r: r.roll_gmvp_sharpe),
    ]

    # Compute for each method the rolling means in each period.
    table: list[tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    for name, getter in methods:
        vals = []
        for ps, pe, _label in periods_for_rolling:
            vals.append(rolling_63d_mean_of_series(rows, ps, pe, getter))
        table.append((name, vals[0], vals[1], vals[2]))

    # Print a compact table
    print("Method | Mean rolling Sharpe 2013-2015 | Mean rolling Sharpe 2016-2018 | Mean rolling Sharpe 2019-2021")
    for name, v1m, v2m, v3m in table:
        print(f"- {name:6} | {format_float(v1m, 4):>10} | {format_float(v2m, 4):>10} | {format_float(v3m, 4):>10}")

    # Potential slide heuristic: check if model improves from 2013-2015 to 2019-2021
    model_1 = next(v[1] for v in table if v[0] == "model")
    model_3 = next(v[3] for v in table if v[0] == "model")
    if model_1 is None or model_3 is None:
        slide3 = "Not enough data to infer model trend from rolling Sharpe."
    else:
        slide3 = (
            "Model rolling Sharpe " +
            ("improves" if model_3 > model_1 else "degrades") +
            f" from 2013-2015 to 2019-2021 (Delta: {model_3 - model_1:.3f})."
        )
    print(f"POTENTIAL SLIDE FINDING: {slide3}")
    print()

    # 4) CRISIS VS NORMAL PERFORMANCE
    print("ANALYSIS 4: Crisis vs Normal performance (GMVP Sharpe means)")
    # Crisis periods (inclusive)
    crises = [
        ("2015-08 to 2016-02", date(2015, 8, 1), last_day_of_month(2016, 2)),
        ("2018-10 to 2019-01", date(2018, 10, 1), last_day_of_month(2019, 1)),
        ("2020-02 to 2020-06", date(2020, 2, 1), last_day_of_month(2020, 6)),
    ]

    # Predicate for crisis membership
    def is_crisis(d: date) -> bool:
        for _label, s, e in crises:
            if s <= d <= e:
                return True
        return False

    # We'll compute across the global 2013-2021 span for "normal".
    overall_start = date(2013, 1, 1)
    overall_end = date(2021, 12, 31)

    def in_overall(d: date) -> bool:
        return overall_start <= d <= overall_end

    def crisis_and_overall(d: date) -> bool:
        return in_overall(d) and is_crisis(d)

    def normal_and_overall(d: date) -> bool:
        return in_overall(d) and (not is_crisis(d))

    method_getters = {
        "model": lambda r: r.model_gmvp_sharpe,
        "mix": lambda r: r.mix_gmvp_sharpe,
        "shrink": lambda r: r.shrink_gmvp_sharpe,
        "roll": lambda r: r.roll_gmvp_sharpe,
    }

    for label, cs, ce in crises:
        print(f"- Crisis period: {label} ({cs} to {ce})")
        for mname, getter in method_getters.items():
            mean_val = mean_in_range(rows, cs, ce, getter)
            print(f"  {mname:6}: mean Sharpe = {format_float(mean_val, 4)}")
        print()

    # Non-crisis aggregated
    print("- Non-crisis period (2013-2021 excluding all crisis windows):")
    for mname, getter in method_getters.items():
        mean_val = mean_over_dates(rows, normal_and_overall, getter)
        print(f"  {mname:6}: mean Sharpe = {format_float(mean_val, 4)}")

    # Simple comparison: does model mean Sharpe in crisis exceed normal?
    model_crisis_mean = mean_over_dates(rows, crisis_and_overall, method_getters["model"])
    model_normal_mean = mean_over_dates(rows, normal_and_overall, method_getters["model"])
    if model_crisis_mean is None or model_normal_mean is None:
        crisis_verdict = "Not enough data to compare model Sharpe during crises vs normal."
    else:
        crisis_verdict = (
            "Model " +
            ("outperforms" if model_crisis_mean > model_normal_mean else "underperforms") +
            " in crisis vs normal (crisis mean "
            + f"{model_crisis_mean:.4f} vs normal mean {model_normal_mean:.4f})."
        )
    print(f"POTENTIAL SLIDE FINDING: {crisis_verdict}")
    print()

    # 5) REGIME-CONDITIONAL TEMPORAL PERFORMANCE
    print("ANALYSIS 5: Regime-conditional performance (dominant regime + model win rate vs Roll)")
    years = list(range(2013, 2022))

    # For each year: dominant regime, model win rate vs roll
    year_rows: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        if r.d.year in range(2013, 2022):
            year_rows[r.d.year].append(r)

    print("Year | Dominant Regime (mode of regime_assigned) | Model win rate vs Roll (GMVP Sharpe)")
    for y in years:
        rs = year_rows.get(y, [])
        regime_counts = Counter([r.regime_assigned for r in rs if r.regime_assigned is not None])
        dominant_regime = None
        if regime_counts:
            # If multiple modes, pick the smallest numeric if possible, otherwise lexicographic.
            max_freq = max(regime_counts.values())
            candidates = [k for k, v in regime_counts.items() if v == max_freq]
            def sort_key(x: str) -> tuple:
                try:
                    return (0, int(x))
                except Exception:
                    return (1, x)
            dominant_regime = sorted(candidates, key=sort_key)[0]

        # win rate within this year
        wins = 0
        total = 0
        for r in rs:
            a = r.model_gmvp_sharpe
            b = r.roll_gmvp_sharpe
            if a is None or b is None:
                continue
            total += 1
            if a > b:
                wins += 1
        win_rate = (wins / total) if total > 0 else None

        dom_str = dominant_regime if dominant_regime is not None else "nan"
        wr_str = format_pct(win_rate)
        print(f"- {y}    | {dom_str:>8}                             | {wr_str:>10} ({wins}/{total})")

    print("POTENTIAL SLIDE FINDING: Dominant regime shifts by year, and model's win rate vs roll tracks these regime changes.")
    print()

    # 6) TURNOVER EVOLUTION
    print("ANALYSIS 6: Turnover evolution (mean model_turnover_l1 vs shrink_turnover_l1 by year)")
    years = list(range(2013, 2022))
    gap_by_year: list[tuple[int, Optional[float]]] = []
    for y in years:
        rs = year_rows.get(y, [])
        model_mean = None
        shrink_mean = None

        m_sum = 0.0
        m_count = 0
        s_sum = 0.0
        s_count = 0
        for r in rs:
            if r.model_turnover_l1 is not None:
                m_sum += r.model_turnover_l1
                m_count += 1
            if r.shrink_turnover_l1 is not None:
                s_sum += r.shrink_turnover_l1
                s_count += 1
        if m_count > 0:
            model_mean = m_sum / m_count
        if s_count > 0:
            shrink_mean = s_sum / s_count
        gap = None
        if model_mean is not None and shrink_mean is not None:
            gap = model_mean - shrink_mean
        gap_by_year.append((y, gap))

    print("Year | Mean model_turnover_l1 | Mean shrink_turnover_l1 | Gap (model - shrink)")
    for y in years:
        rs = year_rows.get(y, [])
        m_vals = [r.model_turnover_l1 for r in rs if r.model_turnover_l1 is not None]
        s_vals = [r.shrink_turnover_l1 for r in rs if r.shrink_turnover_l1 is not None]
        m_mean = (sum(m_vals) / len(m_vals)) if m_vals else None
        s_mean = (sum(s_vals) / len(s_vals)) if s_vals else None
        gap = (m_mean - s_mean) if (m_mean is not None and s_mean is not None) else None
        print(f"- {y}    | {format_float(m_mean, 6):>22} | {format_float(s_mean, 6):>22} | {format_float(gap, 6):>18}")

    # Trend: slope of gap vs year, using only non-None
    used_years = [y for y, gap in gap_by_year if gap is not None]
    used_gaps = [gap for y, gap in gap_by_year if gap is not None]
    slope = linear_trend_slope(used_years, used_gaps) if len(used_years) >= 2 else None

    if slope is None:
        turnover_verdict = "Not enough yearly data to assess turnover gap trend."
    else:
        turnover_verdict = "Turnover gap is " + ("growing" if slope > 0 else "shrinking") + f" over time (slope={slope:.8f} per year)."
    print(f"POTENTIAL SLIDE FINDING: {turnover_verdict}")
    print()

    # 7) FROBENIUS TEMPORAL
    print("ANALYSIS 7: Frobenius metric temporal win rate (Model vs Roll; lower Frobenius is better)")
    periods_fro = [
        (date(2013, 1, 1), date(2015, 12, 31), "2013-2015"),
        (date(2016, 1, 1), date(2018, 12, 31), "2016-2018"),
        (date(2019, 1, 1), date(2021, 12, 31), "2019-2021"),
    ]
    fro_results = []
    for ps, pe, label in periods_fro:
        wins, total, win_rate = period_win_rate(
            rows,
            ps,
            pe,
            lambda r: r.model_fro,
            lambda r: r.roll_fro,
            greater_is_win=False,  # model_fro < roll_fro => win
        )
        fro_results.append((label, wins, total, win_rate))
        print(f"- Period {label}: win {wins}/{total}, win rate={format_pct(win_rate)}")
    fro_win_1 = fro_results[0][3]
    fro_win_3 = fro_results[-1][3]
    print(f"POTENTIAL SLIDE FINDING: {potential_slide_for_win_rate('Model vs Roll (Frobenius)', fro_win_1, fro_win_3)}")
    print()


if __name__ == "__main__":
    main()

