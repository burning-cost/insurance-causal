"""Known UK insurance market shocks for proximity checking.

When an intervention date falls near a known market shock, any causal
estimate from DiD or ITS is potentially confounded. This module provides
a dictionary of shocks and a function to check proximity.
"""

from __future__ import annotations

from typing import Any, Optional
import warnings

# UK insurance market shocks that can confound rate change evaluations.
# Each entry: (approximate_period_label, description)
# Periods are stored as year-quarter strings where relevant, or year strings.
UK_INSURANCE_SHOCKS: dict[str, str] = {
    "2017-Q1": "Ogden rate cut (from 2.5% to -0.75%) — large bodily injury reserve increases",
    "2019-Q3": "Ogden rate partial recovery (from -0.75% to -0.25%)",
    "2020-Q1": "COVID-19 pandemic onset — dramatic reduction in motor claims frequency",
    "2020-Q2": "COVID-19 lockdown — near-zero motor frequency, home claims spike",
    "2021-Q2": "Whiplash reform — Civil Liability Act 2018 in force from May 2021",
    "2022-Q1": "GIPP (General Insurance Pricing Practices) — FCA PS21/5 effective January 2022",
    "2022-Q3": "Ukraine-driven claims inflation peak — used car prices, parts costs surge",
    "2023-Q1": "Motor claims inflation peak — repair costs 30-40% above pre-COVID levels",
    "2023-Q4": "Ogden rate review consultation",
    "2024-Q1": "FCA motor market study — focus on add-on pricing and fair value",
}


def _period_to_year_quarter(period: Any) -> Optional[tuple[int, int]]:
    """Convert a period label to (year, quarter) tuple for proximity checking.

    Handles:
    - Integer year: 2022 -> (2022, 2)  (mid-year)
    - String "YYYY-QN": "2022-Q1" -> (2022, 1)
    - pandas Period: converted via str()
    - datetime-like: uses .year and .month

    Returns None if conversion fails.
    """
    import re

    s = str(period).strip()

    # "YYYY-QN" format
    m = re.match(r"^(\d{4})-Q([1-4])$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "YYYY-MM" or "YYYY-MM-DD"
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        quarter = (month - 1) // 3 + 1
        return year, quarter

    # Pure year integer
    m = re.match(r"^(\d{4})$", s)
    if m:
        return int(m.group(1)), 2  # assume mid-year

    # pandas Period-like: "2022Q1" or "2022Q4"
    m = re.match(r"^(\d{4})Q([1-4])$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None


def _quarters_between(yq1: tuple[int, int], yq2: tuple[int, int]) -> int:
    """Return absolute number of quarters between two (year, quarter) tuples."""
    y1, q1 = yq1
    y2, q2 = yq2
    return abs((y1 * 4 + q1) - (y2 * 4 + q2))


def check_shock_proximity(
    treatment_period: Any,
    proximity_quarters: int = 2,
) -> list[str]:
    """Check whether a treatment period is near any known UK insurance shock.

    Parameters
    ----------
    treatment_period : any
        The intervention period. Accepted formats: "YYYY-QN", "YYYY-MM",
        "YYYY-MM-DD", integer year, pandas Period.
    proximity_quarters : int, default 2
        Number of quarters within which to flag a shock as proximate.

    Returns
    -------
    list of str
        Warning messages for each proximate shock. Empty list if none.

    Examples
    --------
    >>> msgs = check_shock_proximity("2022-Q1")
    >>> print(msgs[0])
    Treatment period 2022-Q1 is within 2 quarters of UK shock: ...
    """
    tp_yq = _period_to_year_quarter(treatment_period)
    if tp_yq is None:
        return []

    messages = []
    for shock_period, shock_desc in UK_INSURANCE_SHOCKS.items():
        shock_yq = _period_to_year_quarter(shock_period)
        if shock_yq is None:
            continue
        dist = _quarters_between(tp_yq, shock_yq)
        if dist <= proximity_quarters:
            messages.append(
                f"Treatment period {treatment_period!r} is within {proximity_quarters} "
                f"quarters of UK market shock at {shock_period}: {shock_desc}. "
                f"Causal estimates may be confounded by this concurrent event."
            )
    return messages
