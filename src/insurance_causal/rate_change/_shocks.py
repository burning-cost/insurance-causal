"""
Known UK insurance market shocks.

A lookup of material external events that can confound rate change evaluations.
If the intervention date is within 2 quarters of any shock, RateChangeEvaluator
emits a UserWarning identifying the potential confounder.

References
----------
Ogden rate changes: Ministry of Justice announcements (2017, 2019).
GIPP: FCA PS21/5 — General Insurance Pricing Practices, effective January 2022.
Whiplash Injury Regulations 2021: Ministry of Justice, Civil Liability Act 2018.
COVID-19 lockdowns: UK Government announcements March 2020, November 2020.
BoE: Bank of England base rate cycle from December 2021 to August 2023.
"""

from __future__ import annotations

import re
import warnings

# Quarter string -> (year, quarter_number) mapping
# Values are (year, q) tuples for arithmetic comparison
UK_INSURANCE_SHOCKS: dict[str, str] = {
    "Ogden rate change (-0.75%)": "2017Q2",
    "Ogden rate change (-0.25%)": "2019Q3",
    "COVID-19 lockdown 1": "2020Q1",
    "COVID-19 lockdown 2": "2020Q4",
    "Whiplash Injury Regulations": "2021Q2",
    "GIPP (PS21/5) implementation": "2022Q1",
    "BoE rapid rate rises begin": "2022Q3",
}


def _parse_quarter_str(q: str) -> tuple[int, int] | None:
    """Parse a quarter string like '2022Q1' or '2022-Q1' into (year, quarter)."""
    if q is None:
        return None
    q_str = str(q).strip().upper().replace("-", "")
    match = re.match(r"^(\d{4})Q([1-4])$", q_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _quarter_distance(q1: tuple[int, int], q2: tuple[int, int]) -> int:
    """Return the absolute distance in quarters between two (year, q) tuples."""
    y1, n1 = q1
    y2, n2 = q2
    return abs((y1 * 4 + n1) - (y2 * 4 + n2))


def check_shock_proximity(
    change_period: str | int,
    proximity_quarters: int = 2,
) -> list[str]:
    """
    Check whether a change_period is within ``proximity_quarters`` of any
    known UK insurance shock.

    Parameters
    ----------
    change_period : str | int
        The intervention period. Only quarter strings (e.g. "2022Q1",
        "2022-Q1") trigger shock lookup. Integer periods are ignored.

    proximity_quarters : int
        Number of quarters either side to flag. Default 2.

    Returns
    -------
    list[str]
        Names of nearby shocks. Empty if none found or period is not a
        quarter string.
    """
    parsed = _parse_quarter_str(str(change_period))
    if parsed is None:
        return []

    nearby: list[str] = []
    for shock_name, shock_q_str in UK_INSURANCE_SHOCKS.items():
        shock_q = _parse_quarter_str(shock_q_str)
        if shock_q is None:
            continue
        dist = _quarter_distance(parsed, shock_q)
        if dist <= proximity_quarters:
            nearby.append(f"{shock_name} ({shock_q_str})")

    return nearby


def warn_if_near_shock(change_period: str | int, proximity_quarters: int = 2) -> list[str]:
    """
    Emit a UserWarning and return shock names if change_period is near a
    known UK insurance shock.

    Returns
    -------
    list[str]
        Shock names emitted as warnings. Empty if none.
    """
    nearby = check_shock_proximity(change_period, proximity_quarters)
    for shock in nearby:
        warnings.warn(
            f"Intervention period '{change_period}' is within {proximity_quarters} "
            f"quarters of a known UK insurance market shock: {shock}. "
            "This may confound the rate change estimate. "
            "Validate that the shock did not differentially affect treated vs control groups.",
            UserWarning,
            stacklevel=4,
        )
    return nearby
