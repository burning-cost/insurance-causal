"""Tests for UK insurance shock proximity checking."""

import pytest
from insurance_causal.rate_change._shocks import (
    check_shock_proximity,
    UK_INSURANCE_SHOCKS,
    _parse_quarter_str,
    _quarter_distance,
)


class TestParseQuarterStr:
    def test_yyyyqn_format(self):
        assert _parse_quarter_str("2022Q1") == (2022, 1)
        assert _parse_quarter_str("2019Q4") == (2019, 4)

    def test_yyyy_qn_format_with_dash(self):
        assert _parse_quarter_str("2022-Q1") == (2022, 1)
        assert _parse_quarter_str("2019-Q4") == (2019, 4)

    def test_lowercase(self):
        assert _parse_quarter_str("2022q1") == (2022, 1)

    def test_unparseable_returns_none(self):
        assert _parse_quarter_str("not_a_period") is None

    def test_integer_returns_none(self):
        # Bare integers are not quarter strings
        assert _parse_quarter_str("2022") is None


class TestQuarterDistance:
    def test_same_quarter(self):
        assert _quarter_distance((2022, 1), (2022, 1)) == 0

    def test_one_quarter_apart(self):
        assert _quarter_distance((2022, 1), (2022, 2)) == 1

    def test_one_year_apart(self):
        assert _quarter_distance((2021, 1), (2022, 1)) == 4

    def test_cross_year(self):
        assert _quarter_distance((2021, 4), (2022, 1)) == 1


class TestCheckShockProximity:
    def test_gipp_flagged(self):
        msgs = check_shock_proximity("2022Q1", proximity_quarters=2)
        assert any("GIPP" in m or "PS21" in m for m in msgs)

    def test_covid_flagged(self):
        msgs = check_shock_proximity("2020Q2", proximity_quarters=2)
        assert any("COVID" in m or "lockdown" in m.lower() for m in msgs)

    def test_ogden_2017_flagged(self):
        msgs = check_shock_proximity("2017Q2", proximity_quarters=2)
        assert any("Ogden" in m for m in msgs)

    def test_benign_period_no_warnings(self):
        msgs = check_shock_proximity("2015Q3", proximity_quarters=2)
        assert len(msgs) == 0

    def test_proximity_threshold_respected(self):
        # 2022-Q3 is 2 quarters from 2022-Q1 (GIPP)
        msgs_2q = check_shock_proximity("2022Q3", proximity_quarters=2)
        msgs_1q = check_shock_proximity("2022Q3", proximity_quarters=1)
        assert len(msgs_2q) >= len(msgs_1q)

    def test_unparseable_period_returns_empty(self):
        msgs = check_shock_proximity("garbage_period")
        assert msgs == []

    def test_uk_insurance_shocks_not_empty(self):
        assert len(UK_INSURANCE_SHOCKS) > 0

    def test_all_shocks_have_valid_periods(self):
        for period in UK_INSURANCE_SHOCKS.keys():
            yq = _parse_quarter_str(period)
            # Shocks are stored as values, not keys
        for period in UK_INSURANCE_SHOCKS.values():
            yq = _parse_quarter_str(period)
            assert yq is not None, f"Could not parse shock period: {period}"
