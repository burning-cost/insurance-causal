"""Tests for UK insurance shock proximity checking."""

import pytest
from insurance_causal.rate_change._shocks import (
    check_shock_proximity,
    UK_INSURANCE_SHOCKS,
    _period_to_year_quarter,
    _quarters_between,
)


class TestPeriodToYearQuarter:
    def test_yyyy_qn_format(self):
        assert _period_to_year_quarter("2022-Q1") == (2022, 1)
        assert _period_to_year_quarter("2019-Q4") == (2019, 4)

    def test_yyyy_mm_format(self):
        yq = _period_to_year_quarter("2022-01")
        assert yq == (2022, 1)
        yq = _period_to_year_quarter("2022-07")
        assert yq == (2022, 3)

    def test_yyyy_mm_dd_format(self):
        yq = _period_to_year_quarter("2022-01-15")
        assert yq == (2022, 1)

    def test_integer_year(self):
        yq = _period_to_year_quarter("2022")
        assert yq == (2022, 2)

    def test_pandas_period_format(self):
        yq = _period_to_year_quarter("2022Q1")
        assert yq == (2022, 1)

    def test_unparseable_returns_none(self):
        assert _period_to_year_quarter("not_a_period") is None


class TestQuartersBetween:
    def test_same_quarter(self):
        assert _quarters_between((2022, 1), (2022, 1)) == 0

    def test_one_quarter_apart(self):
        assert _quarters_between((2022, 1), (2022, 2)) == 1

    def test_one_year_apart(self):
        assert _quarters_between((2021, 1), (2022, 1)) == 4

    def test_cross_year(self):
        assert _quarters_between((2021, 4), (2022, 1)) == 1


class TestCheckShockProximity:
    def test_gipp_flagged(self):
        msgs = check_shock_proximity("2022-Q1", proximity_quarters=2)
        assert any("GIPP" in m or "PS21" in m for m in msgs)

    def test_covid_flagged(self):
        msgs = check_shock_proximity("2020-Q2", proximity_quarters=2)
        assert any("COVID" in m or "lockdown" in m.lower() or "pandemic" in m.lower() for m in msgs)

    def test_ogden_2017_flagged(self):
        msgs = check_shock_proximity("2017-Q2", proximity_quarters=2)
        assert any("Ogden" in m for m in msgs)

    def test_benign_period_no_warnings(self):
        msgs = check_shock_proximity("2015-Q3", proximity_quarters=2)
        assert len(msgs) == 0

    def test_proximity_threshold_respected(self):
        # 2022-Q3 is 2 quarters from 2022-Q1 (GIPP)
        msgs_2q = check_shock_proximity("2022-Q3", proximity_quarters=2)
        msgs_1q = check_shock_proximity("2022-Q3", proximity_quarters=1)
        # With 2-quarter proximity, GIPP at 2022-Q1 should be flagged
        assert len(msgs_2q) >= len(msgs_1q)

    def test_unparseable_period_returns_empty(self):
        msgs = check_shock_proximity("garbage_period")
        assert msgs == []

    def test_uk_insurance_shocks_not_empty(self):
        assert len(UK_INSURANCE_SHOCKS) > 0

    def test_all_shocks_have_valid_periods(self):
        for period in UK_INSURANCE_SHOCKS.keys():
            yq = _period_to_year_quarter(period)
            assert yq is not None, f"Could not parse shock period: {period}"
