"""
Tests for the period encoding utilities in rate_change/_evaluator.py.

_parse_period, _encode_periods, and _encode_change_period are module-level
helpers with no dedicated tests. They handle:

- Quarter string parsing ("2023Q1", "2023-Q1" -> integers)
- Fallback to sorted-rank encoding for arbitrary orderable values
- Mapping a change_period value back through the encoding

These functions are on the critical path for any user who passes quarterly
data (the most common real-world use case). A bug here silently mis-orders
periods or places the change point in the wrong period — which invalidates
the entire DiD/ITS estimate.
"""

import pytest
import pandas as pd

from insurance_causal.rate_change._evaluator import (
    _parse_period,
    _encode_periods,
    _encode_change_period,
)


# ---------------------------------------------------------------------------
# _parse_period
# ---------------------------------------------------------------------------


class TestParsePeriod:
    """Unit tests for quarter-string parsing."""

    def test_standard_quarter_string(self):
        """'2023Q2' should parse to 2023*4 + 2 = 8094."""
        assert _parse_period("2023Q2") == 2023 * 4 + 2

    def test_hyphenated_quarter_string(self):
        """'2023-Q2' (with hyphen) should parse identically to '2023Q2'."""
        assert _parse_period("2023-Q2") == _parse_period("2023Q2")

    def test_case_insensitive(self):
        """Quarter strings should be case-insensitive ('2023q1' == '2023Q1')."""
        assert _parse_period("2023q1") == _parse_period("2023Q1")

    def test_all_four_quarters(self):
        """Q1..Q4 should produce strictly increasing integer values."""
        vals = [_parse_period(f"2023Q{q}") for q in range(1, 5)]
        assert vals == sorted(vals)
        assert len(set(vals)) == 4

    def test_year_boundary(self):
        """2022Q4 must be less than 2023Q1 (year boundary)."""
        assert _parse_period("2022Q4") < _parse_period("2023Q1")

    def test_non_quarter_string_returns_none(self):
        """Non-quarter strings (integers-as-strings, plain text) return None."""
        assert _parse_period("1") is None
        assert _parse_period("period_7") is None
        assert _parse_period("202301") is None  # looks like YYYYMM not quarter

    def test_none_input_returns_none(self):
        """None input should return None (no crash)."""
        assert _parse_period(None) is None

    def test_integer_input_returns_none(self):
        """Integer input is not a quarter string; returns None."""
        assert _parse_period(7) is None


# ---------------------------------------------------------------------------
# _encode_periods
# ---------------------------------------------------------------------------


class TestEncodePeriods:
    """Unit tests for period column encoding."""

    def test_quarter_string_encoding_order(self):
        """Quarters should be encoded in chronological order, dense-ranked from 1."""
        series = pd.Series(["2023Q3", "2023Q1", "2023Q2", "2024Q1"])
        encoded, mapping = _encode_periods(series)
        # Chronological order: Q1=1, Q2=2, Q3=3, 2024Q1=4
        assert mapping["2023Q1"] == 1
        assert mapping["2023Q2"] == 2
        assert mapping["2023Q3"] == 3
        assert mapping["2024Q1"] == 4

    def test_encoded_values_match_mapping(self):
        """encoded series values should match the mapping applied to the input."""
        series = pd.Series(["2023Q2", "2023Q1", "2023Q2", "2023Q3"])
        encoded, mapping = _encode_periods(series)
        for orig, enc in zip(series, encoded):
            assert enc == mapping[orig]

    def test_integer_fallback_encoding(self):
        """Integer period values use rank encoding (1 = smallest integer)."""
        series = pd.Series([5, 3, 7, 3, 5])
        encoded, mapping = _encode_periods(series)
        # Unique sorted: [3, 5, 7] -> {3:1, 5:2, 7:3}
        assert mapping[3] == 1
        assert mapping[5] == 2
        assert mapping[7] == 3

    def test_single_period(self):
        """A series with a single distinct period encodes to 1."""
        series = pd.Series(["2023Q1", "2023Q1", "2023Q1"])
        encoded, mapping = _encode_periods(series)
        assert set(encoded) == {1}

    def test_mixed_non_quarter_strings_use_rank(self):
        """Non-quarter strings fall back to alphabetical rank encoding."""
        series = pd.Series(["period_3", "period_1", "period_2"])
        encoded, mapping = _encode_periods(series)
        # Alphabetical: period_1=1, period_2=2, period_3=3
        assert mapping["period_1"] == 1
        assert mapping["period_2"] == 2
        assert mapping["period_3"] == 3


# ---------------------------------------------------------------------------
# _encode_change_period
# ---------------------------------------------------------------------------


class TestEncodeChangePeriod:
    """Unit tests for change_period mapping."""

    def test_direct_lookup(self):
        """When change_period is already in the mapping, return it directly."""
        mapping = {"2023Q1": 1, "2023Q2": 2, "2023Q3": 3}
        assert _encode_change_period("2023Q2", mapping) == 2

    def test_quarter_string_cross_lookup(self):
        """Hyphenated variant '2023-Q2' should map to same value as '2023Q2'."""
        # The mapping uses "2023Q2" as key, but user may pass "2023-Q2"
        mapping = {"2023Q1": 1, "2023Q2": 2, "2023Q3": 3}
        # "2023-Q2" is not a key but parses to the same integer as "2023Q2"
        result = _encode_change_period("2023-Q2", mapping)
        assert result == 2

    def test_integer_direct_lookup(self):
        """Integer keys work directly."""
        mapping = {3: 1, 5: 2, 7: 3}
        assert _encode_change_period(5, mapping) == 2

    def test_integer_fallback_when_not_in_mapping(self):
        """If change_period is not in mapping but is a valid int, return int(change_period)."""
        mapping = {"2023Q1": 1}
        # change_period=7 not in mapping, falls through to int() cast
        result = _encode_change_period(7, mapping)
        assert result == 7

    def test_unrecognised_string_raises(self):
        """A string that is neither a key, a quarter, nor castable to int raises ValueError."""
        mapping = {"2023Q1": 1, "2023Q2": 2}
        with pytest.raises(ValueError, match="change_period"):
            _encode_change_period("not_a_valid_period", mapping)

    def test_error_message_shows_available_periods(self):
        """ValueError message should mention available period values."""
        mapping = {"2023Q1": 1, "2023Q2": 2}
        with pytest.raises(ValueError, match="2023Q1"):
            _encode_change_period("2025Q1", mapping)


# ---------------------------------------------------------------------------
# Integration: RateChangeEvaluator with quarter string periods
# ---------------------------------------------------------------------------


class TestEvaluatorQuarterPeriods:
    """
    Integration tests confirming RateChangeEvaluator handles quarter-string
    period columns correctly end-to-end.

    This is the most common real-world input format: insurers export data
    with period labelled as "2022Q1", "2022Q2", etc.
    """

    def _make_quarterly_df(self, n_quarters=12, change_quarter="2023Q3", seed=0):
        """Build a minimal DiD-compatible DataFrame with quarter-string periods."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)

        quarters = []
        start_year, start_q = 2021, 1
        for i in range(n_quarters):
            year = start_year + (start_q + i - 1) // 4
            q = ((start_q + i - 1) % 4) + 1
            quarters.append(f"{year}Q{q}")

        # 10 segments: 5 treated, 5 control, each appears in all quarters
        rows = []
        for seg in range(10):
            treated = int(seg >= 5)
            for qt in quarters:
                outcome = 0.5 + 0.02 * rng.standard_normal()
                if treated and qt >= change_quarter:
                    outcome -= 0.03
                rows.append({
                    "segment_id": f"seg_{seg}",
                    "period": qt,
                    "treated": treated,
                    "loss_ratio": outcome,
                    "exposure": float(rng.uniform(50, 150)),
                })

        return pd.DataFrame(rows)

    def test_quarter_periods_did_completes(self):
        """RateChangeEvaluator accepts quarter string periods without error."""
        import warnings
        from insurance_causal.rate_change import RateChangeEvaluator

        df = self._make_quarterly_df()
        ev = RateChangeEvaluator(
            method="did",
            outcome_col="loss_ratio",
            period_col="period",
            treated_col="treated",
            change_period="2023Q3",
            exposure_col="exposure",
            unit_col="segment_id",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ev.fit(df).summary()

        assert result.method == "did"
        assert result.n_periods_pre > 0
        assert result.n_periods_post > 0
        assert result.n_periods_pre + result.n_periods_post == 12

    def test_hyphenated_change_period(self):
        """change_period='2023-Q3' should map to the same period as '2023Q3'."""
        import warnings
        from insurance_causal.rate_change import RateChangeEvaluator

        df = self._make_quarterly_df()

        ev_plain = RateChangeEvaluator(
            method="did",
            outcome_col="loss_ratio",
            period_col="period",
            treated_col="treated",
            change_period="2023Q3",
            exposure_col="exposure",
            unit_col="segment_id",
        )
        ev_hyphen = RateChangeEvaluator(
            method="did",
            outcome_col="loss_ratio",
            period_col="period",
            treated_col="treated",
            change_period="2023-Q3",
            exposure_col="exposure",
            unit_col="segment_id",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = ev_plain.fit(df).summary()
            r2 = ev_hyphen.fit(df).summary()

        # Both should identify the same pre/post split
        assert r1.n_periods_pre == r2.n_periods_pre
        assert r1.n_periods_post == r2.n_periods_post
        # And produce numerically identical estimates
        assert abs(r1.att - r2.att) < 1e-10

    def test_quarter_period_its_fallback(self):
        """ITS path also works with quarter string periods."""
        from insurance_causal.rate_change import RateChangeEvaluator

        df = self._make_quarterly_df(n_quarters=12, change_quarter="2023Q3")
        # Keep only treated rows -> no control group -> auto picks ITS
        df_treated = df[df["treated"] == 1].copy()

        ev = RateChangeEvaluator(
            method="its",
            outcome_col="loss_ratio",
            period_col="period",
            change_period="2023Q3",
            exposure_col="exposure",
            unit_col="segment_id",
        )
        result = ev.fit(df_treated).summary()
        assert result.method == "its"
        assert result.n_periods_pre > 0
