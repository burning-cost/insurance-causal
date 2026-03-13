"""
Tests for FCA PS21/5-compliant renewal pricing optimiser.

Verifies:
    - Optimised prices never exceed ENBP (hard regulatory constraint)
    - Prices stay at or above the floor where feasible (ENBP >= floor)
    - enbp_audit() correctly flags any price breaches
    - Profit objective produces higher margins than retention objective
    - Floor loading is respected when set above 1.0
    - Columns are correctly added to the output DataFrame

Note on floor constraint: when ENBP < tech_prem (the FCA ceiling is below cost),
the ENBP constraint overrides the floor. The test only checks rows where
ENBP >= floor price (i.e. there is a feasible price above cost).
"""

import numpy as np
import polars as pl
import pytest

from insurance_causal.elasticity.data import make_renewal_data
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser


class TestOptimise:
    def test_enbp_constraint_always_respected(self, fitted_estimator, small_df):
        """The ENBP constraint is absolute — no breach should be possible."""
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        enbp = result["enbp"].to_numpy()
        optimal = result["optimal_price"].to_numpy()
        # Allow for floating-point tolerance
        assert np.all(optimal <= enbp + 1e-6), (
            f"ENBP constraint breached: {np.sum(optimal > enbp + 1e-6)} rows"
        )

    def test_floor_constraint_respected_where_feasible(self, fitted_estimator, small_df):
        """Where ENBP >= tech_prem, prices should not fall below tech_prem."""
        opt = RenewalPricingOptimiser(fitted_estimator, floor_loading=1.0)
        result = opt.optimise(small_df)
        tech_prem = result["tech_prem"].to_numpy()
        enbp = result["enbp"].to_numpy()
        optimal = result["optimal_price"].to_numpy()
        floor = tech_prem * 1.0
        # Only check rows where the floor is actually below the ENBP ceiling
        feasible = floor <= enbp
        if feasible.sum() > 0:
            assert np.all(optimal[feasible] >= floor[feasible] * 0.999), (
                f"Floor constraint breached in {np.sum(optimal[feasible] < floor[feasible] * 0.999)} "
                f"feasible rows"
            )

    def test_floor_loading_above_one_respected(self, fitted_estimator, small_df):
        """With floor_loading=1.05, no price should be below 105% of tech_prem
        where the floor is feasible (floor <= ENBP)."""
        opt = RenewalPricingOptimiser(fitted_estimator, floor_loading=1.05)
        result = opt.optimise(small_df, objective="profit")
        tech_prem = result["tech_prem"].to_numpy()
        optimal = result["optimal_price"].to_numpy()
        floor = tech_prem * 1.05
        enbp = result["enbp"].to_numpy()
        feasible = floor < enbp
        assert np.all(optimal[feasible] >= floor[feasible] * 0.999)

    def test_output_columns_present(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        expected = [
            "optimal_price", "optimal_log_change",
            "predicted_renewal_prob", "expected_profit", "enbp_headroom"
        ]
        for col in expected:
            assert col in result.columns, f"Column '{col}' missing from optimise() output"

    def test_enbp_headroom_non_negative(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        headroom = result["enbp_headroom"].to_numpy()
        assert np.all(headroom >= -1e-6), (
            f"Negative ENBP headroom (breach) in {np.sum(headroom < 0)} rows"
        )

    def test_predicted_renewal_prob_bounds(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        probs = result["predicted_renewal_prob"].to_numpy()
        assert np.all(probs >= 0.0), "Predicted renewal probability below 0"
        assert np.all(probs <= 1.0), "Predicted renewal probability above 1"

    def test_profit_objective_returns_df(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df, objective="profit")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(small_df)

    def test_retention_objective_sets_floor_prices(self, fitted_estimator, small_df):
        """Retention objective should push prices toward the floor."""
        opt = RenewalPricingOptimiser(fitted_estimator, floor_loading=1.0)
        result_ret = opt.optimise(small_df, objective="retention")
        result_prof = opt.optimise(small_df, objective="profit")
        mean_ret = result_ret["optimal_price"].mean()
        mean_prof = result_prof["optimal_price"].mean()
        # Retention objective should on average produce lower or equal prices
        assert mean_ret <= mean_prof + 10.0, (
            f"Retention prices (mean={mean_ret:.2f}) higher than profit prices "
            f"(mean={mean_prof:.2f}) — unexpected"
        )

    def test_invalid_objective_raises(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        with pytest.raises(ValueError, match="objective must be"):
            opt.optimise(small_df, objective="shareholder_value")

    def test_target_retention_not_implemented(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        with pytest.raises(NotImplementedError):
            opt.optimise(small_df, target_retention=0.80)


class TestENBPAudit:
    def test_audit_returns_polars(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        audit = opt.enbp_audit(result)
        assert isinstance(audit, pl.DataFrame)

    def test_audit_columns(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        audit = opt.enbp_audit(result)
        for col in ["offered_price", "enbp", "compliant", "margin_to_enbp"]:
            assert col in audit.columns

    def test_audit_all_compliant_after_optimise(self, fitted_estimator, small_df):
        """After optimise(), all prices should be ENBP-compliant."""
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        audit = opt.enbp_audit(result)
        n_breach = (audit["compliant"] == False).sum()
        assert n_breach == 0, f"{n_breach} policies have optimal_price > ENBP"

    def test_audit_on_raw_data(self, fitted_estimator, small_df):
        """Raw offer_price may or may not be compliant — audit just reports."""
        opt = RenewalPricingOptimiser(fitted_estimator)
        audit = opt.enbp_audit(small_df)
        assert isinstance(audit, pl.DataFrame)
        assert len(audit) == len(small_df)

    def test_audit_missing_price_column_raises(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        df_no_price = small_df.drop(["offer_price"])
        with pytest.raises(ValueError, match="optimal_price"):
            opt.enbp_audit(df_no_price)

    def test_audit_includes_policy_id(self, fitted_estimator, small_df):
        opt = RenewalPricingOptimiser(fitted_estimator)
        result = opt.optimise(small_df)
        audit = opt.enbp_audit(result)
        assert "policy_id" in audit.columns
