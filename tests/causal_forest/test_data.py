"""
Tests for make_hte_renewal_data() DGP.
"""

import numpy as np
import polars as pl
import pytest

from insurance_causal.causal_forest.data import make_hte_renewal_data, true_cate_by_ncd


class TestMakeHTERenewalData:
    def test_returns_polars(self):
        df = make_hte_renewal_data(n=100, seed=0)
        assert isinstance(df, pl.DataFrame)

    def test_shape(self):
        df = make_hte_renewal_data(n=500, seed=0)
        assert len(df) == 500

    def test_columns(self):
        df = make_hte_renewal_data(n=100, seed=0)
        expected = [
            "policy_id", "age", "ncd_years", "region", "vehicle_group",
            "channel", "log_price_change", "renewed", "true_cate", "exposure",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_renewed_binary(self):
        df = make_hte_renewal_data(n=500, seed=0)
        unique_vals = set(df["renewed"].unique().to_list())
        assert unique_vals.issubset({0.0, 1.0})

    def test_ncd_range(self):
        df = make_hte_renewal_data(n=500, seed=0)
        assert df["ncd_years"].min() >= 0
        assert df["ncd_years"].max() <= 5

    def test_true_cate_ncd0_lt_ncd5(self):
        """NCD=0 should have more negative CATE than NCD=5."""
        df = make_hte_renewal_data(n=2000, seed=0)
        cate_ncd0 = df.filter(pl.col("ncd_years") == 0)["true_cate"].mean()
        cate_ncd5 = df.filter(pl.col("ncd_years") == 5)["true_cate"].mean()
        assert cate_ncd0 < cate_ncd5, (
            f"Expected true_cate[NCD=0]={cate_ncd0:.3f} < true_cate[NCD=5]={cate_ncd5:.3f}"
        )

    def test_exposure_is_one(self):
        df = make_hte_renewal_data(n=100, seed=0)
        assert (df["exposure"] == 1.0).all()

    def test_reproducible(self):
        df1 = make_hte_renewal_data(n=100, seed=42)
        df2 = make_hte_renewal_data(n=100, seed=42)
        # Use numpy comparison — avoids polars API version differences
        for col in df1.columns:
            np.testing.assert_array_equal(
                df1[col].to_numpy(), df2[col].to_numpy(),
                err_msg=f"Column {col!r} differs between runs with same seed",
            )

    def test_different_seeds(self):
        df1 = make_hte_renewal_data(n=100, seed=1)
        df2 = make_hte_renewal_data(n=100, seed=2)
        assert not np.array_equal(
            df1["log_price_change"].to_numpy(),
            df2["log_price_change"].to_numpy(),
        )


class TestTrueCateByNCD:
    def test_returns_polars(self):
        df = make_hte_renewal_data(n=500, seed=0)
        result = true_cate_by_ncd(df)
        assert isinstance(result, pl.DataFrame)

    def test_columns(self):
        df = make_hte_renewal_data(n=500, seed=0)
        result = true_cate_by_ncd(df)
        assert "ncd_years" in result.columns
        assert "true_cate_mean" in result.columns
        assert "n" in result.columns

    def test_sorted_by_ncd(self):
        df = make_hte_renewal_data(n=1000, seed=0)
        result = true_cate_by_ncd(df)
        ncd_vals = result["ncd_years"].to_list()
        assert ncd_vals == sorted(ncd_vals)

    def test_monotone_true_cate(self):
        """true_cate_mean should increase (become less negative) with NCD."""
        df = make_hte_renewal_data(n=2000, seed=0)
        result = true_cate_by_ncd(df)
        cate_means = result["true_cate_mean"].to_list()
        # Should be monotone increasing
        assert all(cate_means[i] <= cate_means[i + 1] for i in range(len(cate_means) - 1))
