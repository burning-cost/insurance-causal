"""
Elasticity surface visualisation and segment summary.

The elasticity surface is the primary deliverable for UK pricing teams: it shows
how price sensitivity varies across the key risk dimensions (NCD, age, channel,
region). This module wraps the fitted estimator to produce publication-quality
plots and summary tables that actuaries can take directly into rate review.

Design notes:
    - All plots return matplotlib Figure objects so callers control display/save.
    - Segment summaries return polars DataFrames: easy to export to Excel or
      feed into downstream rate-setting tools.
    - plot_surface uses a seaborn-style heatmap approach: rows = first dim,
      columns = second dim. Works best with categorical dimensions.
    - plot_gate uses horizontal bars with error bars for the 95% CI. Easier to
      read than vertical bars when segment labels are long.
"""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import polars as pl


class ElasticitySurface:
    """Visualisation and segment summary for a fitted elasticity estimator.

    Parameters
    ----------
    estimator:
        A fitted :class:`~insurance_causal.elasticity.fit.RenewalElasticityEstimator`.

    Examples
    --------
    >>> from insurance_causal.elasticity.data import make_renewal_data
    >>> from insurance_causal.elasticity.fit import RenewalElasticityEstimator
    >>> from insurance_causal.elasticity.surface import ElasticitySurface
    >>> df = make_renewal_data(n=2000, seed=42)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = RenewalElasticityEstimator(n_estimators=50, catboost_iterations=100)
    >>> est.fit(df, confounders=confounders)
    >>> surface = ElasticitySurface(est)
    >>> summary = surface.segment_summary(df, by="ncd_years")
    """

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def segment_summary(
        self,
        df: pl.DataFrame,
        by: Optional[Union[str, Sequence[str]]] = None,
    ) -> pl.DataFrame:
        """Return a summary table of elasticity by segment.

        Parameters
        ----------
        df:
            Dataset with confounder columns. Must be the same schema as the
            training data.
        by:
            Column name(s) to segment by. If None, returns a single-row
            portfolio summary. Can be a single column name or a list.

        Returns
        -------
        polars.DataFrame with columns: (segment cols), elasticity, ci_lower,
        ci_upper, n, elasticity_at_10pct
        """
        if not hasattr(self.estimator, '_is_fitted') or not self.estimator._is_fitted:
            raise RuntimeError("Estimator is not fitted. Call .fit() first.")

        cate_vals = self.estimator.cate(df)
        lb_vals, ub_vals = self.estimator.cate_interval(df)

        df_aug = df.with_columns([
            pl.Series("_cate", cate_vals),
            pl.Series("_ci_lower", lb_vals),
            pl.Series("_ci_upper", ub_vals),
        ])

        if by is None:
            # Portfolio summary
            return pl.DataFrame({
                "segment": ["portfolio"],
                "elasticity": [float(np.mean(cate_vals))],
                "ci_lower": [float(np.mean(lb_vals))],
                "ci_upper": [float(np.mean(ub_vals))],
                "n": [len(df)],
                "elasticity_at_10pct": [float(np.mean(cate_vals) * np.log(1.1))],
            })

        if isinstance(by, str):
            group_cols = [by]
        else:
            group_cols = list(by)

        result = (
            df_aug
            .group_by(group_cols)
            .agg([
                pl.col("_cate").mean().alias("elasticity"),
                pl.col("_ci_lower").mean().alias("ci_lower"),
                pl.col("_ci_upper").mean().alias("ci_upper"),
                pl.len().alias("n"),
            ])
            .with_columns(
                (pl.col("elasticity") * np.log(1.1)).alias("elasticity_at_10pct")
            )
            .sort(group_cols)
        )
        return result

    def plot_gate(
        self,
        df: pl.DataFrame,
        by: str = "ncd_years",
        ax: Optional[object] = None,
        title: Optional[str] = None,
        color: str = "#1f77b4",
    ) -> object:
        """Bar chart of group average treatment effects with 95% CIs.

        Parameters
        ----------
        df:
            Dataset with confounder columns.
        by:
            Column to group by. Best with low-cardinality categoricals or
            ordinal columns (NCD band, age band, payment method).
        ax:
            Existing matplotlib Axes. If None, creates a new figure.
        title:
            Plot title. Defaults to 'Elasticity by {by}'.
        color:
            Bar colour.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        summary = self.segment_summary(df, by=by)

        labels = summary[by].to_list()
        elas = summary["elasticity"].to_numpy()
        lower = summary["ci_lower"].to_numpy()
        upper = summary["ci_upper"].to_numpy()
        lower_err = elas - lower
        upper_err = upper - elas

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5 + 1)))
        else:
            fig = ax.get_figure()

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, elas, color=color, alpha=0.8, height=0.6)
        ax.errorbar(
            elas, y_pos,
            xerr=[lower_err, upper_err],
            fmt="none", color="black", linewidth=1.5, capsize=4,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(l) for l in labels])
        ax.set_xlabel("Semi-elasticity (change in renewal prob per unit log price change)")
        ax.set_title(title or f"Elasticity by {by}")
        ax.invert_yaxis()

        # Annotate bars with the elasticity value
        for i, (e, lb, ub) in enumerate(zip(elas, lower, upper)):
            ax.text(
                e, i,
                f"  {e:.3f} [{lb:.3f}, {ub:.3f}]",
                va="center", fontsize=8, color="black",
            )

        fig.tight_layout()
        return fig

    def plot_surface(
        self,
        df: pl.DataFrame,
        dims: Sequence[str] = ("ncd_years", "age_band"),
        ax: Optional[object] = None,
        title: Optional[str] = None,
        cmap: str = "RdYlGn",
    ) -> object:
        """Heatmap of elasticity across two dimensions.

        Rows = first dimension, columns = second dimension. Cell colours
        encode the mean CATE for that segment combination.

        Parameters
        ----------
        df:
            Dataset with confounder columns.
        dims:
            Exactly two column names to use as row/column dimensions.
        ax:
            Existing matplotlib Axes. If None, creates a new figure.
        title:
            Plot title.
        cmap:
            Matplotlib colormap. ``"RdYlGn"`` (red = elastic, green = inelastic)
            is the pricing-intuitive default.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if len(dims) != 2:
            raise ValueError(
                f"plot_surface requires exactly 2 dimensions, got {len(dims)}: {dims}"
            )

        row_col, col_col = dims[0], dims[1]
        summary = self.segment_summary(df, by=list(dims))

        row_vals = sorted(summary[row_col].unique().to_list())
        col_vals = sorted(summary[col_col].unique().to_list())

        # Build matrix
        mat = np.full((len(row_vals), len(col_vals)), np.nan)
        row_idx = {v: i for i, v in enumerate(row_vals)}
        col_idx = {v: i for i, v in enumerate(col_vals)}

        for row in summary.iter_rows(named=True):
            r = row_idx.get(row[row_col])
            c = col_idx.get(row[col_col])
            if r is not None and c is not None:
                mat[r, c] = row["elasticity"]

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(max(6, len(col_vals) * 1.2), max(4, len(row_vals) * 0.8))
            )
        else:
            fig = ax.get_figure()

        # Centre colourmap on zero
        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(col_vals)))
        ax.set_xticklabels([str(v) for v in col_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(row_vals)))
        ax.set_yticklabels([str(v) for v in row_vals])
        ax.set_xlabel(col_col)
        ax.set_ylabel(row_col)
        ax.set_title(title or f"Elasticity surface: {row_col} × {col_col}")

        # Annotate cells
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="black",
                    )

        plt.colorbar(im, ax=ax, label="Semi-elasticity")
        fig.tight_layout()
        return fig
