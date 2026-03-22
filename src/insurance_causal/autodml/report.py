"""
ElasticityReport: HTML and JSON audit report for FCA evidence.

UK insurers subject to FCA pricing review (PS21/5) need to document causal
evidence that pricing differentials reflect genuine risk differentiation, not
differential treatment of loyal customers.

This report class takes a fitted PremiumElasticity (or
SelectionCorrectedElasticity) and generates:
1. A formatted HTML report with the dose-response plot, segment table,
   sensitivity analysis, and methodology notes.
2. A JSON summary for programmatic consumption.

The FCA evidence section is written in plain English suitable for inclusion
in a regulatory filing.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, List, Union

import numpy as np

from insurance_causal.autodml._types import EstimationResult, SegmentResult


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Price Elasticity Analysis Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #003366; }}
  h2 {{ color: #005599; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: right; }}
  th {{ background: #f0f4f8; text-align: center; }}
  td:first-child {{ text-align: left; }}
  .highlight {{ background: #fff8e1; font-weight: bold; }}
  .note {{ background: #f9f9f9; padding: 12px; border-left: 4px solid #005599;
            margin: 12px 0; font-size: 0.95em; }}
  .warn {{ background: #fff3e0; padding: 12px; border-left: 4px solid #e65100;
           margin: 12px 0; font-size: 0.95em; }}
  .footer {{ margin-top: 40px; font-size: 0.8em; color: #666; }}
</style>
</head>
<body>
<h1>Price Elasticity Analysis Report</h1>
<p><strong>Generated:</strong> {timestamp}</p>
<p><strong>Method:</strong> Automatic Debiased ML (Riesz Representer)</p>
<p><strong>Library:</strong> insurance-autodml v{version}</p>

<h2>1. Overall Average Marginal Effect (AME)</h2>
<div class="highlight">
  <p>AME = {ame:.4f} &nbsp; (SE = {se:.4f})</p>
  <p>{ci_level}% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]</p>
  <p>P-value: {pvalue:.4f} {stars}</p>
  <p>Observations: {n_obs:,} &nbsp; Folds: {n_folds}</p>
</div>
<div class="note">
<strong>Interpretation:</strong> The average marginal effect of a £1 increase
in premium on the outcome is {ame:.4f} (95% CI: {ci_low:.4f} to {ci_high:.4f}).
This estimate is doubly robust: consistent if either the outcome regression or the
Riesz representer is consistently estimated.
</div>

<h2>2. Segment-Level Effects</h2>
{segment_table}

<h2>3. Sensitivity Analysis</h2>
{sensitivity_section}

<h2>4. Methodology Notes</h2>
<div class="note">
<p><strong>Estimator:</strong> The AME is estimated using the Neyman-orthogonal
double-ML score with a cross-fitted Riesz representer.  Unlike standard double-ML
with continuous treatments, this approach does not require estimation of the
generalised propensity score (GPS), which is ill-posed in renewal portfolios where
selection creates multimodality in the treatment distribution.</p>

<p><strong>Identification:</strong> The estimand is identified under the assumption
that all variables jointly affecting the premium and the outcome are included in X.
Common violations include: omitted risk factors correlated with pricing bands;
competitor price effects; and policyholder switching behaviour.</p>

<p><strong>Cross-fitting:</strong> {n_folds}-fold cross-fitting is used to avoid
Donsker conditions on the nuisance models.  Each fold's nuisance models are fitted
on out-of-fold data only.</p>

<p><strong>Inference:</strong> Standard errors are derived from the empirical
variance of the efficient influence function scores, which achieves the
semiparametric efficiency bound.</p>
</div>

<h2>5. FCA Evidence Summary</h2>
<div class="warn">
<p><strong>Important:</strong> This report provides one quantitative input to a
fair value assessment.  It is not a determination of FCA compliance.  The debiased
ML estimate ({ame:+.4f} per unit premium) quantifies the causal relationship between
premium changes and policyholder outcomes, controlling for observed risk factors (X).
</p>

<p>The identification assumption — that all material confounders are captured in X —
must be assessed by a qualified actuary.  Common violations include omitted risk
factors, competitor price effects, and cohort effects not present in the model.
A statistically significant causal estimate does not by itself demonstrate that
pricing differentials reflect risk differentiation: that judgement requires broader
evidence including market analysis, actuarial sign-off, and qualitative assessment.</p>

<p>Firms preparing evidence for FCA Consumer Duty obligations should treat this
analysis as a supporting tool, not a standalone compliance demonstration.  The
methodology notes in Section 4 should accompany any regulatory submission.</p>
</div>

<div class="footer">
<p>Generated by insurance-autodml &mdash; Burning Cost &mdash; {timestamp}</p>
</div>
</body>
</html>
"""


class ElasticityReport:
    """
    HTML and JSON audit report for price elasticity analysis.

    Designed to produce FCA-ready documentation of causal price elasticity
    estimates from the debiased ML framework.

    Parameters
    ----------
    estimator : PremiumElasticity or SelectionCorrectedElasticity
        A fitted estimator with a valid result_ attribute (after calling
        estimate()).
    segment_results : list of SegmentResult, optional
        Segment-level AME results from effect_by_segment().
    sensitivity_bounds : dict, optional
        Output from SelectionCorrectedElasticity.sensitivity_bounds().
    title : str
        Report title.
    analyst : str
        Name or team responsible for the analysis.
    """

    def __init__(
        self,
        estimator,
        segment_results: Optional[List[SegmentResult]] = None,
        sensitivity_bounds: Optional[dict] = None,
        title: str = "Price Elasticity Analysis",
        analyst: str = "Burning Cost Pricing Team",
    ) -> None:
        self.estimator = estimator
        self.segment_results = segment_results or []
        self.sensitivity_bounds = sensitivity_bounds or {}
        self.title = title
        self.analyst = analyst

        if estimator.result_ is None:
            raise ValueError(
                "The estimator has no result. Call estimate() on the estimator "
                "before passing it to ElasticityReport."
            )
        self._result: EstimationResult = estimator.result_

    def _build_segment_table(self) -> str:
        if not self.segment_results:
            return "<p>No segment results provided.</p>"

        rows = []
        for sr in self.segment_results:
            r = sr.result
            rows.append(
                f"<tr>"
                f"<td>{sr.segment_name}</td>"
                f"<td>{r.estimate:+.4f}</td>"
                f"<td>{r.se:.4f}</td>"
                f"<td>[{r.ci_low:+.4f}, {r.ci_high:+.4f}]</td>"
                f"<td>{r.pvalue:.4f}</td>"
                f"<td>{sr.n_obs:,}</td>"
                f"</tr>"
            )

        return (
            "<table>"
            "<tr><th>Segment</th><th>AME</th><th>SE</th>"
            "<th>95% CI</th><th>P-value</th><th>N</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    def _build_sensitivity_section(self) -> str:
        if not self.sensitivity_bounds:
            return "<p>No sensitivity analysis provided.</p>"

        rows = []
        for gamma, bounds in self.sensitivity_bounds.items():
            rows.append(
                f"<tr>"
                f"<td>{gamma:.1f}</td>"
                f"<td>{bounds['lower']:+.4f}</td>"
                f"<td>{bounds['upper']:+.4f}</td>"
                f"</tr>"
            )

        return (
            "<p>Sensitivity to unobserved selection confounders "
            "(Gamma = odds ratio bound):</p>"
            "<table>"
            "<tr><th>Gamma</th><th>AME lower bound</th><th>AME upper bound</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    def to_html(self, path: Optional[str] = None) -> str:
        """
        Render the report as HTML.

        Parameters
        ----------
        path : str, optional
            If provided, write the HTML to this file path.

        Returns
        -------
        html : str
            Rendered HTML string.
        """
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("insurance-causal")

        r = self._result
        p = r.pvalue
        stars = ""
        if not np.isnan(p):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"

        html = _HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            version=__version__,
            ame=r.estimate,
            se=r.se,
            ci_level=int(r.ci_level * 100),
            ci_low=r.ci_low,
            ci_high=r.ci_high,
            pvalue=p if not np.isnan(p) else float("nan"),
            stars=stars,
            n_obs=r.n_obs,
            n_folds=r.n_folds,
            segment_table=self._build_segment_table(),
            sensitivity_section=self._build_sensitivity_section(),
        )

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def to_json(self, path: Optional[str] = None) -> dict:
        """
        Export the report as a JSON-serialisable dictionary.

        Parameters
        ----------
        path : str, optional
            If provided, write JSON to this file path.

        Returns
        -------
        data : dict
            Report data suitable for downstream consumption.
        """
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("insurance-causal")

        r = self._result
        data = {
            "generated": datetime.now().isoformat(),
            "version": __version__,
            "title": self.title,
            "analyst": self.analyst,
            "overall_ame": {
                "estimate": r.estimate,
                "se": r.se,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "ci_level": r.ci_level,
                "pvalue": r.pvalue if not np.isnan(r.pvalue) else None,
                "n_obs": r.n_obs,
                "n_folds": r.n_folds,
            },
            "segments": [
                {
                    "segment": sr.segment_name,
                    "estimate": sr.result.estimate,
                    "se": sr.result.se,
                    "ci_low": sr.result.ci_low,
                    "ci_high": sr.result.ci_high,
                    "n_obs": sr.n_obs,
                }
                for sr in self.segment_results
            ],
            "sensitivity_bounds": self.sensitivity_bounds,
        }

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        return data
