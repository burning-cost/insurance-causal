"""
Post-hoc rate change evaluation: did our Q3 motor increase work?
================================================================

This script demonstrates `RateChangeEvaluator` on a realistic UK motor
scenario:

    We implemented a 10% rate increase on motor comprehensive in Q3 2023
    for high-risk segments. We left other segments at flat rates as a
    natural control group. Six months later: did loss ratio improve,
    by how much, and is the effect statistically meaningful?

Methods:
  - DiD (Difference-in-Differences): uses the untreated segments as a
    control group to absorb time trends and macro shocks.
  - Parallel trends test: checks that treated and control segments were
    trending similarly before the rate change.

Run on Databricks (recommended):
    Import via Repos, attach to a cluster with insurance-causal installed.
    Expected runtime: under 1 minute.

Run locally:
    pip install "insurance-causal"
    python examples/rate_change_evaluation.py
"""

from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data


# ---------------------------------------------------------------------------
# 1. Synthetic panel dataset
# ---------------------------------------------------------------------------
# 40 segments (e.g. postcode bands), 12 quarterly periods.
# Rate change applied to 25 segments in period 7 (Q3 2023).
# True ATT (average treatment effect on the treated): -5% on loss ratio.

df = make_rate_change_data(
    n_segments=40,
    true_att=-0.05,     # 5pp reduction in loss ratio from the rate change
    random_state=42,
)

print("Panel structure:")
print(f"  Segments:   {df['segment_id'].nunique()}")
print(f"  Periods:    {df['period'].nunique()}")
print(f"  Treated:    {df[df['treated'] == 1]['segment_id'].nunique()} segments")
print(f"  Control:    {df[df['treated'] == 0]['segment_id'].nunique()} segments")
print(f"  Change period: 7  (Q3 2023 in this synthetic dataset)")
print()


# ---------------------------------------------------------------------------
# 2. Fit the evaluator
# ---------------------------------------------------------------------------

evaluator = RateChangeEvaluator(
    method="auto",           # DiD: control group is present
    outcome_col="outcome",
    period_col="period",
    treated_col="treated",
    change_period=7,
    exposure_col="exposure",
    unit_col="segment_id",
)

result = evaluator.fit(df)
print(result.summary())
print()


# ---------------------------------------------------------------------------
# 3. Parallel trends test
# ---------------------------------------------------------------------------
# A key assumption of DiD is that treated and control segments would have
# followed the same trend in the absence of treatment. This test checks
# whether pre-treatment period dummies are jointly zero — a necessary
# (though not sufficient) condition for parallel trends.

pt = evaluator.parallel_trends_test()
print("Parallel trends test:")
print(f"  Joint F-statistic: {pt.joint_pt_fstat:.3f}")
print(f"  p-value:           {pt.joint_pt_pvalue:.3f}")
if pt.joint_pt_pvalue > 0.10:
    print("  Result: pre-treatment trends are parallel (p > 0.10)")
    print("          The DiD identification assumption looks credible.")
else:
    print("  WARNING: parallel trends assumption may be violated.")
    print("          Interpret the ATT with caution.")
print()


# ---------------------------------------------------------------------------
# 4. Interpretation for a rate review
# ---------------------------------------------------------------------------

att = result.att
att_pct = result.att_pct   # % change vs pre-treatment mean (may be None if unknown)
ci_lower = result.ci_lower
ci_upper = result.ci_upper
p_value  = result.p_value
pre_mean = result.pre_mean_treated

print("--- Rate review summary ---")
print(f"Pre-treatment loss ratio (treated segments): {pre_mean:.3f}")
pct_str = f"  ({att_pct:+.1f}% of pre-treatment mean)" if att_pct is not None else ""
print(f"ATT estimate:  {att:+.4f}{pct_str}")
print(f"95% CI:        [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"p-value:       {p_value:.4f}")
print()

if p_value < 0.05:
    print(f"The rate change caused a statistically significant {att_pct:.1f}% reduction")
    print(f"in loss ratio. The 95% CI excludes zero.")
    print()
    print("Next steps:")
    print("  - Use this estimate to calibrate the elasticity assumption in the next rate review")
    print("  - Compare against the pricing team's ex-ante elasticity forecast")
    print("  - If the realised effect is smaller than forecast, revisit the DML elasticity estimate")
else:
    print("The estimate is not statistically significant at the 5% level.")
    print("Either the effect is small, the portfolio is too thin, or")
    print("the parallel trends assumption is violated. Investigate before")
    print("drawing commercial conclusions.")
