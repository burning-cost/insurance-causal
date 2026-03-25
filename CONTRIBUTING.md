# Contributing to insurance-causal

This library is built for UK pricing teams that need causal inference answers — not correlation-based feature importances — from observational insurance data. Contributions that sharpen that use case are welcome.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_causal; print(insurance_causal.__version__)`)
- A minimal reproducible example — the synthetic data generators in the library are the easiest starting point
- What you expected to happen and what actually happened

For inference bugs specifically: include the treatment type (binary/continuous), the confounder structure, and whether the issue is in the first-stage nuisance models or the final causal estimate.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Describe the pricing decision you are trying to make. The DML framework is general but the interesting engineering work is in making it work well on insurance-specific data structures — exposure offsets, zero-inflated claim counts, temporal structure in renewal data. Concrete use cases drive better design decisions than abstract feature requests.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-causal.git
cd insurance-causal
uv sync --dev
uv run pytest
```

The library uses `uv` for dependency management. Python 3.10+ is required. The heavier tests (cross-fitting with real ML models) are marked slow and excluded from the default run:

```bash
uv run pytest --run-slow
```

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation (e.g., "modelling" not "modeling", "programme" not "program" for a statistical programme)
- Docstrings follow NumPy format and cite the relevant paper where a method is non-obvious
- Tests use synthetic data generators — no dependency on external datasets
- The `DoubleMLEstimator` interface is the primary public API; keep it stable

If you are adding a new estimator or treatment effect variant, include a reference to the methodological source and a note on when a pricing actuary would choose this variant over the default.

---

For questions or to discuss ideas before opening an issue, start a [Discussion](https://github.com/burning-cost/insurance-causal/discussions).
