"""
Input validation helpers for insurance-causal.

All public-facing validation lives here. Each function raises a TypeError or
ValueError with a message that names the library, the parameter, and what was
received alongside what was expected. The style is:

    insurance-causal: `treatment` 'foo' not found in DataFrame.
    Available columns: ['age', 'region', 'ncd_years']

These helpers are called at the top of public .fit() / __init__() methods
before any expensive computation begins, so users get an immediate, actionable
error rather than a cryptic traceback from inside numpy or sklearn.
"""

from __future__ import annotations

from typing import Any, Sequence


def check_dataframe(obj: Any, param: str) -> None:
    """Raise TypeError if *obj* is not a pandas or polars DataFrame."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return
    except ImportError:
        pass

    try:
        import polars as pl
        if isinstance(obj, pl.DataFrame):
            return
    except ImportError:
        pass

    raise TypeError(
        f"insurance-causal: `{param}` must be a pandas or polars DataFrame, "
        f"got {type(obj).__name__!r}. "
        "Pass a DataFrame, not None or a numpy array."
    )


def check_not_empty(obj: Any, param: str) -> None:
    """Raise ValueError if a DataFrame or array has zero rows."""
    length = None
    try:
        length = len(obj)
    except TypeError:
        pass
    if length == 0:
        raise ValueError(
            f"insurance-causal: `{param}` is empty (0 rows). "
            "Pass a non-empty DataFrame."
        )


def check_columns_exist(df: Any, columns: Sequence[str], df_param: str = "df") -> None:
    """Raise ValueError if any of *columns* are absent from *df*.

    Works with both pandas and polars DataFrames.
    """
    available = list(df.columns)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"insurance-causal: column(s) {missing} not found in `{df_param}`. "
            f"Available columns: {available}"
        )


def check_column_exists(df: Any, col: str, param: str, df_param: str = "df") -> None:
    """Raise ValueError if a single column is absent from *df*."""
    available = list(df.columns)
    if col not in available:
        raise ValueError(
            f"insurance-causal: `{param}` {col!r} not found in `{df_param}`. "
            f"Available columns: {available}"
        )


def check_column_numeric(df: Any, col: str, param: str) -> None:
    """Raise TypeError if *col* is not a numeric dtype in *df*.

    Supports both pandas and polars.

    Pandas 3.x changed how non-numeric dtypes behave with np.issubdtype: calling
    np.issubdtype(StringDtype(...), np.number) now raises TypeError instead of
    returning False. We catch that and treat it as "not numeric".
    """
    import numpy as np

    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            dtype = df[col].dtype
            try:
                is_numeric = np.issubdtype(dtype, np.number)
            except TypeError:
                # pandas 3.x extension dtypes (StringDtype, CategoricalDtype,
                # ArrowDtype, etc.) raise TypeError from np.issubdtype — treat
                # them as non-numeric.
                is_numeric = False
            if not is_numeric:
                raise TypeError(
                    f"insurance-causal: `{param}` column {col!r} must be numeric, "
                    f"got dtype {dtype!r}. "
                    "Convert to a numeric type (int or float) before fitting."
                )
            return
    except ImportError:
        pass

    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            dtype = df[col].dtype
            numeric_dtypes = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }
            if dtype not in numeric_dtypes:
                raise TypeError(
                    f"insurance-causal: `{param}` column {col!r} must be numeric, "
                    f"got dtype {dtype!r}. "
                    "Convert to a numeric type (int or float) before fitting."
                )
    except ImportError:
        pass


def check_n_splits(value: Any, param: str = "n_folds", minimum: int = 2) -> None:
    """Raise ValueError if *value* is not a valid integer >= *minimum*."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"insurance-causal: `{param}` must be an integer, "
            f"got {type(value).__name__!r} with value {value!r}."
        )
    if value < minimum:
        raise ValueError(
            f"insurance-causal: `{param}` must be >= {minimum}, got {value!r}. "
            f"Standard choice is 5 for most insurance datasets."
        )


def check_n_estimators(value: Any, param: str = "n_estimators") -> None:
    """Raise ValueError if *value* is not a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"insurance-causal: `{param}` must be an integer, "
            f"got {type(value).__name__!r} with value {value!r}."
        )
    if value < 1:
        raise ValueError(
            f"insurance-causal: `{param}` must be >= 1, got {value!r}."
        )


def check_confounders(confounders: Any, param: str = "confounders") -> None:
    """Raise TypeError/ValueError for invalid confounders argument."""
    if confounders is None:
        raise ValueError(
            f"insurance-causal: `{param}` must be provided. "
            "Pass a list of column names for the risk factors that jointly "
            "determine both the treatment and the outcome — e.g. "
            "['age', 'ncd_years', 'vehicle_group', 'region', 'channel']."
        )
    if not isinstance(confounders, (list, tuple)):
        raise TypeError(
            f"insurance-causal: `{param}` must be a list of column names, "
            f"got {type(confounders).__name__!r}."
        )
    if len(confounders) == 0:
        raise ValueError(
            f"insurance-causal: `{param}` is empty. "
            "Provide at least one confounder column."
        )


def check_alpha(value: Any, param: str = "alpha") -> None:
    """Raise ValueError if *value* is not in (0, 1)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"insurance-causal: `{param}` must be a float in (0, 1), "
            f"got {value!r}."
        )
    if not (0.0 < v < 1.0):
        raise ValueError(
            f"insurance-causal: `{param}` must be strictly between 0 and 1, "
            f"got {v!r}."
        )


def check_outcome_type(value: str, valid: tuple, param: str = "outcome_type") -> None:
    """Raise ValueError if *value* is not in *valid*."""
    if value not in valid:
        raise ValueError(
            f"insurance-causal: `{param}` must be one of {valid}, "
            f"got {value!r}."
        )
