import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step


@step
def handle_missing_values_step(
    df: pd.DataFrame,
    strategy: str = "mean",
    cat_method: str = "NA",
    cat_fill_value: str = None
) -> pd.DataFrame:
    """
    Handles missing values using the specified strategy.

    Args:
        df (pd.DataFrame): The input DataFrame with missing values.
        strategy (str): Strategy for numerical columns - 'drop', 'mean', 'median', 'mode', or '0'.
        cat_method (str): Strategy for categorical columns - 'NA' or 'constant'.
        cat_fill_value (str): Constant value to fill for categorical columns if `cat_method` is 'constant'.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """

    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "0"]:
        handler = MissingValueHandler(
            FillMissingValuesStrategy(num_method=strategy, cat_method=cat_method, cat_fill_value=cat_fill_value)
        )
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df