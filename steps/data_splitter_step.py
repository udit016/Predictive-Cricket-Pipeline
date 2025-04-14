import pandas as pd
from typing import Optional, List, Tuple
from zenml import step

from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy


@step(enable_cache=False)
def data_splitter_step(
    data: pd.DataFrame,
    target_column: str,
    columns_to_keep: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    ZenML step that splits the transformed dataset into training and test sets.
    
    Args:
        data (pd.DataFrame): The input DataFrame after feature engineering.
        target_column (str): Name of the target column to predict.
        columns_to_keep (Optional[List[str]]): Specific columns to retain (target will be auto-included).
        test_size (float): Proportion of data to use for the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Tuple containing:
            - X_train (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_train (pd.Series)
            - y_test (pd.Series)
    """
    strategy = SimpleTrainTestSplitStrategy(test_size=test_size, random_state=random_state)
    splitter = DataSplitter(strategy)
    
    X_train, X_test, y_train, y_test = splitter.split(
        data_df=data,
        target_column=target_column,
        columns_to_keep=columns_to_keep,
    )
    
    return X_train, X_test, y_train, y_test