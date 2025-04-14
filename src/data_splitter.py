import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Data Splitting Strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(
        self,
        data_df: pd.DataFrame,
        target_column: str,
        columns_to_keep: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data into training and test sets.
        
        Parameters:
            data_df (pd.DataFrame): The transformed DataFrame after feature engineering, which includes the target column.
            target_column (str): The name of the target column.
            columns_to_keep (list, optional): List of columns to retain (if provided, the target column will be added if missing).
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        pass


# Concrete Strategy for Simple Train-Test Split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the split strategy.
        
        Parameters:
            test_size (float): Fraction of data to use as test set.
            random_state (int): Seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(
        self,
        data_df: pd.DataFrame,
        target_column: str,
        columns_to_keep: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logging.info("Starting simple train-test split using SimpleTrainTestSplitStrategy.")
        # Save to CSV
        data_df.to_csv('output.csv', index=False)
        # Filter columns if specified.
        if columns_to_keep is not None:
            # Ensure target column is included.
            if target_column not in columns_to_keep:
                columns_to_keep.append(target_column)
            missing_cols = [col for col in columns_to_keep if col not in data_df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            data_df = data_df[columns_to_keep]

        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
        # Optionally drop irrelevant columns if they exist.
        drop_cols = ["match_dt", "team1_roster_ids", "team2_roster_ids", "season","ground_id"]
        data_df = data_df.drop(columns=[col for col in drop_cols if col in data_df.columns])
        
        # Split into features and target.
        X = data_df.drop(columns=[target_column])
        y = data_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_test.to_csv('test_data.csv', index=False)
        y_test.to_csv('test_labels.csv', index=False)
        
        logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test


# Context Class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a data splitting strategy.
        
        Parameters:
            strategy (DataSplittingStrategy): The strategy to use for splitting data.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new data splitting strategy.
        
        Parameters:
            strategy (DataSplittingStrategy): The new strategy to be used.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(
        self,
        data_df: pd.DataFrame,
        target_column: str,
        columns_to_keep: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data using the selected strategy.
        
        Parameters:
            data_df (pd.DataFrame): The transformed DataFrame including the target.
            target_column (str): The name of the target column.
            columns_to_keep (list, optional): Columns to retain.
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(data_df, target_column, columns_to_keep)
