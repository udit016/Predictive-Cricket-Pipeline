import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values for both numerical and categorical columns
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, num_method="mean", cat_method="NA", cat_fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with specific methods for numerical
        and categorical columns.

        Parameters:
        num_method (str): Method to fill missing numerical values ('mean', 'median', 'mode', or '0').
        cat_method (str): Method to fill missing categorical values ('NA' or 'constant').
        cat_fill_value (any): The constant value to fill missing categorical values when cat_method is 'constant'.
                              If None and cat_method is 'constant', defaults to "NA".
        """
        self.num_method = num_method
        self.cat_method = cat_method
        self.cat_fill_value = cat_fill_value if cat_fill_value is not None else "NA"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified methods separately for numerical and categorical columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values. Numerical method: {self.num_method}, Categorical method: {self.cat_method}")
        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include="number").columns
        categorical_columns = df_cleaned.select_dtypes(include=["object", "category"]).columns

        # Fill numerical columns
        if self.num_method == "mean":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.num_method == "median":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        elif self.num_method == "mode":
            for col in numeric_columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df_cleaned[col].fillna(mode_val.iloc[0], inplace=True)
        elif self.num_method == "0":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(0)
        else:
            logging.warning(f"Unknown numerical fill method '{self.num_method}'. No numerical missing values handled.")

        # Fill categorical columns
        if self.cat_method == "NA":
            df_cleaned[categorical_columns] = df_cleaned[categorical_columns].fillna("NA")
        elif self.cat_method == "constant":
            df_cleaned[categorical_columns] = df_cleaned[categorical_columns].fillna(self.cat_fill_value)
        else:
            logging.warning(f"Unknown categorical fill method '{self.cat_method}'. No categorical missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # Create an example dataframe with missing values for both numerical and categorical columns
    df = pd.DataFrame({
        "age": [25, None, 30],
        "income": [50000, 60000, None],
        "gender": ["male", None, "female"],
        "occupation": [None, "engineer", "doctor"]
    })
    
    logging.info("Original DataFrame:")
    logging.info(df)

    # Example 1: Fill numerical columns using mean and categorical columns with "NA"
    #missing_value_handler = MissingValueHandler(FillMissingValuesStrategy(num_method="mean", cat_method="NA"))
    #df_filled = missing_value_handler.handle_missing_values(df)
    
    #logging.info("DataFrame after filling missing values (numerical: mean, categorical: NA):")
    #logging.info(df_filled)

    # Example 2: Fill numerical columns with 0 and categorical columns with a constant value (e.g., "Missing")
    #missing_value_handler.set_strategy(FillMissingValuesStrategy(num_method="0", cat_method="constant", cat_fill_value="Missing"))
    #df_filled2 = missing_value_handler.handle_missing_values(df)
    
    #logging.info("DataFrame after filling missing values (numerical: 0, categorical: constant 'Missing'):")
    #logging.info(df_filled2)