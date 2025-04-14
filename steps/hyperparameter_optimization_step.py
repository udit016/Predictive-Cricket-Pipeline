import logging
from typing import Tuple
import pandas as pd
from zenml import step
from src.hyperparameter_optimization import optimize_hyperparameters

@step(enable_cache=False)
def hyperparameter_optimization_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 3,
    n_trials: int = 100
) -> Tuple[object, dict]:
    """
    ZenML step to optimize hyperparameters for the BlendedModel using Optuna.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv_folds (int): Number of cross-validation folds.
        n_trials (int): Number of Optuna trials.
        
    Returns:
        Tuple: (best_model, best_params) where best_model is the trained BlendedModel 
               with the best hyperparameters, and best_params is a dictionary of those parameters.
    """
    logging.info("Starting hyperparameter optimization step.")
    best_model, best_params = optimize_hyperparameters(X_train, y_train, cv_folds, n_trials)
    logging.info("Hyperparameter optimization completed.")
    return best_model, best_params