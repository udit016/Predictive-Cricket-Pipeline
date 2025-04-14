import logging
from typing import Dict, Any

import pandas as pd
from src.model_building import ModelBuilder, BlendedModelStrategy
from zenml import step

@step(enable_cache=False)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgb_params: dict,
    lgbm_params: dict,
    cat_params: dict
) -> Dict[str, Any]:
    """
    ZenML step to build and train the blended model using only the training data.
    
    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target.
        xgb_params (dict): Parameters for XGBClassifier.
        lgbm_params (dict): Parameters for LGBMClassifier.
        cat_params (dict): Parameters for CatBoostClassifier.
        
    Returns:
        dict: Dictionary containing the trained model under key "blended".
    """
    logging.info("Starting model building step using training data only.")
    
    model_builder = ModelBuilder(
        strategy=BlendedModelStrategy(xgb_params, lgbm_params, cat_params)
    )
    
    # Only pass X_train and y_train since no validation set is used.
    trained_models = model_builder.build_model(X_train, y_train)
    return trained_models
