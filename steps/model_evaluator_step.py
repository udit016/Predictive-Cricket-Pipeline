import logging
from typing import Tuple, Dict, Any

import pandas as pd
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_set_name: str = "test",
) -> Tuple[Dict[str, Any], float]:
    # Input type validation
    if not isinstance(X_eval, pd.DataFrame):
        raise TypeError("X_eval must be a pandas DataFrame.")
    if not isinstance(y_eval, pd.Series):
        raise TypeError("y_eval must be a pandas Series.")

    logging.info(f"Starting evaluation on the '{eval_set_name}' set...")

    # If the trained_model is a dict, extract the actual model.
    if isinstance(trained_model, dict):
        if "blended" in trained_model:
            trained_model = trained_model["blended"]
        else:
            raise KeyError("The trained_model dict does not contain the key 'blended'.")

    # Instantiate evaluator and compute metrics
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())
    evaluation_metrics = evaluator.evaluate(trained_model, X_eval, y_eval)

    # Basic validation of the returned metrics
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    f1 = evaluation_metrics.get("F1 Score")
    if f1 is None:
        raise KeyError("F1 Score not found in evaluation metrics.")

    logging.info(f"{eval_set_name.capitalize()} evaluation completed. F1 Score: {f1:.4f}")
    return evaluation_metrics, f1