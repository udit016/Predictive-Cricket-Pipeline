import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self,
        model: Any,  # Accept any model with a predict() method
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluates a classification model on a given dataset.

        Parameters:
            model (Any): The trained classifier (including blended models).
            X_test (pd.DataFrame): Feature set for evaluation. This can be the validation set used in training,
                                   or an external test/deployment set.
            y_test (pd.Series): True labels for the evaluation set.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Classification Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        logging.info("Running predictions using the provided classification model.")
        y_pred = model.predict(X_test)

        logging.info("Computing evaluation metrics for the classification model.")
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "Accuracy": acc,
            "F1 Score": f1,
            "Confusion Matrix": conf_matrix,
            "Classification Report": class_report,
        }

        logging.info(f"Evaluation completed: Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info("Switching to a new model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        logging.info("Starting model evaluation using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)