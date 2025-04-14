import logging
from typing import Any
from zenml import Model, step


@step
def model_loader_step(model_name: str) -> Any:
    """
    Loads the current production classification model from the ZenML Model Registry.

    Args:
        model_name (str): The name of the registered model.

    Returns:
        Any: The loaded model object (e.g., a blended model or pipeline).

    Raises:
        ValueError: If the model cannot be properly loaded.
    """
    logging.info(f"Attempting to load production model: '{model_name}'")

    try:
        model = Model(name=model_name, version="production")
        loaded_model = model.load_artifact()

        if not hasattr(loaded_model, "predict"):
            raise ValueError("The loaded object does not have a predict method. Ensure it's a valid model.")

        logging.info(f"Model '{model_name}' loaded successfully.")
        return loaded_model

    except Exception as e:
        logging.error(f"Failed to load model '{model_name}': {e}")
        raise ValueError(f"Could not load model '{model_name}' from the registry.") from e