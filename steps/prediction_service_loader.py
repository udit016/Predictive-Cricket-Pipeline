import logging
from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def prediction_service_loader_step(
    pipeline_name: str, step_name: str
) -> MLFlowDeploymentService:
    """
    Loads the active MLflow prediction service for a classification model
    deployed by a specified pipeline and step.

    Args:
        pipeline_name (str): The name of the pipeline that deployed the model.
        step_name (str): The step in the pipeline that deployed the model.

    Returns:
        MLFlowDeploymentService: The running MLflow deployment service.

    Raises:
        RuntimeError: If no active deployment is found for the given pipeline and step.
    """
    logging.info(f"Attempting to load MLflow prediction service for pipeline: '{pipeline_name}', step: '{step_name}'")

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=True,  # Optional: ensures only currently running services are considered
    )

    if not existing_services:
        logging.error(f"No active MLflow service found for step '{step_name}' in pipeline '{pipeline_name}'")
        raise RuntimeError(
            f"❌ No running MLflow prediction service deployed by the step '{step_name}' "
            f"in the pipeline '{pipeline_name}'. Make sure the model was deployed and the service is running."
        )

    service = existing_services[0]
    logging.info(f"✅ Found active MLflow deployment service at: {service.prediction_url}")
    return service