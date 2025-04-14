import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Execute the training pipeline
    run = ml_pipeline()

    # Optionally, you could retrieve the trained model from the pipeline run.
    # For example, if your model building step is named "model_building_step":
    # trained_model = run["model_building_step"]
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "to inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

if __name__ == "__main__":
    main()