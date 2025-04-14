import click
from pipelines.deployment_pipeline import deployment_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the win_predictor deployment pipeline."""
    model_name = "win_predictor"

    if stop_service:
        # Stop an existing model prediction service if running.
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        existing_services = model_deployer.find_model_server(
            pipeline_name="deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
            print("[green]Stopped the running prediction service.[/green]")
        else:
            print("[yellow]No running prediction service found.[/yellow]")
        return

    # Set parameters for the deployment pipeline.
    scorecard_folder_path = r"D:\Project\New folder\additional_data"
    train_test_folder_path = r"D:\Project\New folder\Data"
    target_column = "winner_id"

    # Define columns to keep – these must match the schema of your feature-engineered test set.
    cols_to_keep = [
        # Base columns
        "match id",
        "team1_id",
        "team1_roster_ids",
        "team2_id",
        "team2_roster_ids",
        "winner_id",
        "match_dt",
        "season",
        "ground_id",
        "team_count_50runs_last15",
        "team_winp_last5",
        "team1only_avg_runs_last15",
        "team1_winp_team2_last15",
        "ground_avg_runs_last15",
        # New features from toss and lighting effectiveness
        "toss_winner_01",
        "toss_decision_01",
        "team_1_lighting_effectiveness",
        "team_2_lighting_effectiveness",
        # Bowling economy features
        "avgeco_1",
        "avgeco_2",
    ]

    # Add batting performance features for team1 and team2 (assuming maximum 11 players per team)
    for team_num in [1, 2]:
        for order in range(11):
            cols_to_keep.extend([
                f'bat_{team_num}{order}_ar',
                f'bat_{team_num}{order}_sr',
                f'bat_{team_num}{order}_oar',
                f'bat_{team_num}{order}_osr',
                f'bat_{team_num}{order}_out',
            ])

    # Run the deployment pipeline. This pipeline loads the test data, applies feature engineering 
    # and missing value handling, loads a registered model, and evaluates it on the test set.
    evaluation_metrics, f1 = deployment_pipeline(
        scorecard_folder_path=scorecard_folder_path,
        train_test_folder_path=train_test_folder_path,
        target_column=target_column,
        columns_to_keep=cols_to_keep,
        model_name=model_name,
        model_version=None,  # Use latest version if None is provided
    )

    # Print instructions to launch MLflow UI for experiment tracking.
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "to inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

    # Get the active deployer and check if a model service is running.
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name="deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name=model_name,
    )

    if services and services[0]:
        print(
            f"[green]The MLflow prediction server is running at:\n"
            f"    {services[0].prediction_url}\n"
            "To stop the service, re-run this command with the '--stop-service' flag.[/green]"
        )
    else:
        print("[yellow]No prediction service was found after the deployment pipeline run.[/yellow]")

if __name__ == "__main__":
    run_main()