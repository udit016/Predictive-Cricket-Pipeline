import logging
from zenml import pipeline

from steps.data_ingestion_step import load_scorecard_data_step, load_train_test_data_step
from steps.feature_engineering_step import feature_engineering_step
from steps.missing_value_handler_step import handle_missing_values_step
from steps.model_loader_step_step import model_loader_step
from steps.model_evaluator_step import model_evaluator_step

@pipeline
def deployment_pipeline(
    scorecard_folder_path: str,
    train_test_folder_path: str,
    target_column: str,
    columns_to_keep: list,
    model_name: str,
    model_version: str = None  # Optional, defaults to latest if None
):
    # 1. Load match-level, batsman-level, and bowler-level scorecard data
    match_df, batsman_df, bowler_df = load_scorecard_data_step(scorecard_folder_path)

    # 2. Load only the test set (ignore train set here)
    _, test_df = load_train_test_data_step(train_test_folder_path)

    # 3. Feature engineering on the test set
    # For deployment, pass None for train_df since it is not used.
    _, test_fe_df = feature_engineering_step(
        train_df=None,
        test_df=test_df,
        match_df=match_df,
        batsman_df=batsman_df,
        bowler_df=bowler_df,
        columns_to_keep=columns_to_keep
    )

    # 4. Handle missing values in the test set
    # Again, passing None for train_df if not required.
    _, test_cleaned_df = handle_missing_values_step(
        train_df=None,
        test_df=test_fe_df
    )

    # 5. Load the trained model from the model registry
    trained_model = model_loader_step(
        model_name=model_name,
        model_version=model_version
    )

    # 6. Prepare X_test and y_test using the provided columns_to_keep
    X_test = test_cleaned_df[columns_to_keep]
    y_test = test_cleaned_df[target_column]

    # 7. Evaluate the trained model on the test set
    evaluation_metrics, f1 = model_evaluator_step(
        trained_model=trained_model,
        X_eval=X_test,
        y_eval=y_test,
        eval_set_name="test"
    )

    logging.info("Deployment pipeline completed. Test metrics: %s", evaluation_metrics)
    return evaluation_metrics, f1