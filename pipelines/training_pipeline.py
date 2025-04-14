import logging
from zenml import pipeline, Model

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step  # ✅ Updated import
from steps.handle_missing_values_step import handle_missing_values_step
from steps.hyperparameter_optimization_step import hyperparameter_optimization_step
from steps.model_evaluator_step import model_evaluator_step

@pipeline(model=Model(name="win_predictor"))
def ml_pipeline():
    """
    Defines an end-to-end machine learning pipeline for cricket match winner prediction.
    Steps included:
      1. Data ingestion from CSVs
      2. Feature engineering
      3. Missing value handling
      4. Train-test split
      5. Hyperparameter tuning
      6. Model evaluation
    """
    
    # -----------------------------------------------------------------
    # 1. Data Ingestion Step - Load input datasets.
    # -----------------------------------------------------------------
    bat_path = r"D:\Project\New folder\additional_data\batsman_level_scorecard.csv"
    bowl_path = r"D:\Project\New folder\additional_data\bowler_level_scorecard.csv"
    match_path = r"D:\Project\New folder\additional_data\match_level_scorecard.csv"
    train_path = r"D:\Project\New folder\Data\train_data.csv"

    batsman_data = data_ingestion_step(file_path=bat_path)
    bowler_data = data_ingestion_step(file_path=bowl_path)
    match_data = data_ingestion_step(file_path=match_path)
    train_data = data_ingestion_step(file_path=train_path)

    # -----------------------------------------------------------------
    # 2. Define important columns to retain for model training.
    # -----------------------------------------------------------------
    cols_to_keep = [
        'team1', 'team2', 'toss_winner_01', 'toss_decision_01', 'venue', 'city',
        'team_1_lighting_effectiveness', 'team_2_lighting_effectiveness','series_name', 'team_count_50runs_last15', 'team_winp_last5',
        'team1only_avg_runs_last15', 'team1_winp_team2_last15', 'ground_avg_runs_last15',
        'team1_top4_avg_runs', 'team2_top4_avg_runs', 'team1_top4_eco',
        'team2_top4_eco', 'team1_top4_bats_strike', 'team2_top4_bats_strike', 'team1_top2_wck',
        'team2_top2_wck', 'team1_avg_top4_ag', 'team2_avg_top4_ag', 'team1_eco_bot4_ag',
        'team2_eco_bot4_ag', 'team1_eco_top4_ag', 'team2_eco_top4_ag', 'team1_bot4_eco',
        'team2_bot4_eco', 'team1_avg_top4_ve', 'team2_avg_top4_ve', 'team1_eco_top4_ve',
        'team2_eco_top4_ve', 'team1_eco_bot4_ve', 'team2_eco_bot4_ve', 'team1_avg_top4_in',
        'team2_avg_top4_in', 'team1_eco_top4_in', 'team2_eco_top4_in', 'team1_eco_bot4_in',
        'team2_eco_bot4_in', "match_winner_id"
    ]

    # -----------------------------------------------------------------
    # 3. Feature Engineering Step
    # -----------------------------------------------------------------
    train_data = feature_engineering_step(
        df=train_data,
        batsman_scorecard_df=batsman_data,
        bowler_scorecard_df=bowler_data,
        match_scorecard_df=match_data,
        n=15,
        columns_to_keep=cols_to_keep
    )

    # -----------------------------------------------------------------
    # 4. Handle Missing Values
    # -----------------------------------------------------------------
    train_data = handle_missing_values_step(
        df=train_data,
        strategy="0",
        cat_method="NA",
        cat_fill_value="NA"
    )

    # -----------------------------------------------------------------
    # 5. Data Split Step
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = data_splitter_step(
        data=train_data,
        target_column="match_winner_id",
        columns_to_keep=cols_to_keep,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------------------------------------
    # 6. Hyperparameter Optimization Step
    # -----------------------------------------------------------------
    best_model, best_params = hyperparameter_optimization_step(
        X_train=X_train,
        y_train=y_train,
        cv_folds=3,
        n_trials=50
    )

    # -----------------------------------------------------------------
    # 7. Model Evaluation Step
    # -----------------------------------------------------------------
    evaluation_metrics, f1 = model_evaluator_step(
        trained_model=best_model,
        X_eval=X_test,
        y_eval=y_test
    )

    return best_model

if __name__ == "__main__":
    run = ml_pipeline()