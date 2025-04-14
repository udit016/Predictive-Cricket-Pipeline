import pandas as pd
from zenml import step
from src.feature_engineering import (
    TossFeatureEngineering,
    MergedBatsmanFeatures,   # Merged batsman feature transformations.
    MergedTeamFeatures,      # Merged bowler (team) feature transformations.
    LightingEffectivenessFeature,
    TeamRatioFeatures,
    CategoricalEncoder,
    FinalizeFeatures
)

@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame,
    batsman_scorecard_df: pd.DataFrame,
    bowler_scorecard_df: pd.DataFrame,
    match_scorecard_df: pd.DataFrame,
    n: int = 15,
    columns_to_keep: list = None,
) -> pd.DataFrame:
    """Comprehensive cricket-specific feature engineering pipeline using merged batsman and bowler features."""
    
    # 1. Toss-related features.
    df = TossFeatureEngineering().apply_transformation(df)
    
    # 2. Batsman-related features.
    # This merged class computes all batsman performance features (including team assignment and form metrics)
    # based on the historical match and batsman datasets.
    df, batsman_scorecard_df = MergedBatsmanFeatures(
        n_current=n,
        n_opponent=10,
        form_matches=5,
        rel_threshold=20,
        form_threshold=25
    ).apply_transformation(
        df,
        match_data=match_scorecard_df,
        batsman_data=batsman_scorecard_df
    )
    
    # 3. Bowler-related (team) features.
    # This merged class computes all team (bowler) features including team assignment, economy, form,
    # wicket-taking metrics, and more based on the historical match and bowler datasets.
    df, bowler_scorecard_df = MergedTeamFeatures(matches=5).apply_transformation(
        df,
        match_data=match_scorecard_df,
        bowler_data=bowler_scorecard_df
    )
    
    # 4. Lighting conditions and ratio-based features.
    df = LightingEffectivenessFeature().apply_transformation(
        df, match_scorecard_df=match_scorecard_df, n=n
    )
    df = TeamRatioFeatures().apply_transformation(df)
    
    # 5. Encoding categorical variables.
    categorical_cols = ['team1', 'team2', 'venue', 'city', 'series_name', 'season']
    df = CategoricalEncoder(categorical_cols).fit_transform(df)
    df.to_csv('encoded_data.csv', index=False)
    # 6. Finalize features.
    if columns_to_keep is not None:
        df = FinalizeFeatures(columns_to_keep).apply_transformation(df)
    
    return df