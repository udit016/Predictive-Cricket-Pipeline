from typing import NamedTuple
import pandas as pd

class IngestedData(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    batsman_level_scorecard: pd.DataFrame
    bowler_level_scorecard: pd.DataFrame
    match_level_scorecard: pd.DataFrame