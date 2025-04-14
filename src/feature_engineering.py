import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding new feature columns.
        """
        pass

class TossFeatureEngineering(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_transformed = df.copy()
        df_transformed["match_winner_id"] = df_transformed.apply(
            lambda row: 0 if row["winner_id"] == row["team1_id"] else 1,
            axis=1
        )
        df_transformed['toss_winner_01'] = np.where(df_transformed['toss winner'] == df_transformed['team2_id'], 1, 0)
        df_transformed['toss_decision_01'] = np.where(df_transformed['toss decision'] == 'bat', 1, 0)
        return df_transformed

class MergedBatsmanFeatures(FeatureEngineeringStrategy):
    def __init__(self, 
                 n_current: int = 15, 
                 n_opponent: int = 15, 
                 form_matches: int = 5,
                 rel_threshold: float = 20,
                 form_threshold: float = 25):
        """
        :param n_current: Not used explicitly in this merge but reserved for possible current performance.
        :param n_opponent: Not used explicitly in this merge but reserved for possible opponent-specific features.
        :param form_matches: Number of recent matches to consider for form metrics.
        :param rel_threshold: Threshold for a batsman's average runs to be considered "relevant".
        :param form_threshold: Threshold to count a batsman as "in form" (for form count feature).
        """
        self.n_current = n_current
        self.n_opponent = n_opponent
        self.form_matches = form_matches
        self.rel_threshold = rel_threshold
        self.form_threshold = form_threshold
        self.batsman_form_dict = {}  # For last form average runs over recent matches
        
    def apply_transformation(self, train_data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes all team_id-level batsman performance features for train_data using historical
        match_data and batsman_data.
        
        Required keyword arguments:
          - match_data: Historical match-level DataFrame (with columns such as 'match_id', 
                        'team1_id', 'team2_id', 'team1_roster_ids', 'team2_roster_ids', 'toss winner', 
                        'toss decision', 'match_dt', etc.)
          - batsman_data: Historical batsman-level DataFrame (with columns such as 'match id', 
                          'batsman_id', 'match_dt', 'runs', 'strike_rate', 'balls_faced', etc.)
        
        Returns:
          - Updated train_data DataFrame with all new feature columns.
          - Updated batsman_data with a new 'team_id' column added.
        """
        # Validate required inputs.
        if "match_data" not in kwargs:
            raise ValueError("match_data must be provided as a keyword argument.")
        if "batsman_data" not in kwargs:
            raise ValueError("batsman_data must be provided as a keyword argument.")
        
        # Copy datasets to avoid mutating originals.
        hist_match_df = kwargs["match_data"].copy()
        hist_batsman_df = kwargs["batsman_data"].copy()
        train_df = train_data.copy()
        
        # Standardize column names and dates.
        # For historical batsman dataset, rename and convert date.
        hist_batsman_df.rename(columns={"match id": "match_id"}, inplace=True)
        hist_batsman_df['match_dt'] = pd.to_datetime(hist_batsman_df['match_dt'])
        hist_match_df['match_dt'] = pd.to_datetime(hist_match_df['match_dt'])
        
        ####################################################################
        # 1. Add Team Information to Historical Batsman Data
        ####################################################################
        mapping_records = []
        for _, mrow in hist_match_df.iterrows():
            match_id = mrow['match_id']
            team1_id = mrow['team1_id']
            team2_id = mrow['team2_id']
            roster1 = mrow.get('team1_roster_ids', '')
            roster2 = mrow.get('team2_roster_ids', '')
            if pd.notnull(roster1) and roster1.strip() != "":
                for player in roster1.split(':'):
                    try:
                        player_id = float(player)
                        mapping_records.append({'match_id': match_id, 'batsman_id': player_id, 'team_id': team1_id})
                    except:
                        pass
            if pd.notnull(roster2) and roster2.strip() != "":
                for player in roster2.split(':'):
                    try:
                        player_id = float(player)
                        mapping_records.append({'match_id': match_id, 'batsman_id': player_id, 'team_id': team2_id})
                    except:
                        pass
        mapping_df = pd.DataFrame(mapping_records)
        hist_batsman_df = hist_batsman_df.merge(mapping_df, on=['match_id', 'batsman_id'], how='left')
        
        ####################################################################
        # 2. Team Top Batsman Average Feature
        #    Compute each batsman's average runs per team_id.
        ####################################################################
        avg_runs = hist_batsman_df.groupby(['team_id', 'batsman_id'])['runs'].mean().reset_index()
        avg_runs_dict = avg_runs.set_index(['batsman_id', 'team_id'])['runs'].to_dict()
        
        def mean_top4_avg_scores(roster_str, team_id, avg_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            player_ids = roster_str.split(":")
            scores = []
            for player_id in player_ids:
                try:
                    key = (float(player_id), float(team_id))
                    if key in avg_dict:
                        scores.append(avg_dict[key])
                except:
                    continue
            if scores:
                # Take the top 4 highest average scores.
                top4 = sorted(scores, reverse=True)[:4]
                return sum(top4) / len(top4)
            else:
                return 0.0
        
        train_df["team1_top4_avg_runs"] = train_df.apply(
            lambda row: mean_top4_avg_scores(row["team1_roster_ids"], row["team1_id"], avg_runs_dict), axis=1)
        train_df["team2_top4_avg_runs"] = train_df.apply(
            lambda row: mean_top4_avg_scores(row["team2_roster_ids"], row["team2_id"], avg_runs_dict), axis=1)
        
        ####################################################################
        # 3. Team Relative Batsman Count Feature
        #    Count how many batsmen in the roster have an average runs > threshold.
        ####################################################################
        rel_bats = hist_batsman_df[hist_batsman_df['runs'] > self.rel_threshold]
        rel_bats_dict = (
            rel_bats.groupby(['team_id', 'batsman_id'])['runs']
            .mean().reset_index()
            .set_index(['batsman_id', 'team_id'])['runs'].to_dict()
        )
        
        def count_rel_bats(roster_str, team_id, avg_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0
            count = 0
            for player_id in roster_str.split(":"):
                try:
                    key = (float(player_id), float(team_id))
                    if key in avg_dict:
                        count += 1
                except:
                    continue
            return count
        
        train_df["team1_count_rel_bats"] = train_df.apply(
            lambda row: count_rel_bats(row["team1_roster_ids"], row["team1_id"], rel_bats_dict), axis=1)
        train_df["team2_count_rel_bats"] = train_df.apply(
            lambda row: count_rel_bats(row["team2_roster_ids"], row["team2_id"], rel_bats_dict), axis=1)
        
        ####################################################################
        # 4. Team Top 4 Batsman Strike Feature
        #    Using only batsmen who have faced more than 10 balls.
        ####################################################################
        strike_agg = hist_batsman_df.groupby(['team_id', 'batsman_id']).agg({
            'strike_rate': 'mean',
            'balls_faced': 'mean'
        }).reset_index()
        strike_agg = strike_agg[strike_agg['balls_faced'] > 10]
        strike_dict = strike_agg.set_index(['batsman_id', 'team_id'])['strike_rate'].to_dict()
        balls_dict = strike_agg.set_index(['batsman_id', 'team_id'])['balls_faced'].to_dict()
        
        def mean_top4_avg_strikes(roster_str, team_id, strike_dict, balls_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            valid_stats = []
            for pid in roster_str.split(":"):
                try:
                    key = (float(pid), float(team_id))
                    if key in strike_dict and key in balls_dict:
                        valid_stats.append((float(pid), strike_dict[key], balls_dict[key]))
                except:
                    continue
            if valid_stats:
                # Order by balls faced descending; take top 4 strike rates.
                top4 = sorted(valid_stats, key=lambda x: x[2], reverse=True)[:4]
                return sum(x[1] for x in top4) / len(top4)
            else:
                return 0.0
        
        train_df["team1_top4_bats_strike"] = train_df.apply(
            lambda row: mean_top4_avg_strikes(row["team1_roster_ids"], row["team1_id"], strike_dict, balls_dict), axis=1)
        train_df["team2_top4_bats_strike"] = train_df.apply(
            lambda row: mean_top4_avg_strikes(row["team2_roster_ids"], row["team2_id"], strike_dict, balls_dict), axis=1)
        
        ####################################################################
        # 5. Team Top 4 Average Runs Against Feature (Head-to-Head)
        ####################################################################
        # Merge train_df with historical batsman data on match_id.
        comb_df_runs_ag = pd.merge(train_df, hist_batsman_df, on="match_id", how="inner")
        # Define opponent team_id based on train_df. Here we assume that if 'team_id' (from batsman_data) equals team1_id then opponent is team2_id, else team1_id.
        comb_df_runs_ag['opponents'] = comb_df_runs_ag.apply(
            lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1)
        
        grouped_runs_ag = comb_df_runs_ag.groupby(['batsman_id', 'team_id', 'opponents'])['runs'].mean().reset_index()
        grouped_runs_ag.rename(columns={'runs': 'avg_runs_ag'}, inplace=True)
        avg_runs_ag_dict = grouped_runs_ag.set_index(['batsman_id', 'team_id', 'opponents'])['avg_runs_ag'].to_dict()
        
        def mean_top4_ag(roster_str, team_id, opponents, avg_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            scores = []
            for pid in roster_str.split(":"):
                try:
                    key = (float(pid), float(team_id), opponents)
                    if key in avg_dict:
                        scores.append(avg_dict[key])
                except:
                    continue
            if scores:
                top4 = sorted(scores, reverse=True)[:4]
                return sum(top4)/len(top4)
            return 0.0
        
        train_df['team1_avg_top4_ag'] = train_df.apply(
            lambda row: mean_top4_ag(row["team1_roster_ids"], row["team1_id"], row["team2_id"], avg_runs_ag_dict), axis=1)
        train_df['team2_avg_top4_ag'] = train_df.apply(
            lambda row: mean_top4_ag(row["team2_roster_ids"], row["team2_id"], row["team1_id"], avg_runs_ag_dict), axis=1)
        
        ####################################################################
        # 6. Team Top 4 Average Runs in Innings Feature
        ####################################################################
        # For innings, we assume a helper based on toss information.
        def deter_innings(toss_winner, team_id, toss_decision):
            # If team_id is toss winner and they choose to bat, they play the first innings; otherwise second.
            if team_id == toss_winner:
                return 1 if toss_decision == 'bat' else 2
            else:
                return 2 if toss_decision == 'bat' else 1
        
        comb_df_in = pd.merge(train_df, hist_batsman_df, left_on="match_id", right_on="match_id", how="inner")
        # Determine opponent similarly.
        comb_df_in['opponents'] = comb_df_in.apply(
            lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1)
        # Determine innings based on toss info.
        comb_df_in['innings'] = comb_df_in.apply(
            lambda row: deter_innings(row['toss winner'], row['team_id'], row['toss decision']), axis=1)
        grouped_in = comb_df_in.groupby(['batsman_id','team_id','innings'])['runs'].mean().reset_index()
        grouped_in.rename(columns={'runs': 'avg_runs_in'}, inplace=True)
        avg_in_run_dict = grouped_in.set_index(['batsman_id','team_id','innings'])['avg_runs_in'].to_dict()
        
        def mean_top4_in(roster_str, team_id, toss_decision, toss_winner, avg_dict):
            inning = deter_innings(toss_winner, team_id, toss_decision)
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            scores = []
            for pid in roster_str.split(":"):
                try:
                    key = (float(pid), float(team_id), inning)
                    if key in avg_dict:
                        scores.append(avg_dict[key])
                except:
                    continue
            if scores:
                top4 = sorted(scores, reverse=True)[:4]
                return sum(top4)/len(top4)
            else:
                return 0.0
        
        train_df['team1_avg_top4_in'] = train_df.apply(
            lambda row: mean_top4_in(row["team1_roster_ids"], row["team1_id"], row["toss decision"], row["toss winner"], avg_in_run_dict), axis=1)
        train_df['team2_avg_top4_in'] = train_df.apply(
            lambda row: mean_top4_in(row["team2_roster_ids"], row["team2_id"], row["toss decision"], row["toss winner"], avg_in_run_dict), axis=1)
        
        ####################################################################
        # 7. Team Last 5 Average Runs Feature (Batsman Form)
        ####################################################################
        # Build a form dictionary: for each batsman, average runs in their last self.form_matches matches.
        hist_batsman_df.sort_values(by='match_dt', inplace=True)
        for batsman_id in hist_batsman_df['batsman_id'].unique():
            batsman_data = hist_batsman_df[hist_batsman_df['batsman_id'] == batsman_id].sort_values(by='match_dt', ascending=False)
            if batsman_data.empty:
                self.batsman_form_dict[batsman_id] = 0.0
            else:
                recent = batsman_data.head(self.form_matches)
                self.batsman_form_dict[batsman_id] = recent['runs'].mean() if not recent.empty else 0.0
        
        def avg_last5(roster_str, form_dict):
            if pd.isnull(roster_str) or roster_str.strip()=="":
                return 0.0
            scores = []
            for player_id in roster_str.split(":"):
                try:
                    key = float(player_id)
                    if key in form_dict:
                        scores.append(form_dict[key])
                except:
                    continue
            if scores:
                top4 = sorted(scores, reverse=True)[:4]
                return sum(top4)/len(top4)
            else:
                return 0.0
        
        train_df['team1_last5_avg'] = train_df.apply(lambda row: avg_last5(row['team1_roster_ids'], self.batsman_form_dict), axis=1)
        train_df['team2_last5_avg'] = train_df.apply(lambda row: avg_last5(row['team2_roster_ids'], self.batsman_form_dict), axis=1)
        
        ####################################################################
        # 8. Team Batsman Form Count Feature
        #    Count number of batsmen in a roster whose recent average runs >= form_threshold.
        ####################################################################
        def count_player_form(roster_str, form_dict, threshold):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0
            count = 0
            for player_id in roster_str.split(":"):
                try:
                    key = float(player_id)
                    if key in form_dict and form_dict[key] >= threshold:
                        count += 1
                except:
                    continue
            return count
        
        train_df['team1_count_bat_form'] = train_df.apply(
            lambda row: count_player_form(row["team1_roster_ids"], self.batsman_form_dict, self.form_threshold), axis=1)
        train_df['team2_count_bat_form'] = train_df.apply(
            lambda row: count_player_form(row["team2_roster_ids"], self.batsman_form_dict, self.form_threshold), axis=1)
        
        ####################################################################
        # 9. [NEW] Team Top 4 Average Runs at Venue Feature
        ####################################################################
        # Merge train_data and historical batsman data on 'match_id' to compute venue-specific averages.
        combined_df = pd.merge(train_df, hist_batsman_df, on='match_id', how='inner')
        # Group by batsman_id, team_id, and venue to compute the average runs at that venue.
        grouped_df = combined_df.groupby(['batsman_id', 'team_id', 'venue'])['runs'].mean().reset_index()
        grouped_df.rename(columns={'runs': 'avg_runs_ve'}, inplace=True)
        avg_ve_run_dict = {}
        for _, row in grouped_df.iterrows():
            key = (row['batsman_id'], row['team_id'], row['venue'])
            avg_ve_run_dict[key] = row['avg_runs_ve']
        
        def mean_top4_ve(roster, team_id, venue, avg_dict):
            player_ids = roster.split(":")
            valid_player_ids = [(float(player_id), team_id, venue) for player_id in player_ids
                                if (float(player_id), team_id, venue) in avg_dict]
            player_stats = [(player_id, avg_dict[player_id]) for player_id in valid_player_ids]
            top4_players = sorted(player_stats, key=lambda x: x[1], reverse=True)[:4]
            if top4_players:
                return sum(player[1] for player in top4_players) / len(top4_players)
            else:
                return 0.0
        
        train_df['team1_avg_top4_ve'] = train_df.apply(
            lambda row: mean_top4_ve(row["team1_roster_ids"], row["team1_id"], row["venue"], avg_ve_run_dict), axis=1)
        train_df['team2_avg_top4_ve'] = train_df.apply(
            lambda row: mean_top4_ve(row["team2_roster_ids"], row["team2_id"], row["venue"], avg_ve_run_dict), axis=1)
        
        # Return the updated train_data and updated batsman_data
        return train_df, hist_batsman_df

#########################################
# Add Team to Bowler Feature
#########################################
import pandas as pd
from typing import Tuple

# Base class
class FeatureEngineeringStrategy:
    def apply_transformation(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

class MergedTeamFeatures(FeatureEngineeringStrategy):
    def __init__(self, matches: int = 5):
        self.matches = matches
        self.bowler_form_dict = {}

    def apply_transformation(self, train_data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes all team_id-level features for the train_data using the historical match_data and bowler_data.
        
        Required keyword arguments:
           - match_data: Historical match-level DataFrame (includes columns like 'match_id', 'team1_id', 'team2_id',
                         'team1_roster_ids', 'team2_roster_ids', 'match_dt', etc.)
           - bowler_data: Historical bowler-level DataFrame (includes columns like 'match id', 'bowler_id',
                          'match_dt', 'economy', 'runs', 'balls_bowled', 'wicket_count', etc.)
                          
        Returns:
           - Updated train_data DataFrame with all calculated feature columns.
           - Updated bowler_data DataFrame (for example, with team_id assignments added).
        """
        # Validate required inputs.
        if "match_data" not in kwargs:
            raise ValueError("match_data must be provided as a keyword argument.")
        if "bowler_data" not in kwargs:
            raise ValueError("bowler_data must be provided as a keyword argument.")
        
        # Historical datasets
        hist_match_df = kwargs["match_data"].copy()
        hist_bowler_df = kwargs["bowler_data"].copy()
        train_df = train_data.copy()
        
        # Standardize date columns in historical datasets
        hist_bowler_df.rename(columns={"match id": "match_id"}, inplace=True)
        hist_bowler_df['match_dt'] = pd.to_datetime(hist_bowler_df['match_dt'])
        hist_match_df['match_dt'] = pd.to_datetime(hist_match_df['match_dt'])
        
        #################################################################
        # 1. Add Team Information to Bowler Data (from match_data)
        #################################################################
        mapping_records = []
        for _, mrow in hist_match_df.iterrows():
            match_id = mrow['match_id']
            team1_id = mrow['team1_id']
            team2_id = mrow['team2_id']
            roster1 = mrow.get('team1_roster_ids', '')
            roster2 = mrow.get('team2_roster_ids', '')
            if pd.notnull(roster1) and roster1.strip() != "":
                for player in roster1.split(':'):
                    try:
                        player_id = float(player)
                        mapping_records.append({'match_id': match_id, 'bowler_id': player_id, 'team_id': team1_id})
                    except:
                        pass
            if pd.notnull(roster2) and roster2.strip() != "":
                for player in roster2.split(':'):
                    try:
                        player_id = float(player)
                        mapping_records.append({'match_id': match_id, 'bowler_id': player_id, 'team_id': team2_id})
                    except:
                        pass
        mapping_df = pd.DataFrame(mapping_records)
        hist_bowler_df = hist_bowler_df.merge(mapping_df, on=['match_id', 'bowler_id'], how='left')
        
        #################################################################
        # 2. Compute Team Top-Bottom Economy Features using historical bowler data
        #################################################################
        average_eco = hist_bowler_df.groupby(['team_id', 'bowler_id'])['economy'].mean().reset_index()
        average_eco_dict = average_eco.set_index(["bowler_id", "team_id"])["economy"].to_dict()
        
        def mean_eco_top(roster_str, team_id, avg_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            values = []
            for pid in roster_str.split(":"):
                try:
                    key = (float(pid), float(team_id))
                    if key in avg_dict:
                        values.append(avg_dict[key])
                except:
                    continue
            top4 = sorted(values)[:4]
            return sum(top4)/len(top4) if top4 else 0.0
        
        def mean_eco_bot(roster_str, team_id, avg_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            values = []
            for pid in roster_str.split(":"):
                try:
                    key = (float(pid), float(team_id))
                    if key in avg_dict:
                        values.append(avg_dict[key])
                except:
                    continue
            bot4 = sorted(values, reverse=True)[:4]
            return sum(bot4)/len(bot4) if bot4 else 0.0

        # Compute these features on train_data (assumed to have appropriate columns)
        train_df["team1_top4_eco"] = train_df.apply(lambda row: mean_eco_top(row["team1_roster_ids"], row["team1_id"], average_eco_dict), axis=1)
        train_df["team2_top4_eco"] = train_df.apply(lambda row: mean_eco_top(row["team2_roster_ids"], row["team2_id"], average_eco_dict), axis=1)
        train_df["team1_bot4_eco"] = train_df.apply(lambda row: mean_eco_bot(row["team1_roster_ids"], row["team1_id"], average_eco_dict), axis=1)
        train_df["team2_bot4_eco"] = train_df.apply(lambda row: mean_eco_bot(row["team2_roster_ids"], row["team2_id"], average_eco_dict), axis=1)
        
        #################################################################
        # 3. Team Bowler Form Feature (Recent Form Over Last N Matches)
        #################################################################
        def get_bowler_form(bowler_id, df_):
            df_temp = df_[df_['bowler_id'] == bowler_id].sort_values(by='match_dt', ascending=False)
            if df_temp.empty:
                return 0.0
            recent = df_temp.head(self.matches)
            return recent['economy'].mean() if not recent.empty else 0.0

        for bowler_id in hist_bowler_df['bowler_id'].unique():
            self.bowler_form_dict[bowler_id] = get_bowler_form(bowler_id, hist_bowler_df)
        
        def avg_bowler_form(roster_str, form_dict):
            if pd.isnull(roster_str) or roster_str.strip() == "":
                return 0.0
            scores = []
            for pid in roster_str.split(":"):
                try:
                    key = float(pid)
                    if key in form_dict:
                        scores.append(form_dict[key])
                except:
                    continue
            if scores:
                top4 = sorted(scores)[:4]
                return sum(top4)/len(top4)
            return 0.0

        train_df['team1_last5_bowler_form'] = train_df.apply(lambda row: avg_bowler_form(row['team1_roster_ids'], self.bowler_form_dict), axis=1)
        train_df['team2_last5_bowler_form'] = train_df.apply(lambda row: avg_bowler_form(row['team2_roster_ids'], self.bowler_form_dict), axis=1)
        
        #################################################################
        # 4. Team Last5 Economy Feature from recent historical matches
        #################################################################
        hist_bowler_df.sort_values(by=["bowler_id", "match_dt"], inplace=True)
        hist_bowler_df["eco"] = hist_bowler_df["runs"] / (hist_bowler_df["balls_bowled"] / 6 + 1e-6)  # avoid divide-by-zero
        recent_matches = hist_bowler_df.groupby("bowler_id").tail(5)
        avg_dict_last5 = recent_matches.groupby("bowler_id")["eco"].mean().to_dict()

        def eco_last5_top(roster: str) -> float:
            player_ids = roster.split(":")
            eco_list = []
            for pid in player_ids:
                try:
                    key = float(pid)
                    if key in avg_dict_last5:
                        eco_list.append(avg_dict_last5[key])
                except:
                    continue
            if eco_list:
                top4 = sorted(eco_list)[:4]
                return sum(top4) / len(top4)
            return 0.0

        def eco_last5_bot(roster: str) -> float:
            player_ids = roster.split(":")
            eco_list = []
            for pid in player_ids:
                try:
                    key = float(pid)
                    if key in avg_dict_last5:
                        eco_list.append(avg_dict_last5[key])
                except:
                    continue
            if eco_list:
                bot4 = sorted(eco_list, reverse=True)[:4]
                return sum(bot4) / len(bot4)
            return 0.0

        train_df["team1_last5_eco_top"] = train_df["team1_roster_ids"].apply(eco_last5_top)
        train_df["team2_last5_eco_top"] = train_df["team2_roster_ids"].apply(eco_last5_top)
        train_df["team1_last5_eco_bot"] = train_df["team1_roster_ids"].apply(eco_last5_bot)
        train_df["team2_last5_eco_bot"] = train_df["team2_roster_ids"].apply(eco_last5_bot)
        
        #################################################################
        # 5. Team Bowler Form Count Feature
        #################################################################
        def count_player_form_bowl(roster: str, avg_dict) -> int:
            player_ids = roster.split(":")
            count = 0
            for pid in player_ids:
                try:
                    key = float(pid)
                    if key in avg_dict and avg_dict[key] <= 7:
                        count += 1
                except:
                    continue
            return count

        train_df["team1_count_bowl_form"] = train_df["team1_roster_ids"].apply(lambda r: count_player_form_bowl(r, average_eco_dict))
        train_df["team2_count_bowl_form"] = train_df["team2_roster_ids"].apply(lambda r: count_player_form_bowl(r, average_eco_dict))
        
        #################################################################
        # 6. Team Relative Bowler Count Feature
        #################################################################
        rel_bowl = hist_bowler_df[hist_bowler_df["economy"] < 7.5]
        rel_bowl_dict = rel_bowl.set_index(["bowler_id", "team_id"])["economy"].to_dict()
        
        def count_eco_top_relative(roster: str, team_id, avg_dict) -> int:
            player_ids = roster.split(":")
            count = 0
            for pid in player_ids:
                try:
                    key = (float(pid), float(team_id))
                    if key in avg_dict:
                        count += 1
                except:
                    continue
            return count

        train_df["team1_count_rel_bowl"] = train_df.apply(lambda row: count_eco_top_relative(row["team1_roster_ids"], row["team1_id"], rel_bowl_dict), axis=1)
        train_df["team2_count_rel_bowl"] = train_df.apply(lambda row: count_eco_top_relative(row["team2_roster_ids"], row["team2_id"], rel_bowl_dict), axis=1)
        
        #################################################################
        # 7. Team Aggressive Economy Against Feature
        #################################################################
        comb_df = pd.merge(train_df, hist_bowler_df, left_on="match_id", right_on="match_id", how="inner")
        comb_df['opponents'] = comb_df.apply(
            lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1
        )
        group_df = comb_df.groupby(['bowler_id', 'team_id', 'opponents'])['economy'].mean().reset_index()
        group_df.rename(columns={'economy': 'avg_eco_ag'}, inplace=True)
        avg_ag_eco_dict = group_df.set_index(['bowler_id', 'team_id', 'opponents'])['avg_eco_ag'].to_dict()
        
        def mean_top4_eco_ag(roster: str, team_id, opponents, avg_dict) -> float:
            player_ids = roster.split(":")
            stats = [avg_dict.get((float(pid), float(team_id), float(opponents))) 
                     for pid in player_ids if pid.strip() != "" and ((float(pid), float(team_id), float(opponents)) in avg_dict)]
            stats = [s for s in stats if s is not None]
            if stats:
                top4 = sorted(stats)[:4]
                return sum(top4) / len(top4)
            return 0.0
        
        def mean_bot4_eco_ag(roster: str, team_id, opponents, avg_dict) -> float:
            player_ids = roster.split(":")
            stats = [avg_dict.get((float(pid), float(team_id), float(opponents))) 
                     for pid in player_ids if pid.strip() != "" and ((float(pid), float(team_id), float(opponents)) in avg_dict)]
            stats = [s for s in stats if s is not None]
            if stats:
                bot4 = sorted(stats, reverse=True)[:4]
                return sum(bot4) / len(bot4)
            return 0.0
        
        train_df['team1_eco_top4_ag'] = train_df.apply(lambda row: mean_top4_eco_ag(row["team1_roster_ids"], row["team1_id"], row["team2_id"], avg_ag_eco_dict), axis=1)
        train_df['team2_eco_top4_ag'] = train_df.apply(lambda row: mean_top4_eco_ag(row["team2_roster_ids"], row["team2_id"], row["team1_id"], avg_ag_eco_dict), axis=1)
        train_df['team1_eco_bot4_ag'] = train_df.apply(lambda row: mean_bot4_eco_ag(row["team1_roster_ids"], row["team1_id"], row["team2_id"], avg_ag_eco_dict), axis=1)
        train_df['team2_eco_bot4_ag'] = train_df.apply(lambda row: mean_bot4_eco_ag(row["team2_roster_ids"], row["team2_id"], row["team1_id"], avg_ag_eco_dict), axis=1)
        
        #################################################################
        # 8. Team Aggressive Economy at Venue Feature
        #################################################################
        comb_df_ve = pd.merge(train_df, hist_bowler_df, left_on="match_id", right_on="match_id", how="inner")
        # Assumes 'venue' exists in hist_bowler_df.
        group_df_ve = comb_df_ve.groupby(['bowler_id', 'team_id', 'venue'])['economy'].mean().reset_index()
        group_df_ve.rename(columns={'economy': 'avg_eco_ve'}, inplace=True)
        avg_ve_eco_dict = group_df_ve.set_index(['bowler_id', 'team_id', 'venue'])['avg_eco_ve'].to_dict()
        
        def mean_top4_eco_ve(roster: str, team_id, venue, avg_dict) -> float:
            player_ids = roster.split(":")
            stats = [avg_dict.get((float(pid), float(team_id), venue))
                     for pid in player_ids if pid.strip() != "" and ((float(pid), float(team_id), venue) in avg_dict)]
            stats = [s for s in stats if s is not None]
            if stats:
                top4 = sorted(stats)[:4]
                return sum(top4)/len(top4)
            return 0.0
        
        def mean_bot4_eco_ve(roster: str, team_id, venue, avg_dict) -> float:
            player_ids = roster.split(":")
            stats = [avg_dict.get((float(pid), float(team_id), venue))
                     for pid in player_ids if pid.strip() != "" and ((float(pid), float(team_id), venue) in avg_dict)]
            stats = [s for s in stats if s is not None]
            if stats:
                bot4 = sorted(stats, reverse=True)[:4]
                return sum(bot4)/len(bot4)
            return 0.0
        
        train_df['team1_eco_top4_ve'] = train_df.apply(lambda row: mean_top4_eco_ve(row["team1_roster_ids"], row["team1_id"], row["venue"], avg_ve_eco_dict), axis=1)
        train_df['team2_eco_top4_ve'] = train_df.apply(lambda row: mean_top4_eco_ve(row["team2_roster_ids"], row["team2_id"], row["venue"], avg_ve_eco_dict), axis=1)
        train_df['team1_eco_bot4_ve'] = train_df.apply(lambda row: mean_bot4_eco_ve(row["team1_roster_ids"], row["team1_id"], row["venue"], avg_ve_eco_dict), axis=1)
        train_df['team2_eco_bot4_ve'] = train_df.apply(lambda row: mean_bot4_eco_ve(row["team2_roster_ids"], row["team2_id"], row["venue"], avg_ve_eco_dict), axis=1)
        
        #################################################################
        # 9. Team Economy by Innings Feature
        #################################################################
        comb_df_in = pd.merge(train_df, hist_bowler_df, left_on="match_id", right_on="match_id", how="inner")
        comb_df_in['opponents'] = comb_df_in.apply(lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1)
        
        def determine_innings(row):
            if row['team_id'] == row['toss winner']:
                return 1 if row['toss decision'] == 'field' else 2
            else:
                return 2 if row['toss decision'] == 'field' else 1
        comb_df_in['innings'] = comb_df_in.apply(determine_innings, axis=1)
        grouped_in = comb_df_in.groupby(['bowler_id', 'team_id', 'innings'])['economy'].mean().reset_index()
        grouped_in.rename(columns={'economy': 'avg_eco_in'}, inplace=True)
        avg_in_eco_dict = { (row['bowler_id'], row['team_id'], row['innings']): row['avg_eco_in'] 
                            for _, row in grouped_in.iterrows() }
        
        def eco_top4_in(roster: str, team_id: float, toss_decision: str, toss_winner: float, avg_dict: dict) -> float:
            innings = 1 if (team_id == toss_winner and toss_decision == 'field') or (team_id != toss_winner and toss_decision != 'field') else 2
            player_ids = roster.split(":")
            valid = [avg_dict[(float(pid), team_id, innings)]
                     for pid in player_ids if pid.strip() != "" and ((float(pid), team_id, innings) in avg_dict)]
            top4 = sorted(valid)[:4]
            return sum(top4)/len(top4) if top4 else 0.0
        
        def eco_bot4_in(roster: str, team_id: float, toss_decision: str, toss_winner: float, avg_dict: dict) -> float:
            innings = 1 if (team_id == toss_winner and toss_decision == 'field') or (team_id != toss_winner and toss_decision != 'field') else 2
            player_ids = roster.split(":")
            valid = [avg_dict[(float(pid), team_id, innings)]
                     for pid in player_ids if pid.strip() != "" and ((float(pid), team_id, innings) in avg_dict)]
            bot4 = sorted(valid, reverse=True)[:4]
            return sum(bot4)/len(bot4) if bot4 else 0.0
        
        train_df['team1_eco_top4_in'] = train_df.apply(lambda row: eco_top4_in(row["team1_roster_ids"], row["team1_id"], row["toss decision"], row["toss winner"], avg_in_eco_dict), axis=1)
        train_df['team2_eco_top4_in'] = train_df.apply(lambda row: eco_top4_in(row["team2_roster_ids"], row["team2_id"], row["toss decision"], row["toss winner"], avg_in_eco_dict), axis=1)
        train_df['team1_eco_bot4_in'] = train_df.apply(lambda row: eco_bot4_in(row["team1_roster_ids"], row["team1_id"], row["toss decision"], row["toss winner"], avg_in_eco_dict), axis=1)
        train_df['team2_eco_bot4_in'] = train_df.apply(lambda row: eco_bot4_in(row["team2_roster_ids"], row["team2_id"], row["toss decision"], row["toss winner"], avg_in_eco_dict), axis=1)
        
        #################################################################
        # 10. Team Top 2 Wickets Feature
        #################################################################
        avg_wck = hist_bowler_df.groupby(['team_id', 'bowler_id'])['wicket_count'].mean().reset_index()
        avg_wck_dict = avg_wck.set_index(["bowler_id", "team_id"])["wicket_count"].to_dict()
        
        def top2_wck(roster: str, team_id: float, avg_dict: dict) -> float:
            player_ids = roster.split(":")
            wcks = []
            for pid in player_ids:
                try:
                    key = (float(pid), team_id)
                    if key in avg_dict:
                        wcks.append(avg_dict[key])
                except:
                    continue
            top2 = sorted(wcks, reverse=True)[:2]
            return sum(top2)/len(top2) if top2 else 0.0
        
        train_df["team1_top2_wck"] = train_df.apply(lambda row: top2_wck(row["team1_roster_ids"], row["team1_id"], avg_wck_dict), axis=1)
        train_df["team2_top2_wck"] = train_df.apply(lambda row: top2_wck(row["team2_roster_ids"], row["team2_id"], avg_wck_dict), axis=1)
        
        #################################################################
        # 11. Team Top 2 Wickets Against Feature
        #################################################################
        comb_df_wck_ag = pd.merge(train_df, hist_bowler_df, on="match_id", how="inner")
        comb_df_wck_ag['opponents'] = comb_df_wck_ag.apply(
            lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1
        )
        grouped_wck_ag = comb_df_wck_ag.groupby(['bowler_id', 'team_id', 'opponents'])['wicket_count'].mean().reset_index()
        grouped_wck_ag.rename(columns={'wicket_count': 'avg_wck_ag'}, inplace=True)
        avg_ag_wck_dict = { (row['bowler_id'], row['team_id'], row['opponents']): row['avg_wck_ag']
                            for _, row in grouped_wck_ag.iterrows() }
        
        def top2_wck_against(roster: str, team_id: float, opponents: float, avg_dict: dict) -> float:
            player_ids = roster.split(":")
            valid = []
            for pid in player_ids:
                try:
                    key = (float(pid), team_id, opponents)
                    if key in avg_dict:
                        valid.append(avg_dict[key])
                except:
                    continue
            top2 = sorted(valid, reverse=True)[:2]
            return sum(top2)/len(top2) if top2 else 0.0
        
        train_df["team1_wck_top2_ag"] = train_df.apply(lambda row: top2_wck_against(row["team1_roster_ids"], row["team1_id"], row["team2_id"], avg_ag_wck_dict), axis=1)
        train_df["team2_wck_top2_ag"] = train_df.apply(lambda row: top2_wck_against(row["team2_roster_ids"], row["team2_id"], row["team1_id"], avg_ag_wck_dict), axis=1)
        
        #################################################################
        # 12. Team Top 2 Wickets at Venue Feature
        #################################################################
        comb_df_wck_ve = pd.merge(train_df, hist_bowler_df, on="match_id", how="inner")
        grouped_wck_ve = comb_df_wck_ve.groupby(['bowler_id', 'team_id', 'venue'])['wicket_count'].mean().reset_index()
        grouped_wck_ve.rename(columns={'wicket_count': 'avg_wck_ve'}, inplace=True)
        avg_ve_wck_dict = grouped_wck_ve.set_index(['bowler_id', 'team_id', 'venue'])['avg_wck_ve'].to_dict()
        
        def top2_wck_venue(roster: str, team_id: float, venue, avg_dict: dict) -> float:
            player_ids = roster.split(":")
            valid = []
            for pid in player_ids:
                try:
                    key = (float(pid), team_id, venue)
                    if key in avg_dict:
                        valid.append(avg_dict[key])
                except:
                    continue
            top2 = sorted(valid, reverse=True)[:2]
            return sum(top2)/len(top2) if top2 else 0.0
        
        train_df["team1_wck_top2_ve"] = train_df.apply(lambda row: top2_wck_venue(row["team1_roster_ids"], row["team1_id"], row["venue"], avg_ve_wck_dict), axis=1)
        train_df["team2_wck_top2_ve"] = train_df.apply(lambda row: top2_wck_venue(row["team2_roster_ids"], row["team2_id"], row["venue"], avg_ve_wck_dict), axis=1)
        
        #################################################################
        # 13. Team Top 2 Wickets by Innings Feature
        #################################################################
        comb_df_wck_in = pd.merge(train_df, hist_bowler_df, on="match_id", how="inner")
        comb_df_wck_in['opponents'] = comb_df_wck_in.apply(lambda row: row['team2_id'] if row['team_id'] == row.get('team1_id') else row['team1_id'], axis=1)
        comb_df_wck_in['innings'] = comb_df_wck_in.apply(determine_innings, axis=1)
        grouped_wck_in = comb_df_wck_in.groupby(['bowler_id', 'team_id', 'innings'])['wicket_count'].mean().reset_index()
        grouped_wck_in.rename(columns={'wicket_count': 'avg_wck_in'}, inplace=True)
        avg_in_wck_dict = { (row['bowler_id'], row['team_id'], row['innings']): row['avg_wck_in']
                            for _, row in grouped_wck_in.iterrows() }
        
        def wck_top2_in(roster: str, team_id: float, toss_decision: str, toss_winner: float, avg_dict: dict) -> float:
            innings = 1 if (team_id == toss_winner and toss_decision == 'field') or (team_id != toss_winner and toss_decision != 'field') else 2
            player_ids = roster.split(":")
            valid = [avg_dict[(float(pid), team_id, innings)]
                     for pid in player_ids if pid.strip() != "" and ((float(pid), team_id, innings) in avg_dict)]
            top2 = sorted(valid, reverse=True)[:2]
            return sum(top2)/len(top2) if top2 else 0.0
        
        train_df['team1_wck_top2_in'] = train_df.apply(lambda row: wck_top2_in(row["team1_roster_ids"], row["team1_id"], row["toss decision"], row["toss winner"], avg_in_wck_dict), axis=1)
        train_df['team2_wck_top2_in'] = train_df.apply(lambda row: wck_top2_in(row["team2_roster_ids"], row["team2_id"], row["toss decision"], row["toss winner"], avg_in_wck_dict), axis=1)
        
        # Return the updated train_data and updated bowler_data (with team_id information merged)
        return train_df, hist_bowler_df


class TeamRatioFeatures(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Define a list of tuples: (new_column, team1_column, team2_column)
        ratio_features = [
            ('team_ratio_avg', 'team1_top4_avg', 'team2_top4_avg'),
            ('team_ratio_bats', 'team1_top4_bats', 'team2_top4_bats'),
            ('team_ratio_eco_top', 'team1_top4_eco', 'team2_top4_eco'),
            ('team_ratio_eco_bot', 'team1_bot4_eco', 'team2_bot4_eco'),
            ('team_ratio_wck', 'team1_top2_wck', 'team2_top2_wck'),
            ('team_ratio_avg_ag', 'team1_avg_top4_ag', 'team2_avg_top4_ag'),
            ('team_ratio_avg_ve', 'team1_avg_top4_ve', 'team2_avg_top4_ve'),
            ('team_ratio_avg_in', 'team1_avg_top4_in', 'team2_avg_top4_in'),
            ('team_ratio_eco_top_ag', 'team1_eco_top4_ag', 'team2_eco_top4_ag'),
            ('team_ratio_eco_top_ve', 'team1_eco_top4_ve', 'team2_eco_top4_ve'),
            ('team_ratio_eco_top_in', 'team1_eco_top4_in', 'team2_eco_top4_in'),
            ('team_ratio_eco_bot_ag', 'team1_eco_bot4_ag', 'team2_eco_bot4_ag'),
            ('team_ratio_eco_bot_ve', 'team1_eco_bot4_ve', 'team2_eco_bot4_ve'),
            ('team_ratio_eco_bot_in', 'team1_eco_bot4_in', 'team2_eco_bot4_in'),
            ('team_ratio_wck_top_ag', 'team1_wck_top2_ag', 'team2_wck_top2_ag'),
            ('team_ratio_wck_top_ve', 'team1_wck_top2_ve', 'team2_wck_top2_ve'),
            ('team_ratio_wck_top_in', 'team1_wck_top2_in', 'team2_wck_top2_in'),
            ('team_ratio_rel_bats', 'team1_count_rel_bats', 'team2_count_rel_bats'),
            ('team_ratio_rel_bowl', 'team1_count_rel_bowl', 'team2_count_rel_bowl'),
            ('team_ratio_bat_form', 'team1_count_bat_form', 'team2_count_bat_form'),
            ('team_ratio_bowl_form', 'team1_count_bowl_form', 'team2_count_bowl_form'),
            ('team_ratio_last5_avg', 'team1_last5_avg', 'team2_last5_avg'),
            ('team_ratio_last5_eco_top', 'team1_last5_eco_top', 'team2_last5_eco_top'),
            ('team_ratio_last5_eco_bot', 'team1_last5_eco_bot', 'team2_last5_eco_bot')
        ]
        
        df = df.copy()
        for new_col, col1, col2 in ratio_features:
            if col1 in df.columns and col2 in df.columns:
                df[new_col] = df[col1] - df[col2]
            else:
                # If columns are missing, you can set the new column to NaN or 0
                df[new_col] = np.nan
        return df
    
    
class LightingEffectivenessFeature(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Requires that the caller passes:
            match_scorecard_df: pd.DataFrame with historical match data.
        Calculates lighting effectiveness for team1_id and team2_id.
        """
        if "match_scorecard_df" not in kwargs:
            raise ValueError("match_scorecard_df must be provided as a keyword argument.")
        match_df = kwargs["match_scorecard_df"].copy()
        match_df['match_dt'] = pd.to_datetime(match_df['match_dt'])
        
        df_transformed = df.copy()
        df_transformed['match_dt'] = pd.to_datetime(df_transformed['match_dt'])
        
        for team_num, team_id_col in enumerate(['team1_id', 'team2_id'], start=1):
            effectiveness_list = []
            for _, row in df_transformed.iterrows():
                team_id = row[team_id_col]
                lighting = row['lighting']
                match_date = row['match_dt']
                df_team = match_df[
                    (((match_df['team1_id'] == team_id) | (match_df['team2_id'] == team_id)) &
                     (match_df['lighting'] == lighting) &
                     (match_df['match_dt'] < match_date))
                ]
                df_team_last_n = df_team.sort_values(by='match_dt', ascending=False).head(kwargs.get("n", 15))
                wins = df_team_last_n[df_team_last_n['winner_id'] == team_id]
                effectiveness = len(wins) / len(df_team_last_n) if len(df_team_last_n) > 0 else 0
                effectiveness_list.append(effectiveness)
            df_transformed[f"team_{team_num}_lighting_effectiveness"] = effectiveness_list
        return df_transformed

class CategoricalEncoder:
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.label_encoders = {}

    def fit_transform(self, df):
        for col in self.categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def transform(self, df):
        for col in self.categorical_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df
    
    
class FinalizeFeatures(FeatureEngineeringStrategy):
    def __init__(self, columns_to_keep: list):
        self.columns_to_keep = columns_to_keep

    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df[self.columns_to_keep].copy()