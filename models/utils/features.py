# configs/features.py
from typing import List, Set
from pydantic import BaseModel, Field

class BaseFeatureConfig(BaseModel, strict=True):
    """Base configuration for features used in modeling."""
    match_id_col: str = "MatchID"
    home_team_col: str = "HomeTeam"
    away_team_col: str = "AwayTeam"
    date_col: str = "Date"
    league_id_col: str = "LeagueID" # Useful for league-specific effects

    # Target columns needed for training Poisson (goals) or evaluating results
    target_home_goals: str = "FTHG"
    target_away_goals: str = "FTAG"
    target_result: str = "FTR" # H, D, A

    # --- Core Statistical Features (Non-Odds) ---
    core_numerical_features: List[str] = Field(default_factory=lambda: [
        # ELO / Team Strength Indicators
        "HomeTeamELO", "AwayTeamELO",

        # Basic Match Stats (Current Game - Use with caution, potential leakage if used directly for prediction)
        # 'HomeBlockedShots', 'AwayBlockedShots', 'HomeCorners', 'AwayCorners',
        # 'HomeFouls', 'AwayFouls', 'HomeOffsides', 'AwayOffsides',
        # 'HomePassAccuracy', 'AwayPassAccuracy', 'HomePossession', 'AwayPossession',
        # 'HomeRedCards', 'AwayRedCards', 'HomeSaves', 'AwaySaves',
        # 'HomeShots', 'AwayShots', 'HomeShotsInsideBox', 'AwayShotsInsideBox',
        # 'HomeShotsOffTarget', 'AwayShotsOffTarget', 'HomeShotsOutsideBox', 'AwayShotsOutsideBox',
        # 'HomeShotsTarget', 'AwayShotsTarget', 'HomeTotalPasses', 'AwayTotalPasses',
        # 'HomeYellowCards', 'AwayYellowCards', 'AwayExpectedGoals', 'HomeExpectedGoals', # Careful with xG leakage

        # Historical Performance Averages (Lagged - Safer for Prediction)
        # Examples - Add ALL relevant average columns here
        "Away_AvgBlockedShotsAgainst_Away_Last10", "Away_AvgBlockedShotsFor_Away_Last10",
        "Away_AvgCornersAgainst_Away_Last10", "Away_AvgCornersFor_Away_Last10",
        "Away_AvgExpectedGoalsFor_Away_Last10", # Lagged xG is safer
        "Away_AvgFoulsAgainst_Away_Last10", "Away_AvgFoulsFor_Away_Last10",
        "Away_AvgGoalsConceded_Away_Last10", "Away_AvgGoalsScored_Away_Last10",
        "Away_AvgOffsidesAgainst_Away_Last10", "Away_AvgOffsidesFor_Away_Last10",
        "Away_AvgPassAccuracyAgainst_Away_Last10", "Away_AvgPassAccuracyFor_Away_Last10",
        "Away_AvgPossessionAgainst_Away_Last10", "Away_AvgPossessionFor_Away_Last10",
        "Away_AvgRedCardsAgainst_Total_Last10", "Away_AvgRedCardsFor_Total_Last10", # Using Total last 10 as example
        "Away_AvgSavesAgainst_Away_Last10", "Away_AvgSavesFor_Away_Last10",
        "Away_AvgShotsAgainst_Away_Last10", "Away_AvgShotsFor_Away_Last10",
        "Away_AvgShotsInsideBoxAgainst_Away_Last10", "Away_AvgShotsInsideBoxFor_Away_Last10",
        "Away_AvgShotsOffTargetAgainst_Away_Last10", "Away_AvgShotsOffTargetFor_Away_Last10",
        "Away_AvgShotsOutsideBoxAgainst_Away_Last10", "Away_AvgShotsOutsideBoxFor_Away_Last10",
        "Away_AvgShotsTargetAgainst_Away_Last10", "Away_AvgShotsTargetFor_Away_Last10",
        "Away_AvgTotalPassesAgainst_Away_Last10", "Away_AvgTotalPassesFor_Away_Last10",
        "Away_AvgYellowCardsAgainst_Total_Last10", "Away_AvgYellowCardsFor_Total_Last10", # Using Total last 10 as example
        "Away_BTTS_Ratio_Away_Last10", "Away_CleanSheet_Ratio_Away_Last10",
        "Away_FormPoints_Away_Last10",

        "Home_AvgBlockedShotsAgainst_Home_Last10", "Home_AvgBlockedShotsFor_Home_Last10",
        "Home_AvgCornersAgainst_Home_Last10", "Home_AvgCornersFor_Home_Last10",
        "Home_AvgExpectedGoalsFor_Home_Last10", # Lagged xG is safer
        "Home_AvgFoulsAgainst_Home_Last10", "Home_AvgFoulsFor_Home_Last10",
        "Home_AvgGoalsConceded_Home_Last10", "Home_AvgGoalsScored_Home_Last10",
        "Home_AvgOffsidesAgainst_Home_Last10", "Home_AvgOffsidesFor_Home_Last10",
        "Home_AvgPassAccuracyAgainst_Home_Last10", "Home_AvgPassAccuracyFor_Home_Last10",
        "Home_AvgPossessionAgainst_Home_Last10", "Home_AvgPossessionFor_Home_Last10",
        "Home_AvgRedCardsAgainst_Total_Last10", "Home_AvgRedCardsFor_Total_Last10", # Using Total last 10 as example
        "Home_AvgSavesAgainst_Home_Last10", "Home_AvgSavesFor_Home_Last10",
        "Home_AvgShotsAgainst_Home_Last10", "Home_AvgShotsFor_Home_Last10",
        "Home_AvgShotsInsideBoxAgainst_Home_Last10", "Home_AvgShotsInsideBoxFor_Home_Last10",
        "Home_AvgShotsOffTargetAgainst_Home_Last10", "Home_AvgShotsOffTargetFor_Home_Last10",
        "Home_AvgShotsOutsideBoxAgainst_Home_Last10", "Home_AvgShotsOutsideBoxFor_Home_Last10",
        "Home_AvgShotsTargetAgainst_Home_Last10", "Home_AvgShotsTargetFor_Home_Last10",
        "Home_AvgTotalPassesAgainst_Home_Last10", "Home_AvgTotalPassesFor_Home_Last10",
        "Home_AvgYellowCardsAgainst_Total_Last10", "Home_AvgYellowCardsFor_Total_Last10", # Using Total last 10 as example
        "Home_BTTS_Ratio_Home_Last10", "Home_CleanSheet_Ratio_Home_Last10",
        "Home_FormPoints_Home_Last10",

        # League Averages (Can capture baseline expectations)
        "LeagueAvg_AwayCleanSheet_Ratio_Last10", "LeagueAvg_HomeCleanSheet_Ratio_Last10",
        "LeagueAvg_AwayGoalsScored_Last10", "LeagueAvg_HomeGoalsScored_Last10",
        "LeagueAvg_BTTS_Ratio_Last10", "LeagueAvg_TotalGoals_Last10",
    ])

    # --- Betting Odds Features ---
    odds_numerical_features: List[str] = Field(default_factory=lambda: [
        # Closing Odds (Often most predictive)
        "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5",
        "BWCH", "BWCD", "BWCA",
        "IWCH", "IWCD", "IWCA",
        "PSCH", "PSCD", "PSCA", # Pinnacle closing odds
        "WHCH", "WHCD", "WHCA",
        "VCCH", "VCCD", "VCCA",
        "MaxCH", "MaxCD", "MaxCA", "MaxC>2.5", "MaxC<2.5",
        "AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5",

        # Opening Odds (Can indicate initial market sentiment)
        # "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", # etc. for other bookies if needed

        # Asian Handicap Odds
        "AHh", # Handicap line
        "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA", "B365CAHH", "B365CAHA", "PAHH", "PAHA", # Closing AH odds

        # Other Odds (Less common, potentially less useful)
        "GBA", "GBD", "GBH", "SBA", "SBD", "SBH", "SJA", "SJD", "SJH",
        "BSA", "BSD", "BSH", # Bet&Win, Ladbrokes, etc. if available and reliable

        # Implied Probabilities from Odds (Can be engineered)
        # Example: 'ImpliedProbH_B365C', 'ImpliedProbOver25_AvgC'
        # These would be calculated in data_processing.py
    ])

    # --- Categorical Features ---
    # Few obviously useful categorical features beyond Team/League IDs, which might be better handled via target encoding or embeddings.
    # Referee, Formation (often sparse/missing) might be excluded initially.
    core_categorical_features: List[str] = Field(default_factory=list) # e.g., ['LeagueName'] if using it directly

    # --- Utility/Metadata (Not for direct model input unless engineered) ---
    metadata_columns: List[str] = Field(default_factory=lambda: [
        "Country", "Season", "Round", "VenueName", "VenueCity", "Timestamp", "StatusShort", "StatusLong", "StatusElapsed",
        "HomeTeamID", "AwayTeamID", "Referee" # IDs might be used for embeddings/target encoding later
    ])

    @property
    def all_target_cols(self) -> List[str]:
        return [self.target_home_goals, self.target_away_goals, self.target_result]

    @property
    def all_id_cols(self) -> List[str]:
         return [self.match_id_col, self.home_team_col, self.away_team_col, self.date_col, self.league_id_col]

    def get_required_columns(self, include_odds: bool) -> Set[str]:
        """Returns the set of all columns required based on config."""
        cols = set(self.all_id_cols) | set(self.all_target_cols) | set(self.core_numerical_features) | set(self.core_categorical_features)
        if include_odds:
            cols |= set(self.odds_numerical_features)
        return cols

    def get_feature_columns(self, include_odds: bool) -> List[str]:
        """Returns the list of columns to be used as features for the model."""
        features = self.core_numerical_features + self.core_categorical_features
        if include_odds:
            features += self.odds_numerical_features
        # Ensure no targets or IDs leak into features
        features = [f for f in features if f not in self.all_target_cols and f not in self.all_id_cols]
        return sorted(list(set(features))) # Return sorted unique list


# Specific configurations
class FeatureConfigWithOdds(BaseFeatureConfig):
    include_odds: bool = True

class FeatureConfigWithoutOdds(BaseFeatureConfig):
    include_odds: bool = False

# --- Helper function to get the correct config ---
def get_feature_config(include_odds: bool) -> BaseFeatureConfig:
    if include_odds:
        return FeatureConfigWithOdds()
    else:
        return FeatureConfigWithoutOdds()