# feature_calculator.py (Corrected Type Casting - Robust Approach)
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Constants ---
MONGO_STATS_TYPE_MAP_FOR_FEATURES = {
    'Shots on Goal': 'ShotsTarget', 'Shots off Goal': 'ShotsOffTarget', 'Total Shots': 'Shots',
    'Blocked Shots': 'BlockedShots', 'Shots insidebox': 'ShotsInsideBox', 'Shots outsidebox': 'ShotsOutsideBox',
    'Fouls': 'Fouls', 'Corner Kicks': 'Corners', 'Offsides': 'Offsides',
    'Ball Possession': 'Possession', 'Yellow Cards': 'YellowCards', 'Red Cards': 'RedCards',
    'Goalkeeper Saves': 'Saves', 'Total passes': 'TotalPasses', 'Passes accurate': 'PassesAccurate',
    'Passes %': 'PassAccuracy', 'expected_goals': 'ExpectedGoals',
}
ALL_CANONICAL_STATS_BASE = list(MONGO_STATS_TYPE_MAP_FOR_FEATURES.values()) + ['Goals']
STATS_FOR_ROLLING = sorted([f"{base}{suffix}" for base in ALL_CANONICAL_STATS_BASE for suffix in ['For', 'Against']])
STATS_FOR_ROLLING.extend(['Points', 'Is_W', 'Is_D', 'Is_L', 'BTTS_Flag', 'CleanSheet_Flag'])
DEFAULT_WINDOWS = [3, 5, 10, 15]

class FeatureCalculator:

    # --- _extract_match_stats_from_mongo_doc (Keep as is) ---
    def _extract_match_stats_from_mongo_doc(self, mongo_doc: Dict[str, Any], home_team_id: int, away_team_id: int) -> Dict[str, Optional[float]]:
        extracted_stats = {}
        # Use statistics_full first, fallback to statistics (if present)
        stats_list = mongo_doc.get('statistics_full', mongo_doc.get('statistics', []))
        if not isinstance(stats_list, list):
            # Check if it might be directly under fixture_details
            fd = mongo_doc.get('fixture_details', {})
            stats_list_fd = fd.get('statistics_full', fd.get('statistics', []))
            if isinstance(stats_list_fd, list):
                stats_list = stats_list_fd
            else:
                return extracted_stats # Still not found

        for team_stat_block in stats_list:
            if not isinstance(team_stat_block, dict): continue
            team_info = team_stat_block.get('team', {})
            current_team_id = team_info.get('id') if isinstance(team_info, dict) else None
            if not current_team_id: continue

            # Ensure IDs are compared as integers
            try:
                venue_context = 'Home' if int(current_team_id) == int(home_team_id) else ('Away' if int(current_team_id) == int(away_team_id) else None)
            except (ValueError, TypeError):
                venue_context = None # Skip if IDs cannot be compared

            if not venue_context: continue

            stats_items = team_stat_block.get('statistics', [])
            if not isinstance(stats_items, list): continue

            for stat_item in stats_items:
                if not isinstance(stat_item, dict): continue
                stat_type = stat_item.get('type')
                value = stat_item.get('value')
                target_base_name = MONGO_STATS_TYPE_MAP_FOR_FEATURES.get(stat_type)

                if target_base_name:
                    col_name = f"{venue_context}{target_base_name}" # e.g., HomeShotsTarget
                    processed_value = pd.NA
                    if value is not None:
                        if isinstance(value, str):
                             if '%' in value:
                                 try: processed_value = float(value.replace('%',''))
                                 except ValueError: pass
                             elif value.strip().lower() in ['n/a', '-', '', 'null']: pass # Leave as NA
                             else: processed_value = pd.to_numeric(value, errors='coerce')
                        elif isinstance(value, (int, float)):
                             processed_value = float(value) # Convert ints to float
                        else: # Handle other types like bool if necessary
                            processed_value = pd.to_numeric(value, errors='coerce')

                    # Only assign if conversion was successful (not NA)
                    if pd.notna(processed_value):
                        extracted_stats[col_name] = processed_value

        return extracted_stats


    # --- _build_historical_canonical_df (Keep as is, but ensure paths match DB) ---
    def _build_historical_canonical_df(self, historical_matches: List[Dict[str, Any]]) -> pd.DataFrame:
        if not historical_matches: return pd.DataFrame()
        canonical_rows = []; processed_fixture_ids = set()
        for doc in historical_matches:
            try:
                # !!! CRITICAL: Adjust these paths to match your 'matches' collection structure !!!
                fd = doc.get('fixture_details', {})
                fx = fd.get('fixture', {})
                lg = fd.get('league', {})
                tm = fd.get('teams', {})
                gl = fd.get('goals', {})
                sc = fd.get('score', {}); ht_score = sc.get('halftime', {})

                fixture_id = fx.get('id');
                if not fixture_id: continue
                try: fixture_id = int(fixture_id)
                except (ValueError, TypeError): continue
                if fixture_id in processed_fixture_ids: continue
                date_str = fx.get('date');
                if not date_str: continue
                match_date = pd.to_datetime(date_str, errors='coerce', utc=True);
                if pd.isna(match_date): continue
                home_team = tm.get('home', {}); away_team = tm.get('away', {})
                home_id = home_team.get('id'); away_id = away_team.get('id')
                home_name = home_team.get('name'); away_name = away_team.get('name')
                if not all([home_id, away_id, home_name, away_name]): continue
                try: home_id = int(home_id); away_id = int(away_id)
                except (ValueError, TypeError): continue
                fthg = pd.to_numeric(gl.get('home'), errors='coerce')
                ftag = pd.to_numeric(gl.get('away'), errors='coerce')
                if pd.isna(fthg) or pd.isna(ftag): continue
                ftr = 'H' if fthg > ftag else ('A' if fthg < ftag else 'D')
                home_result = 'W' if ftr == 'H' else ('L' if ftr == 'A' else 'D')
                away_result = 'L' if ftr == 'H' else ('W' if ftr == 'A' else 'D')
                home_points = 3 if home_result == 'W' else (1 if home_result == 'D' else 0)
                away_points = 3 if away_result == 'W' else (1 if away_result == 'D' else 0)
                match_stats = self._extract_match_stats_from_mongo_doc(doc, home_id, away_id)
                home_row = {'MatchID': fixture_id, 'Date': match_date, 'TeamID': home_id, 'OpponentID': away_id,'TeamName': home_name, 'Venue': 'Home', 'Result': home_result, 'Points': home_points,'GoalsFor': fthg, 'GoalsAgainst': ftag,}
                for base_stat, canonical_suffix in MONGO_STATS_TYPE_MAP_FOR_FEATURES.items():
                    home_stat_val = match_stats.get(f'Home{canonical_suffix}'); away_stat_val = match_stats.get(f'Away{canonical_suffix}')
                    if home_stat_val is not None: home_row[f'{canonical_suffix}For'] = home_stat_val
                    if away_stat_val is not None: home_row[f'{canonical_suffix}Against'] = away_stat_val
                canonical_rows.append(home_row)
                away_row = {'MatchID': fixture_id, 'Date': match_date, 'TeamID': away_id, 'OpponentID': home_id,'TeamName': away_name, 'Venue': 'Away', 'Result': away_result, 'Points': away_points,'GoalsFor': ftag, 'GoalsAgainst': fthg,}
                for base_stat, canonical_suffix in MONGO_STATS_TYPE_MAP_FOR_FEATURES.items():
                     home_stat_val = match_stats.get(f'Home{canonical_suffix}'); away_stat_val = match_stats.get(f'Away{canonical_suffix}')
                     if away_stat_val is not None: away_row[f'{canonical_suffix}For'] = away_stat_val
                     if home_stat_val is not None: away_row[f'{canonical_suffix}Against'] = home_stat_val
                canonical_rows.append(away_row)
                processed_fixture_ids.add(fixture_id)
            except Exception as e: logger.warning(f"Skipping historical doc build: {e}", exc_info=False); continue
        if not canonical_rows: return pd.DataFrame()
        df_canonical = pd.DataFrame(canonical_rows)

        # Ensure goals are numeric float
        df_canonical['GoalsFor'] = pd.to_numeric(df_canonical['GoalsFor'], errors='coerce')
        df_canonical['GoalsAgainst'] = pd.to_numeric(df_canonical['GoalsAgainst'], errors='coerce')

        # BTTS Flag (1 if both > 0, 0 if either is 0, NA if either is NA)
        df_canonical['BTTS_Flag'] = np.select(
            [
                (df_canonical['GoalsFor'] > 0) & (df_canonical['GoalsAgainst'] > 0),
                pd.notna(df_canonical['GoalsFor']) & pd.notna(df_canonical['GoalsAgainst']) # Condition for 0 if not NA
            ],
            [1.0, 0.0], # Use float for consistency with potential NAs
            default=np.nan
        ).astype(float) # Use standard float type

        # Clean Sheet Flag (1 if GoalsAgainst is 0, 0 if > 0, NA if NA)
        df_canonical['CleanSheet_Flag'] = np.select(
            [
                df_canonical['GoalsAgainst'] == 0,
                pd.notna(df_canonical['GoalsAgainst']) # Condition for 0 if not NA
            ],
            [1.0, 0.0], # Use float
            default=np.nan
        ).astype(float) # Use standard float type

        # W/D/L Flags
        df_canonical['Is_W'] = (df_canonical['Result'] == 'W').astype(float)
        df_canonical['Is_D'] = (df_canonical['Result'] == 'D').astype(float)
        df_canonical['Is_L'] = (df_canonical['Result'] == 'L').astype(float)

        # Ensure numeric types for rolling calculations
        numeric_cols = [col for col in STATS_FOR_ROLLING if col in df_canonical.columns]
        for col in numeric_cols:
            df_canonical[col] = pd.to_numeric(df_canonical[col], errors='coerce')

        # Sort by Team, then Date (Crucial!)
        df_canonical['Date'] = pd.to_datetime(df_canonical['Date']) # Ensure datetime
        df_canonical = df_canonical.sort_values(by=['TeamID', 'Date']).reset_index(drop=True)

        return df_canonical
    # --- calculate_rolling_features_for_match (Keep as is) ---
    def calculate_rolling_features_for_match(
        self, team_id: int, match_date: datetime, historical_canonical_df: pd.DataFrame,
        windows: List[int] = DEFAULT_WINDOWS
    ) -> Dict[str, Any]:
        assert isinstance(team_id, int); assert isinstance(match_date, datetime); assert isinstance(historical_canonical_df, pd.DataFrame)
        if match_date.tzinfo is None: match_date = match_date.replace(tzinfo=timezone.utc)
        team_history = historical_canonical_df[(historical_canonical_df['TeamID'] == team_id) & (historical_canonical_df['Date'] < match_date)].copy()
        if team_history.empty: return {}
        team_history = team_history.sort_values(by='Date', ascending=True)
        calculated_features: Dict[str, Any] = {}; hist_home = team_history[team_history['Venue'] == 'Home']; hist_away = team_history[team_history['Venue'] == 'Away']
        stats_to_avg = [s for s in STATS_FOR_ROLLING if s not in ['Points', 'Is_W', 'Is_D', 'Is_L', 'BTTS_Flag', 'CleanSheet_Flag', 'GoalsFor', 'GoalsAgainst'] and s in team_history.columns]
        ratio_flags = [s for s in ['BTTS_Flag', 'CleanSheet_Flag'] if s in team_history.columns]; points_col = 'Points' if 'Points' in team_history.columns else None
        wdl_flags = [s for s in ['Is_W', 'Is_D', 'Is_L'] if s in team_history.columns]; goals_for_col = 'GoalsFor' if 'GoalsFor' in team_history.columns else None
        goals_against_col = 'GoalsAgainst' if 'GoalsAgainst' in team_history.columns else None
        for W in windows:
            ws = f"_Last{W}"; min_p = 1
            for context_data, context_label in [(team_history, '_Total'), (hist_home, '_Home'), (hist_away, '_Away')]:
                context_data_window = context_data.tail(W); expected_feature_names = []
                if points_col: expected_feature_names.append(f'FormPoints{context_label}{ws}')
                for flag in wdl_flags: expected_feature_names.append(f'{flag[3:]}_Count{context_label}{ws}')
                for stat in stats_to_avg: expected_feature_names.append(f'Avg{stat}{context_label}{ws}')
                if goals_for_col: expected_feature_names.append(f'AvgGoalsScored{context_label}{ws}')
                if goals_against_col: expected_feature_names.append(f'AvgGoalsConceded{context_label}{ws}')
                for flag in ratio_flags: expected_feature_names.append(f'{flag.replace("_Flag", "")}_Ratio{context_label}{ws}')
                if context_data_window.empty:
                     for name in expected_feature_names: calculated_features[name] = np.nan
                     continue
                if points_col: calculated_features[f'FormPoints{context_label}{ws}'] = context_data_window[points_col].sum(min_count=min_p)
                for flag in wdl_flags: calculated_features[f'{flag[3:]}_Count{context_label}{ws}'] = pd.to_numeric(context_data_window[flag], errors='coerce').sum(min_count=min_p)
                for stat in stats_to_avg: calculated_features[f'Avg{stat}{context_label}{ws}'] = context_data_window[stat].mean()
                if goals_for_col: calculated_features[f'AvgGoalsScored{context_label}{ws}'] = context_data_window[goals_for_col].mean()
                else: calculated_features[f'AvgGoalsScored{context_label}{ws}'] = np.nan
                if goals_against_col: calculated_features[f'AvgGoalsConceded{context_label}{ws}'] = context_data_window[goals_against_col].mean()
                else: calculated_features[f'AvgGoalsConceded{context_label}{ws}'] = np.nan
                for flag in ratio_flags:
                     feature_name = f'{flag.replace("_Flag", "")}_Ratio{context_label}{ws}'
                     if flag in context_data_window.columns: calculated_features[feature_name] = context_data_window[flag].mean()
                     else: calculated_features[feature_name] = np.nan
        return calculated_features

    # --- calculate_league_rolling_features_for_match (Keep as is) ---
    def calculate_league_rolling_features_for_match(
        self, league_id: int, season: int, match_date: datetime, historical_matches_all_leagues: pd.DataFrame,
        windows: List[int] = DEFAULT_WINDOWS
        ) -> Dict[str, Any]:
        assert isinstance(league_id, int); assert isinstance(season, int); assert isinstance(match_date, datetime); assert isinstance(historical_matches_all_leagues, pd.DataFrame)
        if match_date.tzinfo is None: match_date = match_date.replace(tzinfo=timezone.utc)
        league_history = historical_matches_all_leagues[(historical_matches_all_leagues['LeagueID'] == league_id) & (historical_matches_all_leagues['Season'] == season) & (historical_matches_all_leagues['Date'] < match_date)].copy()
        if league_history.empty: return {}
        league_history['FTHG'] = pd.to_numeric(league_history['FTHG'], errors='coerce'); league_history['FTAG'] = pd.to_numeric(league_history['FTAG'], errors='coerce')
        league_history.dropna(subset=['FTHG', 'FTAG'], inplace=True);
        if league_history.empty: return {}
        league_history['League_TotalGoals'] = league_history['FTHG'] + league_history['FTAG']
        league_history['League_BTTS_Flag'] = ((league_history['FTHG'] > 0) & (league_history['FTAG'] > 0)).astype(float)
        league_history['League_HomeCleanSheet_Flag'] = (league_history['FTAG'] == 0).astype(float); league_history['League_AwayCleanSheet_Flag'] = (league_history['FTHG'] == 0).astype(float)
        league_history['League_AnyCleanSheet_Flag'] = ((league_history['FTHG'] == 0) | (league_history['FTAG'] == 0)).astype(float)
        league_history = league_history.sort_values(by='Date', ascending=True)
        calculated_features: Dict[str, Any] = {}
        league_metrics_to_avg = ['FTHG', 'FTAG', 'League_TotalGoals', 'League_BTTS_Flag', 'League_HomeCleanSheet_Flag', 'League_AwayCleanSheet_Flag', 'League_AnyCleanSheet_Flag']
        
        for W in windows:
            ws = f"_Last{W}"
            min_p = 1
            window_data = league_history.tail(W)
            expected_feature_names = []
            
            for metric in league_metrics_to_avg:
                base_name = metric.replace('League_', '').replace('_Flag', '_Ratio')
                
                if base_name == 'FTHG':
                    base_name = 'HomeGoals'
                elif base_name == 'FTAG':
                    base_name = 'AwayGoals'
                elif base_name == 'HomeCleanSheet_Ratio':
                    base_name = 'HomeCleanSheetRatio'
                elif base_name == 'AwayCleanSheet_Ratio':
                    base_name = 'AwayCleanSheetRatio'
                elif base_name == 'AnyCleanSheet_Ratio':
                    base_name = 'AnyCleanSheetRatio'
                
                feature_name = f"LeagueAvg_{base_name}{ws}"
                expected_feature_names.append(feature_name)
            
            if window_data.empty:
                for name in expected_feature_names:
                    calculated_features[name] = np.nan
                continue
                
            for metric in league_metrics_to_avg:
                base_name = metric.replace('League_', '').replace('_Flag', '_Ratio')
                
                if base_name == 'FTHG':
                    base_name = 'HomeGoals'
                elif base_name == 'FTAG':
                    base_name = 'AwayGoals'
                elif base_name == 'HomeCleanSheet_Ratio':
                    base_name = 'HomeCleanSheetRatio'
                elif base_name == 'AwayCleanSheet_Ratio':
                    base_name = 'AwayCleanSheetRatio'
                elif base_name == 'AnyCleanSheet_Ratio':
                    base_name = 'AnyCleanSheetRatio'
                
                feature_name = f"LeagueAvg_{base_name}{ws}"
                
                if metric in window_data.columns:
                    calculated_features[feature_name] = window_data[metric].mean()
                else:
                    calculated_features[feature_name] = np.nan
                    
        return calculated_features

    # --- _create_match_id (Keep as is) ---
    def _create_match_id(self, match_date: datetime, home_team_name: str, away_team_name: str) -> Optional[str]:
        assert isinstance(match_date, datetime); assert home_team_name and isinstance(home_team_name, str); assert away_team_name and isinstance(away_team_name, str)
        try:
            if match_date.tzinfo is None: date_utc = match_date.replace(tzinfo=timezone.utc)
            else: date_utc = match_date.astimezone(timezone.utc)
            date_str = date_utc.strftime('%Y%m%d')
            home_cleaned = re.sub(r'\W+', '', home_team_name); away_cleaned = re.sub(r'\W+', '', away_team_name)
            if not home_cleaned or not away_cleaned: return None
            return f"{date_str}_{home_cleaned}_{away_cleaned}"
        except Exception as e: logger.error(f"Error creating MatchID: {e}", exc_info=False); return None

    # --- final_data_assembly (Robust Type Handling) ---
    def final_data_assembly(
        self, base_match_info: Dict[str, Any], raw_api_data: Dict[str, Any],
        elo_data: Dict[str, Optional[int]], team_features: Dict[str, Any],
        league_features: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Combines data, performs cleaning and type casting. Uses robust casting.
        """
        final_doc = {}
        assert isinstance(base_match_info, dict), "base_match_info must be a dictionary"
        assert isinstance(raw_api_data, dict), "raw_api_data must be a dictionary"
        assert isinstance(elo_data, dict), "elo_data must be a dictionary"
        assert isinstance(team_features, dict), "team_features must be a dictionary"
        assert isinstance(league_features, dict), "league_features must be a dictionary"

        try:
            # --- Extract Core Fields (adjust paths based on YOUR 'matches' structure) ---
            # Try to determine structure dynamically - this adds complexity but might be needed
            if 'fixture_details' in base_match_info:
                 fd = base_match_info.get('fixture_details', {})
            elif 'basic_info' in base_match_info and isinstance(base_match_info['basic_info'], list) and base_match_info['basic_info']:
                 # Structure from FixtureDetailsFetcher V1?
                 basic_info = base_match_info['basic_info'][0]
                 fd = { # Reconstruct the expected 'fixture_details' structure
                     'fixture': basic_info.get('fixture', {}), 'league': basic_info.get('league', {}),
                     'teams': basic_info.get('teams', {}), 'goals': basic_info.get('goals', {}),
                     'score': basic_info.get('score', {}), # Need score too
                 }
            elif 'fixture_details_raw' in base_match_info:
                  # Structure from FixtureDetailsFetcher V2?
                  raw_data = base_match_info.get('fixture_details_raw', {})
                  if 'basic_info' in raw_data and isinstance(raw_data['basic_info'], list) and raw_data['basic_info']:
                      basic_info = raw_data['basic_info'][0]
                      fd = {
                         'fixture': basic_info.get('fixture', {}), 'league': basic_info.get('league', {}),
                         'teams': basic_info.get('teams', {}), 'goals': basic_info.get('goals', {}),
                         'score': basic_info.get('score', {}), # Need score too
                     }
                  else: fd = {} # Could not find suitable structure
            else:
                 # Assume flatter structure as last resort
                 logger.warning("Assuming flatter structure for base_match_info in final_data_assembly")
                 fd = {
                     'fixture': base_match_info.get('match_info', {}), # Map 'match_info' to 'fixture'
                     'league': { # Reconstruct league
                         'id': base_match_info.get('league_id'),
                         'name': base_match_info.get('league_name'),
                         'country': base_match_info.get('country'),
                         'season': base_match_info.get('season'),
                         'round': base_match_info.get('round'),
                     },
                     'teams': { # Reconstruct teams
                         'home': base_match_info.get('home_team'),
                         'away': base_match_info.get('away_team'),
                     },
                     'goals': { # Reconstruct goals
                         'home': base_match_info.get('home_goals'), # Assuming these keys
                         'away': base_match_info.get('away_goals'), # Assuming these keys
                     },
                     'score': { # Reconstruct score
                         'halftime': {
                             'home': base_match_info.get('ht_home_goals'), # Assuming these keys
                             'away': base_match_info.get('ht_away_goals'), # Assuming these keys
                         },
                         # Add fulltime/extratime if available in flat structure
                     }
                 }

            # Now extract using the determined 'fd' structure
            fx = fd.get('fixture', {})
            lg = fd.get('league', {})
            tm = fd.get('teams', {})
            gl = fd.get('goals', {})
            sc = fd.get('score', {}); ht_score = sc.get('halftime', {})
            vn = fx.get('venue', {}); st = fx.get('status', {})

            final_doc['fixture_id'] = fx.get('id')
            raw_date = fx.get('date')
            final_doc['Date'] = pd.to_datetime(raw_date, errors='coerce', utc=True)
            final_doc['Timestamp'] = pd.to_numeric(fx.get('timestamp'), errors='coerce')

            if pd.isna(final_doc['Date']):
                 logger.error(f"Could not parse date '{raw_date}' for fixture {final_doc['fixture_id']}. Date will be NaT/None.")

            home_name_base = tm.get('home', {}).get('name')
            away_name_base = tm.get('away', {}).get('name')
            if pd.notna(final_doc.get('Date')) and home_name_base and away_name_base:
                 final_doc['MatchID'] = self._create_match_id(final_doc['Date'], home_name_base, away_name_base)
            else: final_doc['MatchID'] = None

            try: final_doc['LeagueID'] = int(lg.get('id')) if lg.get('id') is not None else None
            except (ValueError, TypeError): final_doc['LeagueID'] = None
            try: final_doc['HomeTeamID'] = int(tm.get('home', {}).get('id')) if tm.get('home', {}).get('id') is not None else None
            except (ValueError, TypeError): final_doc['HomeTeamID'] = None
            try: final_doc['AwayTeamID'] = int(tm.get('away', {}).get('id')) if tm.get('away', {}).get('id') is not None else None
            except (ValueError, TypeError): final_doc['AwayTeamID'] = None
            try: final_doc['Season'] = int(lg.get('season')) if lg.get('season') is not None else None
            except (ValueError, TypeError): final_doc['Season'] = None

            final_doc['LeagueName'] = lg.get('name'); final_doc['Country'] = lg.get('country'); final_doc['Round'] = lg.get('round')
            final_doc['HomeTeam'] = home_name_base; final_doc['AwayTeam'] = away_name_base

            final_doc['FTHG'] = pd.to_numeric(gl.get('home'), errors='coerce').astype(float)
            final_doc['FTAG'] = pd.to_numeric(gl.get('away'), errors='coerce').astype(float)

            if pd.notna(final_doc['FTHG']) and pd.notna(final_doc['FTAG']):
                if final_doc['FTHG'] > final_doc['FTAG']: final_doc['FTR'] = 'H'
                elif final_doc['FTHG'] < final_doc['FTAG']: final_doc['FTR'] = 'A'
                else: final_doc['FTR'] = 'D'
            else: final_doc['FTR'] = None

            final_doc['HTHG'] = pd.to_numeric(ht_score.get('home'), errors='coerce').astype(float)
            final_doc['HTAG'] = pd.to_numeric(ht_score.get('away'), errors='coerce').astype(float)
            # Calculate HTR if possible...

            final_doc['Referee'] = fx.get('referee'); final_doc['VenueName'] = vn.get('name'); final_doc['VenueCity'] = vn.get('city')
            final_doc['StatusLong'] = st.get('long'); final_doc['StatusShort'] = st.get('short')
            final_doc['StatusElapsed'] = pd.to_numeric(st.get('elapsed'), errors='coerce').astype(float)

        except Exception as e:
             logger.error(f"Error during extraction/base casting for fixture {final_doc.get('fixture_id', 'N/A')}: {e}", exc_info=True)
             return {}

        # Add other data blocks
        final_doc.update(elo_data)
        final_doc.update(team_features)
        final_doc.update(league_features)

        # --- Final Cleaning and Type Conversion ---
        cleaned_doc = {}
        integer_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'StatusElapsed',
                        'HomeTeamID', 'AwayTeamID', 'LeagueID', 'Season',
                        'fixture_id', 'Timestamp']
        # --- Ensure 'fixture_id' is treated as string if needed for _id ---
        if 'fixture_id' in final_doc and final_doc['fixture_id'] is not None:
             final_doc['fixture_id'] = str(final_doc['fixture_id']) # Convert to string

        for k, v in final_doc.items():
            if pd.isna(v): cleaned_doc[k] = None
            elif k in integer_cols and k != 'fixture_id': # Exclude fixture_id if it should remain string
                try:
                     if v is not None: cleaned_doc[k] = int(float(v))
                     else: cleaned_doc[k] = None
                except (ValueError, TypeError):
                     logger.warning(f"Could not convert column '{k}' value '{v}' to int for fixture {final_doc.get('fixture_id')}, setting to None.")
                     cleaned_doc[k] = None
            elif isinstance(v, (np.int64, np.int32)): cleaned_doc[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)): cleaned_doc[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, pd.Timestamp): cleaned_doc[k] = v.to_pydatetime()
            elif isinstance(v, pd.BooleanDtype): cleaned_doc[k] = bool(v) if not pd.isna(v) else None
            elif isinstance(v, pd.StringDtype): cleaned_doc[k] = str(v) if not pd.isna(v) else None
            else: cleaned_doc[k] = v # Keep other types (like strings, dicts) as they are

        # Final check for identifier AFTER cleaning
        if not cleaned_doc.get('MatchID') and not cleaned_doc.get('fixture_id'):
             logger.error("Failed to generate or find MatchID/fixture_id during final assembly after cleaning.")
             return {}

        return cleaned_doc

# Singleton instance
feature_calculator = FeatureCalculator()