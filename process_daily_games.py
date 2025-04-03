import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import math

class EnhancedSoccerMatchProcessor:
    """
    Advanced processor for soccer match data with enhanced metrics calculation
    and improved structure for analytics and modeling.
    """
    
    def __init__(self, input_dir: str = "daily_games", output_dir: str = "processed_matches"):
        """Initialize with input and output directories."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_value(self, value: Any) -> Any:
        """Clean string percentage values and convert to float."""
        if isinstance(value, str):
            # Remove % sign and convert to float
            if "%" in value:
                return float(value.strip().replace("%", "")) / 100
            # Convert string numbers to float
            try:
                return float(value)
            except ValueError:
                return value
        return value
    
    def calculate_form_points(self, form_string: Optional[str] = None, 
                              wins: Optional[int] = None, 
                              draws: Optional[int] = None, 
                              losses: Optional[int] = None, 
                              num_games: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate form metrics based on provided wins/draws/losses or a form string.
        
        Args:
            form_string: String of form results (e.g., "WDLWW"). Used if W/D/L/num_games not provided.
            wins: Number of wins.
            draws: Number of draws.
            losses: Number of losses.
            num_games: Total number of games considered.
            
        Returns:
            Dictionary with form metrics (points, ppg, win_rate, efficiency, trend).
        """
        form_metrics = {
            "wins": wins, "draws": draws, "losses": losses, "num_games": num_games,
            "points": 0, "points_per_match": 0, "win_rate": 0, 
            "efficiency": 0, "trend": "neutral" # Default trend
        }

        if wins is not None and draws is not None and losses is not None and num_games is not None and num_games > 0:
            total_matches = num_games
            form_metrics["points"] = (wins * 3) + draws
            form_metrics["points_per_match"] = form_metrics["points"] / total_matches
            form_metrics["win_rate"] = wins / total_matches
            max_possible_points = total_matches * 3
            form_metrics["efficiency"] = form_metrics["points"] / max_possible_points if max_possible_points > 0 else 0
            # Trend calculation requires historical context, which is complex here. 
            # We'll rely on statarea's direct trend if available, or keep neutral.

        elif form_string:
             # Fallback to original form string calculation if specific counts aren't available
            wins = form_string.count('W')
            draws = form_string.count('D')
            losses = form_string.count('L')
            total_matches = wins + draws + losses
            form_metrics.update({ "wins": wins, "draws": draws, "losses": losses, "num_games": total_matches })
            
            if total_matches > 0:
                form_metrics["points"] = (wins * 3) + draws
                form_metrics["points_per_match"] = form_metrics["points"] / total_matches
                form_metrics["win_rate"] = wins / total_matches
                max_possible_points = total_matches * 3
                form_metrics["efficiency"] = form_metrics["points"] / max_possible_points if max_possible_points > 0 else 0

                # Simple trend based on last 5/6 games form string
            if len(form_string) >= 5:
                recent_3_matches = form_string[:3]
                prev_matches = form_string[3:6] if len(form_string) >= 6 else form_string[3:] # Use 2 or 3 previous
                
                recent_points = (recent_3_matches.count('W') * 3) + recent_3_matches.count('D')
                prev_points = (prev_matches.count('W') * 3) + prev_matches.count('D')
                
                if recent_points > prev_points:
                    form_metrics["trend"] = "positive"
                elif recent_points < prev_points:
                    form_metrics["trend"] = "negative"
                else:
                    form_metrics["trend"] = "neutral"
        
        # Round floats for cleaner output
        for key in ["points_per_match", "win_rate", "efficiency"]:
             if isinstance(form_metrics[key], float):
                 form_metrics[key] = round(form_metrics[key], 3)
        
        return form_metrics
    
    def calculate_expected_goals(self, team_stats: Dict[str, Any], venue: str) -> Dict[str, float]:
        """
        Calculate expected goals (xG) using statarea data primarily, falling back to mongodb stats.
        
        Args:
            team_stats: Processed team statistics dictionary
            venue: "home" or "away"
            
        Returns:
            Dictionary with base_xg and adjusted_xg
        """
        xg_metrics = {"base_xg": 0.0, "adjusted_xg": 0.0}
        
        # Try using statarea 15-game data first
        statarea_venue_key = "home" if venue == "home" else "away"
        statarea_data = team_stats.get("statarea_analysis", {}).get(statarea_venue_key, {}).get("last_15_games", {})

        if statarea_data:
            avg_goals_scored = statarea_data.get("avg_goals_scored")
            scoring_prob = statarea_data.get("scoring_probability") # Already a float 0-1
            
            if avg_goals_scored is not None:
                 xg_metrics["base_xg"] = avg_goals_scored
                 # Use scoring probability to adjust, if available and valid
                 if scoring_prob is not None and 0 <= scoring_prob <= 1:
                     xg_metrics["adjusted_xg"] = avg_goals_scored * scoring_prob
                 else:
                     xg_metrics["adjusted_xg"] = avg_goals_scored # Fallback if prob is invalid/missing
            
        else:
            # Fallback to mongodb stats if statarea 15-game data is missing
            mongodb_goals = team_stats.get("goals_overall_mongodb", {}).get("for", {})
            avg_goals_scored = mongodb_goals.get(f"per_game_{venue}", 0)
            xg_metrics["base_xg"] = avg_goals_scored
            # No direct scoring probability in mongodb stats, so adjusted = base
            xg_metrics["adjusted_xg"] = avg_goals_scored
            
        # Round results
        xg_metrics["base_xg"] = round(xg_metrics["base_xg"], 3)
        xg_metrics["adjusted_xg"] = round(xg_metrics["adjusted_xg"], 3)
        
        return xg_metrics
    
    def calculate_match_outcome_probabilities(self, home_team_stats: Dict[str, Any], away_team_stats: Dict[str, Any], 
                                             h2h_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate match outcome probabilities using Poisson distribution, 
        adjusted by defensive strength and potentially blended with H2H/Statarea data.
        Now includes basic and detailed probabilities.
        
        Args:
            home_team_stats: Processed home team statistics
            away_team_stats: Processed away team statistics
            h2h_stats: Processed H2H statistics
            
        Returns:
            Dictionary containing various probability sets.
        """
        # --- 1. Get Base Expected Goals (Adjusted xG) ---
        home_xg = self.calculate_expected_goals(home_team_stats, "home").get("adjusted_xg", 0)
        away_xg = self.calculate_expected_goals(away_team_stats, "away").get("adjusted_xg", 0)
        
        # --- 2. Factor in Opponent's Defense ---
        home_statarea_conceded = home_team_stats.get("statarea_analysis", {}).get("home", {}).get("last_15_games", {}).get("avg_goals_conceded")
        away_statarea_conceded = away_team_stats.get("statarea_analysis", {}).get("away", {}).get("last_15_games", {}).get("avg_goals_conceded")

        home_conceded_avg = home_statarea_conceded if home_statarea_conceded is not None else home_team_stats.get("goals_overall_mongodb", {}).get("against", {}).get("per_game_home", 1)
        away_conceded_avg = away_statarea_conceded if away_statarea_conceded is not None else away_team_stats.get("goals_overall_mongodb", {}).get("against", {}).get("per_game_away", 1)
        
        home_defense_strength = 1 / (home_conceded_avg if home_conceded_avg > 0 else 1)
        away_defense_strength = 1 / (away_conceded_avg if away_conceded_avg > 0 else 1)

        final_home_xg = max(0, home_xg * away_defense_strength)
        final_away_xg = max(0, away_xg * home_defense_strength)

        # --- 3. Calculate Basic Probabilities (1X2, O/U, BTTS) using final XG ---
        poisson_probs_1x2 = self.poisson_match_outcome(final_home_xg, final_away_xg)
        
        basic_probs = {
            "home_win": poisson_probs_1x2["home_win"],
            "draw": poisson_probs_1x2["draw"],
            "away_win": poisson_probs_1x2["away_win"],
            "over_1.5": self.calculate_over_under_probability(final_home_xg, final_away_xg, 1.5),
            "over_2.5": self.calculate_over_under_probability(final_home_xg, final_away_xg, 2.5),
            "over_3.5": self.calculate_over_under_probability(final_home_xg, final_away_xg, 3.5),
            "over_4.5": self.calculate_over_under_probability(final_home_xg, final_away_xg, 4.5), # Added O/U 4.5
            "btts_yes": self.calculate_btts_probability(final_home_xg, final_away_xg)
        }
        # Calculate corresponding 'under' and 'no' probabilities
        basic_probs["under_1.5"] = round(1.0 - basic_probs["over_1.5"], 3)
        basic_probs["under_2.5"] = round(1.0 - basic_probs["over_2.5"], 3)
        basic_probs["under_3.5"] = round(1.0 - basic_probs["over_3.5"], 3)
        basic_probs["under_4.5"] = round(1.0 - basic_probs["over_4.5"], 3) # Added U/O 4.5
        basic_probs["btts_no"] = round(1.0 - basic_probs["btts_yes"], 3)

        # --- ADD DOUBLE CHANCE PROBABILITIES TO BASIC_PROBS ---
        basic_probs["home_draw"] = round(basic_probs["home_win"] + basic_probs["draw"], 3)
        basic_probs["away_draw"] = round(basic_probs["away_win"] + basic_probs["draw"], 3)
        basic_probs["home_away"] = round(basic_probs["home_win"] + basic_probs["away_win"], 3)
        # --- END ADDITION ---

        # --- 4. Calculate Detailed/Combined Probabilities ---
        detailed_probs = self.calculate_detailed_probabilities(final_home_xg, final_away_xg)

        # --- 5. Compile Final Output ---
        all_outcome_probs = {
            "final_xg_home": round(final_home_xg, 3),
            "final_xg_away": round(final_away_xg, 3),
            "most_likely_score": self.get_most_likely_scoreline(final_home_xg, final_away_xg),
            "basic_probabilities": basic_probs, # Now includes double chance and O/U 4.5
            "combined_probabilities": detailed_probs
        }

        return all_outcome_probs
    
    def poisson_match_outcome(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """
        Calculate match outcome probabilities using Poisson distribution.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            
        Returns:
            Dictionary with outcome probabilities
        """
        home_win_prob = 0
        away_win_prob = 0
        draw_prob = 0
        
        # Calculate probabilities for scorelines up to 5 goals per team
        for home_goals in range(6):
            for away_goals in range(6):
                # Calculate Poisson probability for this scoreline
                prob = self.poisson_probability(home_goals, home_xg) * self.poisson_probability(away_goals, away_xg)
                
                # Add to appropriate outcome
                if home_goals > away_goals:
                    home_win_prob += prob
                elif away_goals > home_goals:
                    away_win_prob += prob
                else:
                    draw_prob += prob
        
        # Ensure probabilities sum to 1
        total_prob = home_win_prob + away_win_prob + draw_prob
        
        return {
            "home_win": round(home_win_prob / total_prob, 3),
            "draw": round(draw_prob / total_prob, 3),
            "away_win": round(away_win_prob / total_prob, 3)
        }
    
    def poisson_probability(self, k: int, lamb: float) -> float:
        """Calculate Poisson probability mass function."""
        if lamb <= 0:
            return 1.0 if k == 0 else 0.0
        
        return (lamb ** k) * math.exp(-lamb) / math.factorial(k)
    
    def calculate_over_under_probability(self, home_xg: float, away_xg: float, threshold: float) -> float:
        """Calculate probability of total goals being over a threshold using adjusted xG."""
        over_prob = 0
        # Iterate through possible scores for home and away team
        for home_goals in range(10): # Max 9 goals considered for home
            for away_goals in range(10): # Max 9 goals considered for away
                if home_goals + away_goals > threshold:
                     # Calculate Poisson probability for this scoreline
                    prob = self.poisson_probability(home_goals, home_xg) * self.poisson_probability(away_goals, away_xg)
                    over_prob += prob
        
        # Alternative using total_xg (less accurate for discrete threshold)
        # total_xg = home_xg + away_xg
        # for i in range(int(threshold + 0.5) + 1, 15): # Limit to 15 goals max
        #     over_prob += self.poisson_probability(i, total_xg)
        
        return round(over_prob, 3)
    
    def calculate_btts_probability(self, home_xg: float, away_xg: float) -> float:
        """Calculate both teams to score probability."""
        # Probability of home team not scoring
        home_no_score = self.poisson_probability(0, home_xg)
        
        # Probability of away team not scoring
        away_no_score = self.poisson_probability(0, away_xg)
        
        # Probability of at least one team not scoring
        no_btts_prob = home_no_score + away_no_score - (home_no_score * away_no_score)
        
        # Probability of both teams scoring
        btts_prob = 1 - no_btts_prob
        
        return round(btts_prob, 3)
    
    def calculate_detailed_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """
        Calculate detailed combined outcome probabilities using the Poisson grid.
        Includes Single Outcome + Variable (Goals/BTTS) and 
        Double Chance + Variable (Goals O/U 1.5, 2.5, 3.5 & BTTS). Uses clearer naming for double chance keys.
        
        Args:
            home_xg: Final adjusted expected goals for home team.
            away_xg: Final adjusted expected goals for away team.
            
        Returns:
            Dictionary structured by outcome type containing combined probabilities.
        """
        # Initialize probability accumulators using clearer double chance names
        probs = {
            # --- Single Outcome + Variable ---
            "home_win_and_over_1.5": 0.0, "home_win_and_under_1.5": 0.0,
            "home_win_and_over_2.5": 0.0, "home_win_and_under_2.5": 0.0,
            "home_win_and_over_3.5": 0.0, "home_win_and_under_3.5": 0.0,
            "home_win_and_btts_yes": 0.0, "home_win_and_btts_no": 0.0,
            
            "draw_and_over_1.5": 0.0, "draw_and_under_1.5": 0.0, 
            "draw_and_over_2.5": 0.0, "draw_and_under_2.5": 0.0,
            "draw_and_over_3.5": 0.0, "draw_and_under_3.5": 0.0,
            "draw_and_btts_yes": 0.0, "draw_and_btts_no": 0.0, 

            "away_win_and_over_1.5": 0.0, "away_win_and_under_1.5": 0.0,
            "away_win_and_over_2.5": 0.0, "away_win_and_under_2.5": 0.0,
            "away_win_and_over_3.5": 0.0, "away_win_and_under_3.5": 0.0,
            "away_win_and_btts_yes": 0.0, "away_win_and_btts_no": 0.0,

            # --- Double Chance + Variable (Clearer Names) ---
            "home_or_draw_and_over_1.5": 0.0, "home_or_draw_and_under_1.5": 0.0, 
            "home_or_draw_and_over_2.5": 0.0, "home_or_draw_and_under_2.5": 0.0,
            "home_or_draw_and_over_3.5": 0.0, "home_or_draw_and_under_3.5": 0.0, 
            "home_or_draw_and_btts_yes": 0.0, "home_or_draw_and_btts_no": 0.0,
            
            "away_or_draw_and_over_1.5": 0.0, "away_or_draw_and_under_1.5": 0.0, 
            "away_or_draw_and_over_2.5": 0.0, "away_or_draw_and_under_2.5": 0.0,
            "away_or_draw_and_over_3.5": 0.0, "away_or_draw_and_under_3.5": 0.0, 
            "away_or_draw_and_btts_yes": 0.0, "away_or_draw_and_btts_no": 0.0,

            "home_or_away_and_over_1.5": 0.0, "home_or_away_and_under_1.5": 0.0, 
            "home_or_away_and_over_2.5": 0.0, "home_or_away_and_under_2.5": 0.0,
            "home_or_away_and_over_3.5": 0.0, "home_or_away_and_under_3.5": 0.0, 
            "home_or_away_and_btts_yes": 0.0, "home_or_away_and_btts_no": 0.0,
        }
        
        total_prob_check = 0.0 # For validation

        # Iterate through scorelines to calculate joint probabilities
        for hg in range(10): # Home goals 0-9
            for ag in range(10): # Away goals 0-9
                prob = self.poisson_probability(hg, home_xg) * self.poisson_probability(ag, away_xg)
                total_prob_check += prob
                
                total_goals = hg + ag
                is_over_1_5 = total_goals > 1.5
                is_over_2_5 = total_goals > 2.5
                is_over_3_5 = total_goals > 3.5
                is_btts_yes = hg > 0 and ag > 0

                # --- Accumulate Single Outcome + Variable ---
                if hg > ag: # Home Win
                    if is_over_1_5: probs["home_win_and_over_1.5"] += prob
                    else: probs["home_win_and_under_1.5"] += prob
                    if is_over_2_5: probs["home_win_and_over_2.5"] += prob
                    else: probs["home_win_and_under_2.5"] += prob
                    if is_over_3_5: probs["home_win_and_over_3.5"] += prob
                    else: probs["home_win_and_under_3.5"] += prob
                    if is_btts_yes: probs["home_win_and_btts_yes"] += prob
                    else: probs["home_win_and_btts_no"] += prob 
                
                elif hg == ag: # Draw
                    if is_over_1_5: probs["draw_and_over_1.5"] += prob
                    else: probs["draw_and_under_1.5"] += prob 
                    if is_over_2_5: probs["draw_and_over_2.5"] += prob
                    else: probs["draw_and_under_2.5"] += prob
                    if is_over_3_5: probs["draw_and_over_3.5"] += prob
                    else: probs["draw_and_under_3.5"] += prob
                    if is_btts_yes: probs["draw_and_btts_yes"] += prob 
                    else: probs["draw_and_btts_no"] += prob # Includes 0-0

                else: # Away Win (ag > hg)
                    if is_over_1_5: probs["away_win_and_over_1.5"] += prob
                    else: probs["away_win_and_under_1.5"] += prob
                    if is_over_2_5: probs["away_win_and_over_2.5"] += prob
                    else: probs["away_win_and_under_2.5"] += prob
                    if is_over_3_5: probs["away_win_and_over_3.5"] += prob
                    else: probs["away_win_and_under_3.5"] += prob
                    if is_btts_yes: probs["away_win_and_btts_yes"] += prob
                    else: probs["away_win_and_btts_no"] += prob 

        # --- Calculate Double Chance + Variable Combinations (Using clearer names) ---
        # Home/Draw (1X)
        probs["home_or_draw_and_over_1.5"] = probs["home_win_and_over_1.5"] + probs["draw_and_over_1.5"] 
        probs["home_or_draw_and_under_1.5"] = probs["home_win_and_under_1.5"] + probs["draw_and_under_1.5"] 
        probs["home_or_draw_and_over_2.5"] = probs["home_win_and_over_2.5"] + probs["draw_and_over_2.5"]
        probs["home_or_draw_and_under_2.5"] = probs["home_win_and_under_2.5"] + probs["draw_and_under_2.5"]
        probs["home_or_draw_and_over_3.5"] = probs["home_win_and_over_3.5"] + probs["draw_and_over_3.5"] 
        probs["home_or_draw_and_under_3.5"] = probs["home_win_and_under_3.5"] + probs["draw_and_under_3.5"] 
        probs["home_or_draw_and_btts_yes"] = probs["home_win_and_btts_yes"] + probs["draw_and_btts_yes"]
        probs["home_or_draw_and_btts_no"] = probs["home_win_and_btts_no"] + probs["draw_and_btts_no"]

        # Away/Draw (X2)
        probs["away_or_draw_and_over_1.5"] = probs["away_win_and_over_1.5"] + probs["draw_and_over_1.5"] 
        probs["away_or_draw_and_under_1.5"] = probs["away_win_and_under_1.5"] + probs["draw_and_under_1.5"] 
        probs["away_or_draw_and_over_2.5"] = probs["away_win_and_over_2.5"] + probs["draw_and_over_2.5"]
        probs["away_or_draw_and_under_2.5"] = probs["away_win_and_under_2.5"] + probs["draw_and_under_2.5"]
        probs["away_or_draw_and_over_3.5"] = probs["away_win_and_over_3.5"] + probs["draw_and_over_3.5"] 
        probs["away_or_draw_and_under_3.5"] = probs["away_win_and_under_3.5"] + probs["draw_and_under_3.5"] 
        probs["away_or_draw_and_btts_yes"] = probs["away_win_and_btts_yes"] + probs["draw_and_btts_yes"]
        probs["away_or_draw_and_btts_no"] = probs["away_win_and_btts_no"] + probs["draw_and_btts_no"]

        # Home/Away (12)
        probs["home_or_away_and_over_1.5"] = probs["home_win_and_over_1.5"] + probs["away_win_and_over_1.5"] 
        probs["home_or_away_and_under_1.5"] = probs["home_win_and_under_1.5"] + probs["away_win_and_under_1.5"] 
        probs["home_or_away_and_over_2.5"] = probs["home_win_and_over_2.5"] + probs["away_win_and_over_2.5"]
        probs["home_or_away_and_under_2.5"] = probs["home_win_and_under_2.5"] + probs["away_win_and_under_2.5"]
        probs["home_or_away_and_over_3.5"] = probs["home_win_and_over_3.5"] + probs["away_win_and_over_3.5"] 
        probs["home_or_away_and_under_3.5"] = probs["home_win_and_under_3.5"] + probs["away_win_and_under_3.5"] 
        probs["home_or_away_and_btts_yes"] = probs["home_win_and_btts_yes"] + probs["away_win_and_btts_yes"]
        probs["home_or_away_and_btts_no"] = probs["home_win_and_btts_no"] + probs["away_win_and_btts_no"]

        # Round final probabilities
        rounded_probs = {k: round(v, 3) for k, v in probs.items()}

        # --- Structure the output for clarity (using clearer double chance keys) ---
        structured_output = {
            "single_outcome_combinations": {
                "home_win": {
                    "over_1.5": rounded_probs["home_win_and_over_1.5"],
                    "under_1.5": rounded_probs["home_win_and_under_1.5"],
                    "over_2.5": rounded_probs["home_win_and_over_2.5"],
                    "under_2.5": rounded_probs["home_win_and_under_2.5"],
                    "over_3.5": rounded_probs["home_win_and_over_3.5"],
                    "under_3.5": rounded_probs["home_win_and_under_3.5"],
                    "btts_yes": rounded_probs["home_win_and_btts_yes"],
                    "btts_no": rounded_probs["home_win_and_btts_no"],
                },
                "draw": {
                    "over_1.5": rounded_probs["draw_and_over_1.5"],
                    "under_1.5": rounded_probs["draw_and_under_1.5"],
                    "over_2.5": rounded_probs["draw_and_over_2.5"],
                    "under_2.5": rounded_probs["draw_and_under_2.5"],
                    "over_3.5": rounded_probs["draw_and_over_3.5"],
                    "under_3.5": rounded_probs["draw_and_under_3.5"],
                    "btts_yes": rounded_probs["draw_and_btts_yes"],
                    "btts_no": rounded_probs["draw_and_btts_no"],
                },
                "away_win": {
                    "over_1.5": rounded_probs["away_win_and_over_1.5"],
                    "under_1.5": rounded_probs["away_win_and_under_1.5"],
                    "over_2.5": rounded_probs["away_win_and_over_2.5"],
                    "under_2.5": rounded_probs["away_win_and_under_2.5"],
                    "over_3.5": rounded_probs["away_win_and_over_3.5"],
                    "under_3.5": rounded_probs["away_win_and_under_3.5"],
                    "btts_yes": rounded_probs["away_win_and_btts_yes"],
                    "btts_no": rounded_probs["away_win_and_btts_no"],
                }
            },
            "double_chance_combinations": {
                 "home_or_draw": { # Updated key: 1X
                    "over_1.5": rounded_probs["home_or_draw_and_over_1.5"], 
                    "under_1.5": rounded_probs["home_or_draw_and_under_1.5"], 
                    "over_2.5": rounded_probs["home_or_draw_and_over_2.5"],
                    "under_2.5": rounded_probs["home_or_draw_and_under_2.5"],
                    "over_3.5": rounded_probs["home_or_draw_and_over_3.5"], 
                    "under_3.5": rounded_probs["home_or_draw_and_under_3.5"], 
                    "btts_yes": rounded_probs["home_or_draw_and_btts_yes"],
                    "btts_no": rounded_probs["home_or_draw_and_btts_no"],
                 },
                 "away_or_draw": { # Updated key: X2
                    "over_1.5": rounded_probs["away_or_draw_and_over_1.5"], 
                    "under_1.5": rounded_probs["away_or_draw_and_under_1.5"], 
                    "over_2.5": rounded_probs["away_or_draw_and_over_2.5"],
                    "under_2.5": rounded_probs["away_or_draw_and_under_2.5"],
                    "over_3.5": rounded_probs["away_or_draw_and_over_3.5"], 
                    "under_3.5": rounded_probs["away_or_draw_and_under_3.5"], 
                    "btts_yes": rounded_probs["away_or_draw_and_btts_yes"],
                    "btts_no": rounded_probs["away_or_draw_and_btts_no"],
                 },
                 "home_or_away": { # Updated key: 12
                    "over_1.5": rounded_probs["home_or_away_and_over_1.5"], 
                    "under_1.5": rounded_probs["home_or_away_and_under_1.5"], 
                    "over_2.5": rounded_probs["home_or_away_and_over_2.5"],
                    "under_2.5": rounded_probs["home_or_away_and_under_2.5"],
                    "over_3.5": rounded_probs["home_or_away_and_over_3.5"], 
                    "under_3.5": rounded_probs["home_or_away_and_under_3.5"], 
                    "btts_yes": rounded_probs["home_or_away_and_btts_yes"],
                    "btts_no": rounded_probs["home_or_away_and_btts_no"],
                 }
            }
        }
        
        return structured_output
    
    def get_most_likely_scoreline(self, home_xg: float, away_xg: float) -> str:
        """Calculate the most likely scoreline."""
        max_prob = 0
        likely_score = "0-0"
        
        # Check probabilities for each scoreline up to 5 goals per team for efficiency
        for home_goals in range(6): 
            for away_goals in range(6):
                prob = self.poisson_probability(home_goals, home_xg) * self.poisson_probability(away_goals, away_xg)
                
                if prob > max_prob:
                    max_prob = prob
                    likely_score = f"{home_goals}-{away_goals}"
        
        return likely_score
    
    def extract_team_advanced_stats(self, team_data: Dict[str, Any], team_name: str) -> Dict[str, Any]:
        """
        Extract and calculate advanced team statistics, integrating statarea data.
        
        Args:
            team_data: Raw team data
            team_name: Team name for proper metric extraction
            
        Returns:
            Dictionary with calculated team metrics
        """
        if not team_data:
            return {}
        
        stats = {
            "id": team_data.get("id"),
            "name": team_data.get("name")
        }
        
        # Basic form from mongodb_stats (as fallback)
        mongodb_form_string = team_data.get("mongodb_stats", {}).get("form", "")
        stats["mongodb_form_string"] = mongodb_form_string
        stats["overall_form_analysis"] = self.calculate_form_points(form_string=mongodb_form_string)

        # --- Performance Stats from mongodb_stats ---
        fixtures = team_data.get("mongodb_stats", {}).get("fixtures", {})
        if fixtures:
            # ... (keep existing extraction logic for overall performance) ...
            played_home = fixtures.get("played", {}).get("home", 0)
            played_away = fixtures.get("played", {}).get("away", 0)
            played_total = fixtures.get("played", {}).get("total", 0)
            
            wins_home = fixtures.get("wins", {}).get("home", 0)
            wins_away = fixtures.get("wins", {}).get("away", 0)
            wins_total = fixtures.get("wins", {}).get("total", 0)
            
            draws_home = fixtures.get("draws", {}).get("home", 0)
            draws_away = fixtures.get("draws", {}).get("away", 0)
            draws_total = fixtures.get("draws", {}).get("total", 0)
            
            losses_home = fixtures.get("loses", {}).get("home", 0)
            losses_away = fixtures.get("loses", {}).get("away", 0)
            losses_total = fixtures.get("loses", {}).get("total", 0)
            
            stats["performance_overall_mongodb"] = {
                "home": {
                    "played": played_home, "wins": wins_home, "draws": draws_home, "losses": losses_home,
                    "win_rate": round(wins_home / played_home, 3) if played_home > 0 else 0,
                    "points": (wins_home * 3) + draws_home,
                    "points_per_game": round(((wins_home * 3) + draws_home) / played_home, 3) if played_home > 0 else 0
                },
                "away": {
                    "played": played_away, "wins": wins_away, "draws": draws_away, "losses": losses_away,
                    "win_rate": round(wins_away / played_away, 3) if played_away > 0 else 0,
                    "points": (wins_away * 3) + draws_away,
                    "points_per_game": round(((wins_away * 3) + draws_away) / played_away, 3) if played_away > 0 else 0
                },
                 "total": { # Renamed from 'overall' to avoid confusion
                    "played": played_total, "wins": wins_total, "draws": draws_total, "losses": losses_total,
                    "win_rate": round(wins_total / played_total, 3) if played_total > 0 else 0,
                    "points": (wins_total * 3) + draws_total,
                    "points_per_game": round(((wins_total * 3) + draws_total) / played_total, 3) if played_total > 0 else 0
                }
            }
        
        # --- Goals Stats from mongodb_stats ---
        goals_for = team_data.get("mongodb_stats", {}).get("goals", {}).get("for", {})
        goals_against = team_data.get("mongodb_stats", {}).get("goals", {}).get("against", {})
        
        if goals_for and goals_against:
             # ... (keep existing extraction logic for overall goals) ...
            goals_for_total_home = goals_for.get("total", {}).get("home", 0)
            goals_for_total_away = goals_for.get("total", {}).get("away", 0)
            goals_for_total = goals_for.get("total", {}).get("total", 0)
            
            goals_against_total_home = goals_against.get("total", {}).get("home", 0)
            goals_against_total_away = goals_against.get("total", {}).get("away", 0)
            goals_against_total = goals_against.get("total", {}).get("total", 0)
            
            # Goals per game
            gpg_home = self.clean_value(goals_for.get("average", {}).get("home", 0))
            gpg_away = self.clean_value(goals_for.get("average", {}).get("away", 0))
            gpg_total = self.clean_value(goals_for.get("average", {}).get("total", 0))
            
            # Goals conceded per game
            gcpg_home = self.clean_value(goals_against.get("average", {}).get("home", 0))
            gcpg_away = self.clean_value(goals_against.get("average", {}).get("away", 0))
            gcpg_total = self.clean_value(goals_against.get("average", {}).get("total", 0))
            
            stats["goals_overall_mongodb"] = {
                "for": {
                    "home": goals_for_total_home, "away": goals_for_total_away, "total": goals_for_total,
                    "per_game_home": gpg_home, "per_game_away": gpg_away, "per_game_total": gpg_total
                },
                "against": {
                    "home": goals_against_total_home, "away": goals_against_total_away, "total": goals_against_total,
                    "per_game_home": gcpg_home, "per_game_away": gcpg_away, "per_game_total": gcpg_total
                },
                "difference": {
                    "home": goals_for_total_home - goals_against_total_home,
                    "away": goals_for_total_away - goals_against_total_away,
                    "total": goals_for_total - goals_against_total
                }
            }
            
            # --- Goal Timing from mongodb_stats ---
            if goals_for.get("minute"):
                # ... (keep existing extraction logic for overall goal timing) ...
                 stats["goal_timing_mongodb"] = {
                    "scored": {k: {
                        "total": v.get("total"),
                        "percentage": self.clean_value(v.get("percentage"))
                    } for k, v in goals_for.get("minute", {}).items() if v and isinstance(v, dict)},
                    "conceded": {k: {
                        "total": v.get("total"),
                        "percentage": self.clean_value(v.get("percentage"))
                    } for k, v in goals_against.get("minute", {}).items() if v and isinstance(v, dict)}
                }
                
                # Add first/second half goal distribution
            first_half_scored = sum([
                    stats["goal_timing_mongodb"]["scored"].get(t, {}).get("total", 0) or 0 
                    for t in ["0-15", "16-30", "31-45"]
                ])
            second_half_scored = sum([
                    stats["goal_timing_mongodb"]["scored"].get(t, {}).get("total", 0) or 0
                    for t in ["46-60", "61-75", "76-90", "91-105"] # Include extra time
                ])
                
            first_half_conceded = sum([
                    stats["goal_timing_mongodb"]["conceded"].get(t, {}).get("total", 0) or 0
                    for t in ["0-15", "16-30", "31-45"]
                ])
            second_half_conceded = sum([
                    stats["goal_timing_mongodb"]["conceded"].get(t, {}).get("total", 0) or 0
                    for t in ["46-60", "61-75", "76-90", "91-105"] # Include extra time
                ])
                
            total_scored = first_half_scored + second_half_scored
            total_conceded = first_half_conceded + second_half_conceded
            stats["half_analysis_mongodb"] = {
                    "scored": {
                        "first_half": first_half_scored, "second_half": second_half_scored,
                        "first_half_pct": round(first_half_scored / total_scored, 3) if total_scored > 0 else 0,
                        "second_half_pct": round(second_half_scored / total_scored, 3) if total_scored > 0 else 0
                    },
                    "conceded": {
                        "first_half": first_half_conceded, "second_half": second_half_conceded,
                        "first_half_pct": round(first_half_conceded / total_conceded, 3) if total_conceded > 0 else 0,
                        "second_half_pct": round(second_half_conceded / total_conceded, 3) if total_conceded > 0 else 0
                    }
                }

        # --- Statarea Analysis (5/10/15 games) ---
        statarea_raw = team_data.get("statarea_analysis", {}).get("raw_stats", {})
        stats["statarea_analysis"] = {} # Initialize sub-dictionary
        
        if statarea_raw:
            # Process Home Stats (host_5, host_10, host_15)
            stats["statarea_analysis"]["home"] = {}
            for interval in [15, 10, 5]:
                key = f"host_{interval}"
                if key in statarea_raw:
                    home_stats = statarea_raw[key]
                    
                    # Extract W/D/L counts for form calculation
                    wins = self.clean_value(home_stats.get(f"Number of {team_name} wins"))
                    draws = self.clean_value(home_stats.get(f"Number of {team_name} draws"))
                    losses = self.clean_value(home_stats.get(f"Number of {team_name} loses"))
                    
                    form_calcs = {}
                    if isinstance(wins, (int, float)) and isinstance(draws, (int, float)) and isinstance(losses, (int, float)):
                         form_calcs = self.calculate_form_points(wins=int(wins), draws=int(draws), losses=int(losses), num_games=interval)
                    
                    # Get 1X2 probabilities
                    outcome_prob_raw = home_stats.get("1 X 2", {})
                    outcome_prob = {
                         "win": self.clean_value(outcome_prob_raw.get(team_name)),
                         "draw": self.clean_value(outcome_prob_raw.get("draw")),
                         "loss": self.clean_value(outcome_prob_raw.get("opponent")) # Opponent loss is team win from opponent perspective
                    }

                    stats["statarea_analysis"]["home"][f"last_{interval}_games"] = {
                        "form": form_calcs,
                        "avg_goals_scored": self.clean_value(home_stats.get("Average scored goals per match")),
                        "avg_goals_conceded": self.clean_value(home_stats.get("Average conceded goals per match")),
                        "scoring_probability": self.clean_value(home_stats.get("Chance to score goal next match")),
                        "conceding_probability": self.clean_value(home_stats.get("Chance to conceded goal next match")),
                        "clean_sheets": self.clean_value(home_stats.get("Number of clean sheet matches")),
                        "failed_to_score": self.clean_value(home_stats.get("Failure to score matches")),
                        "over_2_5_pct": self.clean_value(home_stats.get("Matches over 2.5 goals in")) / interval if home_stats.get("Matches over 2.5 goals in") else None,
                        "under_2_5_pct": self.clean_value(home_stats.get("Matches under 2.5 goals in")) / interval if home_stats.get("Matches under 2.5 goals in") else None,
                         "outcome_probabilities_1x2": outcome_prob,
                         # Add more detailed stats if needed, e.g., goal timing for this interval
                         "goal_bands_pct": {k: self.clean_value(v) for k, v in home_stats.get("Goal bands", {}).items()},
                         "btts_pct": self.clean_value(home_stats.get("Team to score", {}).get("both")),
                         # Add other relevant fields from statarea_analysis[key]
                    }
                    # Clean up None values from percentages
                    if stats["statarea_analysis"]["home"][f"last_{interval}_games"]["over_2_5_pct"] is not None:
                         stats["statarea_analysis"]["home"][f"last_{interval}_games"]["over_2_5_pct"] = round(stats["statarea_analysis"]["home"][f"last_{interval}_games"]["over_2_5_pct"], 3)
                    if stats["statarea_analysis"]["home"][f"last_{interval}_games"]["under_2_5_pct"] is not None:
                         stats["statarea_analysis"]["home"][f"last_{interval}_games"]["under_2_5_pct"] = round(stats["statarea_analysis"]["home"][f"last_{interval}_games"]["under_2_5_pct"], 3)


            # Process Away Stats (guest_5, guest_10, guest_15)
            stats["statarea_analysis"]["away"] = {}
            for interval in [15, 10, 5]:
                key = f"guest_{interval}"
                if key in statarea_raw:
                    away_stats = statarea_raw[key]
                    
                    # Extract W/D/L counts for form calculation
                    wins = self.clean_value(away_stats.get(f"Number of {team_name} wins"))
                    draws = self.clean_value(away_stats.get(f"Number of {team_name} draws"))
                    losses = self.clean_value(away_stats.get(f"Number of {team_name} loses"))

                    form_calcs = {}
                    if isinstance(wins, (int, float)) and isinstance(draws, (int, float)) and isinstance(losses, (int, float)):
                        form_calcs = self.calculate_form_points(wins=int(wins), draws=int(draws), losses=int(losses), num_games=interval)

                    # Get 1X2 probabilities
                    outcome_prob_raw = away_stats.get("1 X 2", {})
                    outcome_prob = {
                         "win": self.clean_value(outcome_prob_raw.get(team_name)),
                         "draw": self.clean_value(outcome_prob_raw.get("draw")),
                         "loss": self.clean_value(outcome_prob_raw.get("opponent"))
                    }

                    stats["statarea_analysis"]["away"][f"last_{interval}_games"] = {
                        "form": form_calcs,
                        "avg_goals_scored": self.clean_value(away_stats.get("Average scored goals per match")),
                        "avg_goals_conceded": self.clean_value(away_stats.get("Average conceded goals per match")),
                        "scoring_probability": self.clean_value(away_stats.get("Chance to score goal next match")),
                        "conceding_probability": self.clean_value(away_stats.get("Chance to conceded goal next match")),
                        "clean_sheets": self.clean_value(away_stats.get("Number of clean sheet matches")),
                        "failed_to_score": self.clean_value(away_stats.get("Failure to score matches")),
                        "over_2_5_pct": self.clean_value(away_stats.get("Matches over 2.5 goals in")) / interval if away_stats.get("Matches over 2.5 goals in") else None,
                        "under_2_5_pct": self.clean_value(away_stats.get("Matches under 2.5 goals in")) / interval if away_stats.get("Matches under 2.5 goals in") else None,
                         "outcome_probabilities_1x2": outcome_prob,
                         "goal_bands_pct": {k: self.clean_value(v) for k, v in away_stats.get("Goal bands", {}).items()},
                         "btts_pct": self.clean_value(away_stats.get("Team to score", {}).get("both")),
                         # Add other relevant fields from statarea_analysis[key]
                    }
                     # Clean up None values from percentages
                    if stats["statarea_analysis"]["away"][f"last_{interval}_games"]["over_2_5_pct"] is not None:
                         stats["statarea_analysis"]["away"][f"last_{interval}_games"]["over_2_5_pct"] = round(stats["statarea_analysis"]["away"][f"last_{interval}_games"]["over_2_5_pct"], 3)
                    if stats["statarea_analysis"]["away"][f"last_{interval}_games"]["under_2_5_pct"] is not None:
                         stats["statarea_analysis"]["away"][f"last_{interval}_games"]["under_2_5_pct"] = round(stats["statarea_analysis"]["away"][f"last_{interval}_games"]["under_2_5_pct"], 3)

        # --- Recent Match History (from statarea) ---
        # This part was previously using mongodb_stats recent history, let's keep using statarea's for consistency
        match_history = team_data.get("statarea_analysis", {}).get("match_history", [])
        if match_history:
            stats["recent_form_statarea"] = {}
            for interval in [5, 10, 15]:
                 if len(match_history) >= interval:
                    interval_matches = match_history[:interval]
                    wins_count = sum(1 for m in interval_matches if m.get("result") == "win")
                    draws_count = sum(1 for m in interval_matches if m.get("result") == "draw")
                    losses_count = sum(1 for m in interval_matches if m.get("result") == "loss")
                    goals_scored = sum(m.get("team_goals", 0) for m in interval_matches)
                    goals_conceded = sum(m.get("opponent_goals", 0) for m in interval_matches)

                    form_calcs = self.calculate_form_points(wins=wins_count, draws=draws_count, losses=losses_count, num_games=interval)
                    
                    # Determine current streak based on the most recent match in the interval
                    current_streak = {"type": None, "count": 0}
                    if interval_matches:
                        last_result = interval_matches[0].get("result")
                        streak_count = 0
                        for m in interval_matches:
                            if m.get("result") == last_result:
                                streak_count += 1
                            else:
                                break
                        current_streak = {"type": last_result, "count": streak_count}


                    stats["recent_form_statarea"][f"last_{interval}_games"] = {
                        "matches": interval_matches, # Keep full list only for last_5? Maybe just summary stats needed.
                        "summary": {
                             **form_calcs, # Includes W,D,L, points, ppg, win_rate, efficiency
                "goals_scored": goals_scored,
                "goals_conceded": goals_conceded,
                "goal_difference": goals_scored - goals_conceded,
                            "avg_goals_scored": round(goals_scored / interval, 2),
                            "avg_goals_conceded": round(goals_conceded / interval, 2),
                             "current_streak": current_streak,
                        }
                    }
                    if interval != 5: # Only store full match list for last 5 to save space
                        stats["recent_form_statarea"][f"last_{interval}_games"]["matches"] = f"Summary stats only for last {interval}"
        
        # Remove the old 'recent_form' key if it exists to avoid redundancy
        stats.pop("recent_form", None) 
        
        return stats
    
    def process_h2h_advanced(self, h2h_data: List[Dict[str, Any]], 
                            home_team_id: str, away_team_id: str) -> Dict[str, Any]:
        """
        Process head-to-head data with advanced metrics.
        
        Args:
            h2h_data: Head-to-head match data
            home_team_id: ID of home team
            away_team_id: ID of away team
            
        Returns:
            Advanced H2H statistics
        """
        if not h2h_data:
            return {}
        
        # Use last 10 matches for analysis
        recent_h2h = h2h_data[:10]
        
        # Count stats
        home_team_wins = 0
        away_team_wins = 0
        draws = 0
        
        total_goals = 0
        home_team_goals = 0
        away_team_goals = 0
        
        btts_count = 0
        over_2_5_count = 0
        
        home_team_matches = []
        away_team_matches = []
        
        for match in recent_h2h:
            match_home_id = str(match.get("home_team", {}).get("id"))
            match_away_id = str(match.get("away_team", {}).get("id"))
            home_score = match.get("score", {}).get("home")
            away_score = match.get("score", {}).get("away")
            
            if home_score is None or away_score is None:
                continue
            
            # Record goals
            total_goals += home_score + away_score
            
            # Check if both teams scored
            if home_score > 0 and away_score > 0:
                btts_count += 1
            
            # Check if over 2.5 goals
            if (home_score + away_score) > 2.5:
                over_2_5_count += 1
            
            # Count match outcomes
            if home_score == away_score:
                draws += 1
            else:
                # Identify which team won from fixture perspective
                # Then map to current home/away perspective
                if match_home_id == home_team_id and match_away_id == away_team_id:
                    # Same fixture configuration as current match
                    if home_score > away_score:
                        home_team_wins += 1
                        home_team_goals += home_score
                        away_team_goals += away_score
                    else:
                        away_team_wins += 1
                        home_team_goals += home_score
                        away_team_goals += away_score
                elif match_home_id == away_team_id and match_away_id == home_team_id:
                    # Reverse fixture configuration
                    if home_score > away_score:
                        away_team_wins += 1
                        away_team_goals += home_score
                        home_team_goals += away_score
                    else:
                        home_team_wins += 1
                        away_team_goals += home_score
                        home_team_goals += away_score
            
            # Simplified match record
            match_record = {
                "date": match.get("date"),
                "home_team": match.get("home_team", {}).get("name"),
                "away_team": match.get("away_team", {}).get("name"),
                "score": f"{home_score}-{away_score}",
                "competition": match.get("league", {}).get("name")
            }
            
            # Organize by team perspective
            if match_home_id == home_team_id:
                home_team_matches.append(match_record)
            elif match_away_id == home_team_id:
                away_team_matches.append(match_record)
        
        total_matches = len(recent_h2h)
        if total_matches == 0:
            return {}
        
        h2h_stats = {
            "summary": {
                "home_team_wins": home_team_wins,
                "away_team_wins": away_team_wins,
                "draws": draws,
                "total_matches": total_matches,
                "home_team_win_pct": home_team_wins / total_matches,
                "away_team_win_pct": away_team_wins / total_matches,
                "draw_pct": draws / total_matches,
                "home_team_goals_per_match": home_team_goals / total_matches,
                "away_team_goals_per_match": away_team_goals / total_matches,
                "avg_total_goals": total_goals / total_matches,
                "btts_pct": btts_count / total_matches,
                "over_2_5_pct": over_2_5_count / total_matches
            },
            "matches": recent_h2h[:5]  # Limited to last 5 matches only
        }
        
        return h2h_stats
    
    def _get_top_probable_bets(self, all_outcome_probs: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identifies the top N most probable bets from all calculated probabilities,
        with explicit labels, ranking, and bet type (Simple/Combined).
        """
        all_bets = {} # Dictionary to hold {bet_label: probability}

        # --- Extract and Label Basic Probabilities (Simple Bets) ---
        basic_probs = all_outcome_probs.get("basic_probabilities", {})
        if basic_probs:
            # Simple Outcomes
            all_bets["Home Win"] = basic_probs.get("home_win")
            all_bets["Draw"] = basic_probs.get("draw")
            all_bets["Away Win"] = basic_probs.get("away_win")
            # Simple Over/Under
            all_bets["Over 1.5 Goals"] = basic_probs.get("over_1.5")
            all_bets["Under 1.5 Goals"] = basic_probs.get("under_1.5")
            all_bets["Over 2.5 Goals"] = basic_probs.get("over_2.5")
            all_bets["Under 2.5 Goals"] = basic_probs.get("under_2.5")
            all_bets["Over 3.5 Goals"] = basic_probs.get("over_3.5")
            all_bets["Under 3.5 Goals"] = basic_probs.get("under_3.5")
            all_bets["Over 4.5 Goals"] = basic_probs.get("over_4.5")
            all_bets["Under 4.5 Goals"] = basic_probs.get("under_4.5")
            # Simple BTTS
            all_bets["BTTS Yes"] = basic_probs.get("btts_yes")
            all_bets["BTTS No"] = basic_probs.get("btts_no")
            # Simple Double Chance
            all_bets["Home or Draw"] = basic_probs.get("home_draw") # Renamed from WIN OR DRAW (Home/Draw)
            all_bets["Away or Draw"] = basic_probs.get("away_draw") # Renamed from WIN OR DRAW (Away/Draw)
            all_bets["No Draw (Home or Away Win)"] = basic_probs.get("home_away") # Renamed and clarified

        # --- Extract and Label Combined Probabilities (Combined Bets) ---
        combined_probs = all_outcome_probs.get("combined_probabilities", {})
        if combined_probs:
            # Single Outcome Combinations
            for outcome, variables in combined_probs.get("single_outcome_combinations", {}).items():
                outcome_label = outcome.replace('_', ' ').title()
                for variable, prob in variables.items():
                    # Make variable labels explicit
                    if 'over' in variable:
                        var_label = variable.replace('over_', 'Over ').replace('_', '.') + " Goals"
                    elif 'under' in variable:
                        var_label = variable.replace('under_', 'Under ').replace('_', '.') + " Goals"
                    elif 'btts_yes' in variable:
                        var_label = "BTTS Yes"
                    elif 'btts_no' in variable:
                        var_label = "BTTS No"
                    else:
                        var_label = variable.replace('_', ' ').title() # Fallback
                    all_bets[f"{outcome_label} & {var_label}"] = prob

            # Double Chance Combinations
            for dc_outcome, variables in combined_probs.get("double_chance_combinations", {}).items():
                 # Standardize DC labels
                 if dc_outcome == "home_or_draw": dc_label = "Home or Draw"
                 elif dc_outcome == "away_or_draw": dc_label = "Away or Draw"
                 elif dc_outcome == "home_or_away": dc_label = "No Draw (Home or Away Win)"
                 else: dc_label = dc_outcome.replace('_', ' ').title() # Fallback

                 for variable, prob in variables.items():
                    # Make variable labels explicit
                    if 'over' in variable:
                        var_label = variable.replace('over_', 'Over ').replace('_', '.') + " Goals"
                    elif 'under' in variable:
                        var_label = variable.replace('under_', 'Under ').replace('_', '.') + " Goals"
                    elif 'btts_yes' in variable:
                        var_label = "BTTS Yes"
                    elif 'btts_no' in variable:
                        var_label = "BTTS No"
                    else:
                        var_label = variable.replace('_', ' ').title() # Fallback
                    all_bets[f"{dc_label} + {var_label}"] = prob

        # Filter out None values and invalid probabilities
        valid_bets = {bet: prob for bet, prob in all_bets.items() if prob is not None and 0 <= prob <= 1}
        sorted_bets = sorted(valid_bets.items(), key=lambda item: item[1], reverse=True)

        # Format output with Rank and Type
        top_bets_list = []
        for i, (bet, prob) in enumerate(sorted_bets[:top_n]):
            bet_type = "Combined" if ("&" in bet or "+" in bet) else "Simple"
            top_bets_list.append({
                "rank": i + 1,
                "bet": bet,
                "type": bet_type,
                "probability": f"{prob:.1%}"
            })

        return top_bets_list

    def _find_value_bets(self, all_outcome_probs: Dict[str, Any], market_odds: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifies potential value bets by comparing calculated probability to market odds.
        NOTE: This function expects market_odds to be provided in the input data,
              as direct DB access is not possible here.
        """
        if not market_odds:
            return [{"info": "Market odds data not found in input JSON. Cannot calculate value bets."}]

        value_bets = []
        min_edge_threshold = 0.05 # Minimum 5% edge to be considered significant value

        # --- Map calculated probabilities to potential market odds keys ---
        # This mapping needs to be adjusted based on the *actual* keys used in your market_odds structure
        prob_to_market_map = {
            # Basic 1X2
            "home_win": market_odds.get("1x2", {}).get("home"),
            "draw": market_odds.get("1x2", {}).get("draw"),
            "away_win": market_odds.get("1x2", {}).get("away"),
            # Basic O/U
            "over_1.5": market_odds.get("ou_1_5", {}).get("over"),
            "under_1.5": market_odds.get("ou_1_5", {}).get("under"),
            "over_2.5": market_odds.get("ou_2_5", {}).get("over"),
            "under_2.5": market_odds.get("ou_2_5", {}).get("under"),
            "over_3.5": market_odds.get("ou_3_5", {}).get("over"),
            "under_3.5": market_odds.get("ou_3_5", {}).get("under"),
            "over_4.5": market_odds.get("ou_4_5", {}).get("over"),
            "under_4.5": market_odds.get("ou_4_5", {}).get("under"),
            # Basic BTTS
            "btts_yes": market_odds.get("btts", {}).get("yes"),
            "btts_no": market_odds.get("btts", {}).get("no"),
            # Basic Double Chance
            "home_draw": market_odds.get("double_chance", {}).get("1x"),
            "away_draw": market_odds.get("double_chance", {}).get("x2"),
            "home_away": market_odds.get("double_chance", {}).get("12"),
            # --- Add mappings for combined bets if odds are available ---
            # Example: "home_win_and_over_2.5": market_odds.get("result_ou_2_5", {}).get("home_over"),
            # Example: "home_or_draw_and_btts_yes": market_odds.get("dc_btts", {}).get("1x_yes"),
            # ... etc. This part is highly dependent on your odds data structure
        }

        basic_probs = all_outcome_probs.get("basic_probabilities", {})

        for prob_key, market_odd in prob_to_market_map.items():
            calculated_prob = basic_probs.get(prob_key)

            if calculated_prob is not None and market_odd is not None:
                try:
                    market_odd_float = float(market_odd)
                    if market_odd_float > 1.0: # Basic validation for decimal odds
                        edge = (calculated_prob * market_odd_float) - 1
                        if edge >= min_edge_threshold:
                            value_bets.append({
                                "bet": prob_key.replace('_', ' ').title(),
                                "calculated_prob": f"{calculated_prob:.1%}",
                                "market_odds": market_odd_float,
                                "implied_market_prob": f"{1/market_odd_float:.1%}",
                                "edge": f"{edge:.1%}"
                            })
                except (ValueError, TypeError):
                    # Ignore if market_odd is not a valid number
                    continue

        # TODO: Extend this logic to handle combined probabilities if corresponding market odds are available and mapped

        if not value_bets and market_odds: # Check if odds were present but no value found
             return [{"info": f"No significant value bets found (edge > {min_edge_threshold:.0%})."}]
        elif value_bets:
             # Sort by edge descending
            return sorted(value_bets, key=lambda x: float(x["edge"].strip('%'))/100, reverse=True)
        else: # This case handles when market_odds was None initially
            return [{"info": "Market odds data not found in input JSON. Cannot calculate value bets."}]


    def calculate_match_metrics(self, home_team_stats: Dict[str, Any], away_team_stats: Dict[str, Any],
                              h2h_stats: Dict[str, Any], market_odds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Added market_odds parameter
        """
        Calculate combined match metrics and predictions, including detailed probabilities,
        top probable bets, and potential value bets (if market odds are provided).
        """
        # Get team IDs
        home_team_id = home_team_stats.get("id")
        away_team_id = away_team_stats.get("id")

        # Calculate final match outcome probabilities (now includes basic and combined)
        all_outcome_probs = self.calculate_match_outcome_probabilities(home_team_stats, away_team_stats, h2h_stats)

        # --- Form, Trend, Momentum Comparison (using Statarea data) ---
        form_comparison = {"home": {}, "away": {}}
        for interval in [5, 10, 15]:
             # Home Team (using 'host' stats from statarea)
             home_interval_key = f"last_{interval}_games"
             home_data = home_team_stats.get("statarea_analysis", {}).get("home", {}).get(home_interval_key, {})
             if home_data:
                 form_comparison["home"][home_interval_key] = {
                     "wins": home_data.get("form", {}).get("wins"),
                     "draws": home_data.get("form", {}).get("draws"),
                     "losses": home_data.get("form", {}).get("losses"),
                     "points": home_data.get("form", {}).get("points"),
                     "points_per_match": home_data.get("form", {}).get("points_per_match"),
                     "win_rate": home_data.get("form", {}).get("win_rate"),
                     "efficiency": home_data.get("form", {}).get("efficiency"), # Momentum score
                     "avg_goals_scored": home_data.get("avg_goals_scored"),
                     "avg_goals_conceded": home_data.get("avg_goals_conceded"),
                 }
            
             # Away Team (using 'guest' stats from statarea)
             away_interval_key = f"last_{interval}_games"
             away_data = away_team_stats.get("statarea_analysis", {}).get("away", {}).get(away_interval_key, {})
             if away_data:
                 form_comparison["away"][away_interval_key] = {
                     "wins": away_data.get("form", {}).get("wins"),
                     "draws": away_data.get("form", {}).get("draws"),
                     "losses": away_data.get("form", {}).get("losses"),
                     "points": away_data.get("form", {}).get("points"),
                     "points_per_match": away_data.get("form", {}).get("points_per_match"),
                     "win_rate": away_data.get("form", {}).get("win_rate"),
                     "efficiency": away_data.get("form", {}).get("efficiency"), # Momentum score
                     "avg_goals_scored": away_data.get("avg_goals_scored"),
                     "avg_goals_conceded": away_data.get("avg_goals_conceded"),
                 }

        # --- Key Stats Comparison (Primary source: Statarea 15 games, fallback MongoDB) ---
        key_stats = {"home": {}, "away": {}}
        home_statarea_15 = home_team_stats.get("statarea_analysis", {}).get("home", {}).get("last_15_games", {})
        away_statarea_15 = away_team_stats.get("statarea_analysis", {}).get("away", {}).get("last_15_games", {})
        
        key_stats["home"] = {
            "avg_goals_scored_home": home_statarea_15.get("avg_goals_scored") if home_statarea_15 else home_team_stats.get("goals_overall_mongodb", {}).get("for", {}).get("per_game_home"),
            "avg_goals_conceded_home": home_statarea_15.get("avg_goals_conceded") if home_statarea_15 else home_team_stats.get("goals_overall_mongodb", {}).get("against", {}).get("per_game_home"),
            "clean_sheets_home": home_statarea_15.get("clean_sheets") if home_statarea_15 else None, # Num matches
            "clean_sheet_pct_home": round(home_statarea_15.get("clean_sheets", 0) / 15, 3) if home_statarea_15 and home_statarea_15.get("clean_sheets") is not None else None,
            "failed_to_score_home": home_statarea_15.get("failed_to_score") if home_statarea_15 else None, # Num matches
            "scoring_probability": home_statarea_15.get("scoring_probability") if home_statarea_15 else None,
            "conceding_probability": home_statarea_15.get("conceding_probability") if home_statarea_15 else None,
            "btts_pct": home_statarea_15.get("btts_pct") if home_statarea_15 else None,
            "over_2_5_pct": home_statarea_15.get("over_2_5_pct") if home_statarea_15 else None,
        }
        key_stats["away"] = {
            "avg_goals_scored_away": away_statarea_15.get("avg_goals_scored") if away_statarea_15 else away_team_stats.get("goals_overall_mongodb", {}).get("for", {}).get("per_game_away"),
            "avg_goals_conceded_away": away_statarea_15.get("avg_goals_conceded") if away_statarea_15 else away_team_stats.get("goals_overall_mongodb", {}).get("against", {}).get("per_game_away"),
            "clean_sheets_away": away_statarea_15.get("clean_sheets") if away_statarea_15 else None, # Num matches
            "clean_sheet_pct_away": round(away_statarea_15.get("clean_sheets", 0) / 15, 3) if away_statarea_15 and away_statarea_15.get("clean_sheets") is not None else None,
            "failed_to_score_away": away_statarea_15.get("failed_to_score") if away_statarea_15 else None, # Num matches
             "scoring_probability": away_statarea_15.get("scoring_probability") if away_statarea_15 else None,
            "conceding_probability": away_statarea_15.get("conceding_probability") if away_statarea_15 else None,
            "btts_pct": away_statarea_15.get("btts_pct") if away_statarea_15 else None,
            "over_2_5_pct": away_statarea_15.get("over_2_5_pct") if away_statarea_15 else None,
        }

        # --- Goal Timing Analysis (Using MongoDB stats for overall pattern) ---
        goal_timing_analysis = {"home": {}, "away": {}}
        home_timing = home_team_stats.get("goal_timing_mongodb", {})
        home_half_analysis = home_team_stats.get("half_analysis_mongodb", {}).get("scored", {})
        if home_timing and home_half_analysis:
             goal_timing_analysis["home"] = {
                "first_half_goal_pct": home_half_analysis.get("first_half_pct", 0),
                "second_half_goal_pct": home_half_analysis.get("second_half_pct", 0),
                 "early_goal_pct_0_15": home_timing.get("scored", {}).get("0-15", {}).get("percentage"),
                 "late_goal_pct_76_plus": (home_timing.get("scored", {}).get("76-90", {}).get("percentage") or 0) + 
                                        (home_timing.get("scored", {}).get("91-105", {}).get("percentage") or 0)
            }
        
        away_timing = away_team_stats.get("goal_timing_mongodb", {})
        away_half_analysis = away_team_stats.get("half_analysis_mongodb", {}).get("scored", {})
        if away_timing and away_half_analysis:
             goal_timing_analysis["away"] = {
                "first_half_goal_pct": away_half_analysis.get("first_half_pct", 0),
                "second_half_goal_pct": away_half_analysis.get("second_half_pct", 0),
                 "early_goal_pct_0_15": away_timing.get("scored", {}).get("0-15", {}).get("percentage"),
                 "late_goal_pct_76_plus": (away_timing.get("scored", {}).get("76-90", {}).get("percentage") or 0) +
                                        (away_timing.get("scored", {}).get("91-105", {}).get("percentage") or 0)
            }
            
        # --- Calculate Top Probable Bets (using updated function) ---
        top_probable_bets = self._get_top_probable_bets(all_outcome_probs)

        # --- Calculate Value Bets (if odds provided) ---
        value_bets = self._find_value_bets(all_outcome_probs, market_odds)


        # --- Construct Final Match Metrics Dictionary ---
        match_metrics = {
            "form_trend_momentum_comparison": form_comparison,
            "key_stats_comparison": key_stats,
            "goal_timing_comparison": goal_timing_analysis,
            "predictions": { 
                "outcome_probabilities": all_outcome_probs, 
                "most_likely_score": all_outcome_probs.get("most_likely_score", "N/A"),
                "recommended_bet": self._get_recommended_bet(all_outcome_probs), # Recommended bet logic unchanged for now
                "top_probable_bets": top_probable_bets, # Updated output format
                "value_bets": value_bets 
            },
             "base_xg": {
                 "home": self.calculate_expected_goals(home_team_stats, "home").get("base_xg"),
                 "away": self.calculate_expected_goals(away_team_stats, "away").get("base_xg")
            },
            "predictability": {} # Placeholder for predictability score
        }
        
        return match_metrics
    
    def _get_recommended_bet(self, all_outcome_probs: Dict[str, Any]) -> str:
        """Determine the best bet based on basic and combined probability values,
           prioritizing strong combined bets."""
        
        # Extract basic and combined probabilities safely
        basic_probs = all_outcome_probs.get("basic_probabilities", {})
        combined_probs_structured = all_outcome_probs.get("combined_probabilities", {}) 
        
        if not basic_probs or not isinstance(basic_probs, dict):
             return "Insufficient basic probability data"
        if not combined_probs_structured or not isinstance(combined_probs_structured, dict):
             print("Warning: Combined probabilities missing or invalid for recommendation.") 
             combined_probs_structured = {"single_outcome_combinations": {}, "double_chance_combinations": {}} 


        # --- Helper to safely get nested combined probabilities (Updated for new double chance keys) ---
        def get_nested_prob(category, outcome, variable, default=0.0):
            try:
                # Map the common outcome names to the new keys if needed for double chance
                if category == "double_chance_combinations":
                     if outcome == "home_draw": outcome = "home_or_draw"
                     elif outcome == "away_draw": outcome = "away_or_draw"
                     elif outcome == "home_away": outcome = "home_or_away"
                return combined_probs_structured[category][outcome][variable]
            except KeyError:
                return default
        
        # --- Extract Basic Probabilities ---
        home_win = basic_probs.get("home_win", 0)
        draw = basic_probs.get("draw", 0)
        away_win = basic_probs.get("away_win", 0)
        over_2_5 = basic_probs.get("over_2.5", 0)
        under_2_5 = basic_probs.get("under_2.5", 0)
        btts_yes = basic_probs.get("btts_yes", 0)
        btts_no = basic_probs.get("btts_no", 0)
        home_draw_prob = basic_probs.get("home_draw", 0) # 1X basic prob
        away_draw_prob = basic_probs.get("away_draw", 0) # X2 basic prob

        # --- Extract Single Outcome + Variable Probabilities (No changes here) ---
        home_win_over_25 = get_nested_prob("single_outcome_combinations", "home_win", "over_2.5")
        home_win_under_25 = get_nested_prob("single_outcome_combinations", "home_win", "under_2.5")
        home_win_btts_yes = get_nested_prob("single_outcome_combinations", "home_win", "btts_yes")
        home_win_btts_no = get_nested_prob("single_outcome_combinations", "home_win", "btts_no")
        
        draw_over_25 = get_nested_prob("single_outcome_combinations", "draw", "over_2.5")
        draw_under_25 = get_nested_prob("single_outcome_combinations", "draw", "under_2.5")
        draw_btts_yes = get_nested_prob("single_outcome_combinations", "draw", "btts_yes")
        draw_btts_no = get_nested_prob("single_outcome_combinations", "draw", "btts_no")

        away_win_over_25 = get_nested_prob("single_outcome_combinations", "away_win", "over_2.5")
        away_win_under_25 = get_nested_prob("single_outcome_combinations", "away_win", "under_2.5")
        away_win_btts_yes = get_nested_prob("single_outcome_combinations", "away_win", "btts_yes")
        away_win_btts_no = get_nested_prob("single_outcome_combinations", "away_win", "btts_no")

        # --- Extract Double Chance + Variable Probabilities (using helper and updated outcome names) ---
        home_or_draw_over_25 = get_nested_prob("double_chance_combinations", "home_or_draw", "over_2.5")
        home_or_draw_under_25 = get_nested_prob("double_chance_combinations", "home_or_draw", "under_2.5")
        home_or_draw_btts_yes = get_nested_prob("double_chance_combinations", "home_or_draw", "btts_yes")
        home_or_draw_btts_no = get_nested_prob("double_chance_combinations", "home_or_draw", "btts_no")
        
        away_or_draw_over_25 = get_nested_prob("double_chance_combinations", "away_or_draw", "over_2.5")
        away_or_draw_under_25 = get_nested_prob("double_chance_combinations", "away_or_draw", "under_2.5")
        away_or_draw_btts_yes = get_nested_prob("double_chance_combinations", "away_or_draw", "btts_yes")
        away_or_draw_btts_no = get_nested_prob("double_chance_combinations", "away_or_draw", "btts_no")
        
        # --- Define Thresholds ---
        simple_high_prob_threshold = 0.65 
        combined_high_prob_threshold = 0.55 
        double_chance_threshold = 0.70 
        medium_prob_threshold = 0.55 

        # --- Determine Best Bet - Order of Priority ---

        # 1. High Probability Simple Outcomes 
        if home_win > simple_high_prob_threshold: return f"Home Win ({home_win:.1%})"
        if away_win > simple_high_prob_threshold: return f"Away Win ({away_win:.1%})"
        if over_2_5 > simple_high_prob_threshold: return f"Over 2.5 Goals ({over_2_5:.1%})"
        if under_2_5 > simple_high_prob_threshold: return f"Under 2.5 Goals ({under_2_5:.1%})"
        if btts_yes > simple_high_prob_threshold: return f"Both Teams To Score ({btts_yes:.1%})"
        if btts_no > simple_high_prob_threshold: return f"BTTS No ({btts_no:.1%})"

        # 2. High Probability Single Outcome + Variable Bets
        if home_win_over_25 > combined_high_prob_threshold: return f"Home Win & Over 2.5 ({home_win_over_25:.1%})"
        if home_win_under_25 > combined_high_prob_threshold: return f"Home Win & Under 2.5 ({home_win_under_25:.1%})"
        if home_win_btts_yes > combined_high_prob_threshold: return f"Home Win & BTTS Yes ({home_win_btts_yes:.1%})"
        if home_win_btts_no > combined_high_prob_threshold: return f"Home Win & BTTS No ({home_win_btts_no:.1%})"
        if away_win_over_25 > combined_high_prob_threshold: return f"Away Win & Over 2.5 ({away_win_over_25:.1%})"
        if away_win_under_25 > combined_high_prob_threshold: return f"Away Win & Under 2.5 ({away_win_under_25:.1%})"
        if away_win_btts_yes > combined_high_prob_threshold: return f"Away Win & BTTS Yes ({away_win_btts_yes:.1%})"
        if away_win_btts_no > combined_high_prob_threshold: return f"Away Win & BTTS No ({away_win_btts_no:.1%})"
        if draw < medium_prob_threshold: # Only consider draw combos if draw itself isn't highly likely
            if draw_over_25 > combined_high_prob_threshold: return f"Draw & Over 2.5 ({draw_over_25:.1%})"
            if draw_under_25 > combined_high_prob_threshold: return f"Draw & Under 2.5 ({draw_under_25:.1%})"
            if draw_btts_yes > combined_high_prob_threshold: return f"Draw & BTTS Yes ({draw_btts_yes:.1%})"
            if draw_btts_no > combined_high_prob_threshold: return f"Draw & BTTS No ({draw_btts_no:.1%})"

        # 3. High Probability Double Chance + Variable Bets (Using new variables)
        if home_or_draw_over_25 > combined_high_prob_threshold: return f"WIN OR DRAW (Home/Draw) + Over 2.5 ({home_or_draw_over_25:.1%})"
        if home_or_draw_under_25 > combined_high_prob_threshold: return f"WIN OR DRAW (Home/Draw) + Under 2.5 ({home_or_draw_under_25:.1%})"
        if home_or_draw_btts_yes > combined_high_prob_threshold: return f"WIN OR DRAW (Home/Draw) + BTTS Yes ({home_or_draw_btts_yes:.1%})"
        if home_or_draw_btts_no > combined_high_prob_threshold: return f"WIN OR DRAW (Home/Draw) + BTTS No ({home_or_draw_btts_no:.1%})"
        if away_or_draw_over_25 > combined_high_prob_threshold: return f"WIN OR DRAW (Away/Draw) + Over 2.5 ({away_or_draw_over_25:.1%})"
        if away_or_draw_under_25 > combined_high_prob_threshold: return f"WIN OR DRAW (Away/Draw) + Under 2.5 ({away_or_draw_under_25:.1%})"
        if away_or_draw_btts_yes > combined_high_prob_threshold: return f"WIN OR DRAW (Away/Draw) + BTTS Yes ({away_or_draw_btts_yes:.1%})"
        if away_or_draw_btts_no > combined_high_prob_threshold: return f"WIN OR DRAW (Away/Draw) + BTTS No ({away_or_draw_btts_no:.1%})"

        # 4. Standard Double Chances (Using basic_probs['home_draw'] etc.)
        if home_draw_prob > double_chance_threshold and home_win >= away_win: return f"WIN OR DRAW (Home/Draw) ({home_draw_prob:.1%})"
        if away_draw_prob > double_chance_threshold and away_win >= home_win: return f"WIN OR DRAW (Away/Draw) ({away_draw_prob:.1%})"
        if home_draw_prob > double_chance_threshold and home_draw_prob >= away_draw_prob: return f"WIN OR DRAW (Home/Draw) ({home_draw_prob:.1%})"
        if away_draw_prob > double_chance_threshold and away_draw_prob > home_draw_prob: return f"WIN OR DRAW (Away/Draw) ({away_draw_prob:.1%})"

        # 5. Consider Draw if it meets the medium threshold 
        if draw > medium_prob_threshold: return f"Draw ({draw:.1%})"
            
        # 6. Lower probability "Lean" bets 
        lean_candidates = {
             f"Lean Home Win & Over 2.5 ({home_win_over_25:.1%})": home_win_over_25,
             f"Lean Home Win & Under 2.5 ({home_win_under_25:.1%})": home_win_under_25,
             f"Lean Home Win & BTTS Yes ({home_win_btts_yes:.1%})": home_win_btts_yes,
             f"Lean Home Win & BTTS No ({home_win_btts_no:.1%})": home_win_btts_no,
             f"Lean Away Win & Over 2.5 ({away_win_over_25:.1%})": away_win_over_25,
             f"Lean Away Win & Under 2.5 ({away_win_under_25:.1%})": away_win_under_25,
             f"Lean Away Win & BTTS Yes ({away_win_btts_yes:.1%})": away_win_btts_yes,
             f"Lean Away Win & BTTS No ({away_win_btts_no:.1%})": away_win_btts_no,
        }
        lean_candidates = {k: v for k, v in lean_candidates.items() if v > 0.40} 

        if lean_candidates:
            best_lean_bet = max(lean_candidates, key=lean_candidates.get)
            simple_lean_probs = {
                f"Lean Home Win ({home_win:.1%})": home_win,
                f"Lean Away Win ({away_win:.1%})": away_win,
                f"Lean Over 2.5 Goals ({over_2_5:.1%})": over_2_5,
                f"Lean Under 2.5 Goals ({under_2_5:.1%})": under_2_5,
                f"Lean BTTS Yes ({btts_yes:.1%})": btts_yes,
                f"Lean BTTS No ({btts_no:.1%})": btts_no,
            }
            simple_lean_probs = {k: v for k,v in simple_lean_probs.items() if v > 0.40} # Changed threshold to match lean_candidates
            
            if simple_lean_probs:
                 best_simple_lean_prob = max(simple_lean_probs.values())
                 # Compare best lean combo prob vs best simple lean prob
                 if lean_candidates[best_lean_bet] > best_simple_lean_prob + 0.05: # Prefer combo if significantly higher
                      return best_lean_bet
                 else: # Otherwise, recommend the simpler lean bet
                     best_simple_lean_bet = max(simple_lean_probs, key=simple_lean_probs.get)
                     return best_simple_lean_bet
            else: # If no simple leans qualify, return the best combo lean
                return best_lean_bet


        # Fallback to simple "Lean" bets if no combos qualified previously
        if home_win > away_win and home_win > draw: return f"Lean Home Win ({home_win:.1%})" 
        if away_win > home_win and away_win > draw: return f"Lean Away Win ({away_win:.1%})" 
        if over_2_5 > under_2_5: return f"Lean Over 2.5 Goals ({over_2_5:.1%})"
        if btts_yes > btts_no: return f"Lean BTTS Yes ({btts_yes:.1%})"
        if under_2_5 > over_2_5: return f"Lean Under 2.5 Goals ({under_2_5:.1%})"
        if btts_no > btts_yes: return f"Lean BTTS No ({btts_no:.1%})"
        
        # Absolute fallback: Highest basic probability
        strongest_1x2 = max(home_win, draw, away_win)
        if strongest_1x2 == home_win: return f"Highest Prob: Home Win ({home_win:.1%})"
        if strongest_1x2 == draw: return f"Highest Prob: Draw ({draw:.1%})"
        if strongest_1x2 == away_win: return f"Highest Prob: Away Win ({away_win:.1%})"
        
        return "No clear value bet"
    
    def calculate_predictability_score(self, home_stats: Dict[str, Any], away_stats: Dict[str, Any], 
                                      h2h_stats: Dict[str, Any], match_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates a predictability score based on the consistency of various metrics.
        
        Args:
            home_stats: Processed home team statistics.
            away_stats: Processed away team statistics.
            h2h_stats: Processed H2H statistics.
            match_metrics: Calculated match metrics including Poisson predictions.
            
        Returns:
            Dictionary with predictability score (0-10) and reasoning text.
        """
        score = 5.0 # Start with a neutral score
        reasons = []
        max_score_contribution = 1.0 # Max contribution per check
        min_score_contribution = -1.0 # Min contribution per check

        # --- Metric Access Helpers ---
        def get_statarea_metric(stats, venue, interval, metric_key, default=None):
            try:
                return stats["statarea_analysis"][venue][f"last_{interval}_games"][metric_key]
            except KeyError:
                return default

        def get_h2h_metric(h2h, metric_key, default=None):
             try:
                 return h2h["summary"][metric_key]
             except KeyError:
                 return default

        def get_poisson_prob(metrics, prob_key, default=None):
            try:
                return metrics["predictions"]["outcome_probabilities"]["basic_probabilities"][prob_key]
            except KeyError:
                 return default

        # --- Comparisons ---

        # 1. Recent Form (Statarea 5/15 games) vs. H2H Form
        home_ppg_5 = get_statarea_metric(home_stats, "home", 5, "form", {}).get("points_per_match")
        away_ppg_5 = get_statarea_metric(away_stats, "away", 5, "form", {}).get("points_per_match")
        home_ppg_15 = get_statarea_metric(home_stats, "home", 15, "form", {}).get("points_per_match")
        away_ppg_15 = get_statarea_metric(away_stats, "away", 15, "form", {}).get("points_per_match")
        h2h_home_win_pct = get_h2h_metric(h2h_stats, "home_team_win_pct")
        h2h_away_win_pct = get_h2h_metric(h2h_stats, "away_team_win_pct")

        if all(v is not None for v in [home_ppg_15, away_ppg_15, h2h_home_win_pct, h2h_away_win_pct]):
            recent_home_adv = home_ppg_15 > away_ppg_15
            h2h_home_adv = h2h_home_win_pct > h2h_away_win_pct
            if recent_home_adv == h2h_home_adv:
                score += max_score_contribution * 0.5 # Smaller contribution
                reasons.append("Recent form trend aligns with H2H historical advantage.")
            else:
                score += min_score_contribution * 0.5
                reasons.append("Recent form trend conflicts with H2H historical advantage.")
        
        # 2. Goal Averages (Statarea 15) vs. H2H Averages
        home_avg_scored_15 = get_statarea_metric(home_stats, "home", 15, "avg_goals_scored")
        away_avg_scored_15 = get_statarea_metric(away_stats, "away", 15, "avg_goals_scored")
        h2h_home_avg = get_h2h_metric(h2h_stats, "home_team_goals_per_match")
        h2h_away_avg = get_h2h_metric(h2h_stats, "away_team_goals_per_match")

        if all(v is not None for v in [home_avg_scored_15, away_avg_scored_15, h2h_home_avg, h2h_away_avg]):
            # Compare scoring difference consistency
            diff_15 = abs(home_avg_scored_15 - away_avg_scored_15)
            diff_h2h = abs(h2h_home_avg - h2h_away_avg)
            if abs(diff_15 - diff_h2h) < 0.5: # If goal difference pattern is similar
                score += max_score_contribution * 0.75
                reasons.append("Recent goal scoring averages (last 15) show similar patterns to H2H averages.")
            else:
                score += min_score_contribution * 0.75
                reasons.append("Recent goal scoring averages (last 15) differ significantly from H2H averages.")
                
        # 3. Poisson Predictions vs. Statarea Probabilities (15 games)
        poisson_home_win = get_poisson_prob(match_metrics, "home_win")
        poisson_away_win = get_poisson_prob(match_metrics, "away_win")
        statarea_home_win = get_statarea_metric(home_stats, "home", 15, "outcome_probabilities_1x2", {}).get("win")
        statarea_away_win = get_statarea_metric(away_stats, "away", 15, "outcome_probabilities_1x2", {}).get("win") # Note: this is away team's win prob when playing away

        if all(v is not None for v in [poisson_home_win, poisson_away_win, statarea_home_win, statarea_away_win]):
            poisson_favors_home = poisson_home_win > poisson_away_win
            # Statarea comparison is tricky - compare home team's win prob at home vs away team's win prob away
            statarea_favors_home = statarea_home_win > statarea_away_win 
            if poisson_favors_home == statarea_favors_home:
                 score += max_score_contribution
                 reasons.append("Poisson model outcome prediction aligns with Statarea's 15-game outcome probabilities.")
            else:
                 score += min_score_contribution
                 reasons.append("Poisson model outcome prediction conflicts with Statarea's 15-game outcome probabilities.")

        # 4. Over/Under 2.5 Goals Consistency (Poisson vs Statarea 15)
        poisson_over_25 = get_poisson_prob(match_metrics, "over_2.5")
        # Use average of home/away statarea O/U pct
        home_over_25_pct = get_statarea_metric(home_stats, "home", 15, "over_2_5_pct")
        away_over_25_pct = get_statarea_metric(away_stats, "away", 15, "over_2_5_pct")
        
        if all(v is not None for v in [poisson_over_25, home_over_25_pct, away_over_25_pct]):
            statarea_avg_over_25 = (home_over_25_pct + away_over_25_pct) / 2
            if abs(poisson_over_25 - statarea_avg_over_25) < 0.15: # Threshold for agreement
                score += max_score_contribution
                reasons.append("Poisson Over/Under 2.5 prediction aligns well with Statarea's historical O/U rate.")
            else:
                score += min_score_contribution
                reasons.append("Poisson Over/Under 2.5 prediction differs significantly from Statarea's historical O/U rate.")

        # 5. BTTS Consistency (Poisson vs Statarea 15)
        poisson_btts = get_poisson_prob(match_metrics, "btts_yes")
        home_btts_pct = get_statarea_metric(home_stats, "home", 15, "btts_pct")
        away_btts_pct = get_statarea_metric(away_stats, "away", 15, "btts_pct")

        if all(v is not None for v in [poisson_btts, home_btts_pct, away_btts_pct]):
             statarea_avg_btts = (home_btts_pct + away_btts_pct) / 2
             if abs(poisson_btts - statarea_avg_btts) < 0.15: # Threshold for agreement
                 score += max_score_contribution
                 reasons.append("Poisson BTTS prediction aligns well with Statarea's historical BTTS rate.")
             else:
                 score += min_score_contribution
                 reasons.append("Poisson BTTS prediction differs significantly from Statarea's historical BTTS rate.")
                 
        # 6. Recent Volatility (Std Dev of recent results could be added here if match results were stored)
        # Placeholder: Check if recent form (last 5) is very different from longer term (last 15)
        if home_ppg_5 is not None and home_ppg_15 is not None:
            if abs(home_ppg_5 - home_ppg_15) > 0.75: # Significant change in recent form points
                 score -= 0.5 # Reduce score slightly for volatility
                 reasons.append("Home team shows significant change between short-term and longer-term form.")
        if away_ppg_5 is not None and away_ppg_15 is not None:
            if abs(away_ppg_5 - away_ppg_15) > 0.75:
                 score -= 0.5
                 reasons.append("Away team shows significant change between short-term and longer-term form.")


        # Normalize score to 0-10 range
        final_score = max(0, min(10, round(score, 1)))
        
        # Generate final reason text
        if not reasons:
            reason_text = "Insufficient comparative data to reliably assess predictability."
        elif final_score >= 7.5:
            reason_text = f"High predictability ({final_score}/10). Key indicators generally align. Reasons: " + " ".join(reasons)
        elif final_score <= 3.5:
            reason_text = f"Low predictability ({final_score}/10). Conflicting signals across metrics. Reasons: " + " ".join(reasons)
        else:
            reason_text = f"Moderate predictability ({final_score}/10). Some alignment, some conflicts. Reasons: " + " ".join(reasons)

        return {"score": final_score, "reason": reason_text}
    
    def process_match_advanced(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single match with enhanced analytics and structure.
        """
        try:
            with open(file_path, 'r') as f:
                match_data = json.load(f)
            
            fixture_info = match_data.get("fixture_info", {})
            league_info = match_data.get("league", {})
            home_team_data = match_data.get("teams", {}).get("home", {})
            away_team_data = match_data.get("teams", {}).get("away", {})
            # --- Attempt to load market odds if present ---
            market_odds_data = match_data.get("market_odds") # Assuming top-level key 'market_odds'
            
            if not fixture_info or not league_info or not home_team_data or not away_team_data:
                 print(f"Skipping {file_path}: Missing essential basic data.")
                 return {}
            
            home_team_name = home_team_data.get("name", "")
            away_team_name = away_team_data.get("name", "")
            
            home_team_stats = self.extract_team_advanced_stats(home_team_data, home_team_name)
            away_team_stats = self.extract_team_advanced_stats(away_team_data, away_team_name)
            
            h2h_data = match_data.get("h2h", [])
            h2h_stats = self.process_h2h_advanced(
                h2h_data, 
                str(home_team_data.get("id", "")), 
                str(away_team_data.get("id", ""))
            )
            
            # Calculate metrics, passing market odds if available
            match_analysis_predictions = self.calculate_match_metrics(
                home_team_stats, away_team_stats, h2h_stats, market_odds_data # Pass odds here
            )

            # Calculate Predictability Score
            predictability_result = self.calculate_predictability_score(
                home_team_stats, away_team_stats, h2h_stats, match_analysis_predictions
            )
            match_analysis_predictions["predictability"] = predictability_result

            # Create the enhanced processed structure
            processed_data = {
                "match_info": {
                    "id": fixture_info.get("id"),
                    "date": fixture_info.get("date"),
                    "venue": fixture_info.get("venue", {}).get("name"),
                    "city": fixture_info.get("venue", {}).get("city"),
                    "referee": fixture_info.get("referee"),
                    "status": fixture_info.get("status", {}).get("long")
                },
                "league": {
                    "id": league_info.get("id"),
                    "name": league_info.get("name"),
                    "country": league_info.get("country"),
                    "season": league_info.get("season"),
                    "round": league_info.get("round")
                },
                "teams": {
                    "home": home_team_stats, 
                    "away": away_team_stats 
                },
                "head_to_head": h2h_stats,
                "match_analysis": match_analysis_predictions 
                # Potentially include the raw market_odds used, for traceability
                # "market_odds_used": market_odds_data 
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def process_all_matches_advanced(self):
        """Process all match files with advanced analytics."""
        match_count = 0
        success_count = 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json') and file != "standings.json" and not file.startswith('games_summary'):
                    input_path = os.path.join(root, file)
                    match_count += 1
                    
                    # Create output filename and directory structure
                    rel_path = os.path.relpath(root, self.input_dir)
                    output_dir = os.path.join(self.output_dir, rel_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file)
                    
                    # Process the file
                    print(f"Processing: {input_path}")
                    processed_data = self.process_match_advanced(input_path)
                    
                    if processed_data:
                        # Save the processed data to JSON
                        with open(output_path, 'w') as f:
                            json.dump(processed_data, f, indent=2)
                        print(f"Saved processed data to: {output_path}")
                        success_count += 1
                    else:
                        print(f"Failed to process: {input_path}")
        
        print(f"Processed {success_count}/{match_count} matches successfully")


# Run the enhanced processor
if __name__ == "__main__":
    processor = EnhancedSoccerMatchProcessor(
        input_dir="daily_games",
        output_dir="processed_matches"
    )
    
    # Process all match files with advanced analytics
    processor.process_all_matches_advanced()
    
    print("Enhanced match processing completed!")