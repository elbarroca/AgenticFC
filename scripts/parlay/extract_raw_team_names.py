# extract_raw_team_names.py
import pandas as pd
from pathlib import Path
import json
import argparse
import re
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# --- Configuration ---
# Assuming this script is in the root of your AGENTICFC888 project directory
BASE_DIR = Path("/Users/barroca888/Downloads/Agenticfc/AgenticFC888") # Your project root
PREDICTIONS_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
OUTPUT_DIR = BASE_DIR / 'scripts' / 'config_generation_helpers'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR_FOR_TEAM_MAPPING = BASE_DIR / 'get_data' / 'api_football' / 'db_ids' # Where team_id_mappings.py might be

DEFAULT_OOF_CSV_INPUT_FILE = 'level0_oof_predictions_with_odds.csv'
DEFAULT_MATCH_ID_COL = 'MatchID'

RAW_EXTRACTED_STRINGS_JSON = OUTPUT_DIR / 'a1_raw_extracted_team_strings.json'
SUGGESTED_NORMALIZATIONS_JSON = OUTPUT_DIR / 'a2_suggested_team_normalizations.json'
POTENTIAL_MATCHID_SPLITS_JSON = OUTPUT_DIR / 'a3_potential_matchid_splits.json'


def generate_camel_case_spaced_version(name: str) -> str:
    """Converts CamelCase or PascalCase to space-separated."""
    if not name: return ""
    # Add space before capital letters, but not if it's the first letter or preceded by another capital/number
    # This also handles numbers like 1FCKoln -> 1 FC Koln
    spaced = re.sub(r"(\B[A-Z0-9])", r" \1", name)
    # Handle cases like FCKoln -> FC Koln (if FCKoln was the input)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    # Handle numbers followed by letters e.g. Mainz05 -> Mainz 05
    spaced = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", spaced)
    spaced = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", spaced)
    return spaced.strip()

def try_load_existing_team_id_mapping() -> Dict:
    """Tries to load TEAM_ID_MAPPING if team_id_mappings.py exists and is importable."""
    try:
        # This assumes team_id_mappings.py is structured to allow importing TEAM_ID_MAPPING
        # You might need to adjust sys.path if it's not directly importable
        # For simplicity, if you have it as a JSON, load it here.
        # If it's in a .py file, you'd ideally refactor that .py to be easily importable.
        
        # Attempting a placeholder load - replace with your actual loading if TEAM_ID_MAPPING is in a .py
        # from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
        # return TEAM_ID_MAPPING
        print("INFO: Attempting to load an existing TEAM_ID_MAPPING (placeholder).")
        # This is a placeholder. If you have your TEAM_ID_MAPPING in a JSON or accessible .py, load it here.
        # For now, returning an empty dict if not found easily.
        existing_mapping_path = CONFIG_DIR_FOR_TEAM_MAPPING / "team_id_mappings_for_bootstrap.json" # Example path
        if existing_mapping_path.exists():
            with open(existing_mapping_path, 'r') as f:
                return json.load(f)
        return {} # Return empty if not found
    except Exception as e:
        print(f"Warning: Could not load existing TEAM_ID_MAPPING: {e}. Proceeding without it.")
        return {}

def main(args):
    print(f"--- Starting Raw Team Name Extraction from {args.input_csv_path} ---")
    
    input_file = Path(args.input_csv_path)
    if not input_file.exists():
        print(f"CRITICAL: Input CSV file not found: {input_file}"); return

    try:
        df = pd.read_csv(input_file, usecols=[args.match_id_col], low_memory=False, dtype={args.match_id_col: str})
        print(f"Loaded {args.match_id_col} column from {input_file.name}. Shape: {df.shape}")
    except Exception as e:
        print(f"CRITICAL: Error loading CSV: {e}"); return

    if args.match_id_col not in df.columns:
        print(f"CRITICAL: MatchID column '{args.match_id_col}' not found."); return

    # --- Attempt to load existing canonical names to help with splitting ---
    # This assumes your TEAM_ID_MAPPING keys are your canonical names
    existing_canonical_names = list(try_load_existing_team_id_mapping().keys())
    # Sort by length descending to match longer names first during splitting attempts
    existing_canonical_names_sorted = sorted(existing_canonical_names, key=len, reverse=True)
    print(f"  Loaded {len(existing_canonical_names_sorted)} existing canonical team names to aid splitting.")


    all_raw_team_substrings: Set[str] = set()
    potential_matchid_splits: Dict[str, List[Tuple[str, str]]] = {} # "FullTeamsPart" -> [("HomeRaw", "AwayRaw"), ...]
    suggested_normalizations_dict: Dict[str, str] = {} # "RawSubstring" -> "Suggested Spaced Version"

    unique_match_ids = df[args.match_id_col].dropna().unique()
    print(f"Processing {len(unique_match_ids)} unique MatchIDs...")

    for match_id_str in tqdm(unique_match_ids, desc="Extracting Team Strings"):
        original_match_id = match_id_str # Keep for reference

        # 1. Extract teams part (remove date prefix if YYYYMMDD_)
        parts = match_id_str.split('_', 1)
        teams_part = match_id_str
        is_dated_format = False
        if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8:
            teams_part = parts[1]
            is_dated_format = True
        
        # Add the full teams_part to raw substrings
        all_raw_team_substrings.add(teams_part)
        if teams_part not in suggested_normalizations_dict:
            suggested_normalizations_dict[teams_part] = generate_camel_case_spaced_version(teams_part)

        # 2. Attempt to split teams_part into Home and Away
        home_cand_raw, away_cand_raw = None, None

        # Scenario A: Standard Underscore Separation (e.g., YYYYMMDD_HomeTeam_AwayTeam)
        if is_dated_format and teams_part.count('_') >= 1: # At least one more underscore for H_A
            # Try splitting by the first underscore in the teams_part
            # This assumes HomeTeam does not contain underscores, but AwayTeam might
            home_try, away_try = teams_part.split('_', 1)
            if home_try and away_try: # Both must be non-empty
                home_cand_raw, away_cand_raw = home_try, away_try
        
        # Scenario B: No underscore in teams_part (e.g., YYYYMMDD_HomeTeamAwayTeam)
        # Or, if Scenario A didn't yield a good split, try splitting concatenated names
        if not (home_cand_raw and away_cand_raw) and '_' not in teams_part:
            # Try splitting based on known canonical names (longest first)
            found_split = False
            for known_name_canon in existing_canonical_names_sorted:
                # Normalize known name for matching against concatenated string
                norm_known_name = re.sub(r'[^a-zA-Z0-9]', '', known_name_canon).lower()
                norm_teams_part = re.sub(r'[^a-zA-Z0-9]', '', teams_part).lower()

                if norm_teams_part.startswith(norm_known_name):
                    # Find the original form of the matched part
                    # This is tricky. We need to find how many chars of 'teams_part' correspond to 'norm_known_name'
                    len_matched = 0
                    temp_match_str = ""
                    for char_idx in range(len(teams_part)):
                        temp_match_str += teams_part[char_idx]
                        if re.sub(r'[^a-zA-Z0-9]', '', temp_match_str).lower() == norm_known_name:
                            len_matched = len(temp_match_str)
                            break
                    
                    if len_matched > 0 and len_matched < len(teams_part):
                        home_cand_raw = teams_part[:len_matched]
                        away_cand_raw = teams_part[len_matched:]
                        found_split = True
                        break # Found a plausible split based on known home team
            
            if not found_split: # Generic CamelCase split as last resort for concatenated
                s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', teams_part) # Add space before capital if preceded by lowercase
                s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)    # Add space if lowercase/digit followed by capital
                potential_words = s2.split(' ')
                if len(potential_words) == 2 : # Simple HomeAway
                    home_cand_raw, away_cand_raw = potential_words[0], potential_words[1]
                # More complex splits (e.g., Paris Saint Germain) are harder here without full dictionary matching

        if home_cand_raw and away_cand_raw:
            all_raw_team_substrings.add(home_cand_raw)
            all_raw_team_substrings.add(away_cand_raw)
            if home_cand_raw not in suggested_normalizations_dict:
                suggested_normalizations_dict[home_cand_raw] = generate_camel_case_spaced_version(home_cand_raw)
            if away_cand_raw not in suggested_normalizations_dict:
                suggested_normalizations_dict[away_cand_raw] = generate_camel_case_spaced_version(away_cand_raw)
            
            if original_match_id not in potential_matchid_splits:
                potential_matchid_splits[original_match_id] = []
            potential_matchid_splits[original_match_id].append((home_cand_raw, away_cand_raw))
        else: # Could not split reliably
            if original_match_id not in potential_matchid_splits:
                 potential_matchid_splits[original_match_id] = [("COULD_NOT_SPLIT", teams_part)]


    # --- Save Outputs ---
    sorted_raw_strings = sorted([s for s in list(all_raw_team_substrings) if len(s) > 1 and not s.isdigit()])
    with open(RAW_EXTRACTED_STRINGS_JSON, 'w') as f:
        json.dump(sorted_raw_strings, f, indent=4)
    print(f"\nSaved {len(sorted_raw_strings)} unique raw team strings/substrings to: {RAW_EXTRACTED_STRINGS_JSON}")

    filtered_suggested_normalizations = {
        k: v for k, v in suggested_normalizations_dict.items() if k.lower() != v.lower().replace(" ", "") and k !=v and len(k)>1
    }
    with open(SUGGESTED_NORMALIZATIONS_JSON, 'w') as f:
        json.dump(dict(sorted(filtered_suggested_normalizations.items())), f, indent=4)
    print(f"Saved {len(filtered_suggested_normalizations)} suggested normalizations to: {SUGGESTED_NORMALIZATIONS_JSON}")

    with open(POTENTIAL_MATCHID_SPLITS_JSON, 'w') as f:
        json.dump(dict(sorted(potential_matchid_splits.items())), f, indent=4)
    print(f"Saved {len(potential_matchid_splits)} potential MatchID splits to: {POTENTIAL_MATCHID_SPLITS_JSON}")
    
    print_next_steps()


def print_next_steps():
    print("\n--- Next Steps ---")
    print(f"1. Manually review '{RAW_EXTRACTED_STRINGS_JSON}'.")
    print(f"   From this list, identify all unique CANONICAL team names (e.g., 'Standard Liege', 'Paris Saint Germain').")
    print(f"2. Review '{POTENTIAL_MATCHID_SPLITS_JSON}'.")
    print(f"   This shows how MatchIDs were split. Verify if 'HomeRaw' and 'AwayRaw' are correct.")
    print(f"   Use this to identify raw team name variations that appear in your MatchIDs.")
    print(f"3. Review '{SUGGESTED_NORMALIZATIONS_JSON}'.")
    print(f"   This suggests how raw extracted strings (like 'StandardLiege') could map to space-separated versions ('Standard Liege').")
    print(f"4. Create/Update your `TEAM_ID_MAPPING` dictionary (e.g., in team_id_mappings.py or a new JSON to be loaded):")
    print("   TEAM_ID_MAPPING = {")
    print("     \"Standard Liege\": {\"country\": \"Belgium\", \"statarea_id\": \"733\", ..., \"alternative_names\": [\"Standard\", \"StandardLiege\"]},")
    print("     \"Paris Saint Germain\": {\"country\": \"France\", ..., \"alternative_names\": [\"Paris SG\", \"PSG\", \"ParisSaintGermain\"]},")
    print("     # For each canonical name, add its details and any raw variations as 'alternative_names'.")
    print("   }")
    print(f"5. Create/Update your `TEAM_NAME_NORMALIZATION` dictionary (e.g., in team_id_mappings.py or a new JSON):")
    print("   # This maps very raw/concatenated forms from MatchIDs directly to a canonical name found in TEAM_ID_MAPPING keys.")
    print("   TEAM_NAME_NORMALIZATION = {")
    print("     \"StandardLiegeZulteWaregem\": null, # This whole string is not a team, so map to null or handle via split")
    print("     \"StandardLiege\": \"Standard Liege\",")
    print("     \"ZulteWaregem\": \"Zulte Waregem\",")
    print("     \"ParisSaintGermain\": \"Paris Saint Germain\",")
    print("     \"KVCOosterlo\": \"KVC Westerlo\", # Map raw form to your chosen canonical name")
    print("     # ... etc. The goal is that any distinct team string part from a MatchID gets mapped to a canonical name.")
    print("   }")
    print("Your `TeamNameResolver` class will then use these two comprehensive mappings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and suggest normalizations for team names from MatchID column.")
    parser.add_argument("--input_csv_path", type=str, 
                        default=str(PREDICTIONS_DIR / DEFAULT_OOF_CSV_INPUT_FILE),
                        help=f"Path to the input OOF CSV file. Default: {PREDICTIONS_DIR / DEFAULT_OOF_CSV_INPUT_FILE}")
    parser.add_argument("--match_id_col", type=str, default=DEFAULT_MATCH_ID_COL,
                        help=f"Name of the MatchID column in the CSV. Default: {DEFAULT_MATCH_ID_COL}")
    
    args = parser.parse_args()
    main(args)