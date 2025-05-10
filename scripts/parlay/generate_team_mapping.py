import pandas as pd
import os
import glob
import sys
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path for consistent imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Attempt to import existing mappings ---
try:
    from models.utils.config import TEAM_NAME_MAPPING as EXISTING_CSV_MAP, RAW_CSV_DIR
    logging.info("Successfully imported existing TEAM_NAME_MAPPING and RAW_CSV_DIR from config.")
except ImportError:
    logging.error("Fatal: Could not import TEAM_NAME_MAPPING or RAW_CSV_DIR from models.utils.config. Ensure config.py is accessible and correctly structured.")
    sys.exit(1) # Exit if config is essential and missing

# --- Import fuzzy matching ---
try:
    from thefuzz import process as fuzz_process
    FUZZY_ENABLED = True
    logging.info("Successfully imported 'thefuzz' for fuzzy matching suggestions.")
except ImportError:
    FUZZY_ENABLED = False
    logging.warning("'thefuzz' library not found. Fuzzy matching suggestions disabled. Run 'pip install thefuzz python-Levenshtein' to enable.")

try:
    # Assuming TEAM_ID_MAPPING contains the 'standard' names we want to map to
    from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING as STANDARD_NAME_MAP
    logging.info("Successfully imported standard team names (TEAM_ID_MAPPING).")
    STANDARD_NAMES = list(STANDARD_NAME_MAP.keys()) # Keep as list for fuzzy matching
except ImportError:
    logging.warning("Could not import TEAM_ID_MAPPING from get_data.api_football.db_ids.team_id_mappings. Suggestions based on standard names will be unavailable.")
    STANDARD_NAME_MAP = {}
    STANDARD_NAMES = []

# --- Helper Functions ---

def find_csv_files(directory: str) -> list[str]:
    """Finds all CSV files recursively within a directory."""
    if not os.path.isdir(directory):
        logging.error(f"Specified CSV directory does not exist: {directory}")
        return []
    pattern = os.path.join(directory, '**', '*.csv')
    files = glob.glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} CSV files in '{directory}'.")
    return files

def extract_team_names_from_csv(filepath: str) -> set[str]:
    """Extracts unique team names from potential Home/Away columns of a CSV."""
    team_names = set()
    possible_cols = ['HomeTeam', 'AwayTeam', 'Home', 'Away', 'HT', 'AT']
    cols_to_read = []

    try:
        # Peek at header to find relevant columns without reading full file initially
        header = pd.read_csv(filepath, nrows=0, low_memory=False).columns.tolist()
    except UnicodeDecodeError:
        try:
            header = pd.read_csv(filepath, nrows=0, encoding='ISO-8859-1', low_memory=False).columns.tolist()
        except Exception as e:
            logging.warning(f"Could not read header for {os.path.basename(filepath)}: {e}")
            return team_names # Skip file if header can't be read

    cols_to_read = [col for col in possible_cols if col in header]

    if not cols_to_read:
        logging.debug(f"No relevant team columns found in {os.path.basename(filepath)}. Skipping.")
        return team_names

    try:
        # Read only the necessary columns
        try:
            df = pd.read_csv(filepath, usecols=cols_to_read, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, usecols=cols_to_read, encoding='ISO-8859-1', low_memory=False)

        for col in cols_to_read:
            # Standardize processing: dropna, convert to string, strip whitespace
            unique_in_col = df[col].dropna().astype(str).str.strip().unique()
            team_names.update(unique_in_col)

    except Exception as e:
        logging.warning(f"Error processing team names in {os.path.basename(filepath)}: {e}")

    # Remove any empty strings that might have resulted
    team_names.discard('')
    return team_names

def suggest_mappings(csv_names: set[str], existing_csv_map: dict, standard_name_map: dict) -> dict:
    """
    Identifies unmapped CSV names and suggests potential standard names and their IDs,
    using exact, case-insensitive, and fuzzy matching if available.
    Returns dict: csv_name -> (suggested_standard_name, score, type, ids_dict | None)
    """
    unmapped_csv_names = csv_names - set(existing_csv_map.keys())
    # Store suggestions: csv_name -> (suggested_standard, score, type, ids_dict)
    suggestions = {}
    SIMILARITY_THRESHOLD = 80 # Minimum score for fuzzy match suggestion (0-100)
    standard_names_list = list(standard_name_map.keys()) # For fuzzy matching

    logging.info(f"Found {len(unmapped_csv_names)} unique team names from CSVs that are not keys in the current TEAM_NAME_MAPPING.")

    mapped_values = set(existing_csv_map.values())
    truly_unmapped = {name for name in unmapped_csv_names if name not in mapped_values}

    logging.info(f"{len(truly_unmapped)} names need potential mapping entries.")

    if not standard_names_list:
        logging.warning("No standard names available to suggest mappings against.")
        for csv_name in sorted(list(truly_unmapped)):
            suggestions[csv_name] = ("???", 0, "No Standard Names", None)
        return suggestions

    for csv_name in sorted(list(truly_unmapped)):
        suggested_standard = "???"
        match_score = 0
        match_type = "Manual Required"
        ids = None

        # 1. Check for exact match (case-sensitive) in standard names
        if csv_name in standard_name_map:
            suggested_standard = csv_name
            match_score = 100
            match_type = "Exact Match (Standard)"
            ids = standard_name_map.get(suggested_standard) # Get IDs
        else:
            # 2. Check for case-insensitive match
            potential_matches_ci = [s_name for s_name in standard_names_list if s_name.lower() == csv_name.lower()]
            if len(potential_matches_ci) == 1:
                suggested_standard = potential_matches_ci[0]
                match_score = 100 # Treat case-insensitive as good match
                match_type = "Case-Insensitive Match"
                ids = standard_name_map.get(suggested_standard) # Get IDs
            elif FUZZY_ENABLED:
                # 3. Use fuzzy matching if no exact/case-insensitive match found
                # Use the original standard_name_map keys (the list standard_names_list) for matching
                best_match = fuzz_process.extractOne(csv_name, standard_names_list)
                if best_match and best_match[1] >= SIMILARITY_THRESHOLD:
                    suggested_standard = best_match[0]
                    match_score = best_match[1]
                    match_type = "Fuzzy Match"
                    ids = standard_name_map.get(suggested_standard) # Get IDs

        suggestions[csv_name] = (suggested_standard, match_score, match_type, ids)

    return suggestions

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Team Mapping Suggestion Script ---")

    csv_files = find_csv_files(RAW_CSV_DIR)
    if not csv_files:
        logging.error("No CSV files found to process. Exiting.")
        sys.exit(1)

    all_csv_team_names = set()
    for i, f in enumerate(csv_files):
        logging.debug(f"Scanning file {i+1}/{len(csv_files)}: {os.path.basename(f)}...")
        all_csv_team_names.update(extract_team_names_from_csv(f))

    if not all_csv_team_names:
         logging.error("No team names could be extracted from any CSV file. Exiting.")
         sys.exit(1)

    logging.info(f"Total unique, non-empty team names found in CSVs: {len(all_csv_team_names)}")

    # --- Reporting ---
    print("\n" + "="*70) # Wider separator
    print(" Team Name Mapping Analysis & Suggestions (Pasteable Format)")
    print("="*70)

    print(f"\nUnique names found in CSV files: {len(all_csv_team_names)}")
    print(f"Names currently mapped as keys in config.TEAM_NAME_MAPPING: {len(EXISTING_CSV_MAP)}")
    if STANDARD_NAMES:
        print(f"Available 'standard' names for mapping: {len(STANDARD_NAMES)}")
        print(f"Fuzzy matching suggestions: {'Enabled' if FUZZY_ENABLED else 'Disabled (install thefuzz)'}")

    # Pass the full STANDARD_NAME_MAP to suggest_mappings
    suggested_mappings_details = suggest_mappings(all_csv_team_names, EXISTING_CSV_MAP, STANDARD_NAME_MAP)

    if suggested_mappings_details:
        print("\n--- Mappings to Add/Review in models/utils/config.py ---")
        print("# Copy and paste the lines below into your TEAM_NAME_MAPPING dictionary.")
        print("# Review suggestions carefully, especially fuzzy matches.")
        # print("# Format: 'CSV_Name': 'Suggested_Standard_Name', # Match Type (Score % | IDs)") # Commented out as per request
        print("-" * 30)
        # Sort by CSV name for consistent output
        for csv_name, (suggested, score, type, ids) in sorted(suggested_mappings_details.items()):
            comment = "" # Start with empty comment
            if suggested == "???":
                 comment = " # Manual mapping required."
            else:
                 comment = f" # {type}"
                 if score > 0 and score < 100: # Add score only if not perfect match
                     comment += f" ({score}%)"
                 if ids:
                     # Safely get IDs, default to '?' if missing
                     stat_id = ids.get('statarea_id', '?')
                     mongo_id = ids.get('mongodb_id', '?')
                     comment += f" | IDs: stat={stat_id}, mongo={mongo_id}"
                 elif suggested != "???": # Add note if IDs missing for a suggested name
                      comment += " | IDs: Not Found!"


            # Handle potential quotes in names for clean output using repr()
            csv_name_repr = repr(csv_name)
            suggested_repr = repr(suggested) # Use repr for suggested name too

            print(f"    {csv_name_repr}: {suggested_repr},{comment}")

        print("-" * 30)
        print(f"\nPlease update the TEAM_NAME_MAPPING dictionary in 'models/utils/config.py'.")
    else:
        print("\n--- Verification Result ---")
        print("All unique team names found in the CSV files appear to be handled:")
        print("  - Either they are keys in the TEAM_NAME_MAPPING.")
        print("  - Or they match a standard name (value) already used in the mapping.")
        print("The current mapping seems comprehensive based on the scanned files.")

    print("\n" + "="*70)
    logging.info("--- Team Mapping Suggestion Script Finished ---")