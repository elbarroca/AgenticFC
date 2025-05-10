# generate_market_definitions.py
import json
from pathlib import Path
import re

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / 'models' / 'config'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_PATH = CONFIG_DIR / 'parlay_market_definitions.json'

CORE_PREDICTION_KEYS = [
    # Lambdas - We generally don't treat raw lambdas as parlayable markets directly for this structure
    # 'expected_HG', 'expected_AG', # Let's exclude these from PARLAY_MARKET_DEFINITIONS for now
    # Single outcomes
    'prob_H', 'prob_D', 'prob_A',
    'prob_1X', 'prob_12', 'prob_X2',
    'prob_O05', 'prob_U05',
    'prob_O15', 'prob_U15',
    'prob_O25', 'prob_U25',
    'prob_O35', 'prob_U35',
    'prob_O45', 'prob_U45',
    'prob_BTTS_Y', 'prob_BTTS_N',
    # Multiple outcomes - Match Result + Goals
    'prob_H_and_O05', 'prob_D_and_O05', 'prob_A_and_O05',
    'prob_H_and_U05', 'prob_D_and_U05', 'prob_A_and_U05',
    'prob_H_and_O15', 'prob_D_and_O15', 'prob_A_and_O15',
    'prob_H_and_U15', 'prob_D_and_U15', 'prob_A_and_U15',
    'prob_H_and_O25', 'prob_D_and_O25', 'prob_A_and_O25',
    'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
    'prob_H_and_O35', 'prob_D_and_O35', 'prob_A_and_O35',
    'prob_H_and_U35', 'prob_D_and_U35', 'prob_A_and_U35',
    'prob_H_and_O45', 'prob_D_and_O45', 'prob_A_and_O45',
    'prob_H_and_U45', 'prob_D_and_U45', 'prob_A_and_U45',
    # Multiple outcomes - Double Chance + Goals
    'prob_1X_and_O05', 'prob_12_and_O05', 'prob_X2_and_O05',
    'prob_1X_and_U05', 'prob_12_and_U05', 'prob_X2_and_U05',
    'prob_1X_and_O15', 'prob_12_and_O15', 'prob_X2_and_O15',
    'prob_1X_and_U15', 'prob_12_and_U15', 'prob_X2_and_U15',
    'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25',
    'prob_1X_and_U25', 'prob_12_and_U25', 'prob_X2_and_U25',
    'prob_1X_and_O35', 'prob_12_and_O35', 'prob_X2_and_O35',
    'prob_1X_and_U35', 'prob_12_and_U35', 'prob_X2_and_U35',
    'prob_1X_and_O45', 'prob_12_and_O45', 'prob_X2_and_O45',
    'prob_1X_and_U45', 'prob_12_and_U45', 'prob_X2_and_U45',
    # Multiple outcomes - Match Result + BTTS
    'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y', 'prob_A_and_BTTS_Y',
    'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
    # Multiple outcomes - Double Chance + BTTS
    'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y',
    'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
    # Multiple outcomes - Goals + BTTS (Added O/U 0.5 and 1.5 variants)
    'prob_O05_and_BTTS_Y', 'prob_O05_and_BTTS_N',
    'prob_U05_and_BTTS_Y', 'prob_U05_and_BTTS_N', # U0.5_BTTS_Y is impossible, U0.5_BTTS_N is just U0.5
    'prob_O15_and_BTTS_Y', 'prob_O15_and_BTTS_N',
    'prob_U15_and_BTTS_Y', 'prob_U15_and_BTTS_N',
    'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N',
    'prob_U25_and_BTTS_Y', 'prob_U25_and_BTTS_N',
    'prob_O35_and_BTTS_Y', 'prob_O35_and_BTTS_N',
    'prob_U35_and_BTTS_Y', 'prob_U35_and_BTTS_N',
    'prob_O45_and_BTTS_Y', 'prob_O45_and_BTTS_N',
    'prob_U45_and_BTTS_Y', 'prob_U45_and_BTTS_N',
]


def generate_market_label_and_base(market_base_from_prob_key: str) -> tuple:
    """Generates a readable market label and identifies base components."""
    # market_base_from_prob_key is like "H", "O25", "H_and_O15", "1X_and_BTTS_Y"
    
    label = market_base_from_prob_key
    base_target_suffixes = [] # These are suffixes like "H", "O15", "BTTS_Y"
    op = None

    if "_and_" in label:
        op = "and"
        parts = label.split("_and_")
        base_target_suffixes = [parts[0], parts[1]]
        # Create a more compact label
        label = f"{parts[0]}{parts[1].replace('BTTS_','BTTS')}" # e.g. HBTTSY, 1XO15
    else: # Single outcomes
        base_target_suffixes = [label]
        # Simple label adjustments
        if label == "1X": label = "HomeOrDraw"
        elif label == "X2": label = "DrawOrAway"
        elif label == "12": label = "HomeOrAway"
        elif label == "BTTS_Y": label = "BTTSYes"
        elif label == "BTTS_N": label = "BTTSNo"
        elif label.startswith("O") or label.startswith("U"): # O05, U25
            pass # Keep as is, e.g. O05, U25
    
    # Further simplify label by removing underscores if any remain from single parts
    label = label.replace("_", "")
    return label, base_target_suffixes, op


def generate_parlay_market_definitions(core_keys: list) -> dict:
    definitions = {}
    temp_market_labels = set() # To ensure unique generated market_labels

    for key in core_keys:
        if not key.startswith("prob_"):
            continue

        prob_suffix_as_key = key # e.g., "prob_H", "prob_H_and_O25"
        market_base = key[len("prob_"):] # e.g., "H", "H_and_O25"
        
        market_label, base_target_suffixes, op = generate_market_label_and_base(market_base)
        
        # Ensure unique market_label (dictionary key)
        final_market_label = market_label
        counter = 1
        while final_market_label in temp_market_labels:
            final_market_label = f"{market_label}_{counter}"
            counter += 1
        temp_market_labels.add(final_market_label)

        target_col = f"target_{market_base}" # Always based on the original market_base
        
        # Construct full base_target column names
        actual_base_targets = [f"target_{bts}" for bts in base_target_suffixes]

        # Define conflict group (can be refined further if needed)
        conflict_group = market_base
        if op == "or" and ("H" in market_base or "D" in market_base or "A" in market_base) and \
           not market_base.startswith(("O", "U", "BTTS")):
            conflict_group = "DC_Result" # For 1X, X2, 12
        elif (market_base.startswith("O") or market_base.startswith("U")) and "_and_" not in market_base:
            conflict_group = f"OU{market_base[1:]}" # OU05, OU15
        elif market_base.startswith("BTTS") and "_and_" not in market_base:
            conflict_group = "BTTS_Outcome"
            
        definitions[final_market_label] = {
            "prob_suffix": prob_suffix_as_key, # This is the key from CORE_PREDICTION_KEYS
            "target_col": target_col,
            "base_targets": actual_base_targets,
            "op": op,
            "conflict_group": conflict_group
        }
        
    # Manual adjustments for specific impossible/redundant outcomes
    if "U05BTTSYes" in definitions: # from prob_U05_and_BTTS_Y
        print("Note: U0.5 and BTTS Yes is an impossible outcome. Removing 'U05BTTSYes'.")
        definitions.pop("U05BTTSYes", None)
    if "U05BTTSNo" in definitions: # from prob_U05_and_BTTS_N
        # U0.5 and BTTS No is simply U0.5. We should ensure U0.5 is defined and remove this redundant one.
        # Or, if U0.5 is not explicitly a market label, rename this one.
        # For now, let's assume U0.5 (market_label "U05") exists.
        # The target_col for U05BTTSNo would be target_U05_and_BTTS_N
        # Base targets would be ["target_U05", "target_BTTS_N"]
        # This is actually a valid combo, meaning exactly 0-0.
        # The conflict group should be OU05.
        definitions["U05BTTSNo"]["conflict_group"] = "OU05"
        # Its base targets are ["target_U05", "target_BTTS_N"], op "and". This seems correct.

    return definitions

if __name__ == "__main__":
    print("--- Generating PARLAY_MARKET_DEFINITIONS ---")
    parlay_market_definitions = generate_parlay_market_definitions(CORE_PREDICTION_KEYS)
    
    print(f"\nGenerated {len(parlay_market_definitions)} market definitions.")
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(parlay_market_definitions, f, indent=4)
    
    print(f"\nSuccessfully saved PARLAY_MARKET_DEFINITIONS to: {OUTPUT_JSON_PATH}")
    print("\nPlease review the generated JSON file for accuracy, especially:")
    print("- Dictionary keys (market_label)")
    print("- 'prob_suffix' (should match CORE_PREDICTION_KEYS entries)")
    print("- 'target_col' (e.g., target_H, target_H_and_O25)")
    print("- 'base_targets' (e.g., ['target_H', 'target_O25']) and 'op'")

    print("\nSample of generated definitions (first 5 and a dual example):")
    count = 0
    dual_example_shown = False
    for k, v in parlay_market_definitions.items():
        if count < 5:
            print(f"'{k}': {v}")
            count += 1
        if "_and_" in v['prob_suffix'] and not dual_example_shown :
            if count >=5: # only print if not already printed
                 print(f"'{k}': {v}")
            dual_example_shown = True
        if count >=5 and dual_example_shown:
            break
    # Find and print specific example like 1X_O15
    if "1XO15" in parlay_market_definitions:
        print("\nExample for 1X_O15 (key '1XO15'):")
        print(f"'1XO15': {parlay_market_definitions['1XO15']}")
    elif "HomeOrDrawO15" in parlay_market_definitions: # Check alternative label
        print("\nExample for 1X_O15 (key 'HomeOrDrawO15'):")
        print(f"'HomeOrDrawO15': {parlay_market_definitions['HomeOrDrawO15']}")