#!/usr/bin/env python3
"""
Team Information Consolidator

This script processes all teams from team_id_mappings and league_id_mappings,
compiling comprehensive information including IDs, league, country, and
alternative names into a single JSON file.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import sys
import os

# Add the project root to the path so we can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from get_data
from get_data.api_football.db_ids.league_id_mappings import LEAGUE_ID_MAPPING
from models.utils.config import INITIAL_TEAM_ID_MAPPING as TEAM_ID_MAPPING

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Output file paths
OUTPUT_DIR = Path("models/config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONSOLIDATED_OUTPUT_FILE = OUTPUT_DIR / "consolidated_team_info.json"
MISSING_INFO_FILE = OUTPUT_DIR / "teams_missing_info.json"

def create_country_league_mapping() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a mapping from country to available leagues in that country.
    
    Returns:
        Dict mapping country names to list of league information
    """
    country_to_leagues = {}
    
    for league_name, league_info in LEAGUE_ID_MAPPING.items():
        # Extract country from league name (format is typically "League Name (Country)")
        if "(" in league_name and ")" in league_name:
            country = league_name.split("(")[1].split(")")[0].strip()
            
            if country not in country_to_leagues:
                country_to_leagues[country] = []
            
            country_to_leagues[country].append({
                "name": league_name,
                "statarea_id": league_info.get("statarea_id"),
                "mongodb_id": league_info.get("mongodb_id"),
                "directory_name": league_info.get("directory_name"),
                "form_chars": league_info.get("form_chars", ["W", "D", "L"])
            })
    
    return country_to_leagues

def get_alternative_names(team_name: str, team_info: Dict[str, Any]) -> List[str]:
    """
    Extract alternative names for a team from the team_info dictionary.
    
    Args:
        team_name: The canonical team name
        team_info: Team information dictionary
        
    Returns:
        List of alternative names
    """
    alt_names = []
    
    # Check the "alternative_names" key (from team_id_mappings.py)
    if "alternative_names" in team_info:
        alt_names.extend(team_info["alternative_names"])
    
    # Check the "alt" key (from team_league_mapper.py)
    if "alt" in team_info:
        alt_names.extend(team_info["alt"])
    
    # Ensure uniqueness and remove the canonical name if it's in the alt list
    return list(set(alt for alt in alt_names if alt != team_name))

def consolidate_team_information() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Process all teams and consolidate their information.
    
    Returns:
        Tuple of (consolidated team info dict, teams with missing info dict)
    """
    country_to_leagues = create_country_league_mapping()
    consolidated_data = {}
    missing_info = {}
    
    for team_name, team_info in TEAM_ID_MAPPING.items():
        country = team_info.get("country")
        statarea_id = team_info.get("statarea_id")
        mongodb_id = team_info.get("mongodb_id")
        
        # Get or infer league information
        leagues = []
        if country and country in country_to_leagues:
            leagues = country_to_leagues[country]
        
        # Get alternative names
        alt_names = get_alternative_names(team_name, team_info)
        
        # Create consolidated entry
        team_entry = {
            "canonical_name": team_name,
            "country": country,
            "statarea_id": statarea_id,
            "mongodb_id": mongodb_id,
            "leagues": [{"name": league["name"], "id": league["mongodb_id"]} for league in leagues],
            "alt_names": alt_names
        }
        
        consolidated_data[team_name] = team_entry
        
        # Check for missing information
        missing = []
        if not country or country == "Unknown":
            missing.append("country")
        if not statarea_id:
            missing.append("statarea_id")
        if not mongodb_id:
            missing.append("mongodb_id")
        if not leagues:
            missing.append("league")
            
        if missing:
            missing_info[team_name] = {
                "missing_fields": missing,
                "current_info": team_entry
            }
    
    return consolidated_data, missing_info

def main():
    logger.info("Starting team information consolidation...")
    
    # Process all teams
    consolidated_data, missing_info = consolidate_team_information()
    
    # Write consolidated data to file
    with open(CONSOLIDATED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
    
    # Write missing info to file
    with open(MISSING_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(missing_info, f, indent=2, ensure_ascii=False)
    
    # Report statistics
    total_teams = len(consolidated_data)
    teams_with_issues = len(missing_info)
    
    logger.info(f"Processed {total_teams} teams total")
    logger.info(f"Found {teams_with_issues} teams with missing information")
    logger.info(f"Consolidated data written to {CONSOLIDATED_OUTPUT_FILE}")
    logger.info(f"Missing information report written to {MISSING_INFO_FILE}")

if __name__ == "__main__":
    main()