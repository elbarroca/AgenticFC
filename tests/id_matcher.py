import os
import logging
from pymongo import MongoClient, ReadPreference
from dotenv import load_dotenv
import unicodedata
import re

# Import your existing team data
from get_data.db_ids.team_data import TEAM_DATA
from get_data.db_ids.league_ids import LEAGUE_IDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_to_mongodb():
    """Connect to MongoDB using environment variables"""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    
    if not mongo_uri:
        logger.error("MONGO_URI environment variable not set")
        raise ValueError("MONGO_URI environment variable not set")
    
    try:
        # Use SECONDARY_PREFERRED to allow reading from secondary nodes
        client = MongoClient(mongo_uri, 
                            serverSelectionTimeoutMS=20000,
                            readPreference='secondaryPreferred')
        
        # Try to ping using secondary preferred
        client.admin.command('ping', read_preference=ReadPreference.SECONDARY_PREFERRED)
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def get_teams_from_mongodb(client, date_range=None):
    """
    Extract unique teams and their IDs from the MongoDB games collection
    Returns a dictionary of {team_name: mongo_id}
    """
    teams_dict = {}
    db = client['games']
    
    # Get all month collections
    collections = [coll for coll in db.list_collection_names() if coll.startswith('month_')]
    
    for collection_name in collections:
        collection = db[collection_name]
        
        # Find all game documents
        for doc in collection.find({}):
            if 'data' not in doc:
                continue
                
            # Process leagues
            if 'leagues' in doc['data']:
                for league_id, league_data in doc['data']['leagues'].items():
                    if 'matches' not in league_data:
                        continue
                        
                    # Process matches
                    for match in league_data['matches']:
                        if 'home_team' in match and 'away_team' in match:
                            # Add home team
                            if 'name' in match['home_team'] and 'id' in match['home_team']:
                                home_name = match['home_team']['name']
                                home_id = match['home_team']['id']
                                teams_dict[home_name] = home_id
                                
                            # Add away team
                            if 'name' in match['away_team'] and 'id' in match['away_team']:
                                away_name = match['away_team']['name']
                                away_id = match['away_team']['id']
                                teams_dict[away_name] = away_id
    
    logger.info(f"Found {len(teams_dict)} unique teams in MongoDB")
    return teams_dict

def get_leagues_from_mongodb(client):
    """
    Extract unique leagues and their IDs from the MongoDB games collection
    Returns a dictionary of {league_name: mongo_id}
    """
    leagues_dict = {}
    db = client['games']
    
    # Get all month collections
    collections = [coll for coll in db.list_collection_names() if coll.startswith('month_')]
    
    for collection_name in collections:
        collection = db[collection_name]
        
        # Find all game documents
        for doc in collection.find({}):
            if 'data' not in doc:
                continue
                
            # Process leagues
            if 'leagues' in doc['data']:
                for league_id, league_data in doc['data']['leagues'].items():
                    if 'name' in league_data and 'country' in league_data:
                        league_name = league_data['name']
                        country = league_data['country']
                        # Include country in the name to avoid ambiguity
                        full_name = f"{league_name} ({country})"
                        leagues_dict[full_name] = league_id
    
    logger.info(f"Found {len(leagues_dict)} unique leagues in MongoDB")
    return leagues_dict

def create_id_mapping(mongo_teams, team_data):
    """Create a mapping between MongoDB team IDs and API team IDs with improved matching"""
    def normalize_name(name):
        # Remove accents
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        # Convert to lowercase
        name = name.lower()
        # Replace common abbreviations
        name = name.replace(' utd', ' united').replace(' fc', '').replace(' cf', '')
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        return name.strip()
    
    mapping = {}
    missing_teams = []
    
    # Prepare normalized versions of team_data names for faster lookup
    normalized_team_data = {}
    for td_name, td_info in team_data.items():
        normalized_team_data[normalize_name(td_name)] = (td_name, td_info)
    
    for team_name, mongo_id in mongo_teams.items():
        normalized_name = normalize_name(team_name)
        
        # Try exact match with normalized name
        if normalized_name in normalized_team_data:
            original_name, td_info = normalized_team_data[normalized_name]
            api_id = td_info["api_id"]
            mapping[mongo_id] = {
                "name": team_name,
                "api_name": original_name,
                "api_id": api_id,
                "country": td_info["country"]
            }
            continue
            
        # Try partial match
        matched = False
        best_match = None
        highest_score = 0
        
        for norm_td_name, (original_td_name, td_info) in normalized_team_data.items():
            # Simple partial match
            if normalized_name in norm_td_name or norm_td_name in normalized_name:
                # Calculate match score based on length of common substring
                common = set(normalized_name.split()).intersection(set(norm_td_name.split()))
                score = len(common) / max(len(normalized_name.split()), len(norm_td_name.split()))
                
                if score > highest_score:
                    highest_score = score
                    best_match = (original_td_name, td_info)
        
        # Use best match if score is above threshold
        if highest_score > 0.3 and best_match:
            original_td_name, td_info = best_match
            api_id = td_info["api_id"]
            mapping[mongo_id] = {
                "name": team_name,
                "api_name": original_td_name,
                "api_id": api_id,
                "country": td_info["country"],
                "match_score": highest_score
            }
            logger.info(f"Fuzzy matched: '{team_name}' -> '{original_td_name}' (score: {highest_score:.2f})")
            matched = True
        
        if not matched:
            missing_teams.append(team_name)
    
    logger.info(f"Created mapping for {len(mapping)} teams")
    if missing_teams:
        logger.warning(f"Could not find mappings for {len(missing_teams)} teams: {missing_teams[:10]}...")
    
    return mapping

def create_league_mapping(mongo_leagues, statarea_league_ids):
    """
    Create a mapping between MongoDB league IDs and statarea league IDs
    This is a simplistic approach and may need manual review
    """
    # First, extract league names from MongoDB
    mongo_league_names = {v: k for k, v in mongo_leagues.items()}
    
    # For now, just create a simple mapping file with IDs from both sources
    # The actual matching will likely require manual verification
    mapping = {}
    
    for mongo_id in mongo_league_names:
        league_name = mongo_league_names[mongo_id]
        # Add entry with placeholder for statarea ID
        mapping[mongo_id] = {
            "name": league_name,
            "statarea_id": "unknown"
        }
    
    logger.info(f"Created league ID mapping skeleton for {len(mapping)} leagues")
    return mapping

def save_mappings_to_file(team_mapping, league_mapping):
    """Save both mappings as separate Python files with a more readable format"""
    # Save team mappings
    with open("team_id_mappings.py", "w") as f:
        # Write file header
        f.write("# Team ID Mapping: Team Name | Statarea ID | MongoDB ID\n\n")
        f.write("TEAM_ID_MAPPING = {\n")
        
        for mongo_id, info in sorted(team_mapping.items(), key=lambda x: x[1].get('api_name', x[1]['name'])):
            team_name = info.get('api_name', info['name'])
            api_id = info['api_id']
            country = info['country']
            
            f.write(f"    # {team_name} | {api_id} | {mongo_id}\n")
            f.write(f"    \"{team_name}\": {{\n")
            f.write(f"        \"statarea_id\": \"{api_id}\",\n")
            f.write(f"        \"mongodb_id\": \"{mongo_id}\",\n")
            f.write(f"        \"country\": \"{country}\"\n")
            f.write(f"    }},\n")
        
        f.write("}\n")
    
    logger.info(f"Saved team mappings to team_id_mappings.py")
    
    # Save league mappings
    with open("league_id_mappings.py", "w") as f:
        # Write file header
        f.write("# League ID Mapping: League Name | Statarea ID | MongoDB ID\n\n")
        f.write("LEAGUE_ID_MAPPING = {\n")
        
        for mongo_id, info in sorted(league_mapping.items(), key=lambda x: x[1]['name']):
            league_name = info['name']
            statarea_id = info['statarea_id']
            
            f.write(f"    # {league_name} | {statarea_id} | {mongo_id}\n")
            f.write(f"    \"{league_name}\": {{\n")
            f.write(f"        \"statarea_id\": \"{statarea_id}\",\n")
            f.write(f"        \"mongodb_id\": \"{mongo_id}\"\n")
            f.write(f"    }},\n")
        
        f.write("}\n")
    
    logger.info(f"Saved league mappings to league_id_mappings.py")

def main():
    # Connect to MongoDB
    client = connect_to_mongodb()
    
    try:
        # Get teams from MongoDB
        mongo_teams = get_teams_from_mongodb(client)
        
        # Get leagues from MongoDB
        mongo_leagues = get_leagues_from_mongodb(client)
        
        # Create team mapping
        team_mapping = create_id_mapping(mongo_teams, TEAM_DATA)
        
        # Create league mapping scaffold
        league_mapping = create_league_mapping(mongo_leagues, LEAGUE_IDS)
        
        # Save mappings to file
        save_mappings_to_file(team_mapping, league_mapping)
        
        # Print some stats
        logger.info(f"Successfully created mapping for {len(team_mapping)} teams")
        logger.info(f"Team coverage: {len(team_mapping)}/{len(mongo_teams)} MongoDB teams mapped ({len(team_mapping)/len(mongo_teams)*100:.1f}%)")
        logger.info(f"Created league mapping structure for {len(league_mapping)} leagues")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()