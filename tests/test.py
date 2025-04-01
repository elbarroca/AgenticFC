import http.client
import os
import json
import unicodedata
from datetime import datetime
from dotenv import load_dotenv
from get_data.statarea.league_ids import LEAGUE_IDS

# Load environment variables
load_dotenv()

# API configuration
API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please create a .env file with API_FOOTBALL_KEY")

API_HOST = "api-football-v1.p.rapidapi.com"
BASE_ENDPOINT = "/v3/fixtures"

def normalize_text(text):
    """Convert Unicode characters to their closest ASCII equivalents and simplify text"""
    if not text:
        return ""
    
    # Normalize Unicode characters (ë -> e, ü -> u, etc.)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    
    # Remove any remaining special characters and simplify
    simplified = ''.join(c for c in ascii_text if c.isalnum() or c in (' ', '-'))
    return simplified.strip()

def get_user_date():
    """Prompt user to enter a date or use today's date"""
    today = datetime.now().strftime("%Y-%m-%d")
    while True:
        user_input = input(f"Enter date (YYYY-MM-DD) [default: {today}]: ").strip()
        if not user_input:
            return today
        try:
            datetime.strptime(user_input, "%Y-%m-%d")
            return user_input
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

def simplify_fixture_data(fixture_data):
    """Extract required fields including logos with normalized names"""
    return {
        "fixture_id": fixture_data["fixture"]["id"],
        "league": {
            "id": fixture_data["league"]["id"],
            "name": normalize_text(fixture_data["league"]["name"]),
            "country": normalize_text(fixture_data["league"]["country"]),
            "logo": fixture_data["league"]["logo"],
            "round": normalize_text(fixture_data["league"]["round"])
        },
        "teams": {
            "home": {
                "id": fixture_data["teams"]["home"]["id"],
                "name": normalize_text(fixture_data["teams"]["home"]["name"]),
                "logo": fixture_data["teams"]["home"]["logo"]
            },
            "away": {
                "id": fixture_data["teams"]["away"]["id"],
                "name": normalize_text(fixture_data["teams"]["away"]["name"]),
                "logo": fixture_data["teams"]["away"]["logo"]
            }
        }
    }

def sanitize_folder_name(name):
    """Make league name filesystem-safe with normalized text"""
    normalized = normalize_text(name)
    # Replace spaces with underscores and remove special chars
    return "".join(c if c.isalnum() else "_" for c in normalized).strip("_")

def make_api_request(league_id, date):
    """Make API request with error handling"""
    conn = http.client.HTTPSConnection(API_HOST)
    endpoint = f"{BASE_ENDPOINT}?date={date}&league={league_id}&season=2024"
    
    try:
        conn.request("GET", endpoint, headers={
            'x-rapidapi-key': API_KEY,
            'x-rapidapi-host': API_HOST
        })
        
        res = conn.getresponse()
        if res.status == 200:
            return json.loads(res.read().decode("utf-8"))
        print(f"Error {res.status} for league {league_id}: {res.reason}")
        return None
        
    except Exception as e:
        print(f"Request failed for league {league_id}: {str(e)}")
        return None
    finally:
        conn.close()

def save_fixtures(data, date):
    """Save processed data to organized folders with normalized names"""
    if not data or not data.get("response"):
        return False
    
    # Get original league info
    original_league = data["response"][0]["league"]
    
    # Create sanitized version for storage
    league_info = {
        "id": original_league["id"],
        "name": normalize_text(original_league["name"]),
        "country": normalize_text(original_league["country"]),
        "logo": original_league["logo"],
        "round": normalize_text(original_league["round"])
    }
    
    # Create directory structure
    safe_name = sanitize_folder_name(league_info["name"])
    base_path = os.path.join("Games", date, safe_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Prepare fixtures data
    fixtures = []
    for fixture in data["response"]:
        fixtures.append(simplify_fixture_data(fixture))
    
    # Save to JSON
    output_path = os.path.join(base_path, f"fixtures_{date}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "date": date,
            "league": league_info,
            "fixtures": fixtures
        }, f, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves Unicode
    
    print(f"Saved {len(fixtures)} fixtures for {league_info['name']}")
    return True

def main():
    date = get_user_date()
    print(f"\nFetching fixtures for {date}...\n")
    
    for league_id in LEAGUE_IDS:
        data = make_api_request(league_id, date)
        if data:
            save_fixtures(data, date)

if __name__ == "__main__":
    try:
        import dotenv
    except ImportError:
        print("Error: Required package not found. Please install with:")
        print("pip install python-dotenv")
        exit(1)
    
    main()