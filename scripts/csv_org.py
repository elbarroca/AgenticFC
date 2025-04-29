import os
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dictionary to map division codes to full names and folder names
DIVISION_MAPPING = {
    'D1': {'name': 'Bundesliga1', 'folder': 'Bundesliga'},
    'D2': {'name': 'Bundesliga2', 'folder': 'Bundesliga'},
    'E0': {'name': 'PremierLeague', 'folder': 'England'},
    'E1': {'name': 'Championship', 'folder': 'England'},
    'I1': {'name': 'SerieA', 'folder': 'Italy'},
    'I2': {'name': 'SerieB', 'folder': 'Italy'},
    'SP1': {'name': 'LaLiga', 'folder': 'Spain'},
    'SP2': {'name': 'LaLiga2', 'folder': 'Spain'},
    'F1': {'name': 'Ligue1', 'folder': 'France'},
    'F2': {'name': 'Ligue2', 'folder': 'France'},
    'N1': {'name': 'Eredivisie', 'folder': 'Netherlands'},
    'P1': {'name': 'LigaPortugal', 'folder': 'Portugal'},
    'B1': {'name': 'JupilerLeague', 'folder': 'Belgium'},
    'T1': {'name': 'SuperLig', 'folder': 'Turkey'},
    'G1': {'name': 'GreekSuperLeague', 'folder': 'Greece'}
}

def parse_date(date_str):
    """Try multiple date formats to parse the date string"""
    date_formats = [
        '%d/%m/%y',    # 15/08/15
        '%d/%m/%Y',    # 15/08/2015
        '%Y-%m-%d',    # 2015-08-15
        '%d-%m-%Y',    # 15-08-2015
        '%d-%m-%y',    # 15-08-15
        '%Y/%m/%d'     # 2015/08/15
    ]
    
    for date_format in date_formats:
        try:
            return pd.to_datetime(date_str, format=date_format)
        except:
            continue
    return None

def get_league_and_year(csv_path):
    try:
        # Read first few rows to get better date detection
        df = pd.read_csv(csv_path, nrows=5)
        
        if 'Div' not in df.columns or 'Date' not in df.columns:
            logger.warning(f"Required columns not found in {csv_path}")
            return None, None, None
        
        # Get division
        division = df['Div'].iloc[0]
        league_info = DIVISION_MAPPING.get(division, {'name': division, 'folder': 'Other'})
        league_name = league_info['name']
        folder_name = league_info['folder']
        
        # Try to parse dates from multiple rows
        for date_str in df['Date'].dropna():
            parsed_date = parse_date(date_str)
            if parsed_date is not None:
                return league_name, folder_name, parsed_date.year
        
        logger.warning(f"Could not parse any dates in {csv_path}")
        return league_name, folder_name, None
            
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {str(e)}")
        return None, None, None

def organize_files():
    root_dir = Path('football_data_db')
    
    # Track processed files to handle duplicates
    processed_files = {}
    
    # First, scan all files and collect information
    all_files_info = []
    
    logger.info("Scanning all CSV files...")
    
    # Walk through all subdirectories
    for folder_path in root_dir.iterdir():
        if not folder_path.is_dir():
            continue
            
        logger.info(f"Scanning directory: {folder_path}")
        
        # Process all CSV files in the directory
        csv_files = list(folder_path.glob('*.csv'))
        total_files = len(csv_files)
        
        logger.info(f"Found {total_files} CSV files in {folder_path}")
        
        for idx, csv_file in enumerate(csv_files, 1):
            logger.info(f"Analyzing file {idx}/{total_files}: {csv_file.name}")
            
            league_name, folder_name, year = get_league_and_year(csv_file)
            
            if league_name and folder_name and year:
                all_files_info.append({
                    'source': csv_file,
                    'league_name': league_name,
                    'folder_name': folder_name,
                    'year': year
                })
            else:
                logger.warning(f"Could not process {csv_file}")
    
    # Create league folders and move files
    logger.info("Organizing files into league folders...")
    
    # First, remove existing league folders if they exist
    for league_info in DIVISION_MAPPING.values():
        folder_path = root_dir / league_info['folder']
        if folder_path.exists():
            shutil.rmtree(folder_path)
    
    # Now organize files into league folders
    for file_info in all_files_info:
        # Create league folder if it doesn't exist
        league_folder = root_dir / file_info['folder_name']
        league_folder.mkdir(exist_ok=True)
        
        # Generate new filename
        base_name = f"{file_info['league_name']}_{file_info['year']}"
        key = f"{file_info['folder_name']}_{base_name}"
        
        # Handle duplicates
        counter = processed_files.get(key, 0) + 1
        processed_files[key] = counter
        
        new_name = f"{base_name}.csv" if counter == 1 else f"{base_name}_part{counter}.csv"
        new_path = league_folder / new_name
        
        try:
            # Move file to new location
            shutil.move(file_info['source'], new_path)
            logger.info(f"Moved {file_info['source'].name} to {new_path}")
        except Exception as e:
            logger.error(f"Error moving {file_info['source']}: {str(e)}")

    # Clean up empty directories
    logger.info("Cleaning up empty directories...")
    for folder_path in root_dir.iterdir():
        if folder_path.is_dir() and not any(folder_path.iterdir()):
            try:
                folder_path.rmdir()
                logger.info(f"Removed empty directory: {folder_path}")
            except Exception as e:
                logger.error(f"Error removing directory {folder_path}: {str(e)}")

    logger.info("File organization completed")

if __name__ == "__main__":
    try:
        organize_files()
        logger.info("Organization process completed successfully")
    except Exception as e:
        logger.error(f"Critical error during execution: {str(e)}")