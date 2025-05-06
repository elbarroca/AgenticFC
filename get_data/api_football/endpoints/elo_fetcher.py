# elo_fetcher.py (Updated _fetch_team_elo_history method)
import os
import sys
import soccerdata as sd
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, Set, Union
import logging
import pathlib
import re
from dotenv import load_dotenv

# Add project root to path if needed (adjust relative path)
# project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
# sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# --- Configuration & Caching ---
load_dotenv() # Load environment variables from .env file if needed

TEAM_NAME_MAPPING = { # Using the combined mapping from previous step
  "1899 Hoffenheim": "Hoffenheim", "AC Milan": "Milan", "ADO Den Haag": "Den Haag",
  "AFC Hermannstadt": "AFC Hermannstadt", "AZ Alkmaar": "AZ Alkmaar", "Aalborg": "Aalborg",
  "Academico Viseu": "Academico Viseu", "Ajaccio GFCO": "Ajaccio GFCO", "Ajax": "Ajax",
  "Alanyaspor": "Alanyaspor", "Albacete": "Albacete", "Almere City FC": "Almere City",
  "Almeria": "Almeria", "Alverca": "Alverca", "Amiens": "Amiens", "Anderlecht": "Anderlecht",
  "Angers": "Angers", "Annecy": "Annecy", "Antalyaspor": "Antalyaspor", "Antwerp": "Antwerp",
  "Apoel": "Apoel", "Arouca": "Arouca", "Arsenal": "Arsenal FC", "Aston Villa": "Aston Villa",
  "Atalanta": "Atalanta", "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Atletico",
  "Auxerre": "Auxerre", "BB Bodrumspor": "BB Bodrumspor", "Barcelona": "Barça", "Bari": "Bari",
  "Bastia": "Bastia", "Bayer Leverkusen": "Leverkusen", "Beerschot Wilrijk": "Beerschot VA",
  "Benfica": "Benfica", "Benfica B": "Benfica B", "Besiktas": "Besiktas", "Betis": "Betis",
  "Blackburn": "Blackburn", "Boavista": "Boavista", "Bologna": "Bologna",
  "Borussia Dortmund": "Dortmund", "Borussia Monchengladbach": "M'gladbach", "Bournemouth": "Bournemouth",
  "Brentford": "Brentford", "Brescia": "Brescia", "Brighton": "Brighton", "Bristol City": "Bristol City",
  "Brondby": "Brondby", "Burgos": "Burgos", "Burnley": "Burnley", "CD Eldense": "Eldense",
  "Cadiz": "Cadiz", "Caen": "Caen", "Cagliari": "Cagliari", "Cambuur": "Cambuur", "Cardiff": "Cardiff",
  "Carrarese": "Carrarese", "Cartagena": "Cartagena", "Casa Pia": "Casa Pia", "Castellon": "Castellon",
  "Catanzaro": "Catanzaro", "Celta Vigo": "Celta", "Celtic": "Celtic", "Cercle Brugge": "Cercle Brugge",
  "Cesena": "Cesena", "Charleroi": "Charleroi", "Chaves": "Chaves", "Chelsea": "Chelsea FC",
  "Cittadella": "Cittadella", "Clermont Foot": "Clermont", "Club Brugge": "Club Brugge", "Como": "Como",
  "Cordoba": "Cordoba", "Cosenza": "Cosenza", "Coventry": "Coventry", "Cracovia Krakow": "Cracovia Krakow",
  "Cremonese": "Cremonese", "Crvena Zvezda": "Crvena Zvezda", "Crystal Palace": "Crystal Palace",
  "De Graafschap": "Graafschap", "Den Bosch": "Den Bosch", "Deportivo Alaves": "Alaves",
  "Deportivo La Coruna": "La Coruna", "Derby": "Derby", "Dinamo Bucuresti": "Dinamo Bucuresti",
  "Dordrecht": "Dordrecht", "Dunkerque": "Dunkerque", "Eibar": "Eibar",
  "Eintracht Braunschweig": "Braunschweig", "Eintracht Frankfurt": "Ein Frankfurt", "Elche": "Elche",
  "Emmen": "FC Emmen", "Empoli": "Empoli", "Espanyol": "Espanol", "Estac Troyes": "Troyes",
  "Estoril": "Estoril", "Estrela Da Amadora": "Estrela", "Everton": "Everton", "Excelsior": "Excelsior",
  "Eyupspor": "Eyupspor", "FC Augsburg": "Augsburg", "FC Botosani": "FC Botosani",
  "FC Copenhagen": "FC København", "FC Dender": "FC Dender", "FC Eindhoven": "FC Eindhoven",
  "FC Koln": "FC Koln", "FC Lugano": "FC Lugano", "FC Midtjylland": "FC Midtjylland",
  "FC Nordsjaelland": "FC Nordsjaelland", "FC Porto": "Porto", "FC Porto B": "Porto B",
  "FC Schalke 04": "Schalke 04", "FC St. Pauli": "St Pauli", "FCSB": "FCSB",
  "FSV Mainz 05": "Mainz 05", "Famalicao": "Famalicao", "Farense": "Farense",
  "Farul Constanta": "Farul Constanta", "Feirense": "Feirense", "Felgueiras": "Felgueiras",
  "Fenerbahce": "Fenerbahce", "Ferencvarosi TC": "Ferencvarosi TC", "Feyenoord": "Feyenoord",
  "Fiorentina": "Fiorentina", "Fortuna Dusseldorf": "Fortuna Dusseldorf", "Fortuna Sittard": "For Sittard",
  "Frosinone": "Frosinone", "Fulham": "Fulham", "GKS Katowice": "GKS Katowice", "Galatasaray": "Galatasaray",
  "Genk": "Genk", "Genoa": "Genoa", "Gent": "Gent", "Getafe": "Getafe", "Gil Vicente": "Gil Vicente",
  "Girona": "Girona", "Gloria Buzau": "Gloria Buzau", "Go Ahead Eagles": "Go Ahead Eagles",
  "Gornik Zabrze": "Gornik Zabrze", "Goztep": "Goztep", "Granada CF": "Granada",
  "Grenoble Foot 38": "Grenoble", "Groningen": "Groningen", "Guimaraes": "Guimaraes",
  "Guingamp": "Guingamp", "HNK Gorica": "HNK Gorica", "HNK Hajduk Split": "HNK Hajduk Split",
  "HNK Rijeka": "HNK Rijeka", "HNK Sibenik": "Sibenik", "Hamburger SV": "Hamburg",
  "Hannover 96": "Hannover", "Hatayspor": "Hatayspor", "Heerenveen": "Heerenveen",
  "Helmond Sport": "Helmond Sport", "Heracles": "Heracles", "Holstein Kiel": "Holstein Kiel",
  "Huesca": "Huesca", "Hull City": "Hull", "Inter": "Internazionale", "Ipswich": "Ipswich",
  "Istanbul Basaksehir": "Buyuksehyr", "Jagiellonia": "Jagiellonia", "Jahn Regensburg": "Regensburg",
  "Jong AZ": "Jong AZ", "Jong Ajax": "Jong Ajax", "Jong PSV": "Jong PSV", "Jong Utrecht": "Jong Utrecht",
  "Juve Stabia": "Juve Stabia", "Juventus": "Juve", "KV Mechelen": "Mechelen",
  "KVC Westerlo": "Westerlo", "Karlsruher SC": "Karlsruhe", "Kasimpasa": "Kasimpasa",
  "Kayserispor": "Kayserispor", "Konyaspor": "Konyaspor", "Korona Kielce": "Korona Kielce",
  "Kortrijk": "Kortrijk", "Las Palmas": "Las Palmas", "Laval": "Laval", "Lazio": "Lazio",
  "Le Havre": "Le Havre", "Lecce": "Lecce", "Lech Poznan": "Lech", "Lechia Gdansk": "Lechia Gdansk",
  "Leeds": "Leeds", "Leganes": "Leganes", "Legia Warszawa": "Legia Warszawa", "Leicester": "Leicester",
  "Leiria": "Leiria", "Leixoes": "Leixoes", "Lens": "Lens", "Levante": "Levante", "Lille": "Lille",
  "Liverpool": "Liverpool FC", "Lorient": "Lorient", "Luton": "Luton", "Lyngby": "Lyngby",
  "Lyon": "Lyon", "MVV": "MVV", "Mafra": "Mafra", "Malaga": "Malaga", "Mallorca": "Mallorca",
  "Manchester City": "Man City", "Manchester United": "Man United", "Mantova": "Mantova",
  "Maritimo": "Maritimo", "Marseille": "Marseille", "Martigues": "Martigues", "Metz": "Metz",
  "Middlesbrough": "Middlesbrough", "Millwall": "Millwall", "Mirandes": "Mirandes", "Modena": "Modena",
  "Monaco": "Monaco", "Montpellier": "Montpellier", "Monza": "Monza", "Moreirense": "Moreirense",
  "Motor Lublin": "Motor Lublin", "NAC Breda": "NAC Breda", "NEC Nijmegen": "Nijmegen",
  "NK Dinamo Zagreb": "NK Dinamo Zagreb", "NK Lokomotiva Zagreb": "NK Lokomotiva Zagreb",
  "NK Osijek": "NK Osijek", "NK Slaven Belupo": "NK Slaven Belupo", "NK Varazdin": "NK Varazdin",
  "Nacional": "Nacional", "Nantes": "Nantes", "Napoli": "SSC Napoli", "Newcastle": "Newcastle",
  "Nice": "Nice", "Norwich": "Norwich", "Nottingham Forest": "Nott'm Forest",
  "OH Leuven": "Oud-Heverlee Leuven", "Oliveirense": "Oliveirense", "Olympiakos Piraeus": "Olympiakos",
  "Osasuna": "Osasuna", "Otelul Galati": "Otelul Galati", "Oviedo": "Oviedo", "Oxford United": "Oxford",
  "PAOK": "PAOK", "PEC Zwolle": "Zwolle", "PSV": "PSV Eindhoven", "Pacos Ferreira": "Paços Ferreira",
  "Pafos FC": "Pafos FC", "Palermo": "Palermo", "Panathinaikos": "Panathinaikos", "Paris FC": "Paris FC",
  "Paris Saint-Germain": "PSG", "Parma": "Parma", "Pau": "Pau FC", "Penafiel": "Penafiel",
  "Piast Gliwice": "Piast Gliwice", "Pisa": "Pisa", "Plymouth": "Plymouth",
  "Pogon Szczecin": "Pogon Szczecin", "Portimonense": "Portimonense", "Portsmouth": "Portsmouth",
  "Preston": "Preston", "QPR": "QPR", "RB Leipzig": "RB Leipzig", "Racing Santander": "Santander",
  "Radomiak Radom": "Radomiak Radom", "Rakow Czestochowa": "Rakow Czestochowa", "Randers FC": "Randers",
  "Rangers": "Rangers", "Rapid Bucuresti": "Rapid Bucuresti", "Rapid Vienna": "Rapid Vienna",
  "Rayo Vallecano": "Vallecano", "Real Madrid": "Real", "Real Sociedad": "Sociedad", "Red Star": "Red Star",
  "Reggiana": "Reggiana", "Reims": "Reims", "Rennes": "Rennes", "Rio Ave": "Rio Ave",
  "Rizespor": "Rizespor", "Roda": "Roda", "Rodez": "Rodez", "Roma": "Roma", "AS Roma": "Roma", # Added mapping from original image
  "SC Braga": "Braga", "SC Freiburg": "Freiburg", "SC Paderborn 07": "Paderborn",
  "SSV Ulm 1846": "SSV Ulm 1846", "SV Darmstadt 98": "Darmstadt", "SV Elversberg": "Elversberg",
  "Salernitana": "Salernitana", "Sampdoria": "Sampdoria", "Samsunspor": "Samsunspor",
  "Santa Clara": "Santa Clara", "Sassuolo": "Sassuolo", "Sepsi OSK": "Sepsi OSK", "Sevilla": "Sevilla",
  "Sheffield Utd": "Sheffield United", "Sheffield Wed": "Sheffield Weds", "Silkeborg": "Silkeborg",
  "Sivasspor": "Sivasspor", "Slask Wroclaw": "Slask Wroclaw", "Slavia Praha": "Slavia Praha",
  "SonderjyskE": "SonderjyskE", "Southampton": "Southampton", "SpVgg Greuther Furth": "Greuther Furth",
  "Sparta Rotterdam": "Sparta Rotterdam", "Spezia": "Spezia", "Sporting CP": "Sp Lisbon",
  "Sporting Gijon": "Sp Gijon", "St Truiden": "St Truiden", "Stade Brestois 29": "Brest",
  "Stal Mielec": "Stal Mielec", "Standard Liege": "Standard", "Stoke City": "Stoke",
  "Strasbourg": "Strasbourg", "Sturm Graz": "Sturm Graz", "Sudtirol": "Sudtirol",
  "Sunderland": "Sunderland", "Swansea": "Swansea", "TSC Backa Topola": "TSC Backa Topola",
  "Telstar": "Telstar", "Tenerife": "Tenerife", "Tondela": "Tondela", "Torino": "Torino",
  "Torreense": "Torreense", "Tottenham Hotspur": "Tottenham", "Toulouse": "Toulouse",
  "Trabzonspor": "Trabzonspor", "Twente": "Twente", "UTA Arad": "UTA Arad", "Udinese": "Udinese",
  "Union Berlin": "Union Berlin", "Union St. Gilloise": "St. Gilloise", "Unione Venezia": "Venezia",
  "Universitatea Cluj": "Universitatea Cluj", "Utrecht": "Utrecht", "VVV Venlo": "VVV Venlo",
  "Valencia": "Valencia", "Valladolid": "Valladolid", "Vejle": "Vejle", "Verona": "Verona",
  "VfB Stuttgart": "Stuttgart", "VfL Bochum": "Bochum", "VfL Wolfsburg": "Wolfsburg",
  "Viborg": "Viborg", "Viktoria Plzen": "Viktoria Plzen", "Villarreal": "Villarreal",
  "Vitesse": "Vitesse", "Vizela": "Vizela", "Volendam": "Volendam", "Waalwijk": "Waalwijk",
  "Watford": "Watford", "Werder Bremen": "Werder Bremen", "West Brom": "West Brom",
  "West Ham": "West Ham", "Willem II": "Willem II", "Wolves": "Wolves",
  "Zaglebie Lubin": "Zaglebie Lubin", "Zaragoza": "Zaragoza", "FC OSS": "TOP Oss",
  "FC Heidenheim": "Heidenheim", "Neftchi Baku": "Neftchi", "Vikingur Reykjavik": "Víkingur",
  "Puszcza Niepołomice": "Puszcza", "Hanácká": "Hanácká Slavia", "Valmiera BSS": "Valmiera",
  "Krems Rehberg": "Krems Rehberg"
}

# Cache for team Elo histories to minimize API calls via soccerdata's cache
TEAM_ELO_HISTORIES_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
FAILED_ELO_FETCH_TEAMS: Set[str] = set()

# !!! DEBUG: Clear caches !!!
logger.warning("!!! ELO FETCHER DEBUG: Clearing in-memory cache and failed teams set !!!")
TEAM_ELO_HISTORIES_CACHE.clear()
FAILED_ELO_FETCH_TEAMS.clear()
# Consider clearing soccerdata cache manually: rm -rf ~/soccerdata/data/ClubElo/*

class EloFetcher:
    _instance = None
    _elo_reader: Optional[sd.ClubElo] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EloFetcher, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        try:
            self._elo_reader = sd.ClubElo()
            cache_dir = pathlib.Path(self._elo_reader.data_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ClubElo reader initialized. Using cache path: {cache_dir}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize ClubElo reader: {e}", exc_info=True)
            self._elo_reader = None

    def _get_reader(self) -> sd.ClubElo:
        assert self._initialized and self._elo_reader is not None, "EloFetcher not initialized or ClubElo reader failed to load."
        return self._elo_reader

    def _fetch_team_elo_history(self, team_name: str) -> Optional[pd.DataFrame]:
        """Fetches and caches Elo history for a team, applying name mapping and variations."""
        assert isinstance(team_name, str) and team_name, "Team name must be a non-empty string"
        original_team_name = team_name.strip()
        logger.debug(f"ELO Fetch History: Received request for '{original_team_name}'")

        if original_team_name in TEAM_ELO_HISTORIES_CACHE:
            logger.debug(f"ELO Fetch History: Cache hit for '{original_team_name}'. Returning cached data (Type: {type(TEAM_ELO_HISTORIES_CACHE[original_team_name])}).")
            return TEAM_ELO_HISTORIES_CACHE[original_team_name]
        if original_team_name in FAILED_ELO_FETCH_TEAMS:
            logger.debug(f"ELO Fetch History: Hit known failed team '{original_team_name}'. Returning None.")
            return None

        elo_reader = self._get_reader()

        # --- Use Enhanced Name Variation Logic ---
        base_variant = TEAM_NAME_MAPPING.get(original_team_name, original_team_name)
        name_variations_set = { # Use a set for uniqueness
            base_variant,
            original_team_name,
            original_team_name.replace(" & ", " "),
            original_team_name.replace("FC ", "").replace(" FC", ""),
            original_team_name.split(" (")[0],
            re.sub(r'\b(II|2)\b', '', original_team_name).strip(),
            original_team_name.split(" ")[0],
            ' '.join(original_team_name.split(" ")[:2]),
            original_team_name.replace("BSS", "").strip(),
            original_team_name.replace("Rehberg", "").strip(),
        }
        if "/" in original_team_name: name_variations_set.update(p.strip() for p in original_team_name.split("/"))
        if " & " in original_team_name: name_variations_set.update(p.strip() for p in original_team_name.split(" & "))
        name_variations = sorted(list(name_variations_set - {None, ''}), key=lambda x: (x != base_variant, x != original_team_name))
        logger.debug(f"ELO Fetch History: Trying variations for '{original_team_name}': {name_variations}")
        # --- End Enhanced Name Variation Logic ---

        for name_variant in name_variations:
            if not name_variant: continue
            logger.debug(f"ELO Fetch History: Attempting soccerdata.read_team_history('{name_variant}')")
            try:
                history_df = elo_reader.read_team_history(name_variant)
                if history_df is not None and not history_df.empty:
                    logger.info(f"ELO Fetch History: SUCCESS reading history for variant '{name_variant}' (Team: '{original_team_name}')")
                    # Standardize index and 'to' column
                    index_name = history_df.index.name if history_df.index.name else 'from'
                    if history_df.index.name is None: history_df.index.name = 'from'
                    history_df.index = pd.to_datetime(history_df.index, errors='coerce').tz_localize(None)
                    history_df['to'] = pd.to_datetime(history_df['to'], errors='coerce').dt.tz_localize(None)
                    history_df.dropna(subset=[history_df.index.name, 'to'], inplace=True)
                    if history_df.empty:
                        logger.warning(f"ELO Fetch History: History read for '{name_variant}', but all rows dropped due to invalid dates.")
                        continue
                    TEAM_ELO_HISTORIES_CACHE[original_team_name] = history_df
                    logger.info(f"Successfully cached Elo data for '{original_team_name}' using variant '{name_variant}'")
                    return history_df

            except ValueError as ve:
                logger.debug(f"ELO Fetch History: ValueError for variant '{name_variant}' (Team: '{original_team_name}'): {ve}")
                continue
            except FileNotFoundError:
                logger.debug(f"ELO Fetch History: FileNotFoundError for variant '{name_variant}' (Team: '{original_team_name}')")
                try:
                    expected_path = pathlib.Path(elo_reader.data_dir) / f"{name_variant}.csv"
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception: pass
                continue
            except KeyError as ke:
                logger.warning(f"ELO Fetch History: KeyError reading ELO history for variant '{name_variant}' (Team: '{original_team_name}'): {ke}. Likely soccerdata parsing issue.", exc_info=False)
                continue
            except Exception as e:
                logger.warning(f"ELO Fetch History: Error reading ELO history for variant '{name_variant}' (Team: '{original_team_name}'): {type(e).__name__} - {e}", exc_info=False)
                continue

        logger.warning(f"ELO Fetch History: Could not find Elo data for team '{original_team_name}' after trying all variants.")
        FAILED_ELO_FETCH_TEAMS.add(original_team_name)
        TEAM_ELO_HISTORIES_CACHE[original_team_name] = None
        return None

    def get_elo_on_date(self, team_name: str, match_date: datetime) -> Optional[int]:
        """Finds the ClubElo rating for a team on a specific date using cached history."""
        assert isinstance(team_name, str) and team_name, "Team name must be a non-empty string"
        assert isinstance(match_date, datetime), "Match date must be a datetime object"

        history_df = self._fetch_team_elo_history(team_name)
        if history_df is None: return None

        match_date_naive = match_date.astimezone(timezone.utc).replace(tzinfo=None)

        try:
            relevant_elo_row = history_df[
                (history_df.index <= match_date_naive) & (history_df['to'] >= match_date_naive)
            ]

            if not relevant_elo_row.empty:
                elo_value = relevant_elo_row.iloc[0]['elo']
                assert pd.notna(elo_value), f"ELO value is NaN in relevant row for {team_name} on {match_date_naive.date()}"
                return int(round(elo_value))
            else:
                past_elos = history_df[history_df.index < match_date_naive].sort_index(ascending=False)
                if not past_elos.empty:
                    latest_past_elo_row = past_elos.iloc[0]
                    elo_value = latest_past_elo_row['elo']
                    to_date = latest_past_elo_row['to']
                    if pd.notna(elo_value) and pd.notna(to_date) and (match_date_naive - to_date <= timedelta(days=90)):
                        return int(round(elo_value))
            return None
        except Exception as e:
            logger.error(f"Error processing Elo history lookup for {team_name} on {match_date_naive.date()}: {e}", exc_info=True)
            return None

    def get_elos_for_match(self, home_team_name: str, away_team_name: str, match_date: datetime) -> Tuple[Optional[int], Optional[int]]:
        """Convenience function to get ELO for both teams for a match."""
        home_elo = self.get_elo_on_date(home_team_name, match_date)
        away_elo = self.get_elo_on_date(away_team_name, match_date)
        return home_elo, away_elo

# Singleton instance
elo_fetcher = EloFetcher()