# config.py
import os
from pydantic import BaseModel, Field , constr, field_validator
from typing import List, Set, Dict, Any, Optional

# --- Paths ---
# Use absolute paths or paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up 3 levels from utils/config.py
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_CSV_DIR = os.path.join(PROJECT_ROOT, 'football_data_db')  # Directory containing CSV files
UNIFIED_DATA_PATH = os.path.join(DATA_DIR, 'unified_data', 'mongo.parquet')

# MongoDB Configuration
MONGO_URI = 'mongodb://root:RicardoMongoDB@74.50.127.165:27017/admin'  # Updated URI
MONGO_DB_NAME = 'agenticfc'
MONGO_COLLECTION_NAME = 'matches'
MONGO_DAILY_GAMES_COLLECTION = 'daily_games'
MONGO_MATCH_PROCESSOR_COLLECTION = 'match_processor'
MONGO_MATCHES_COLLECTION = 'matches'
MONGO_ODDS_COLLECTION = 'odds'
MONGO_STANDINGS_COLLECTION = 'standings'
MONGO_STATAREA_STATS_COLLECTION = 'statarea_stats'
MONGO_TEAM_SEASON_FIXTURES_COLLECTION = 'team_season_fixtures'

# Team Name Mapping for standardization
TEAM_NAME_MAPPING = {
   'AEK': '???', # Manual mapping required.
    'AVS': '???', # Manual mapping required.
    'AZ Alkmaar': 'AZ Alkmaar', # Exact Match (Standard) | IDs: stat=201, mongo=201
    'Aachen': '???', # Manual mapping required.
    'Aalen': '???', # Manual mapping required.
    'Academica': 'Academico_Viseu', # Fuzzy Match (85%) | IDs: stat=238, mongo=238
    'Ad. Demirspor': '???', # Manual mapping required.
    'Ahlen': '???', # Manual mapping required.
    'Ajaccio': 'Ajaccio GFCO', # Fuzzy Match (90%) | IDs: stat=98, mongo=98
    'Ajaccio GFCO': 'Ajaccio GFCO', # Exact Match (Standard) | IDs: stat=98, mongo=98
    'Ajax': 'Ajax', # Exact Match (Standard) | IDs: stat=194, mongo=194
    'Akhisar Belediyespor': '???', # Manual mapping required.
    'Alanyaspor': 'Alanyaspor', # Exact Match (Standard) | IDs: stat=996, mongo=996
    'Alaves': 'Alaves', # Exact Match (Standard) | IDs: stat=542, mongo=542
    'Albacete': 'Albacete', # Exact Match (Standard) | IDs: stat=722, mongo=722
    'Albinoleffe': '???', # Manual mapping required.
    'Alcorcon': '???', # Manual mapping required.
    'Alcoyano': '???', # Manual mapping required.
    'Alessandria': '???', # Manual mapping required.
    'Alicante': '???', # Manual mapping required.
    'Almere City': 'Almere City', # Exact Match (Standard) | IDs: stat=419, mongo=419
    'Almeria': 'Almeria', # Exact Match (Standard) | IDs: stat=723, mongo=723
    'Altay': '???', # Manual mapping required.
    'Amiens': 'Amiens', # Exact Match (Standard) | IDs: stat=87, mongo=87
    'Amorebieta': '???', # Manual mapping required.
    'Ancona': '???', # Manual mapping required.
    'Anderlecht': 'Anderlecht', # Exact Match (Standard) | IDs: stat=554, mongo=554
    'Andorra': '???', # Manual mapping required.
    'Angers': 'Angers', # Exact Match (Standard) | IDs: stat=77, mongo=77
    'Ankaragucu': '???', # Manual mapping required.
    'Annecy': 'Annecy', # Exact Match (Standard) | IDs: stat=3012, mongo=3012
    'Antalyaspor': 'Antalyaspor', # Exact Match (Standard) | IDs: stat=1005, mongo=1005
    'Antwerp': 'Antwerp', # Exact Match (Standard) | IDs: stat=740, mongo=740
    'Apollon': '???', # Manual mapping required.
    'Arezzo': '???', # Manual mapping required.
    'Aris': 'Paris FC', # Fuzzy Match (90%) | IDs: stat=114, mongo=114
    'Arles': '???', # Manual mapping required.
    'Arouca': 'Arouca', # Exact Match (Standard) | IDs: stat=240, mongo=240
    'Arsenal': 'Arsenal', # Exact Match (Standard) | IDs: stat=42, mongo=42
    'Ascoli': '???', # Manual mapping required.
    'Asteras Tripolis': '???', # Manual mapping required.
    'Aston Villa': 'Aston Villa', # Exact Match (Standard) | IDs: stat=66, mongo=66
    'Atalanta': 'Atalanta', # Exact Match (Standard) | IDs: stat=499, mongo=499
    'Ath Bilbao B': 'Ath Bilbao', # Fuzzy Match (95%) | IDs: stat=531, mongo=531
    'Athens Kallithea': '???', # Manual mapping required.
    'Atromitos': '???', # Manual mapping required.
    'Augsburg': 'Augsburg', # Exact Match (Standard) | IDs: stat=170, mongo=170
    'Auxerre': 'Auxerre', # Exact Match (Standard) | IDs: stat=108, mongo=108
    'Avellino': '???', # Manual mapping required.
    'Aves': 'Chaves', # Fuzzy Match (90%) | IDs: stat=223, mongo=223
    'Barcelona': 'Barcelona', # Exact Match (Standard) | IDs: stat=529, mongo=529
    'Barcelona B': 'Barcelona', # Fuzzy Match (95%) | IDs: stat=529, mongo=529
    'Bari': 'Bari', # Exact Match (Standard) | IDs: stat=508, mongo=508
    'Barnsley': 'Burnley', # Fuzzy Match (80%) | IDs: stat=44, mongo=44
    'Bastia': 'Bastia', # Exact Match (Standard) | IDs: stat=1305, mongo=1305
    'Bayern Munich': '???', # Manual mapping required.
    'Beauvais': '???', # Manual mapping required.
    'Beerschot VA': 'Beerschot VA', # Exact Match (Standard) | IDs: stat=263, mongo=263
    'Beira Mar': '???', # Manual mapping required.
    'Belenenses': '???', # Manual mapping required.
    'Benevento': '???', # Manual mapping required.
    'Benfica': 'Benfica', # Exact Match (Standard) | IDs: stat=211, mongo=211
    'Bergen': '???', # Manual mapping required.
    'Besancon': '???', # Manual mapping required.
    'Besiktas': 'Besiktas', # Exact Match (Standard) | IDs: stat=549, mongo=549
    'Betis': 'Betis', # Exact Match (Standard) | IDs: stat=543, mongo=543
    'Beziers': '???', # Manual mapping required.
    'Bielefeld': '???', # Manual mapping required.
    'Birmingham': '???', # Manual mapping required.
    'Blackburn': 'Blackburn', # Exact Match (Standard) | IDs: stat=67, mongo=67
    'Blackpool': '???', # Manual mapping required.
    'Boavista': 'Boavista', # Exact Match (Standard) | IDs: stat=222, mongo=222
    'Bochum': 'Bochum', # Exact Match (Standard) | IDs: stat=176, mongo=176
    'Bodrumspor': 'BB_Bodrumspor', # Fuzzy Match (95%) | IDs: stat=3583, mongo=3583
    'Bologna': 'Bologna', # Exact Match (Standard) | IDs: stat=500, mongo=500
    'Bolton': '???', # Manual mapping required.
    'Bordeaux': '???', # Manual mapping required.
    'Boulogne': 'Bologna', # Fuzzy Match (80%) | IDs: stat=500, mongo=500
    'Bourg Peronnas': '???', # Manual mapping required.
    'Bournemouth': 'Bournemouth', # Exact Match (Standard) | IDs: stat=35, mongo=35
    'Braunschweig': 'Braunschweig', # Exact Match (Standard) | IDs: stat=744, mongo=744
    'Brentford': 'Brentford', # Exact Match (Standard) | IDs: stat=55, mongo=55
    'Brescia': 'Brescia', # Exact Match (Standard) | IDs: stat=518, mongo=518
    'Brest': 'Brest', # Exact Match (Standard) | IDs: stat=106, mongo=106
    'Brighton': 'Brighton', # Exact Match (Standard) | IDs: stat=51, mongo=51
    'Bristol City': 'Bristol City', # Exact Match (Standard) | IDs: stat=56, mongo=56
    'Burghausen': '???', # Manual mapping required.
    'Burgos': 'Burgos', # Exact Match (Standard) | IDs: stat=9580, mongo=9580
    'Burnley': 'Burnley', # Exact Match (Standard) | IDs: stat=44, mongo=44
    'Bursaspor': '???', # Manual mapping required.
    'Burton': '???', # Manual mapping required.
    'Buyuksehyr': 'Buyuksehyr', # Exact Match (Standard) | IDs: stat=564, mongo=564
    'CA Bastia': 'Bastia', # Fuzzy Match (90%) | IDs: stat=1305, mongo=1305
    'CZ Jena': '???', # Manual mapping required.
    'Cadiz': 'Cadiz', # Exact Match (Standard) | IDs: stat=724, mongo=724
    'Caen': 'Caen', # Exact Match (Standard) | IDs: stat=88, mongo=88
    'Cagliari': 'Cagliari', # Exact Match (Standard) | IDs: stat=490, mongo=490
    'Cambuur': 'Cambuur', # Exact Match (Standard) | IDs: stat=420, mongo=420
    'Cardiff': 'Cardiff', # Exact Match (Standard) | IDs: stat=43, mongo=43
    'Carpi': '???', # Manual mapping required.
    'Carrarese': 'Carrarese', # Exact Match (Standard) | IDs: stat=1581, mongo=1581
    'Cartagena': 'Cartagena', # Exact Match (Standard) | IDs: stat=5262, mongo=5262
    'Casa Pia': 'Casa Pia', # Exact Match (Standard) | IDs: stat=4716, mongo=4716
    'Castellon': 'Castellon', # Exact Match (Standard) | IDs: stat=5254, mongo=5254
    'Catania': '???', # Manual mapping required.
    'Catanzaro': 'Catanzaro', # Exact Match (Standard) | IDs: stat=1687, mongo=1687
    'Celta': 'Celta', # Exact Match (Standard) | IDs: stat=538, mongo=538
    'Cercle Brugge': 'Cercle Brugge', # Exact Match (Standard) | IDs: stat=741, mongo=741
    'Cesena': 'Cesena', # Exact Match (Standard) | IDs: stat=509, mongo=509
    'Chambly': '???', # Manual mapping required.
    'Charleroi': 'Charleroi', # Exact Match (Standard) | IDs: stat=736, mongo=736
    'Charlton': 'Luton', # Fuzzy Match (80%) | IDs: stat=1359, mongo=1359
    'Chateauroux': '???', # Manual mapping required.
    'Chaves': 'Chaves', # Exact Match (Standard) | IDs: stat=223, mongo=223
    'Chelsea': 'Chelsea', # Exact Match (Standard) | IDs: stat=49, mongo=49
    'Chievo': '???', # Manual mapping required.
    'Cittadella': 'Cittadella', # Exact Match (Standard) | IDs: stat=510, mongo=510
    'Ciudad de Murcia': '???', # Manual mapping required.
    'Clermont': 'Clermont', # Exact Match (Standard) | IDs: stat=99, mongo=99
    'Club Brugge': 'Club Brugge', # Exact Match (Standard) | IDs: stat=569, mongo=569
    'Colchester': '???', # Manual mapping required.
    'Como': 'Como', # Exact Match (Standard) | IDs: stat=895, mongo=895
    'Concarneau': '???', # Manual mapping required.
    'Cordoba': 'Cordoba', # Exact Match (Standard) | IDs: stat=713, mongo=713
    'Cosenza': 'Cosenza', # Exact Match (Standard) | IDs: stat=10137, mongo=10137
    'Cottbus': '???', # Manual mapping required.
    'Coventry': 'Coventry', # Exact Match (Standard) | IDs: stat=1346, mongo=1346
    'Cremonese': 'Cremonese', # Exact Match (Standard) | IDs: stat=520, mongo=520
    'Creteil': '???', # Manual mapping required.
    'Crewe': '???', # Manual mapping required.
    'Crotone': '???', # Manual mapping required.
    'Crystal Palace': 'Crystal Palace', # Exact Match (Standard) | IDs: stat=52, mongo=52
    'Darmstadt': 'Darmstadt', # Exact Match (Standard) | IDs: stat=181, mongo=181
    'Den Bosch': 'Den_Bosch', # Fuzzy Match | IDs: stat=421, mongo=421
    'Den Haag': 'Den Haag', # Exact Match (Standard) | IDs: stat=198, mongo=198
    'Dender': 'FC Dender', # Fuzzy Match (90%) | IDs: stat=6215, mongo=6215
    'Denizlispor': '???', # Manual mapping required.
    'Derby': 'Derby', # Exact Match (Standard) | IDs: stat=69, mongo=69
    'Dijon': 'Sp Gijon', # Fuzzy Match (80%) | IDs: stat=731, mongo=731
    'Doncaster': '???', # Manual mapping required.
    'Dordrecht': 'Dordrecht', # Exact Match (Standard) | IDs: stat=409, mongo=409
    'Dortmund': 'Dortmund', # Exact Match (Standard) | IDs: stat=165, mongo=165
    'Dresden': '???', # Manual mapping required.
    'Duisburg': '???', # Manual mapping required.
    'Dunkerque': 'Dunkerque', # Exact Match (Standard) | IDs: stat=1304, mongo=1304
    'Eibar': 'Eibar', # Exact Match (Standard) | IDs: stat=545, mongo=545
    'Ein Frankfurt': 'Ein Frankfurt', # Exact Match (Standard) | IDs: stat=169, mongo=169
    'Elche': 'Elche', # Exact Match (Standard) | IDs: stat=797, mongo=797
    'Eldense': 'Eldense', # Exact Match (Standard) | IDs: stat=9692, mongo=9692
    'Elversberg': 'Elversberg', # Exact Match (Standard) | IDs: stat=1660, mongo=1660
    'Empoli': 'Empoli', # Exact Match (Standard) | IDs: stat=511, mongo=511
    'Erzgebirge Aue': '???', # Manual mapping required.
    'Erzurum BB': '???', # Manual mapping required.
    'Essen': '???', # Manual mapping required.
    'Est Amadora': '???', # Manual mapping required.
    'Estoril': 'Estoril', # Exact Match (Standard) | IDs: stat=230, mongo=230
    'Estrela': 'Estrela', # Exact Match (Standard) | IDs: stat=15130, mongo=15130
    'Eupen': '???', # Manual mapping required.
    'Everton': 'Everton', # Exact Match (Standard) | IDs: stat=45, mongo=45
    'Evian Thonon Gaillard': '???', # Manual mapping required.
    'Excelsior': 'Excelsior', # Exact Match (Standard) | IDs: stat=196, mongo=196
    'Extremadura UD': '???', # Manual mapping required.
    'Eyupspor': 'Eyupspor', # Exact Match (Standard) | IDs: stat=3588, mongo=3588
    'FC Brussels': 'FC Koln', # Fuzzy Match (86%) | IDs: stat=192, mongo=192
    'FC Emmen': 'FC Emmen', # Exact Match (Standard) | IDs: stat=208, mongo=208
    'FC Koln': 'FC Koln', # Exact Match (Standard) | IDs: stat=192, mongo=192
    'Famalicao': 'Famalicao', # Exact Match (Standard) | IDs: stat=242, mongo=242
    'Farense': 'Farense', # Exact Match (Standard) | IDs: stat=231, mongo=231
    'Feirense': 'Feirense', # Exact Match (Standard) | IDs: stat=213, mongo=213
    'Fenerbahce': 'Fenerbahce', # Exact Match (Standard) | IDs: stat=611, mongo=611
    'FeralpiSalo': 'Pisa', # Fuzzy Match (90%) | IDs: stat=801, mongo=801
    'Ferrol': '???', # Manual mapping required.
    'Feyenoord': 'Feyenoord', # Exact Match (Standard) | IDs: stat=209, mongo=209
    'Fiorentina': 'Fiorentina', # Exact Match (Standard) | IDs: stat=502, mongo=502
    'Foggia': '???', # Manual mapping required.
    'For Sittard': 'For Sittard', # Exact Match (Standard) | IDs: stat=205, mongo=205
    'Fortuna Dusseldorf': 'Fortuna Dusseldorf', # Exact Match (Standard) | IDs: stat=158, mongo=158
    'Frankfurt FSV': '???', # Manual mapping required.
    'Freiburg': 'Freiburg', # Exact Match (Standard) | IDs: stat=160, mongo=160
    'Frosinone': 'Frosinone', # Exact Match (Standard) | IDs: stat=512, mongo=512
    'Fuenlabrada': '???', # Manual mapping required.
    'Fulham': 'Fulham', # Exact Match (Standard) | IDs: stat=36, mongo=36
    'Galatasaray': 'Galatasaray', # Exact Match (Standard) | IDs: stat=645, mongo=645
    'Gallipoli': '???', # Manual mapping required.
    'Gaziantep': '???', # Manual mapping required.
    'Genclerbirligi': '???', # Manual mapping required.
    'Genk': 'Genk', # Exact Match (Standard) | IDs: stat=742, mongo=742
    'Genoa': 'Genoa', # Exact Match (Standard) | IDs: stat=495, mongo=495
    'Gent': 'Gent', # Exact Match (Standard) | IDs: stat=631, mongo=631
    'Germinal': '???', # Manual mapping required.
    'Getafe': 'Getafe', # Exact Match (Standard) | IDs: stat=546, mongo=546
    'Giannina': '???', # Manual mapping required.
    'Gil Vicente': 'Gil Vicente', # Exact Match (Standard) | IDs: stat=762, mongo=762
    'Gimnastic': '???', # Manual mapping required.
    'Giresunspor': '???', # Manual mapping required.
    'Girona': 'Girona', # Exact Match (Standard) | IDs: stat=547, mongo=547
    'Go Ahead Eagles': 'Go Ahead Eagles', # Exact Match (Standard) | IDs: stat=410, mongo=410
    'Goztep': 'Goztep', # Exact Match (Standard) | IDs: stat=994, mongo=994
    'Graafschap': 'Graafschap', # Exact Match (Standard) | IDs: stat=199, mongo=199
    'Granada': 'Granada', # Exact Match (Standard) | IDs: stat=715, mongo=715
    'Granada 74': 'Granada', # Fuzzy Match (95%) | IDs: stat=715, mongo=715
    'Grenoble': 'Grenoble', # Exact Match (Standard) | IDs: stat=101, mongo=101
    'Greuther Furth': 'Greuther Furth', # Exact Match (Standard) | IDs: stat=178, mongo=178
    'Groningen': 'Groningen', # Exact Match (Standard) | IDs: stat=202, mongo=202
    'Grosseto': '???', # Manual mapping required.
    'Guadalajara': '???', # Manual mapping required.
    'Gubbio': '???', # Manual mapping required.
    'Gueugnon': '???', # Manual mapping required.
    'Guimaraes': 'Guimaraes', # Exact Match (Standard) | IDs: stat=224, mongo=224
    'Guingamp': 'Guingamp', # Exact Match (Standard) | IDs: stat=90, mongo=90
    'Hamburg': 'Hamburg', # Exact Match (Standard) | IDs: stat=175, mongo=175
    'Hannover': 'Hannover', # Exact Match (Standard) | IDs: stat=166, mongo=166
    'Hansa Rostock': '???', # Manual mapping required.
    'Hatayspor': 'Hatayspor', # Exact Match (Standard) | IDs: stat=3575, mongo=3575
    'Heerenveen': 'Heerenveen', # Exact Match (Standard) | IDs: stat=210, mongo=210
    'Heidenheim': '???', # Manual mapping required.
    'Heracles': 'Heracles', # Exact Match (Standard) | IDs: stat=206, mongo=206
    'Hercules': 'Heracles', # Fuzzy Match (88%) | IDs: stat=206, mongo=206
    'Hertha': '???', # Manual mapping required.
    'Hoffenheim': 'Hoffenheim', # Exact Match (Standard) | IDs: stat=167, mongo=167
    'Holstein Kiel': 'Holstein Kiel', # Exact Match (Standard) | IDs: stat=191, mongo=191
    'Huddersfield': '???', # Manual mapping required.
    'Huesca': 'Huesca', # Exact Match (Standard) | IDs: stat=726, mongo=726
    'Hull': 'Hull', # Exact Match (Standard) | IDs: stat=64, mongo=64
    'Ibiza': '???', # Manual mapping required.
    'Ingolstadt': '???', # Manual mapping required.
    'Inter': 'Inter', # Exact Match (Standard) | IDs: stat=505, mongo=505
    'Ionikos': '???', # Manual mapping required.
    'Ipswich': 'Ipswich', # Exact Match (Standard) | IDs: stat=57, mongo=57
    'Istanbulspor': '???', # Manual mapping required.
    'Istres': '???', # Manual mapping required.
    'Jaen': '???', # Manual mapping required.
    'Juve Stabia': 'Juve Stabia', # Exact Match (Standard) | IDs: stat=863, mongo=863
    'Juventus': 'Juventus', # Exact Match (Standard) | IDs: stat=496, mongo=496
    'Kaiserslautern': '???', # Manual mapping required.
    'Karabukspor': '???', # Manual mapping required.
    'Karagumruk': '???', # Manual mapping required.
    'Karlsruhe': 'Karlsruhe', # Exact Match (Standard) | IDs: stat=785, mongo=785
    'Kasimpasa': 'Kasimpasa', # Exact Match (Standard) | IDs: stat=1004, mongo=1004
    'Kayserispor': 'Kayserispor', # Exact Match (Standard) | IDs: stat=1001, mongo=1001
    'Kerkyra': '???', # Manual mapping required.
    'Kifisia': '???', # Manual mapping required.
    'Koblenz': '???', # Manual mapping required.
    'Konyaspor': 'Konyaspor', # Exact Match (Standard) | IDs: stat=607, mongo=607
    'Kortrijk': 'Kortrijk', # Exact Match (Standard) | IDs: stat=734, mongo=734
    'Lamia': '???', # Manual mapping required.
    'Larisa': '???', # Manual mapping required.
    'Las Palmas': 'Las Palmas', # Exact Match (Standard) | IDs: stat=534, mongo=534
    'Latina': '???', # Manual mapping required.
    'Laval': 'Laval', # Exact Match (Standard) | IDs: stat=433, mongo=433
    'Lazio': 'Lazio', # Exact Match (Standard) | IDs: stat=487, mongo=487
    'Le Havre': 'Le Havre', # Exact Match (Standard) | IDs: stat=111, mongo=111
    'Le Mans': '???', # Manual mapping required.
    'Lecce': 'Lecce', # Exact Match (Standard) | IDs: stat=867, mongo=867
    'Lecco': 'Lecce', # Fuzzy Match (80%) | IDs: stat=867, mongo=867
    'Leeds': 'Leeds', # Exact Match (Standard) | IDs: stat=63, mongo=63
    'Leganes': 'Leganes', # Exact Match (Standard) | IDs: stat=537, mongo=537
    'Leicester': 'Leicester', # Exact Match (Standard) | IDs: stat=46, mongo=46
    'Leiria': 'Leiria', # Exact Match (Standard) | IDs: stat=4662, mongo=4662
    'Leixoes': 'Leixoes', # Exact Match (Standard) | IDs: stat=244, mongo=244
    'Lens': 'Lens', # Exact Match (Standard) | IDs: stat=116, mongo=116
    'Leonesa': '???', # Manual mapping required.
    'Levadeiakos': '???', # Manual mapping required.
    'Levante': 'Levante', # Exact Match (Standard) | IDs: stat=539, mongo=539
    'Leverkusen': 'Leverkusen', # Exact Match (Standard) | IDs: stat=168, mongo=168
    'Libourne': '???', # Manual mapping required.
    'Lierse': '???', # Manual mapping required.
    'Lille': 'Lille', # Exact Match (Standard) | IDs: stat=79, mongo=79
    'Liverpool': 'Liverpool', # Exact Match (Standard) | IDs: stat=40, mongo=40
    'Livorno': '???', # Manual mapping required.
    'Llagostera': '???', # Manual mapping required.
    'Lleida': '???', # Manual mapping required.
    'Logrones': '???', # Manual mapping required.
    'Lokeren': '???', # Manual mapping required.
    'Lorca': 'Mallorca', # Fuzzy Match (90%) | IDs: stat=798, mongo=798
    'Lorient': 'Lorient', # Exact Match (Standard) | IDs: stat=97, mongo=97
    'Lugo': '???', # Manual mapping required.
    'Luton': 'Luton', # Exact Match (Standard) | IDs: stat=1359, mongo=1359
    'Lyon': 'Lyon', # Exact Match (Standard) | IDs: stat=80, mongo=80
    "M'gladbach": "M'gladbach", # Exact Match (Standard) | IDs: stat=163, mongo=163
    'Magdeburg': '???', # Manual mapping required.
    'Mainz': 'Mainz 05', # Fuzzy Match (90%) | IDs: stat=164, mongo=164
    'Malaga': 'Malaga', # Exact Match (Standard) | IDs: stat=535, mongo=535
    'Malaga B': 'Malaga', # Fuzzy Match (95%) | IDs: stat=535, mongo=535
    'Mallorca': 'Mallorca', # Exact Match (Standard) | IDs: stat=798, mongo=798
    'Mantova': 'Mantova', # Exact Match (Standard) | IDs: stat=1693, mongo=1693
    'Maritimo': 'Maritimo', # Exact Match (Standard) | IDs: stat=214, mongo=214
    'Marseille': 'Marseille', # Exact Match (Standard) | IDs: stat=81, mongo=81
    'Martigues': 'Martigues', # Exact Match (Standard) | IDs: stat=3200, mongo=3200
    'Mechelen': 'Mechelen', # Exact Match (Standard) | IDs: stat=266, mongo=266
    'Messina': '???', # Manual mapping required.
    'Metz': 'Metz', # Exact Match (Standard) | IDs: stat=112, mongo=112
    'Middlesbrough': 'Middlesbrough', # Exact Match (Standard) | IDs: stat=70, mongo=70
    'Milan': 'Milan', # Exact Match (Standard) | IDs: stat=489, mongo=489
    'Millwall': 'Millwall', # Exact Match (Standard) | IDs: stat=58, mongo=58
    'Milton Keynes Dons': '???', # Manual mapping required.
    'Mirandes': 'Mirandes', # Exact Match (Standard) | IDs: stat=799, mongo=799
    'Modena': 'Modena', # Exact Match (Standard) | IDs: stat=899, mongo=899
    'Monaco': 'Monaco', # Exact Match (Standard) | IDs: stat=91, mongo=91
    'Montpellier': 'Montpellier', # Exact Match (Standard) | IDs: stat=82, mongo=82
    'Monza': 'Monza', # Exact Match (Standard) | IDs: stat=1579, mongo=1579
    'Moreirense': 'Moreirense', # Exact Match (Standard) | IDs: stat=215, mongo=215
    'Mouscron': '???', # Manual mapping required.
    'Mouscron-Peruwelz': '???', # Manual mapping required.
    'Munich 1860': '???', # Manual mapping required.
    'Murcia': '???', # Manual mapping required.
    'NAC Breda': 'NAC Breda', # Exact Match (Standard) | IDs: stat=203, mongo=203
    'Nacional': 'Nacional', # Exact Match (Standard) | IDs: stat=225, mongo=225
    'Nancy': '???', # Manual mapping required.
    'Nantes': 'Nantes', # Exact Match (Standard) | IDs: stat=83, mongo=83
    'Napoli': 'Napoli', # Exact Match (Standard) | IDs: stat=492, mongo=492
    'Naval': 'Laval', # Fuzzy Match (80%) | IDs: stat=433, mongo=433
    'Newcastle': 'Newcastle', # Exact Match (Standard) | IDs: stat=34, mongo=34
    'Nice': 'Nice', # Exact Match (Standard) | IDs: stat=84, mongo=84
    'Nijmegen': 'Nijmegen', # Exact Match (Standard) | IDs: stat=413, mongo=413
    'Nimes': '???', # Manual mapping required.
    'Niort': '???', # Manual mapping required.
    'Nocerina': '???', # Manual mapping required.
    'Norwich': 'Norwich', # Exact Match (Standard) | IDs: stat=71, mongo=71
    "Nott'm Forest": "Nott'm Forest", # Exact Match (Standard) | IDs: stat=65, mongo=65
    'Novara': '???', # Manual mapping required.
    'Numancia': '???', # Manual mapping required.
    'Nurnberg': '???', # Manual mapping required.
    'OFI Crete': '???', # Manual mapping required.
    'Oberhausen': '???', # Manual mapping required.
    'Offenbach': '???', # Manual mapping required.
    'Olhanense': '???', # Manual mapping required.
    'Olympiakos': 'Olympiakos', # Exact Match (Standard) | IDs: stat=553, mongo=553
    'Oostende': '???', # Manual mapping required.
    'Orleans': '???', # Manual mapping required.
    'Osasuna': 'Osasuna', # Exact Match (Standard) | IDs: stat=727, mongo=727
    'Osmanlispor': '???', # Manual mapping required.
    'Osnabruck': '???', # Manual mapping required.
    'Oud-Heverlee Leuven': 'Oud-Heverlee Leuven', # Exact Match (Standard) | IDs: stat=260, mongo=260
    'Oviedo': 'Oviedo', # Exact Match (Standard) | IDs: stat=718, mongo=718
    'Oxford': 'Oxford', # Exact Match (Standard) | IDs: stat=1338, mongo=1338
    'PAOK': 'PAOK', # Exact Match (Standard) | IDs: stat=619, mongo=619
    'PSV Eindhoven': 'PSV Eindhoven', # Exact Match (Standard) | IDs: stat=197, mongo=197
    'Pacos Ferreira': 'Pacos Ferreira', # Exact Match (Standard) | IDs: stat=234, mongo=234
    'Paderborn': 'Paderborn', # Exact Match (Standard) | IDs: stat=185, mongo=185
    'Padova': '???', # Manual mapping required.
    'Palermo': 'Palermo', # Exact Match (Standard) | IDs: stat=522, mongo=522
    'Panathinaikos': 'Panathinaikos', # Exact Match (Standard) | IDs: stat=617, mongo=617
    'Panetolikos': '???', # Manual mapping required.
    'Panionios': '???', # Manual mapping required.
    'Panserraikos': '???', # Manual mapping required.
    'Paris FC': 'Paris FC', # Exact Match (Standard) | IDs: stat=114, mongo=114
    'Paris SG': 'Paris SG', # Exact Match (Standard) | IDs: stat=85, mongo=85
    'Parma': 'Parma', # Exact Match (Standard) | IDs: stat=523, mongo=523
    'Pau FC': 'Pau FC', # Exact Match (Standard) | IDs: stat=1297, mongo=1297
    'Penafiel': 'Penafiel', # Exact Match (Standard) | IDs: stat=235, mongo=235
    'Pendikspor': '???', # Manual mapping required.
    'Perugia': '???', # Manual mapping required.
    'Pescara': '???', # Manual mapping required.
    'Peterboro': '???', # Manual mapping required.
    'Piacenza': '???', # Manual mapping required.
    'Pisa': 'Pisa', # Exact Match (Standard) | IDs: stat=801, mongo=801
    'Platanias': '???', # Manual mapping required.
    'Plymouth': 'Plymouth', # Exact Match (Standard) | IDs: stat=1357, mongo=1357
    'Poli Ejido': '???', # Manual mapping required.
    'Ponferradina': '???', # Manual mapping required.
    'Pordenone': '???', # Manual mapping required.
    'Portimonense': 'Portimonense', # Exact Match (Standard) | IDs: stat=216, mongo=216
    'Porto': 'Porto', # Exact Match (Standard) | IDs: stat=212, mongo=212
    'Portogruaro': 'Porto', # Fuzzy Match (90%) | IDs: stat=212, mongo=212
    'Portsmouth': 'Portsmouth', # Exact Match (Standard) | IDs: stat=1355, mongo=1355
    'Preston': 'Preston', # Exact Match (Standard) | IDs: stat=59, mongo=59
    'Preußen Münster': '???', # Manual mapping required.
    'Pro Vercelli': '???', # Manual mapping required.
    'QPR': 'QPR', # Exact Match (Standard) | IDs: stat=72, mongo=72
    'Quevilly Rouen': '???', # Manual mapping required.
    'RB Leipzig': 'RB Leipzig', # Exact Match (Standard) | IDs: stat=173, mongo=173
    'RWD Molenbeek': '???', # Manual mapping required.
    'Ravenna': '???', # Manual mapping required.
    'Rayo Majadahonda': '???', # Manual mapping required.
    'Reading': '???', # Manual mapping required.
    'Real Madrid': 'Real Madrid', # Exact Match (Standard) | IDs: stat=541, mongo=541
    'Real Madrid B': 'Real Madrid', # Fuzzy Match (95%) | IDs: stat=541, mongo=541
    'Real Union': '???', # Manual mapping required.
    'Recreativo': '???', # Manual mapping required.
    'Red Star': 'Red Star', # Exact Match (Standard) | IDs: stat=104, mongo=104
    'Regensburg': 'Regensburg', # Exact Match (Standard) | IDs: stat=177, mongo=177
    'Reggiana': 'Reggiana', # Exact Match (Standard) | IDs: stat=880, mongo=880
    'Reggina': 'Reggiana', # Fuzzy Match (93%) | IDs: stat=880, mongo=880
    'Reims': 'Reims', # Exact Match (Standard) | IDs: stat=93, mongo=93
    'Rennes': 'Rennes', # Exact Match (Standard) | IDs: stat=94, mongo=94
    'Reus Deportiu': '???', # Manual mapping required.
    'Rimini': '???', # Manual mapping required.
    'Rio Ave': 'Rio Ave', # Exact Match (Standard) | IDs: stat=226, mongo=226
    'Rizespor': 'Rizespor', # Exact Match (Standard) | IDs: stat=1007, mongo=1007
    'Roda': 'Roda', # Exact Match (Standard) | IDs: stat=414, mongo=414
    'Roda JC': 'Roda', # Fuzzy Match (90%) | IDs: stat=414, mongo=414
    'Rodez': 'Rodez', # Exact Match (Standard) | IDs: stat=1301, mongo=1301
    'Roeselare': '???', # Manual mapping required.
    'Roma': 'Roma', # Exact Match (Standard) | IDs: stat=497, mongo=497
    'Roosendaal': '???', # Manual mapping required.
    'Rotherham': '???', # Manual mapping required.
    'Rouen': '???', # Manual mapping required.
    'Sabadell': '???', # Manual mapping required.
    'Salamanca': '???', # Manual mapping required.
    'Salernitana': 'Salernitana', # Exact Match (Standard) | IDs: stat=514, mongo=514
    'Sampdoria': 'Sampdoria', # Exact Match (Standard) | IDs: stat=498, mongo=498
    'Samsunspor': 'Samsunspor', # Exact Match (Standard) | IDs: stat=3603, mongo=3603
    'Sandhausen': '???', # Manual mapping required.
    'Santa Clara': 'Santa Clara', # Exact Match (Standard) | IDs: stat=227, mongo=227
    'Sassuolo': 'Sassuolo', # Exact Match (Standard) | IDs: stat=488, mongo=488
    'Schalke 04': 'Schalke 04', # Exact Match (Standard) | IDs: stat=174, mongo=174
    'Scunthorpe': '???', # Manual mapping required.
    'Sedan': '???', # Manual mapping required.
    'Seraing': '???', # Manual mapping required.
    'Sete': '???', # Manual mapping required.
    'Setubal': '???', # Manual mapping required.
    'Sevilla': 'Sevilla', # Exact Match (Standard) | IDs: stat=536, mongo=536
    'Sevilla B': 'Sevilla', # Fuzzy Match (95%) | IDs: stat=536, mongo=536
    'Sheffield United': 'Sheffield United', # Exact Match (Standard) | IDs: stat=62, mongo=62
    'Sheffield Weds': 'Sheffield Weds', # Exact Match (Standard) | IDs: stat=62, mongo=74
    'Siena': '???', # Manual mapping required.
    'Sivasspor': 'Sivasspor', # Exact Match (Standard) | IDs: stat=1002, mongo=1002
    'Sochaux': '???', # Manual mapping required.
    'Sociedad B': 'Sociedad', # Fuzzy Match (95%) | IDs: stat=548, mongo=548
    'Southampton': 'Southampton', # Exact Match (Standard) | IDs: stat=41, mongo=41
    'Southend': '???', # Manual mapping required.
    'Sp Braga': 'Sp Braga', # Exact Match (Standard) | IDs: stat=217, mongo=217
    'Sp Lisbon': 'Sp Lisbon', # Exact Match (Standard) | IDs: stat=228, mongo=228
    'Spal': '???', # Manual mapping required.
    'Sparta': 'Sparta Rotterdam', # Fuzzy Match (90%) | IDs: stat=426, mongo=426
    'Sparta Rotterdam': 'Sparta Rotterdam', # Exact Match (Standard) | IDs: stat=426, mongo=426
    'Spezia': 'Spezia', # Exact Match (Standard) | IDs: stat=515, mongo=515
    'St Etienne': '???', # Manual mapping required.
    'St Pauli': 'St Pauli', # Exact Match (Standard) | IDs: stat=186, mongo=186
    'St Truiden': 'St Truiden', # Exact Match (Standard) | IDs: stat=735, mongo=735
    'St. Gilloise': 'St. Gilloise', # Exact Match (Standard) | IDs: stat=1393, mongo=1393
    'Standard': 'Standard', # Exact Match (Standard) | IDs: stat=733, mongo=733
    'Stoke': 'Stoke', # Exact Match (Standard) | IDs: stat=75, mongo=75
    'Strasbourg': 'Strasbourg', # Exact Match (Standard) | IDs: stat=95, mongo=95
    'Stuttgart': 'Stuttgart', # Exact Match (Standard) | IDs: stat=172, mongo=172
    'Sudtirol': 'Sudtirol', # Exact Match (Standard) | IDs: stat=1578, mongo=1578
    'Sunderland': 'Sunderland', # Exact Match (Standard) | IDs: stat=746, mongo=746
    'Swansea': 'Swansea', # Exact Match (Standard) | IDs: stat=76, mongo=76
    'Tenerife': 'Tenerife', # Exact Match (Standard) | IDs: stat=719, mongo=719
    'Ternana': '???', # Manual mapping required.
    'Tondela': 'Tondela', # Exact Match (Standard) | IDs: stat=218, mongo=218
    'Torino': 'Torino', # Exact Match (Standard) | IDs: stat=503, mongo=503
    'Tottenham': 'Tottenham', # Exact Match (Standard) | IDs: stat=47, mongo=47
    'Toulouse': 'Toulouse', # Exact Match (Standard) | IDs: stat=96, mongo=96
    'Tours': '???', # Manual mapping required.
    'Trabzonspor': 'Trabzonspor', # Exact Match (Standard) | IDs: stat=998, mongo=998
    'Trapani': '???', # Manual mapping required.
    'Treviso': '???', # Manual mapping required.
    'Triestina': '???', # Manual mapping required.
    'Trofense': '???', # Manual mapping required.
    'Troyes': 'Troyes', # Exact Match (Standard) | IDs: stat=110, mongo=110
    'Tubize': '???', # Manual mapping required.
    'Twente': 'Twente', # Exact Match (Standard) | IDs: stat=415, mongo=415
    'UCAM Murcia': '???', # Manual mapping required.
    'Udinese': 'Udinese', # Exact Match (Standard) | IDs: stat=494, mongo=494
    'Ulm': 'SSV Ulm 1846', # Fuzzy Match (90%) | IDs: stat=1652, mongo=1652
    'Umraniyespor': '???', # Manual mapping required.
    'Uniao Madeira': '???', # Manual mapping required.
    'Union Berlin': 'Union Berlin', # Exact Match (Standard) | IDs: stat=182, mongo=182
    'Unterhaching': '???', # Manual mapping required.
    'Utrecht': 'Utrecht', # Exact Match (Standard) | IDs: stat=207, mongo=207
    'VVV Venlo': 'VVV Venlo', # Exact Match (Standard) | IDs: stat=204, mongo=204
    'Valence': 'Valencia', # Fuzzy Match (80%) | IDs: stat=532, mongo=532
    'Valencia': 'Valencia', # Exact Match (Standard) | IDs: stat=532, mongo=532
    'Valenciennes': 'Valencia', # Fuzzy Match (84%) | IDs: stat=532, mongo=532
    'Valladolid': 'Valladolid', # Exact Match (Standard) | IDs: stat=720, mongo=720
    'Vallecano': 'Vallecano', # Exact Match (Standard) | IDs: stat=728, mongo=728
    'Vannes': '???', # Manual mapping required.
    'Varese': 'Carrarese', # Fuzzy Match (82%) | IDs: stat=1581, mongo=1581
    'Vecindario': '???', # Manual mapping required.
    'Venezia': 'Venezia', # Exact Match (Standard) | IDs: stat=517, mongo=517
    'Verona': 'Verona', # Exact Match (Standard) | IDs: stat=504, mongo=504
    'Vicenza': '???', # Manual mapping required.
    'Villarreal': 'Villarreal', # Exact Match (Standard) | IDs: stat=533, mongo=533
    'Villarreal B': 'Villarreal', # Fuzzy Match (95%) | IDs: stat=533, mongo=533
    'Virtus Entella': '???', # Manual mapping required.
    'Virtus Lanciano': '???', # Manual mapping required.
    'Vitesse': 'Vitesse', # Exact Match (Standard) | IDs: stat=200, mongo=200
    'Vizela': 'Vizela', # Exact Match (Standard) | IDs: stat=810, mongo=810
    'Volendam': 'Volendam', # Exact Match (Standard) | IDs: stat=416, mongo=416
    'Volos NFC': '???', # Manual mapping required.
    'Waalwijk': 'Waalwijk', # Exact Match (Standard) | IDs: stat=417, mongo=417
    'Waasland-Beveren': '???', # Manual mapping required.
    'Waregem': '???', # Manual mapping required.
    'Wasquehal': '???', # Manual mapping required.
    'Watford': 'Watford', # Exact Match (Standard) | IDs: stat=38, mongo=38
    'Wehen': '???', # Manual mapping required.
    'Werder Bremen': 'Werder Bremen', # Exact Match (Standard) | IDs: stat=162, mongo=162
    'West Brom': 'West Brom', # Exact Match (Standard) | IDs: stat=60, mongo=60
    'West Ham': 'West Ham', # Exact Match (Standard) | IDs: stat=48, mongo=48
    'Westerlo': 'Westerlo', # Exact Match (Standard) | IDs: stat=261, mongo=261
    'Wigan': '???', # Manual mapping required.
    'Willem II': 'Willem II', # Exact Match (Standard) | IDs: stat=195, mongo=195
    'Wolfsburg': 'Wolfsburg', # Exact Match (Standard) | IDs: stat=161, mongo=161
    'Wolves': 'Wolves', # Exact Match (Standard) | IDs: stat=39, mongo=39
    'Wurzburger Kickers': '???', # Manual mapping required.
    'Wycombe': '???', # Manual mapping required.
    'Xanthi': '???', # Manual mapping required.
    'Xerez': '???', # Manual mapping required.
    'Yeni Malatyaspor': 'Alanyaspor', # Fuzzy Match (81%) | IDs: stat=996, mongo=996
    'Yeovil': '???', # Manual mapping required.
    'Zaragoza': 'Zaragoza', # Exact Match (Standard) | IDs: stat=732, mongo=732
    'Zwolle': 'Zwolle', # Exact Match (Standard) | IDs: stat=193, mongo=193
}

# --- Pydantic Models for Schema Validation ---
TeamNameStrict = constr(min_length=1)

class TeamDetails(BaseModel):
    country: str = Field(default="TODO: Unknown Country")
    statarea_id: Optional[str] = None
    mongodb_id: Optional[str] = None
    alt: List[str] = Field(default_factory=list)

    @field_validator("alt", mode="before")
    @classmethod
    def ensure_alt_is_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            # Allow a single string to be converted to a list with one string
            if isinstance(v, str):
                return [v]
            raise ValueError("alt must be a list, a string, or None")
        return v
    
    class Config:
        validate_assignment = True


TeamIdMappingSchema = Dict[TeamNameStrict, TeamDetails]
TeamNameNormalizationSchema = Dict[TeamNameStrict, TeamNameStrict]


# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'unified_data'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'saved_models'), exist_ok=True) # Ensure model dir exists too

# --- Paths ---
# Use absolute paths or paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is at project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'historical_matches.csv') # Example
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'features.parquet') # Example
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
ELO_SAVE_PATH = os.path.join(MODEL_DIR, 'elo_calculator.joblib')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Prediction Configuration ---
class PredictionConfig(BaseModel, strict=True):
    """Configuration settings for generating the final ranked predictions."""

    # --- Core Ranking Settings ---
    n_top_predictions: int = Field(
        default=8, ge=5, le=15,
        description="Number of top ranked predictions to generate per match."
    )
    min_probability_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Minimum probability for any outcome (single or dual) to be considered before ranking."
    )

    # --- Single Market Inclusion ---
    include_1x2_singles: List[str] = Field(
        default=['prob_H', 'prob_D', 'prob_A'],
        description="List of 1X2 probability keys to include as single predictions."
    )
    include_dc_singles: List[str] = Field(
        default=['prob_1X', 'prob_X2', 'prob_12'],
        description="List of Double Chance probability keys (calculated) to include as single predictions."
    )
    include_ou_singles: List[str] = Field(
        default=[
            'prob_O15', 'prob_U15', 'prob_O25', 'prob_U25', 'prob_O35', 'prob_U35',
        ],
        description="List of Over/Under probability keys to include as single predictions."
    )
    include_btts_singles: List[str] = Field(
        default=['prob_BTTS_Y', 'prob_BTTS_N'],
        description="List of BTTS probability keys to include as single predictions."
    )
    include_goal_band_singles: List[str] = Field(
        default=[
            'prob_goals_0_1', 'prob_goals_2_3', 'prob_goals_2_4', 'prob_goals_3_plus',
        ],
        description="List of Goal Band probability keys to include as single predictions."
    )

    # --- Dual Market Inclusion (using ACCURATE scoreline summation results) ---
    include_1x2_ou25_duals: List[str] = Field(
        default=[
            'prob_H_and_O25', 'prob_D_and_O25', 'prob_A_and_O25',
            'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
        ],
        description="List of 1X2 & O/U 2.5 dual probability keys (calculated via scoreline) to include."
    )
    include_dc_ou25_duals: List[str] = Field(
         default=[
            'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25',
            'prob_1X_and_U25', 'prob_12_and_U25', 'prob_X2_and_U25',
         ],
        description="List of DC & O/U 2.5 dual probability keys (calculated via scoreline) to include."
    )
    include_1x2_btts_duals: List[str] = Field(
        default=[
            'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y', 'prob_A_and_BTTS_Y',
            'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
        ],
        description="List of 1X2 & BTTS dual probability keys (calculated via scoreline) to include."
    )
    include_dc_btts_duals: List[str] = Field(
        default=[
            'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y',
            'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
        ],
        description="List of DC & BTTS dual probability keys (calculated via scoreline) to include."
    )
    include_ou25_btts_duals: List[str] = Field(
        default=[
            'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N',
            'prob_U25_and_BTTS_Y', 'prob_U25_and_BTTS_N',
        ],
        description="List of O/U 2.5 & BTTS dual probability keys (calculated via scoreline) to include."
    )

    # --- Output Formatting ---
    required_match_info_cols: List[str] = Field(
        default=['MatchID', 'HomeTeam', 'AwayTeam'],
        description="Columns required from the input DataFrame for prediction formatting."
    )

    # --- Internal Helper ---
    def get_allowed_prediction_keys(self) -> Set[str]:
        """Returns a set of all probability keys to be considered for ranking."""
        allowed_keys = set()
        allowed_keys.update(self.include_1x2_singles)
        allowed_keys.update(self.include_dc_singles)
        allowed_keys.update(self.include_ou_singles)
        allowed_keys.update(self.include_btts_singles)
        allowed_keys.update(self.include_goal_band_singles)
        allowed_keys.update(self.include_1x2_ou25_duals)
        allowed_keys.update(self.include_dc_ou25_duals)
        allowed_keys.update(self.include_1x2_btts_duals)
        allowed_keys.update(self.include_dc_btts_duals)
        allowed_keys.update(self.include_ou25_btts_duals)
        return allowed_keys

# --- Optional: Model Parameter Configs ---
class PoissonModelParams(BaseModel, strict=True):
    """Default Hyperparameters for the Poisson model."""
    alpha: float = Field(default=1e-5, ge=0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-4, gt=0)

UNIQUE_TEAMS_JSON_STRING = """
{
  "teams": [
    "1074\\u00c7ank\\u0131r\\u0131spor", "1461Trabzon", "1899Hoffenheim", "1910Oradea", "1920Mara\\u015fspor",
    "1FCHeidenheim", "1FCKaiserslautern", "1FCK\\u00f6ln", "1FCMagdeburg", "1FCN\\u00fcrnberg",
    "1\\u00baDezembro", "24Erzincanspor", "68AksarayBelediyespor", "76I\\u011fd\\u0131rBelediyespor",
    "ABCopenhagen", "ACHorsens", "ACMilan", "ACSBerceni", "ACSForestaSuceava", "ACSPoliTimisoara",
    "ACSSirineasa", "ACV", "ADCeutaFC", "ADO20", "ADODenHaag", "ADPortomosense", "AEKAthensFC",
    "AEKLarnaca", "AEL", "AELKallonis", "AEZakakiou", "AFCAmsterdam", "AFCFylde", "AFCHermannstadt",
    "AFCLeopards", "AFCRushdenDiamonds", "AFCTelfordUnited", "AFCWimbledon", "AGCaennaise",
    "AIKStockholm", "AJFano", "AOKKerkyra", "ARS\\u00e3oMartinho", "ASATarguMures", "ASECK",
    "ASEupen", "ASFAYennenga", "ASFB", "ASJeunesseEsch", "ASKKlagenfurt", "ASPTTBrest", "ASRoma",
    "ASSStillMutzig", "ASTrencin", "ASVDeDijk", "ASWH", "AZAlkmaar", "AZPicerno", "Aabenraa",
    "Aabyh\\u00f8j", "Aalborg", "AalborgFreja", "Aarhus", "Aarhus1900", "AarhusFremad", "Aarup",
    "Aberdeen", "AcademiaAnzo\\u00e1tegui", "Academico", "AcademicoViseu", "Academico_Viseu",
    "AccringtonST", "Acero", "Acharnaikos", "Achilles29", "AchillesVeen", "Acireale", "Adamswiller",
    "AdanaDemirspor", "Adanaspor", "AdmiraWacker", "Ad\\u0131yaman1954", "AerostarBacau",
    "AfjetAfyonspor", "Afyonkarahisarspor", "Agde", "Agerb\\u00e6kStarup", "AgricolaBorcea",
    "AgrotikosAsteras", "AhironasOnisilos", "Aiginiakos", "Aiglon", "AirbusUK", "Aische",
    "AittitosSpaton", "Aixoise", "AjaccioGFCO", "Ajax", "AjaxAmateurs", "Ajka", "Ajman",
    "AkademijaPandev", "AkhisarBelediye", "AkhmatGrozny", "Akragas", "Akritas", "Akron", "Aktobe",
    "AlAhliJeddah", "AlAhly", "AlArabiSC", "AlDuhailSC", "AlEttifaq", "AlFateh", "AlFayha",
    "AlHazm", "AlHilalAlNassrStars", "AlHilalSaudiFC", "AlIttihadFC", "AlJazira", "AlNasr",
    "AlNassr", "AlQadisiyahFC", "AlRaed", "AlRayyanSC", "AlRiffa", "AlSadd", "AlShabab",
    "AlShamal", "AlTaawon", "AlWahdaFC", "AlWakrah", "AlWaslFC", "AlWehdaClub", "Alanyaspor",
    "Alashkert", "Alatri", "Alaves", "Albacete", "Albalonga", "AlberesArgel\\u00e8s",
    "AlbertQu\\u00e9vyMons", "Albi", "Albissola", "Alcains", "Alcanenense", "Alcides", "Alcione",
    "Alcochetense", "Alcora", "AldershotTown", "AlemanniaAachen", "AlemanniaWaldalgesheim",
    "Alen\\u00e7on", "Alexandria", "Alfaro", "Alfonsine", "AlfretonTown", "Algar", "Algeciras",
    "Aljustrelense", "AlkiOroklini", "Aller\\u00f8d", "Alles\\u00f8", "AlloaAthletic",
    "AllobrogesAsafia", "Almansa", "Almaz\\u00e1n", "AlmereCityFC", "Almeria", "Alsbach",
    "AltaLisboa", "Altglienicke", "Altona93", "Altrincham", "Alt\\u0131nda\\u011fBelediyesispor",
    "Alt\\u0131nordu", "Aluminij", "Alverca", "AlwaysReady", "Alzira", "Amarante", "Ambrosiana",
    "Amed", "Amiens", "AmiensAC", "Amn\\u00e9ville", "Amora", "Anadia", "AnadoluSel\\u00e7ukspor",
    "Anadolu\\u00dcsk\\u00fcdar", "AnagennisiKarditsas", "Anaitasuna", "Anaune", "Anaunia",
    "Anderlecht", "Andratx", "Andr\\u00e9zieux", "Angers", "AngletGenets", "Angoul\\u00eame",
    "Angrense", "Anif", "AnkaraDemirspor", "Ankaraspor", "AnkerWismar", "AnnanAthletic", "Annecy",
    "Anorthosis", "Ansbach", "Antalyaspor", "Antequera", "Antoniano", "Antwerp", "ApoelNicosia",
    "ApollonLimassol", "ApollonPontou", "ApollonSmirnis", "April25", "AraguaFC", "Arakl\\u0131spor",
    "Arandina", "Ararat", "AraratArmenia", "Arconatese", "ArdaKardzhali", "ArenasGetxo", "Arenteiro",
    "Argentr\\u00e9duPlessis", "ArgesPitesti", "ArisAvato", "ArisThessalonikis", "ArkaGdynia",
    "ArleseyTown", "Armacenenses", "ArminiaBielefeld", "Arnavutk\\u00f6yBelediyespor", "Arnedo",
    "Arosa", "Arouca", "Arras", "ArroncheseBenfica", "Arsenal", "ArsenalTivat", "ArsenalTula",
    "ArsenalU21", "Arsinspor", "ArtvinHopaspor", "Arzachena", "ArzignanoValchiampo", "Ashdod",
    "AstonVilla", "AstonVillaU21", "Astur", "Atalanta", "AtalantaII", "AthleticClub",
    "AthleticClubII", "AthleticoMarseille", "Atlas", "AtlasDelmenhorst", "AtleticoArteixo",
    "AtleticoMadrid", "Atl\\u00e9ticoArteixo", "Atl\\u00e9ticoAstorga", "Atl\\u00e9ticoBaleares",
    "Atl\\u00e9ticoCP", "Atl\\u00e9ticoLugones", "Atl\\u00e9ticoMadridII", "Atl\\u00e9ticoMalveira",
    "Atl\\u00e9ticoPaso", "Atl\\u00e9ticoPulpile\\u00f1o", "Atl\\u00e9ticoTordesillas",
    "AtromitosAthinon", "Atzeneta", "Aubagne", "Aubel", "Aubervilliers", "Auch", "Auda",
    "AudaceCerignola", "AugsburgII", "AumundVegesack", "Auray", "AustriaKlagenfurt",
    "AustriaLustenau", "AustriaSalzburg", "AustriaVienna", "Autol", "Auxerre", "AuxerreII",
    "Avarta", "AvenirFootLoz\\u00e8re", "Avion", "AvironBayonnais", "AvoineOCC", "Avranches",
    "AvvSwift", "Av\\u00e2ntulValeaM\\u0103rului", "Axiopolis", "Ayd\\u0131nspor", "Ayd\\u0131nspor1923",
    "AyiaNapa", "AyrUtd", "Ayval\\u0131kg\\u00fcc\\u00fcBelediyespor", "Azuaga", "B1908", "B1913",
    "B36Torshavn", "B93", "BAK", "BAK07", "BBBodrumspor", "BFCDynamo", "BFCPreussen", "BGA",
    "BKHacken", "BKVEl\\u0151re", "BSCHastedt", "BSCYoungBoys", "BSF", "BSGChemieLeipzig",
    "BWLohne", "Backa", "BadKreuznach", "BadWimsbach", "Badajoz", "Badalona", "BagnoleseBM",
    "BahlingerSC", "BalaTown", "Balagne", "Balatonf\\u00fcredi", "Balingen", "Ballkani",
    "Balmazujvaros", "Baltika", "Bal\\u00e7ovaYa\\u015famspor", "Bal\\u0131kesirspor", "BamberBridge",
    "BanburyUnited", "Band\\u0131rmaspor", "BangkokGlass", "BangorCity", "Banjole",
    "Ban\\u00edkOstrava", "Barakaldo", "Barbad\\u00e1s", "Barbastro", "Barbate", "Barcelona",
    "Barendrecht", "Bari", "BarmbekUhlenhorst", "Barnet", "BarockstadtFuldaLehn", "Baronie",
    "BarracasCentral", "Barrosas", "Barrow", "BasaraMainz", "BasingstokeTown", "BassanoVirtus",
    "Bastia", "BastiaBorgo", "BateBorisov", "BathCity", "BatmanPetrolspor", "Baunatal",
    "Bayburt\\u0130\\u00d6\\u0130", "BayerLeverkusen", "BayernAlzenau", "BayernM\\u00fcnchen",
    "BayernM\\u00fcnchenII", "Bayonne", "Bayrampa\\u015faspor", "Bayreuth", "Beasain", "Beaune",
    "BecerrilCampos", "Bedford", "BedfordTown", "Bednja", "Beerschot", "BeerschotWilrijk",
    "BeijingGuoan", "BeiraMar", "BeitarJerusalem", "Bekescsaba1912", "BelediyeDerincespor",
    "BelenensesII", "Belfort", "BelgranoCordoba", "Beli\\u0161\\u0107e", "Bellinzona", "Benfica",
    "BenficaB", "BenficaCasteloBranco", "Benig\\u00e0nim", "Ben\\u00e1tkynadJizerou", "Berck",
    "Berg", "BergamaBelediyespor", "Berganti\\u00f1os", "BergenDal", "Bergerac", "BergischGladbach",
    "Berkum", "Beroe", "Berre", "Bersenbr\\u00fcck", "BertemLeefdaal", "Ber\\u00e7o",
    "BesaDob\\u00ebrdoll", "Besan\\u00e7on", "Besiktas", "Beylerbeyispor", "BeypazariSekerspor",
    "Bezanija", "Be\\u0142chat\\u00f3w", "Biancavilla", "Biarritz", "Biberach", "Bicskei",
    "BielBienne", "Biesheim", "BihorOradea", "Bilje", "Bilogora91", "Birkirkara", "Bisceglie",
    "Bischofshofen", "BishopAuckland", "Bistra", "Bistrica", "Bjelovar", "Blackburn", "Blagnac",
    "BlancMesnil", "BlauwGeel", "Blejoi", "Blois", "BlumenthalerSV", "BneiSakhnin", "Boavista",
    "Bobigny", "BocaJuniors", "BodoGlimt", "BognorRegisTown", "Bohemians", "Bohemians1905",
    "Boiro", "Bolbec", "Bologna", "Boluspor", "Bol\\u00edvar", "BonnerSC", "BoracBanjaLuka",
    "BoracCacak", "BordeauxII", "BorehamWood", "Borgo", "BorinciJarmina", "BorussiaDortmund",
    "BorussiaDortmundII", "BorussiaHildesheim", "BorussiaMgladbachII", "BorussiaMonchengladbach",
    "BorussiaM\\u00f6nchengladbach", "Bosch_ADODenHaag", "Bosch_AEKLarnaca", "Bosch_AFCAmsterdam",
    "Bosch_Achilles29", "Bosch_Ajax", "Bosch_AlmereCityFC", "Bosch_Cambuur", "Bosch_DeGraafschap",
    "Bosch_DeTreffers", "Bosch_Dordrecht", "Bosch_Emmen", "Bosch_Excelsior", "Bosch_FCEindhoven",
    "Bosch_FCOSS", "Bosch_FCVolendam", "Bosch_FortunaSittard", "Bosch_GOAheadEagles",
    "Bosch_Groningen", "Bosch_HelmondSport", "Bosch_Heracles", "Bosch_JongAZ", "Bosch_JongAjax",
    "Bosch_JongPSV", "Bosch_JongUtrecht", "Bosch_MVV", "Bosch_NACBreda", "Bosch_NECNijmegen",
    "Bosch_NivoSparta", "Bosch_PECZwolle", "Bosch_Roda", "Bosch_SpartaRotterdam", "Bosch_TEC",
    "Bosch_Telstar", "Bosch_Twente", "Bosch_UNA", "Bosch_VVVVenlo", "Bosch_Vitesse",
    "Bosch_Waalwijk", "Bosch_WillemII", "Botafogo", "BotevPlovdiv", "BotevVratsa",
    "Bourgenbresse01", "BourgesFoot", "BourgoinJallieu", "Bournemouth", "Boyabat1868Spor",
    "Bozner", "Brabantia", "Brabrand", "BracknellTown", "Bradford", "Bragadiru", "Bragan\\u00e7a",
    "Braila", "Braintree", "Brann", "Bravo", "Bra\\u0219ovSteagulRena\\u0219te", "Breidablik",
    "Breitenrain", "BremerSV", "BremerSv", "Breno", "Brentford", "BrentfordB", "Brescia",
    "Bressuire", "Breteuil", "Bre\\u017eice", "Brighton", "BrightonU21", "BrightonU23", "Brindisi",
    "BrinjeGrosuplje", "BrisbaneRoar", "BristolCity", "BristolRovers", "Brito", "Bromley",
    "Brondby", "Br\\u00e9tignyFoot", "Br\\u00f8nsh\\u00f8j", "Br\\u00fchl", "BskBijeloBrdo", "BucaFK",
    "BucakO\\u011fuzhanspor", "Bucaspor", "BuckieThistle", "BudafokiLC", "BudapestHonved",
    "BuducnostDobanovci", "BuducnostPodgorica", "Buje", "Bunyodkor", "BurgessHillTown",
    "Burgos", "Burnley", "BurtonAlbion", "Bury", "BuryTown", "Buxton", "Bu\\u00f1ol",
    "BytoviaByt\\u00f3w", "B\\u00e9thune", "B\\u00f6blingen", "B\\u00fchlertal", "CALaPaz",
    "CChartres", "CDCalahorra", "CDCoremarca", "CDCoria", "CDOlivaiseMoscavide", "CFIAlicante",
    "CFOsBelenenses", "CFR1907Cluj", "CFRPforzheim", "CFTalavera", "CPBBRennes",
    "CSASteauaBucure\\u015fti", "CSAfumati", "CSBalotesti", "CSCCayenne",
    "CSDinamoBucure\\u0219ti", "CSKA1948", "CSKAMoscow", "CSKASofia", "CSKPivara",
    "CSLuceafarulOradea", "CSMBac\\u0103u", "CSMRamnicuValcea", "CSMRe\\u015fi\\u0163a",
    "CSMioveni", "CSUnireaTarlungeni", "CSUniversitateaCraiova", "Cacere\\u00f1o", "Cadiz",
    "Caen", "CaernarfonTown", "Cagliari", "Calahorra", "Calais", "CalaisBeauMarais",
    "CalcioCastelfiorentino", "Caldas", "CaldieroTerme", "Calvi", "Calvina", "CalvoSotelo",
    "Camacha", "Cambrai", "CambrianClydach", "CambridgeUnited", "Cambuur", "Camon", "Canelas2010",
    "CanetRoussillon", "Cannes", "CannetRocheville", "Cantolagua", "Capelle", "Cappellen",
    "Carapinheirense", "Cardassar", "Cardiff", "CarinaGubin", "CarlZeissJena", "Carlisle",
    "CarmarthenTown", "Carquefou", "Carrarese", "CartusiaKartuzy", "CasaPia", "Casertana",
    "Castell\\u00f3n", "Castelvetro", "CastroDaire", "Castrovillari", "Catanzaro", "Cavese",
    "Cay\\u00f3n", "Cazalegas", "Ca\\u00e7adoresdasTaipas", "Ceahl\\u0103ulPiatraNeam\\u0163",
    "Ceares", "Celje", "CeltaVigo", "CeltadeVigoII", "Celtic", "CercleBrugge", "CerezoOsaka",
    "Cerveira", "Cesena", "CetateDeva", "Ceuta", "Challans", "Cham", "Chamali\\u00e8res",
    "ChamblyThelleFC", "Chamb\\u00e9ry", "Chania", "Chantilly", "Charitoise", "Charleroi",
    "Charlotte", "ChasselayMDA", "ChassieuD\\u00e9cines", "Chaumont", "Chauny", "Chauray",
    "Chauvigny", "Chaves", "ChavesII", "Chelsea", "ChelseaU21", "ChelseaU23", "Cheltenham",
    "CheminBasdAvignon", "ChemnitzerFC", "Cherbourg", "ChernoMoreVarna", "Chester",
    "Chesterfield", "Che\\u0142miankaChe\\u0142m", "Chiclana", "ChiemgauTraunstein",
    "ChindiaTargoviste", "Chisola", "ChlumecnadCidlinou", "ChojniczankaChojnice", "Cholet",
    "Chorley", "Chornomorets", "ChrobryG\\u0142og\\u00f3w", "Chrudim", "Ch\\u00e2teaubriant",
    "Ciliverghe", "Cinf\\u00e3es", "Cittadella", "Cittanovese", "Citt\\u00e0diCampobasso",
    "Citt\\u00e0diFasano", "Citt\\u00e0diFoligno", "CityPiratesAntwerpen", "CiudaddeLucena",
    "Cizrespor", "CjarlinsMuzane", "ClermontFoot", "CliftonvilleFC", "Clodiense", "ClubAmerica",
    "ClubBruggeII", "ClubBruggeKV", "ClubFranciscain", "ClubTijuana", "ClusesScionzier", "Clyde",
    "Coimbr\\u00f5es", "CollinadOro", "Colmar", "ColoColo", "Colomiers", "ColonSantaFe",
    "ColumbusCrew", "Comillas", "Como", "Compi\\u00e8gne", "Compostela", "ComunaRecea",
    "Concordia", "ConcordiaBasel", "Condeixa", "Confian\\u00e7a", "Conquense", "Cordoba",
    "CoriglianoCalabro", "CorkCity", "Cornell\\u00e0", "CoronaBra\\u015fov", "Correggese",
    "Corte", "Cortes", "Coruchense", "Coruxo", "CorvinulHunedoara", "Cosenza", "Coulaines",
    "Courseulles", "Coutada", "CovaDePiedade", "Covadonga", "CoveRangers", "Coventry",
    "Cowdenbeath", "CracoviaKrakow", "Crato", "CrawleyTown", "Creil", "Crema", "Cremonese",
    "Crevillente", "Crikvenica", "CristoAtl\\u00e9tico", "Cri\\u015fulChi\\u015fineuCri\\u015f",
    "CroatiaZmijavci", "CroixFootballIC", "CrusadersFC", "CruzAzul", "CrystalPalace",
    "CrystalPalaceU21", "CsSoimiiPancota", "Csakvar", "Csikszereda", "CsvApeldoorn", "Cukaricki",
    "CulturalLeonesa", "Cuneo", "Cura\\u00e7ao", "Cven", "C\\u00e1dizII", "C\\u00e2maradeLobos",
    "DCUnited", "DEM", "DUNO", "DVS33Ermelo", "DaciaBuiucani", "DaciaUnireaBraila",
    "DagenhamRedbridge", "DakovoCroatia", "Dalum", "Darlington", "Dartford",
    "Dar\\u0131caGen\\u00e7lerbirli\\u011fi", "DeGraafschap", "DeTreffers", "DebreceniVSC",
    "Deinze", "Dekani", "Delbr\\u00fcckerSC", "Den", "Den_Bosch", "Denderhoutem", "DenizliBB",
    "DeportivaMinera", "DeportivoAlav\\u00e9sII", "DeportivoLaCoruna", "DeportivoMurcia",
    "Derby", "DerehamTown", "DerryCity", "Dersimspor", "DesenzanoCalvina", "Desna", "DesselSport",
    "Deutz", "Dhamk", "DiablesNoirs", "Diefflen", "Dieppe", "Diest", "DigenisYpsonas",
    "Dikkelvenne", "Dila", "Dilj", "DinamoBatumi", "DinamoBrest", "DinamoBucuresti",
    "DinamoMinsk", "DinamoMoscow", "DinamoTbilisi", "DinamoTirana", "DinamoVranje",
    "DinamoZagreb", "DinamoZagrebII", "DinanL\\u00e9hon", "Dinsheim", "Diocesano",
    "DiosgyoriVTK", "Diyarbekirspor", "Djoliba", "DjurgardensIF", "Djursland", "Dnipro1",
    "DniproDnipropetrovsk", "Doln\\u00fdKub\\u00edn", "DolomitiBellunesi", "DonBenito", "Dongen",
    "DorchesterTown", "DordoiBishkek", "Dordrecht", "Dornbirn", "DorogiFC", "Douanes",
    "Doubravka", "Dover", "Dovo", "Doxa", "DoxaDramas", "DrachtsterBoys", "Drancy",
    "DrapeauFoug\\u00e8res", "Drita", "Drouais", "DubravaZagreb", "DugoSelo", "Dugopolje",
    "Dukagjini", "DuklaBansk\\u00e1Bystrica", "DuklaPraha", "Dumbarton", "Dumbr\\u0103vi\\u0163a",
    "Dumiense", "DunaharasztiMTK", "DunajskaStreda", "DunareaCalarasi",
    "Duna\\u00fajv\\u00e1rosP\\u00e1lhalma", "Dundalk", "Dundee", "DundeeUtd", "Dunfermline",
    "Dunkerque", "DynamoDresden", "DynamoKyiv", "D\\u00fczcespor", "ESAnzinSaintAubin",
    "ESMTK", "EVV", "EastFife", "Eastleigh", "EbbsfleetUnited", "Ebreichsdorf", "Ebro",
    "Edirnespor", "EendrachtAalst", "EendrachtTermien", "EendrachtWervik", "Egaleo", "Egen",
    "Eger", "EgnatiaRrogozhin\\u00eb", "Eibar", "Eichst\\u00e4tt", "Eilenburg",
    "Eimsb\\u00fcttelerTV", "EinheitWernigerode", "EintrachtBadKreuznach", "EintrachtBamberg",
    "EintrachtBraunschweig", "EintrachtCelle", "EintrachtFrankfurt", "EintrachtFrankfurtII",
    "EintrachtNorderstedt", "EintrachtStadtallendorf", "EintrachtTrier", "Eirense", "Ejea",
    "Ekranas", "ElEjido", "ElPalmar", "ElanaToru\\u0144", "Elaz\\u0131\\u011fBelediyesporFK",
    "Elaz\\u0131\\u011fspor", "Elche", "ElcheII", "Eldense", "EleneGrotenberge", "Eltersdorf",
    "El\\u00c1lamo", "El\\u00e9ctrico", "Emmen", "Empoli", "EnergieCottbus", "Enosis",
    "EntenteSStGratien", "Epinal", "Erbaaspor", "ErgeneVelime\\u015fespor", "Ergotelis",
    "Ermis", "Erokspor", "ErzgebirgeAUE", "ErzinSpor", "ErzincanRefahiyespor", "Esbjerg",
    "EsbjergIF92", "Escobedo", "Eski\\u015fehirspor", "EspalySaintMarcel", "Espanyol",
    "EspanyolII", "Espinho", "Esquelbecq", "EstacTroyes", "Estarreja", "Este", "Estepona",
    "Estoril", "EstorilU23", "Estrela", "EstrelaVendasNovas", "EstudantesAfricanos",
    "EthnikosAchna", "EthnikosAsteras", "EtimesgutBelediyespor", "EtoileFilante", "Europa",
    "EuropaFc", "EuropaPoint", "Everton", "EvertondeVina", "EvianTG", "Excelsior", "Excelsior31",
    "ExcelsiorMaassluis", "ExcelsiorStJoseph", "ExcelsiorVirton", "ExeterCity", "Extremadura",
    "Extremadura1924", "Ey\\u00fcpspor", "F91Dudelange", "FA2000", "FC08Homburg", "FC08Villingen",
    "FCAarau", "FCAndorra", "FCAstana", "FCAstoriaWalldorf", "FCAstraGiurgiu", "FCAugsburg",
    "FCBWLinz", "FCBacau", "FCBasel1893", "FCBocholt", "FCBotosani", "FCBrasov",
    "FCCarlZeissJena", "FCCartagena", "FCChiasso", "FCClinceni", "FCCopenhagen", "FCDallas",
    "FCDender", "FCDifferdange03", "FCEindhoven", "FCFredericia", "FCGie\\u00dfen", "FCGutersloh",
    "FCHalifaxTown", "FCHatvan", "FCHeidenheim", "FCHelsingor", "FCIngolstadt04", "FCJuarez",
    "FCKaiserslautern", "FCLebbeke", "FCLevadiaTallinn", "FCLiefering", "FCLienden", "FCLisse",
    "FCLugano", "FCLuzern", "FCMagdeburg", "FCMariupol", "FCMessina", "FCMidtjylland", "FCMinsk",
    "FCNoah", "FCNordsjaelland", "FCNottingen", "FCNurnberg", "FCOSS", "FCObermais",
    "FCPolitehnicaTimisoara", "FCPorto", "FCPortoB", "FCRielasingenArlen", "FCRostov",
    "FCRotWei\\u00dfErfurt", "FCSB", "FCSRObernai", "FCSTGallen", "FCSaarbrucken",
    "FCSaarbr\\u00fccken", "FCSantaColoma", "FCSchaffhausen", "FCSchalke04", "FCSchweinfurt05",
    "FCSerpa", "FCSion", "FCSochi", "FCStPauli", "FCThun", "FCUFA", "FCUrartu", "FCVaduz",
    "FCViktoriaKoln", "FCViktoriaK\\u00f6ln", "FCVolendam", "FCVoluntari", "FCWIL1900",
    "FCWinterthur", "FCWurzburgerKickers", "FCZurich", "FCsGravenzande", "FHhafnarfjordur",
    "FIUK", "FKBecej", "FKCrvenaZvezda", "FKJablonec", "FKKo\\u0161ice", "FKKukesi", "FKLiepaja",
    "FKMladostPodgorica", "FKPartizan", "FKRabotnicki", "FKSarajevo", "FKSpartakZdrepcevaKRV",
    "FKTobolKostanay", "FKTrayal", "FKVentspils", "FKVozdovac", "FKZalgirisVilnius", "FKZlatibor",
    "FSVFrankfurt", "FSVMainz05", "FSVMainz05II", "FSVZwickau", "FTBraunschweig", "FVEngers07",
    "FVRavensburg", "FabrilBarreiro", "Fabr\\u00e8gues", "Fafe", "FalkenseeFinkenkrug", "Falkirk",
    "Famalicao", "Fanfulla", "Farense", "Farnborough", "FarulConstanta", "FatihKarag\\u00fcmr\\u00fck",
    "FatsaBelediyespor", "Feh\\u00e9rv\\u00e1rFC", "FeigniesAulnoye", "Feirense", "Felgueiras1932",
    "Fenerbahce", "Feralpisalo", "Ferdinandovac", "FerencvarosiTC", "Fermana", "FerreiradeAves",
    "Ferreiras", "Fethiyespor", "Feurs", "Feyenoord", "FeyenoordU21", "Feytiat", "Fezzanese",
    "FidelisAndria", "Filia\\u015fi", "Fiorentina", "Fiorenzuola", "FirstVienna", "Flamengo",
    "FleetwoodTown", "Fleury91", "FloraTallinn", "Floriana", "FloridsdorferAC", "Fluminense",
    "Foc\\u015fani", "Foix", "Fokikos", "FolaEsch", "Fondi", "Fontinhas", "Forbach", "ForestGreen",
    "Forli", "Formigine", "Foron", "FortunaDusseldorf", "FortunaD\\u00fcsseldorf", "FortunaKoln",
    "FortunaRegensburg", "FortunaSittard", "ForwardMadison", "Fos", "Fostiras", "Franciacorta",
    "FrancsBorains", "Fraserburgh", "Freamunde", "Frederiksv\\u00e6rk", "Fredrikstad",
    "FreiburgII", "Frem", "FremadAmager", "Freyming", "Frosinone", "Fr\\u00e9jusStRapha\\u00ebl",
    "Fr\\u00fddekM\\u00edstek", "Fuentes", "Fulham", "FulhamU21", "FurianiAgliani", "Fyn",
    "F\\u00e1tima", "F\\u0103urei", "GAZMetanMedias", "GILVicente", "GKSKatowice", "GOAheadEagles",
    "GOES", "GO\\u0160KDubrovnik", "GO\\u0160KGabela", "GRAP", "GVI", "GVVVVeenendaal", "Gafanha",
    "Gafetense", "Gais", "Gaj", "Galatasaray", "GalliaLucciana", "Gamaches", "GambaOsaka",
    "GambarognoContone", "Gandzasar", "Ganshoren", "Gap", "GarbarniaKrak\\u00f3w", "Garrel",
    "Gateshead", "Gavionenses", "Gavorrano", "GazelecFCAjaccio", "Gaziantepspor",
    "Gaziosmanpa\\u015faspor", "Gazi\\u015fehirGaziantep", "GedaniaGdansk", "Geel", "Geispolsheim",
    "Gelbison", "Gemert", "Genk", "Genoa", "Gent", "GermaniaEgestorf", "GermaniaHalberstadt",
    "GermaniaLeer", "GermaniaWindeck", "Gernika", "Getafe", "GianaErminio", "Gieres",
    "Gij\\u00f3nIndustrial", "Gillingham", "Gimn\\u00e1sticaSegoviana", "Gimn\\u00e1sticaTorrelavega",
    "Gin\\u00e1sioFigueirense", "Gin\\u00e1siodeAlcoba\\u00e7a", "Girona", "Giugliano", "Gjilani",
    "Globo", "GloriaBuz\\u0103u", "Gobelins", "Gocza\\u0142kowiceZdr\\u00f3j", "GoldenLion",
    "Gondomar", "Gonfreville", "Gorica", "GornikZabrze", "GosportBorough", "Gouveia", "Goztepe",
    "Gozzano", "Grace", "Graciosa", "Grafi\\u010dar", "GranadaCF", "GranadaII", "GrandQuevilly",
    "GranicarLaze", "Grani\\u010dar\\u017dupanja", "Granville", "Grasse", "Grasshoppers",
    "Gravelines", "Gravina", "GrazerAK", "GreifswalderFC", "Gremio", "Grenoble", "Greve",
    "Grimsby", "Grobni\\u010dan\\u010cavle", "GroeneSter", "Groningen", "GryfWejherowo",
    "Gr\\u00e6sr\\u00f8dderne", "Gr\\u00f6dig", "Gr\\u00f6digII", "GuadalajaraChivas", "Guadalupe",
    "GuangzhouEvergrandeFC", "Guarda", "GuardaDesportiva", "Guarnizo", "Guichen",
    "GuidoniaMontecelio1937", "Guijuelo", "Guimaraes", "GuimaraesB", "Guingamp", "GuiseleyAFC",
    "Gunzwil", "Gurten", "GwarekTarnowskieG\\u00f3ry", "GyirmotSE", "GyoriETOFC", "GziraUnited",
    "G\\u00e9menos", "G\\u00e9vora", "G\\u00f3rnikPolkowice", "G\\u00f3rnik\\u0141\\u0119czna",
    "G\\u00f6lc\\u00fckspor", "G\\u00fcm\\u00fc\\u015fhanespor", "HB", "HBKoge", "HHC", "HIK",
    "HJKhelsinki", "HNKCibalia", "HNKGorica", "HNKHajdukSplit", "HNKHajdukSplitII", "HNKRijeka",
    "HOKalken", "HSCHannover", "HSVODIN59", "HVCVQuick", "HVVTeWerve", "HZVV", "Hacettepe",
    "Hades", "Haguenau", "HalideEdipAd\\u0131var", "HallescherFC", "Halmstad", "HamKam",
    "HamburgerSV", "HamburgerSVII", "HamiltonAcademical", "HammarbyFF", "HamrunSpartans",
    "HangYuen", "Hannover96", "Han\\u00e1ck\\u00e1", "HapoelBeerSheva", "HapoelHaifa",
    "HapoelKatamon", "HapoelTelAviv", "Hard", "Harelbeke", "HarkemaseBoys", "HaroDeportivo",
    "HarrogateTown", "HarrowBorough", "Harte", "Hartlepool", "HassaniaAgadir", "HastingsUnited",
    "Hatayspor", "Haubourdin", "Haugesund", "HautsLyonnais", "HavantWville", "Havelse",
    "HayesYeadingUnited", "Hazebrouck", "HeartOfMidlothian", "Hebar1918", "Hedensted",
    "Heerenveen", "HeeslingerSC", "Heimstetten", "Heinenoord", "Heist", "Hekimo\\u011fluTrabzon",
    "HelmondSport", "Helsingborg", "HemmingenWesterfeld", "Hendon", "Heracles", "Hereford",
    "HerefordUnited", "Herlev", "Hern\\u00e1nCort\\u00e9s", "Hersted\\u00f8ster", "HerthaBSC",
    "HerthaBSCII", "HerthaBerlin", "HessenKassel", "HeurTongeren", "Hibernian", "Hilden",
    "Hiller\\u00f8d", "HinckleyUnited", "HitchinTown", "Hlu\\u010d\\u00edn", "Hobro", "Hoek",
    "HoffenheimII", "Hohenems", "Holb\\u00e6kBI", "Hollandia", "Hollenbach", "HollufPile",
    "Holstebro", "HolsteinKiel", "HolzheimerSG", "HombourgHaut", "HomeUnited", "Hoogezand",
    "Hoogstraten", "Houilles", "HradecKr\\u00e1lov\\u00e9", "Hrvace", "HrvatskiDragovoljac",
    "Hsc21", "Huesca", "Huldenberg", "HullCity", "Hunedoara", "HungerfordTown", "Hurac\\u00e1nMelilla",
    "HutnikKrak\\u00f3w", "Hu\\u00e9torT\\u00e1jar", "Hu\\u0219anaHu\\u0219i", "Hvidovre",
    "Hwaebul", "HydeUnited", "Hy\\u00e8res", "H\\u00e9rcules", "H\\u00e9rculesII", "H\\u00e9v\\u00edz",
    "IFBrommapojkarna", "IFElfsborg", "IFKGoteborg", "IFKNorrkoping", "IFKVarnamo",
    "IMTNoviBeograd", "IbizaIslasPitiusas", "Ideal", "Ijsselmeervogels", "Ilirija", "Illertissen",
    "Illueca", "IllzachModenheim", "Imolese", "ImortalAlbufeira", "Imotski", "Imst",
    "IndependientedelValle", "Indjija", "Inter", "InterClubdEscaldes", "InterMiami", "InterTurku",
    "InterZapresic", "Intercity", "InternacionaldeMadrid", "InvernessCT", "Ipatinga", "Ipswich",
    "Iraklis", "IraklisPsachna", "IrtyshOmsk", "Ish\\u00f8j", "Isparta32Spor", "IstanbulBasaksehir",
    "Istra1961", "Ivry", "Iv\\u00e1ncsa", "Izarra", "JAArmenti\\u00e8res", "JSHercules",
    "JVCCuijk", "JadranLP", "JadranPore\\u010d", "Jagiellonia", "JagielloniaII", "Jagodina",
    "JahnRegensburg", "Jammerbugt", "Jarun", "Jastrz\\u0119bie", "Javor", "JedinstvoUb", "Jelgava",
    "Jenbach", "JeonbukMotors", "Jette", "JeziorakI\\u0142awa", "JiulPetro\\u015fani", "Joane",
    "JodanBoys", "JohorDarulTakzimFC", "JongAZ", "JongAjax", "JongPSV", "JongUtrecht",
    "JoveEspa\\u00f1ol", "JubiloIwata", "JuraSudFoot", "JuveStabia", "JuvenesDogana",
    "JuventudTorremolinos", "Juventus", "JuventusBucuresti", "JuventusU23", "KAAGentII",
    "KAAkureyri", "KFCUerdingen05", "KFUMOslo", "KFUMRoskilde", "KIKlaksvik", "KKontich",
    "KLeagueStars", "KOZAF", "KRCGenkII", "KRCGent", "KRReykjavik", "KSZO1929", "KVCWesterlo",
    "KVMechelen", "KageraSugar", "Kahramanmara\\u015fspor", "Kahta02Spor", "KairatAlmaty", "Kaisar",
    "KaiserslauternII", "Kalamata", "Kalisz", "KaljuNomme", "Kallithea", "KalmarFF", "Kalundborg",
    "Kaposvar", "KaracabeyBelediyespor", "KaradenizEre\\u011fliBSK", "Karak\\u00f6pr\\u00fcBelediyespor",
    "KardemirKarabukspor", "KarkonoszeJeleniaG\\u00f3ra", "Karlovac", "Karlovac1919", "KarlovyVary",
    "KarlsruherSC", "Karmiotissa", "KarpatyKrosno", "Kartalspor", "Karvin\\u00e1", "Kar\\u015f\\u0131yaka",
    "Kashima", "Kasimpasa", "Kastamonuspor1966", "Katwijk", "Kauno\\u017dalgiris", "Kavala",
    "KawasakiFrontale", "KayapinarBelediyespor", "KayseriErciyesspor", "Kayserispor",
    "Kecskem\\u00e9tiTE", "Kemerspor2003", "KepezBelediyespor", "KeshlaFC", "Kestelspor", "Ketsch",
    "KetteringTown", "Ke\\u00e7i\\u00f6reng\\u00fcc\\u00fc", "Khalidiya", "KickersEmden",
    "KickersOffenbach", "KidderminsterHarriers", "Kilmarnock", "KingsLynnTown", "Kissamikos",
    "KisvardaFC", "Kitchee", "Kjellerup", "Klatovy", "Knokke", "KocaeliBirlikspor", "Kocaelispor",
    "Koetzingue", "KoldingB", "KoldingIF", "KolosKovalivka", "Kolovec", "KolpingBoys", "Kolubara",
    "KoninklijkeHFC", "Konyaspor", "Koper", "KoronaKielce", "Koronc\\u00f3", "Kortrijk",
    "KotwicaKo\\u0142obrzeg", "KozakkenBoys", "Krasnodar", "KremsRehberg", "Kri\\u017eevci", "Krk",
    "Kruoja", "KrupanaVrbasu", "KryvbasKR", "Kr\\u0161ko", "KuPS", "Kufstein", "Kurilovec",
    "Kustosija", "Kutjevo", "Ku\\u017aniaJawor", "Ku\\u017aniaUstro\\u0144", "KyotoSanga", "KyzylZhar",
    "K\\u00e4storf", "K\\u00f6lnII", "K\\u00f6rfezSpor", "K\\u00f6rfez\\u0130skenderunspor",
    "K\\u00fctahyaspor", "K\\u00fc\\u00e7\\u00fck\\u00e7ekmeceSinopspor", "K\\u0131rklarelispor",
    "K\\u0131r\\u0131khanspor", "K\\u0131r\\u0131kkaleB\\u00fcy\\u00fckAnadolu",
    "K\\u0131r\\u015fehirBelediyespor", "K\\u0131z\\u0131lcab\\u00f6l\\u00fckspor",
    "LDAlajuelense", "LEHavre", "LEtratLaTourSportif", "LHospitalet", "LNZCherkasy", "LONGA30",
    "LSF", "LaChauxdeFonds", "LaCh\\u00e2taigneraie", "LaFiorita", "LaLouvi\\u00e8reCentre",
    "LaNuc\\u00eda", "LaRocheVF", "LaSeynesurMerEntente", "LaSuze", "LaTourSaintClair",
    "Laakkwartier", "Laci", "Lagoa", "Lagos", "Laguna", "Lajense", "Lancy", "LandskronaBoIS",
    "Langreo", "Lannion", "Lanzarote", "Laredo", "Larne", "LasPalmas", "LasPalmasII", "LasRozas",
    "LaskJuniorsLinz", "LaskLinz", "LatteDolce", "Lausanne", "Lauwe", "Lavagnese", "Laval",
    "Lazio", "LeBouscat", "LeHavre", "LeHavreFrileuse", "LePaysduValois", "LePoir\\u00e9surVie",
    "LePontetUS", "LePortel", "LePuyFoot", "Lealtad", "Leamington", "Lecce", "LechPoznan",
    "LechPozna\\u0144II", "LechiaDzier\\u017coni\\u00f3w", "LechiaGdansk", "LechiaZielonaG\\u00f3ra",
    "Leeds", "Leganes", "LegiaWarszawa", "LegiaWarszawaII", "LegionoviaLegionowo", "LegnagoSalus",
    "Leher", "Leicester", "LeicesterCityU21", "LeicesterCityU23", "Leioa", "Leixoes", "Lenense",
    "Lens", "LensII", "Lentigione", "Leobendorf", "LesHerbiers", "LesLilas", "LesMureaux",
    "LesUlis", "Levadiakos", "Levante", "LevanteII", "Levico", "LevskiKrumovgrad", "LevskiSofia",
    "LeytonOrient", "Le\\u00e7a", "Libertas", "Licata", "Lichtenberg", "LierseKempenzonen",
    "Liffr\\u00e9", "Ligorna", "Lille", "LilleII", "Lillestrom", "Limoges", "Limonest",
    "LinaresDeportivo", "LinasMontlhery", "Lincoln", "LincolnRedImpsFC", "Linense", "Linfield",
    "Linth", "Lippstadt08", "Liptovsk\\u00fdMikul\\u00e1\\u0161", "Liverpool", "Livingston",
    "Li\\u00e8ge", "Llanera", "Llapi", "LleidaEsportiu", "Llerenense", "Locarno", "Locri1909",
    "LokerenTemse", "LokomotivMoscow", "LokomotivPlovdiv", "LokomotiveLeipzig", "LokomotiviTbilisi",
    "LommelUnited", "Lons", "LoonPlage", "LorcaDeportiva", "LorcaFC", "Lorient", "LorientII",
    "LosAngelesFC", "LosAngelesGalaxy", "LouhansCuiseaux", "LouisvilleCity", "Louletano", "Loures",
    "Lourinhanense", "Loznica", "Lucchese", "Luckenwalde", "Lucko", "Ludogorets", "LuganoII",
    "Lumezzane", "LuneburgerSKHansa", "LupaRoma", "Luparense", "LusitanoFCV",
    "LusitanoGC\\u00c9vora", "Lusitanos", "Lusitano\\u00c9vora1911", "Lusit\\u00e2nia",
    "Lusit\\u00e2niaLourosa", "Luton", "Luzenac", "Lu\\u00e7on", "Lyngby", "Lyon", "LyonDuchere",
    "LyraLierseBerlaar", "Lyseng", "Lystrup", "L\\u00ed\\u0161e\\u0148", "L\\u00fcnerSV", "MCAlger",
    "MEAPNisou", "MLSAllStars", "MSVDuisburg", "MTKBudapest", "MVV", "MaccabiBneiRaina",
    "MaccabiHaifa", "MaccabiNetanya", "MaccabiParisUJA", "MaccabiPetahTikva", "MaccabiTelAviv",
    "Macclesfield", "Maceratese", "Machico", "MachidaZelvia", "Macouria", "Macva", "Mafra",
    "Magenta", "Magpies", "Maidenhead", "MaidstoneUtd", "Majestic", "MakedonijaGjP", "Makkabi",
    "MaksimirZagreb", "Malaga", "Mallorca", "MalmoFF", "Maltepespor", "MamelodiSundowns", "Manacor",
    "ManchaReal", "ManchesterCity", "ManchesterCityU21", "ManchesterUnited",
    "ManchesterUnitedU21", "MandelUnited", "ManisaBBSK", "Manisaspor", "Manises", "Manresa",
    "MansfieldTown", "Mantes78", "Mantova", "MarMenor", "Marbella", "Marchamalo", "Marck",
    "MardinBB", "MariadaFonte", "Maribor", "Mariehamn", "Marienlyst", "Marignane", "MarignaneUS",
    "MarinadiRagusa", "Marine", "Marinhense", "Marino", "MarinodeLuanco", "Maritimo",
    "MaritsaPlovdiv", "Mari\\u00e1nsk\\u00e9L\\u00e1zn\\u011b", "Marko", "Marnaval", "Marsala",
    "Marsaxlokk", "Marseille", "MarseilleEndoume", "Marsonia", "MarssacSRDT", "Martigues",
    "Mar\\u00edtimoII", "Matelica", "Matera", "MatlockTown", "MeauxAcademy", "MechelenU21",
    "MedjimurjeCakovec", "Meerbusch", "Meerssen", "Meinerzhagen", "MelbourneVictory", "Melfi",
    "Melilla", "MelillaCD", "MenemenBelediyespor", "Mensajero", "Merelbeke", "Merelinense",
    "Mersin\\u0130dmanyurdu", "Merstham", "Mestre", "MetalKharkiv", "MetalacGM", "Metalist",
    "Metalist1925Kharkiv", "Metaloglobus", "MetalulBuz\\u0103u", "Metz", "MetzII", "Meux",
    "Mezokovesdzsory", "Mezzolara", "Middelfart", "Middlesbrough", "Midtjylland", "MiedzLegnica",
    "Millonarios", "Millwall", "MilsamiOrhei", "Minai", "MinaurBaiaMare", "MinnesotaUnitedFC",
    "Mirandela", "Mirandes", "Mis\\u00e9rieuxTr\\u00e9voux", "MladaBoleslav", "MladostLucani",
    "MladostNoviSad", "MladostPetrinja", "Mladost\\u017ddralovi", "Modena", "Modri\\u010da",
    "MohunBagan", "Moitense", "Molde", "Mollerussa", "Monaco", "MonacoII", "Moncarapachense",
    "Mondercange", "Mondinense", "Mondorf", "Monheim", "Monnaie", "Monopoli", "Mons",
    "Montagnarde", "Montalegre", "Montceau", "MontdeMarsan", "MonteCarlo", "Monterrey",
    "MontevarchiCalcio", "Monthey", "Montijo", "Montpellier", "MontpellierII", "Monza",
    "Mon\\u00e7\\u00e3o", "Morbio", "Morecambe", "Moreirense", "Mornar", "Mors\\u00f8", "Morton",
    "Mort\\u00e1gua", "MorudVeflinge", "Moslavina", "Mosteirense", "Motherwell", "MotorLublin",
    "Mouilleron", "Moulins", "MoulinsYzeureFoot03", "Moura", "Mtsapere", "Mulhouse", "Mura",
    "Mutilvera", "Mutschelbach", "Mu\\u011flaspor", "Mu\\u015fMenderesspor", "M\\u00e9rida",
    "M\\u00e9ridaAD", "M\\u00f3stoles", "M\\u00fbrsErign\\u00e9", "NACBreda", "NECNijmegen",
    "NKDomzale", "NKLokomotivaZagreb", "NKOsijek", "NKOsijekII", "NKSlavenBelupo", "NKVarazdin",
    "NKZadar", "NKZagreb", "Nacional", "Naestved", "Nafta", "NagoyaGrampus", "NagyecsedRSE",
    "NancyII", "Nanterre", "Nantes", "NantesII", "Napoli", "Napredak", "Naval1\\u00badeMaio",
    "Navalcarnero", "Navbahor", "NazilliBelediyespor", "NeaSalamis", "Necaxa", "Nedeli\\u0161\\u0107e",
    "Nedeljanec", "Neftchi", "NeftchiBaku", "Nehaj", "Neman", "NeretvanacOpuzen",
    "Neubrandenburg04", "NeuchatelXamaxFC", "Neudrossenfeld", "Neustrelitz",
    "Nev\\u015fehirBelediyespor", "NewMexicoUnited", "NewYorkRedBulls", "Newcastle",
    "NewcastleUnitedU21", "NewportCounty", "NewtownAFC", "Nice", "NiceII", "Nieciecza", "Nijlen",
    "NikiVolos", "Ninove", "NivoSparta", "Ni\\u011fdeAnadolu", "Nogent", "NoisyleGrand", "Nola1925",
    "Noordwijk", "NorthCarolina", "NorthLeigh", "Northampton", "Norwich", "NorwichCityU21",
    "NottinghamForest", "NottsCounty", "Nouaille", "Novelda", "NoviPazar", "Novigrad", "Nublense",
    "NuneatonTown", "NuovaCosenza", "Nyiregyhaza", "NykobingFC", "N\\u00e6sby", "N\\u00eemesII",
    "N\\u00f8rresundby", "N\\u0153uxlesMines", "N\\u0160Drava", "ODDBallklubb", "OElvas",
    "OFCOostzaan", "OFI", "OFIerapetra", "OFKBeograd", "OFKVr\\u0161ac", "OHLeuven", "OJCRosmalen",
    "OKS", "ONSSneek", "OSFives", "OSS20", "Oberachern", "Oberlauterbach", "Oberneuland",
    "OcnaMure\\u0219", "Odder", "Odense", "OdorheiuSecuiesc", "OdraOpole", "OffenburgerFV",
    "Ofspor", "Oissel", "OlSaintMarcellin", "Olancho", "Olbia", "Oldham", "Oleiros", "Oleksandria",
    "OlimpiaElbl\\u0105g", "OlimpiaGrudzi\\u0105dz", "OlimpiaSatuMare", "OlimpiaZambr\\u00f3w",
    "OlimpicZ\\u0103rne\\u015fti", "OlimpijaLjubljana", "OlimpikDonetsk", "OliveiraHospital",
    "Oliveirense", "Olot", "OlsaBrakel", "OlympiaWijgmaal", "OlympiadaLympion", "Olympiakos",
    "OlympiakosPiraeus", "OlympiakosVolos", "OlympicCharleroi", "OlympiqueAl\\u00e8s",
    "OlympiqueLyonnaisII", "OlympiqueMarcquois", "OlympiqueMarseilleII", "OlympiquePavillais",
    "OlympiqueStQuentin", "OlympiquedAl\\u00e8s", "OlympiquedeValence", "Ol\\u00edmpicX\\u00e0tiva",
    "Ol\\u00edmpicodoMontijo", "OmladinacGornjaVrba", "Omonia29isMaiou", "OmoniaAradippou",
    "OmoniaNicosia", "Onti\\u00f1ena", "Opatija", "Opava", "Oper\\u00e1rio", "OptikRathenow",
    "OrangeCountySC", "OranjeWit", "Ordabasy", "Orduspor", "Orhangazispor", "OrientalDragon",
    "OrientalLisboa", "Orihuela", "Orijent1919", "OriolikOriovac", "OrlandoCitySC",
    "OrlandoPirates", "OrvaultSF", "OsLimianos", "OsMarialvas", "Osasuna", "OsasunaII",
    "Osmaniyespor", "OstersundsFK", "Othellos", "Otterup", "Ouagadougou", "Oudenaarde", "OurenseCF",
    "Oviedo", "OxfordCity", "OxfordUnited", "O\\u0163elul", "PAEEK", "PAOK", "PASGiannina", "PAU",
    "PECZwolle", "PROPiacenza", "PROVercelli", "PSGII", "PSVEindhoven", "Pachuca", "PacosFerreira",
    "Pafos", "Paganese", "Paide", "Paks", "Palermo", "Palmeiras", "Palmese", "Paloma",
    "PanachaikiFC", "Panader\\u00edaPulido", "Panathinaikos", "PanduriiTGJIU", "Panegialios",
    "Panelefsiniakos", "Panev\\u0117\\u017eys", "Panthrakikos", "Paphos", "Papuk", "Paradiso",
    "Pardubice", "Paredes", "ParisFC", "ParisSaintGermain", "ParlaEscuela", "Parma", "Partick",
    "Partizani", "Paterna", "PatroEisden", "PaulhanP\\u00e9zenas", "Payasspor", "PaysNeslois",
    "PaysdeCassel", "Pazarspor", "PedrasRubras", "PedrasSalgadas", "Pedr\\u00f3g\\u00e3oS\\u00e3oPedro",
    "PelicansSC", "PelikaanS", "Pelister", "Penafiel", "Penarol", "Peniche", "Penybont", "Peralada",
    "Pergolettese", "PerthGlory", "Peterborough", "Petrocub", "PetrodeLuanda", "PetrolulPloiesti",
    "Pet\\u0159\\u00ednPlze\\u0148", "Pevidem", "Peyia", "Pe\\u00f1aAzagresa", "Pe\\u00f1aDeportiva",
    "Pe\\u00f1aSport", "Pfeddersheim", "Ph\\u00f6nixL\\u00fcbeck", "Pianese", "PiastGliwice",
    "Pierikos", "PilicaBia\\u0142obrzegi", "Pineto", "Pinhalnovense", "PinzgauSaalfelden",
    "PirinBlagoevgrad", "Pirmasens", "Pisa", "Pistoiese", "Pitoma\\u010da", "PlessisRobinson92",
    "Plopeni", "Plouzan\\u00e9", "Plymouth", "PlymouthParkway", "Plzen", "Pl\\u00e9dran", "Poblense",
    "Podbeskidzie", "Podbrezov\\u00e1", "PodhaleNowyTarg", "Poggibonsi", "PogonSzczecin",
    "Pogo\\u0144GrodMazowiecki", "Pogo\\u0144Siedlce", "Pogo\\u0144SzczecinII", "Poissy", "Poitiers",
    "Polessya", "Polet", "Police", "PolitehnicaIasi", "PoloniaBytom", "PoloniaWarszawa",
    "Polonia\\u015aroda", "Pombal", "Ponikve", "Pontarlier", "PontcharraStLoup", "Pontedera",
    "Pontevedra", "PontivyGSI", "Pope\\u0219tiLeordeni", "PortVale", "Portalegrense",
    "Portimonense", "Portosantense", "Portsmouth", "Portugalete", "Porz", "Posu\\u0161je", "Potenza",
    "Pozoblanco", "Praiense", "Prat", "Prato", "Preston", "PreussenMunster", "PrimoracBiograd",
    "Primorje", "Prishtina", "Prixl\\u00e8sM\\u00e9zi\\u00e8res", "ProPatria", "ProPiacenza",
    "ProSesto", "ProgresNiederkorn", "Progresso", "ProgresulPecica", "ProgresulSpartac",
    "ProleterNoviSAD", "Proodeftiki", "Provin", "PuenteGenil", "PuskasAcademy",
    "PuszczaNiepo\\u0142omice", "PyunikYerevan", "P\\u00e1pa", "P\\u00e9csiMFC", "P\\u00earoPinheiro",
    "P\\u00edsek", "P\\u0159epe\\u0159e", "P\\u0159\\u00edbram", "QPR", "Qabala", "Qarabag",
    "QatarSC", "QueenoftheSouth", "QueensPark", "Quevilly", "Quick20", "QuickBoys",
    "Quimperl\\u00e9", "Quintanar", "QuintanardelRey", "RAALLaLouvi\\u00e8re", "RAD", "RBLeipzig",
    "RCBoboDioulasso", "RCCalais", "RCCatarroja", "RCK", "RCSaintJoseph", "REDStarFC93",
    "RFCSeraing", "RFCWetteren", "RFKNoviSad", "RKAVVolendam", "RKSVNuenen", "RNKSplit",
    "RSCAnderlechtII", "RWDM", "RaboPeixe", "RacingClermontois", "RacingColombes92",
    "RacingFCUnionLuxembourg", "RacingFerrol", "RacingMechelen", "RacingMurcia", "RacingRioja",
    "RacingRoma", "RacingSantander", "RadjevacKrupanj", "Radnicki1923", "RadnickiNIS",
    "RadnickiPirot", "RadnikSurdulica", "Radni\\u010dkiNoviBeograd", "Radni\\u010dkiSrMitrovica",
    "RadomiakRadom", "Radomlje", "Radomsko", "RaduniaSt\\u0119\\u017cyca", "Rahimo", "RaithRovers",
    "RajaCasablanca", "Rak\\u00f3wCz\\u0119stochowa", "RamlingenEhlershausen", "RandersFC", "Rangers",
    "Rann\\u00e9eLaGuerche", "Rantzau", "RaonlEtape", "Rapid", "RapidSymphorinois", "RapidVienna",
    "Rapperswil", "RayoVallecano", "RayoZuliano", "ReadingU23", "Real", "RealAvil\\u00e9s",
    "RealBetis", "RealBetisII", "RealEspporClub", "RealForteQuerceta", "RealJa\\u00e9n",
    "RealMadrid", "RealMadridII", "RealMurcia", "RealOviedoII", "RealSaltLake", "RealSociedad",
    "RealSociedadII", "RealUni\\u00f3n", "RealValladolidII", "RealVicenza", "RealZaragozaII",
    "Real\\u00c1vila", "Rebecq", "Rebordosa", "Recanatese", "RecoltaGheorgheDoja",
    "RecreativoCanaviais", "RecreativoHuelva", "RedBullSalzburg", "Reggiana", "Reims", "ReimsII",
    "ReimsSainteAnne", "RekordBielskoBia\\u0142a", "Renate", "Rende", "Rennes", "Renova",
    "Resende", "ResoviaRzesz\\u00f3w", "Restrup", "Reus", "Reusrath", "Reutlingen", "Revel",
    "Rezzato", "RhoneVallee", "Ribadumia", "Ribeir\\u00e3o", "Ried", "Rieti", "Riga",
    "RijnsburgseBoys", "Rijnvogels", "Rinc\\u00f3n", "RioAve", "RioAveII", "RipensiaTimisoara",
    "RiverPlate", "Rizespor", "RoburSiena", "Roccella", "Rochdale", "RocheSaintGenest",
    "Rochefort", "Roda", "Rodez", "Roga\\u0161ka", "RomaU19", "RomaW", "Romorantin", "Ronse",
    "Rops", "Rosenborg", "Roskilde", "RossCounty", "RotWeissAhlen", "RotWei\\u00dfDarmstadt",
    "RotWei\\u00dfEssen", "RotenburgerSV", "RotweissErfurt", "RotweissEssen", "RotweissOberhausen",
    "Rovereto", "Rovers", "Royal", "RoyalExcelMouscron", "RoyeNoyon", "Ro\\u00dfbachVerscheid",
    "Rubin", "Rub\\u00ed", "RuchChorz\\u00f3w", "RuchWysokieMazowieckie", "Rudar",
    "RudarMurskoSredi\\u0161\\u0107e", "RudarVelenje", "Rudersdal", "Rudes", "RuffiacMalestroit",
    "RuhLviv", "RukhVynnyky", "RumillyValli\\u00e8res", "RupelBoom", "Ru\\u017eomberok",
    "R\\u00e1ckeveVAFC", "R\\u00e2mnicuS\\u0103rat", "R\\u00f6this", "R\\u00f8dovre", "R\\u012bgasFS",
    "SAKKlagenfurt", "SAM\\u00e9rignac", "SCBastia", "SCBraga", "SCBragaB", "SCCovilha", "SCEgedal",
    "SCFreiburg", "SCGenemuiden", "SCMGloriaBuz\\u0103u", "SCMZal\\u0103u", "SCPaderborn07",
    "SCRAltach", "SCWienerNeustadt", "SDCPutten", "SDLogro\\u00f1\\u00e9s", "SFHamborn",
    "SGSonnenhofGrossaspach", "SGTertreHautrage", "SGVFreiberg", "SJCNoordwijk", "SKNSTPolten",
    "SKRACz\\u0119stochowa", "SKVorwartsSteyr", "SONABEL", "SPAL", "SSMonopoli", "SSReyes",
    "SSVJahnRegensburg", "SSVJeddeloh", "SSVULM1846", "SSVUlm1846", "SSVgVelbert", "STJohnstone",
    "STMirren", "SVBabelsberg03", "SVDarmstadt98", "SVDrochtersenAssel", "SVDrochtersenassel",
    "SVElversberg", "SVGReichenau", "SVInnsbruck", "SVKapfenberg", "SVLafnitz", "SVMattersburg",
    "SVMeppen", "SVRodinghausen", "SVSandhausen", "SVVScheveningen", "SVWehen", "SVZW", "SabahFA",
    "Sabl\\u00e9", "Saburtalo", "Sacavenense", "Saguntino", "SaintAmand", "SaintBerthevin",
    "SaintBrieuc", "SaintColombanLocmin\\u00e9", "SaintCyrCollonges", "SaintDenis", "SaintEtienne",
    "SaintEtienneSeltz", "SaintFlour", "SaintJeanBeaulieu", "SaintLouisNeuweg", "SaintMalo",
    "SaintMaximin", "SaintMeziery", "SaintOmer", "SaintPauloise", "SaintPriest", "SaintRenan",
    "SainteGenevi\\u00e8ve", "SainteMarienne", "Saint\\u00c9tienneII", "Sakaryaspor",
    "SalamancaUDS", "Salernitana", "SalfordCity", "Salgueiros", "SalinieresAiguesMortes",
    "SalisburyCity", "Salitas", "Salmrohr", "Salsomaggiore", "SaltashUnited", "Sambenedettese",
    "Sammaurese", "Samobor", "Sampdoria", "SampdoriaU19", "Samsunspor", "SanAgust\\u00edn",
    "SanAntonio", "SanDiegoLoyal", "SanDonatoTavarnelle", "SanFernandoCD", "SanJoseEarthquakes",
    "SanJuan", "SanLorenzo", "SanRoqueLepe", "SanTirso", "SanTommaso", "SancaktepeBelediyespor",
    "SandecjaNowyS\\u0105cz", "SanfrecceHiroshima", "SangiulianoCity", "Sanjoanense", "Sanluque\\u00f1o",
    "SannoisStGratien", "Sansepolcro", "SantAndreu", "SantaAmalia", "SantaClara", "SantaIria",
    "SantaMariaFC", "SantaMartaPenagui\\u00e3o", "Santarcangelo", "Santos", "SantosLaguna",
    "SarmientoJunin", "Sarpsborg08FF", "SarreUnion", "Sarrebourg", "Sarreguemines", "Sarto",
    "Sar\\u0131yer", "SassoMarconiZola", "Sassuolo", "Saumur", "Savignanese", "Savoia",
    "ScHauenstein", "Schalke04II", "Schiltigheim", "SchottJena", "SchottMainz", "SchwabenAugsburg",
    "SchwarzWei\\u00dfBregenz", "SchwarzWei\\u00dfRehden", "Schwaz", "Schwechat",
    "Schw\\u00e4bischHall", "Sch\\u00f6nberg", "Seekirchen", "Selongey", "Semendrija1924",
    "Sementina", "Sendim", "Senlis", "SepsiOSKSfantuGheorghe", "SeraingUnited", "Seravezza",
    "Seregno", "Sere\\u010f", "SerhatArdahanspor", "SerikBelediyespor", "SerquignyNassandres",
    "Sertanense", "ServetteFC", "Sesimbra", "SestaoRiver", "SestriLevante", "Sesvete", "Sevilla",
    "SevillaAtletico", "SeyssinetPariset", "Sf\\u00eentulGheorghe", "ShakhtarDonetsk",
    "ShakhtarDonetskU21", "ShakhterKaragandy", "ShakhterSoligorsk", "ShamrockRovers", "SharjahFC",
    "SheffieldUtd", "SheffieldWednesday", "SheriffTiraspol", "ShimizuSpulse", "Shirak",
    "Shkendija", "Shkupi1927", "Shrewsbury", "SiarkaTarnobrzeg", "Sibenik", "SiculaLeonzio",
    "SiegburgerSV", "Siegendorf", "SigmaOlomouc", "Siirtspor", "Siirt\\u0130l\\u00d6zel\\u0130daresi",
    "Silivrispor", "Silkeborg", "SilkeborgKFUM", "Silla", "Sillam\\u00e4eKalev", "Silves", "Silvolde",
    "SindjelicBeograd", "SintElooisWinkel", "Sintra", "Sintrense", "Siofok", "Siracusa", "Sirius",
    "SirokiBrijeg", "Sivas4Eyl\\u00fcl", "SivasBelediyespor", "Sivasspor", "SkelmersdaleUnited",
    "SkenderbeuKorce", "Skive", "Skovbakken", "Sk\\u00f6vdeAIK", "SlagelseBI", "SlaskWroclaw",
    "Slatina", "SlaviaPraha", "SlaviaSofia", "SlavojVy\\u0161ehrad", "SlavonijaPo\\u017eega",
    "SliemaWanderers", "SligoRovers", "Sloboda", "SlobodaTuzla", "SlobodaUzice", "SlogaMravince",
    "SlovanBratislava", "SlovanLiberec", "SlovanVelvary", "Slov\\u00e1cko", "SochauxII",
    "Socu\\u00e9llamos", "Sogndal", "Sok\\u00f3\\u0142Kleczew", "Sok\\u00f3\\u0142Ostr\\u00f3da",
    "Solares", "SolihullMoors", "Solin", "Solr\\u00f8d", "Somaspor", "Sonderjyske", "Sorrento",
    "Sourense", "SouthShields", "Southampton", "SouthamptonU21", "Southport", "SpGijon",
    "SpVggGreutherFurth", "SpVggGreutherF\\u00fcrth", "SpVggUnterhaching", "Spakenburg",
    "SpartaEnschede", "SpartaNijkerk", "SpartaPetegem", "SpartaPraha", "SpartaRotterdam",
    "SpartaRotterdamII", "SpartakMoscow", "SpartakTrnava", "SpartakosKitiou", "SpartaksJurmala",
    "SpelleVenhaus", "SpennymoorTown", "Spezia", "Spi\\u0161sk\\u00e1Nov\\u00e1Ves",
    "SportfreundeDorfmerkingen", "SportfreundeLotte", "SportfreundeSiegen", "SportingCP",
    "SportingCPB", "SportingGijon", "SportingGij\\u00f3nII", "SportingLie\\u015fti",
    "SportingdeLourel", "Sportlust46", "SportulChiscani", "SportulSnagov", "SpouwenMopertingen",
    "Sprimont", "StJosephSFc", "StLouisCityII", "StMaloUS", "StMaurLusitanos", "StPatricksAthl",
    "StPhilbertGdLieu", "StPryv\\u00e9StHilaire", "StSerninduBois", "StTruiden", "StadeBordelais",
    "StadeBrestois29", "StadeBriochin", "StadeB\\u00e9thunois", "StadeLausanneOuchy",
    "StadeMayennais", "StadeMontois", "StadeNyonnais", "StadePlabennec", "StadePontivy",
    "StadeYgossais", "StadlPaura", "Stadlau", "StainesTown", "StalBrzeg", "StalMielec",
    "StalRzesz\\u00f3w", "StalStalowaWola", "StandardLiege", "StandardLi\\u00e8geII", "Staphorst",
    "StargardSzczeci\\u0144ski", "StartJe\\u0142owa", "SteDoCo", "StellaMarisDouarnenez",
    "Stevenage", "StiintaMiroslava", "Still", "StilonGorz\\u00f3w", "StirlingAlbion", "Stjarnan",
    "StockportCounty", "StokeCity", "StokeCityU21", "StomilOlsztyn", "Stourbridge", "Straelen",
    "Stranraer", "Strasbourg", "StrasbourgKoenigshoffen", "Stromsgodset", "Struga", "Stubai",
    "SturmGraz", "StuttgartII", "StuttgarterKickers", "Sudtirol", "SuduvaMarijampole",
    "SultanbeyliBelediyespor", "Sumqay\\u0131t", "Sunderland", "Surkhon", "Sutjeska", "SuttonUtd",
    "SvMorlautern", "Svendborg", "Swansea", "SwanseaCityU23", "SwindonSupermarine", "SwindonTown",
    "Swit", "Sydalliancen", "Sydvest", "Szeged2011", "Szeksz\\u00e1rd", "Szentl\\u0151rincSE",
    "SzolnokiMAVFC", "SzombathelyiHaladas", "S\\u00e1rv\\u00e1ri", "S\\u00e2nmartin",
    "S\\u00e3oJo\\u00e3oVer", "S\\u00e3oRoque", "S\\u00e3oRoqueA\\u00e7ores", "S\\u00e8te",
    "S\\u00e9nartMoissy", "S\\u00e9ny\\u0151Carnifex", "S\\u00e9n\\u00e9", "S\\u00f8nderborg",
    "S\\u0103n\\u0103tateaCluj", "S\\u0259bail", "TARennes", "TEC", "TSCBackaTopola", "TSGalaxy",
    "TSV1860Munich", "TSV1860M\\u00fcnchen", "TSVHartberg", "TSVKirchberg", "TSVSteinbach",
    "TaborSe\\u017eana", "Tadamon", "TadamonSour", "TaffsWell", "Talavera", "TalleresCordoba",
    "Tamaraceite", "Tamworth", "Taranto", "Tarazona", "Tarbes", "Tardienta",
    "Tarsus\\u0130dmanYurdu", "TatvanGen\\u00e7lerbirli\\u011fi", "Tavistock",
    "Tav\\u015fanl\\u0131Linyitspor", "TeamWienerLinien", "Tefana", "Tehni\\u010dar1974",
    "Tekirda\\u011fspor", "TekstilacOd\\u017eaci", "Teleoptik", "Telstar", "Temnic1924",
    "TempoOverijse", "Tenerife", "Teningen", "Tepecikspor", "Teplice", "TerLeede", "Teramo",
    "Terrassa", "Teruel", "TeutaDurr\\u00ebs", "TeutoniaOttensen", "Thaon", "TheNewSaints",
    "Theix", "ThesSport", "ThimphuCity", "ThionvilleLusitanos", "ThistedFC", "Thonon\\u00c9vian",
    "Thrasyvoulos", "Tienen", "Tirana", "Tirsense", "TiszafurediVSE", "TiszakecskeFC",
    "TivertonTown", "Tj\\u00e6reborg", "Tlaxcala", "Tocha", "ToekomstMenen", "TokatBldPlevnespor",
    "Tokatspor", "TokyoVerdy", "Toledo", "Tolosa", "Tomares", "Tondela", "Torcatense", "Torino",
    "TorpedoKutaisi", "TorpedoZhodino", "Torquay", "Torreense", "Torres", "Tottenham",
    "TottenhamHotspurU21", "Toulon", "ToulonLeLas", "Toulouse", "ToulouseM\\u00e9tropole",
    "TrabzonKanuni", "Trabzonspor", "Trancoso", "Tranmere", "Trapani1905", "Trastevere", "Trento",
    "Triglav", "Trikala", "Tritium", "Trnje", "Troina", "TrouvilleDeauville", "Tr\\u00e9lissac",
    "Tskhinvali", "TuRU1880D\\u00fcsseldorf", "TuSBWK\\u00f6nigsdorf", "TuSErndtebruck",
    "TuSKoblenz", "TuSRWKoblenz", "Tudelano", "Tunari", "Turgutluspor", "Turris",
    "TurrisOltulTM\\u0103gurele", "Tur\\u00e9gano", "Tuttocuoio", "TuzlaCity", "Tuzlaspor", "Twente",
    "Tychy71", "Tyrnavos", "T\\u00e1borsko", "T\\u00e5rnbyFF", "T\\u00f6k\\u00f6l",
    "T\\u00fcrkg\\u00fcc\\u00fcAtaspor", "T\\u00fcrksporAugsburg", "T\\u0159inec", "UCraiova1948",
    "UDI19", "UDLogro\\u00f1\\u00e9s", "UDParachique", "UDRioMaior", "UDSanFernando", "UDSanPedro",
    "UDdaSerra", "UESantJulia", "UESantaColoma", "UMMSalal", "UMadeira", "UNA", "UNAMPumas",
    "UNFP", "URK", "URSLVis\\u00e9", "USBlavozy", "USFA", "Ubberud", "UcamMurcia", "Udinese",
    "Ujpest", "UlsanHyundaiFC", "Unami", "UniaSkierniewice", "UniaTurza\\u015al\\u0105ska",
    "UnionBerlin", "UnionCosnoise", "UnionF\\u00fcrstenwalde", "UnionNettetal", "UnionSaintJean",
    "UnionStGilloise", "UnionTitusPetange", "UnionistasdeSalamanca", "UnireaAlbaIulia",
    "UnireaConstan\\u021ba", "UnireaDej", "UnireaSlobozia", "UnireaUngheni", "UniversitateaCluj",
    "UniversitateaCraiova", "Uni\\u00e3oAlmeirim", "Uni\\u00e3oIdanhense", "Uni\\u00e3oMontemor",
    "Uni\\u00e3oSantar\\u00e9m", "Uni\\u00e3odeCoimbra", "Uni\\u00e3odeLeiria", "Uni\\u00e3odeTomar",
    "Uni\\u00f3nSurYaiza", "Urawa", "UtaArad", "Uta\\u015fU\\u015fakspor", "Utebo", "Utrecht",
    "Utrillas", "Uz\\u00e8sPontduGard", "VOC", "VPS", "VRI", "VSK\\u00c5rhus", "VVCS", "VVVVenlo",
    "ValadaresGaia", "ValdIze", "ValdinievoleMontecatini", "ValeFormoso", "ValenceFC", "Valencia",
    "ValenciaII", "Valenciano", "ValenciennesII", "Valerenga", "Valladolid", "ValleEg\\u00fc\\u00e9s",
    "VallettaFC", "ValmieraBSS", "ValurReykjavik", "VanBB", "VancouverWhitecaps", "Vand\\u0153uvre",
    "Vanl\\u00f8se", "VardarSkopje", "Varde", "VardeIFElite", "Varea", "VarteksVara\\u017edin",
    "Varzim", "Vasas", "VascodaGama", "VascodaGamaVidigueira", "Vaslui", "Vecs\\u00e9s", "VejgaardB",
    "Vejle", "Velarde", "Velay", "VelezSarsfield", "Vele\\u017e", "VendsysselFF",
    "Vend\\u00e9eFontenay", "Venezia", "Venray", "VenturaCountyFusion", "Verden04",
    "VerdunBelleville", "VeresRivne", "Veria", "Verl", "Verlaine", "Verona", "Versailles",
    "Vesoul", "Vestsj\\u00e6lland", "VfBLubeck", "VfBL\\u00fcbeck", "VfBOldenburg", "VfBStuttgart",
    "VfLBochum", "VfLOldenburg", "VfLOsnabruck", "VfLOsnabr\\u00fcck", "VfLWolfsburg", "VfRAalen",
    "VflBochum", "Vianense", "Vibonese", "Viborg", "Viby", "Vic", "VicenzaVirtus", "VictoriaCF",
    "VictoriaCump\\u0103na", "VictoriaHamburg", "Vierzon", "VierzonFC", "VigorCarpaneto",
    "ViitorulConstanta", "ViitorulDomne\\u015fti", "ViitorulIanca", "Viitorul\\u015eelimb\\u0103r",
    "Viking", "VikingurGota", "VikingurReykjavik", "Viktoria", "ViktoriaAschaffenburg",
    "ViktoriaBerlin", "ViktoriaGriesheim", "ViktoriaJ\\u00fcchenGarz", "Viktoria\\u017di\\u017ekov",
    "VilaFlor", "VilaMe\\u00e3", "VilaPouca", "VilaReal", "Vilafranquense", "VilaineAtlantique",
    "Vilamarxant", "VilardePerdizes", "Vilarinho", "Vilaverdense", "Vildbjerg", "VilladeFortuna",
    "Villafranca", "Villajoyosa", "Villalb\\u00e9s", "Villamuriel", "Villanovense", "Villarreal",
    "VillarrealII", "VillarrealIII", "Villarrobledo", "Villarrubia", "Villefranche", "Villejuif",
    "VillemombleSports", "Villenave", "VilleneuvedAscq", "VillersHoulgateCF", "VilleruptThil",
    "Vilzing", "Vimenor", "Vinhais", "Vinogradar", "Vire", "Virtus", "VirtusFrancavilla",
    "VirtusVerona", "ViryCh\\u00e2tillon", "VisPesaro", "Viseu_AcademicoViseu", "Viseu_Alverca",
    "Viseu_Arouca", "Viseu_BeiraMar", "Viseu_Benfica", "Viseu_BenficaB", "Viseu_Boavista",
    "Viseu_CasaPia", "Viseu_Chaves", "Viseu_Estoril", "Viseu_Estrela", "Viseu_FCPorto",
    "Viseu_FCPortoB", "Viseu_Famalicao", "Viseu_Farense", "Viseu_Feirense",
    "Viseu_Felgueiras1932", "Viseu_GILVicente", "Viseu_Guimaraes", "Viseu_Leixoes", "Viseu_Mafra",
    "Viseu_Maritimo", "Viseu_Moreirense", "Viseu_Nacional", "Viseu_Naval1\\u00badeMaio",
    "Viseu_Oliveirense", "Viseu_PacosFerreira", "Viseu_Penafiel", "Viseu_Portimonense",
    "Viseu_RioAve", "Viseu_SCBraga", "Viseu_SantaClara", "Viseu_SportingCP", "Viseu_Tondela",
    "Viseu_Torreense", "Viseu_UMadeira", "Viseu_Uni\\u00e3odeLeiria", "Viseu_VitoriaSetubal",
    "Viseu_Vizela", "VisselKobe", "Viterbese", "Vitesse", "VitoriaDaConquista", "VitoriaSetubal",
    "Vitr\\u00e9", "Vit\\u00f3riadeSernache", "Viveiro", "Vizela", "VllazniaShkod\\u00ebr",
    "Voitsberg", "Vojvodina", "VolendamII", "VoluntariII", "Volvic", "VoorwaartsZwevezele",
    "Vordingborg", "VorsklaPoltava", "Vrap\\u010deZagreb", "Vukovar", "VuteksSloga", "VvDeMeern",
    "Vvsb", "Vy\\u0161kov", "V\\u00e9lez", "V\\u00e9nissieux", "WSGWattens", "Waalwijk",
    "Waaslandbeveren", "WackerBurghausen", "WackerInnsbruck", "WackerInnsbruckAm",
    "WaldhofMannheim", "Waldkirch", "Walsall", "WartaPozna\\u0144", "Watford", "Wealdstone",
    "WegbergBeeck", "WeicheFlensburg", "Weiz", "WellingUnited", "Wels", "WerderBremen",
    "WerderBremenII", "WeselLackhausen", "WestBrom", "WestBromwichAlbionU21",
    "WestBromwichAlbionU23", "WestHam", "WestHamUnitedU21", "WestHamUnitedU23",
    "WesternSydneyWanderers", "WestfaliaRhynern", "Westhoek", "Westlandia", "Weymouth",
    "Widzew\\u0141\\u00f3d\\u017a", "WieczystaKrak\\u00f3w", "Wiedenbr\\u00fcck", "WienerViktoria",
    "WigrySuwa\\u0142ki", "Wilhelmshaven", "WillemII", "Wimpassing", "Winterswijk", "WislaKrakow",
    "WislaPlock", "Wis\\u0142aPu\\u0142awy", "Wittemheim", "Wittenhorst",
    "Wi\\u015blanieJa\\u015bkowice", "Woking", "Wolfenbuttel", "WolfsbergerAC",
    "WolvertemMerchtem", "Wolves", "WolvesU21", "WorcesterCity", "WormatiaWorms", "Wrexham",
    "WuppertalerSV", "W\\u00f3lczankaWPe\\u0142ki\\u0144ska", "W\\u00f6rgl", "W\\u00fcrzburgerKickers",
    "W\\u0142oc\\u0142aviaW\\u0142oc\\u0142awek", "XanthiFC", "XerezDeportivo", "Xylotympou", "Yeclano",
    "YeniAmasyaspor", "YeniOrduspor", "Yeni\\u00c7orumspor", "YeovilTown", "Ye\\u015filBursa",
    "YimpasYozgatspor", "YokohamaFMarinos", "Yomraspor", "York", "Yozgatspor1959", "Ytrac", "Yutz",
    "YverdonSport", "ZFCMeuselwitz", "ZaglebieLubin", "ZaglebieSosnowiec", "Zagora", "Zagorec",
    "Zag\\u0142\\u0119bieLubinII", "ZalaegerszegiTE", "Zamora", "ZaraBelediyespor", "Zaragoza",
    "ZariaBalti", "Zarkovo", "ZawiszaBydgoszcz", "ZbrojovkaBrno", "Zelina", "ZeljeznicarSarajevo",
    "Zempl\\u00ednMichalovce", "Zemun", "Zenit", "ZenitSaintPetersburg", "ZepperenBrustem",
    "Zestafoni", "Zimbru", "Zira", "Zlat\\u00e9Moravce", "Zlin", "ZniczPruszk\\u00f3w",
    "ZonguldakK\\u00f6m\\u00fcrspor", "ZoryaLuhansk", "Zrinjski", "ZrinskiJurjevac", "ZulteWaregem",
    "Zwaluwen", "ZwarteLeeuw", "Zweibr\\u00fccken", "como", "trelleborgsFF", "tstFodbold",
    "\\u00c1gueda", "\\u00c1guiasdoMoradal", "\\u00c1guilas", "\\u00c7anakkaleDardanel",
    "\\u00c7ankayaFK", "\\u00c7ank\\u0131r\\u0131spor", "\\u00c7ar\\u015fambaspor", "\\u00c7atalcaspor",
    "\\u00c7engelk\\u00f6yspor", "\\u00c7ineMadranspor", "\\u00c7orumspor", "\\u00c7ubukspor",
    "\\u00c9pernay", "\\u00c9vreux", "\\u00c9vreux27", "\\u00d8sterbro", "\\u00dast\\u00ednadLabem",
    "\\u00dcmraniyespor", "\\u00dcnyespor", "\\u010cardaMartjanci", "\\u010celik", "\\u010cepin",
    "\\u010cesk\\u00e9Bud\\u011bjovice", "\\u0130neg\\u00f6lKafkasGen\\u00e7lik", "\\u0130neg\\u00f6lspor",
    "\\u0130nterBak\\u0131", "\\u0130stanbulG\\u00fcng\\u00f6renspor", "\\u0130stanbulspor",
    "\\u0130\\u00e7el\\u0130dmanyurduSpor", "\\u0141KS\\u0141\\u00f3d\\u017a", "\\u0141ag\\u00f3w",
    "\\u015al\\u0105skWroc\\u0142awII", "\\u015al\\u0119zaWroc\\u0142aw", "\\u015awitSkolwin",
    "\\u015eanl\\u0131urfaspor", "\\u017delezni\\u010darPan\\u010devo", "\\u017dilina",
    "\\u0218oimiiLipova"
  ]
}
"""

INITIAL_TEAM_ID_MAPPING: TeamIdMappingSchema = {
    # Entries from your original script, updated/confirmed by snippet
    "Hoffenheim": {"country": "Germany", "statarea_id": "167", "mongodb_id": "167", "alt": ["1899Hoffenheim"]},
    "Milan": {"country": "Italy", "statarea_id": "489", "mongodb_id": "489", "alt": ["AC Milan", "AC_Milan"]},
    "Den Haag": {"country": "Netherlands", "statarea_id": "198", "mongodb_id": "198", "alt": ["ADODenHaag", "ADO Den Haag"]}, # Snippet used "Den Haag" as key
    "AFC Hermannstadt": {"country": "Romania", "statarea_id": "2579", "mongodb_id": "2579", "alt": ["AFC_Hermannstadt"]}, # My canonical "AFC Hermannstadt", snippet had "AFC_Hermannstadt"
    "AZ Alkmaar": {"country": "Netherlands", "statarea_id": "201", "mongodb_id": "201", "alt": ["AZAlkmaar"]},
    "Aalborg": {"country": "Denmark", "statarea_id": "402", "mongodb_id": "402", "alt": ["AalborgBK"]},
    "Academico Viseu": {"country": "Portugal", "statarea_id": "238", "mongodb_id": "238", "alt": ["Academico_Viseu"]}, # My canonical "Academico Viseu", snippet had "Academico_Viseu"
    "Ajaccio GFCO": {"country": "France", "statarea_id": "98", "mongodb_id": "98", "alt": ["GFC Ajaccio"]},
    "Ajax": {"country": "Netherlands", "statarea_id": "194", "mongodb_id": "194", "alt": []},
    "Alanyaspor": {"country": "Turkey", "statarea_id": "996", "mongodb_id": "996", "alt": []},
    "Albacete": {"country": "Spain", "statarea_id": "722", "mongodb_id": "722", "alt": []},
    "Almere City": {"country": "Netherlands", "statarea_id": "419", "mongodb_id": "419", "alt": ["Almere_City_FC", "Almere City FC"]}, # Snippet used "Almere City" as key
    "Almeria": {"country": "Spain", "statarea_id": "723", "mongodb_id": "723", "alt": []},
    "Alverca": {"country": "Portugal", "statarea_id": "4724", "mongodb_id": "4724", "alt": []},
    "Amiens": {"country": "France", "statarea_id": "87", "mongodb_id": "87", "alt": ["AmiensSC"]},
    "Anderlecht": {"country": "Belgium", "statarea_id": "554", "mongodb_id": "554", "alt": []},
    "Angers": {"country": "France", "statarea_id": "77", "mongodb_id": "77", "alt": ["Angers SCO"]},
    "Annecy": {"country": "France", "statarea_id": "3012", "mongodb_id": "3012", "alt": ["Annecy FC"]},
    "Antalyaspor": {"country": "Turkey", "statarea_id": "1005", "mongodb_id": "1005", "alt": []},
    "Antwerp": {"country": "Belgium", "statarea_id": "740", "mongodb_id": "740", "alt": ["Royal Antwerp"]},
    "Apoel": {"country": "Cyprus", "statarea_id": "2247", "mongodb_id": "2247", "alt": ["APOEL Nicosia"]},
    "Arouca": {"country": "Portugal", "statarea_id": "240", "mongodb_id": "240", "alt": []},
    "Arsenal": {"country": "England", "statarea_id": "42", "mongodb_id": "42", "alt": []},
    "Aston Villa": {"country": "England", "statarea_id": "66", "mongodb_id": "66", "alt": ["Aston_Villa"]}, # Snippet key "Aston Villa"
    "Atalanta": {"country": "Italy", "statarea_id": "499", "mongodb_id": "499", "alt": []},
    "Ath Bilbao": {"country": "Spain", "statarea_id": "531", "mongodb_id": "531", "alt": ["Athletic Club", "Athletic Bilbao", "Athletic_Club"]}, # Snippet key "Ath Bilbao"
    "Ath Madrid": {"country": "Spain", "statarea_id": "530", "mongodb_id": "530", "alt": ["Atletico Madrid", "Atletico_Madrid"]}, # Snippet key "Ath Madrid"
    "Auxerre": {"country": "France", "statarea_id": "108", "mongodb_id": "108", "alt": ["AJ Auxerre"]},
    "BB Bodrumspor": {"country": "Turkey", "statarea_id": "3583", "mongodb_id": "3583", "alt": ["Bodrumspor", "BB_Bodrumspor"]}, # Snippet key "BB_Bodrumspor"
    "Barcelona": {"country": "Spain", "statarea_id": "529", "mongodb_id": "529", "alt": ["FC Barcelona"]},
    "Bari": {"country": "Italy", "statarea_id": "508", "mongodb_id": "508", "alt": ["SSC Bari"]},
    "Bastia": {"country": "France", "statarea_id": "1305", "mongodb_id": "1305", "alt": ["SC Bastia"]},
    "Leverkusen": {"country": "Germany", "statarea_id": "168", "mongodb_id": "168", "alt": ["Bayer Leverkusen", "Bayer_Leverkusen"]}, # Snippet key "Leverkusen"
    "Beerschot VA": {"country": "Belgium", "statarea_id": "263", "mongodb_id": "263", "alt": ["Beerschot Wilrijk", "Beerschot", "Beerschot_Wilrijk"]}, # Snippet key "Beerschot VA"
    "Benfica": {"country": "Portugal", "statarea_id": "211", "mongodb_id": "211", "alt": ["SL Benfica"]},
    "Benfica B": {"country": "Portugal", "statarea_id": "229", "mongodb_id": "229", "alt": ["Benfica_B"]}, # Snippet key "Benfica_B"
    "Besiktas": {"country": "Turkey", "statarea_id": "549", "mongodb_id": "549", "alt": []},
    "Betis": {"country": "Spain", "statarea_id": "543", "mongodb_id": "543", "alt": ["Real Betis"]},
    "Blackburn": {"country": "England", "statarea_id": "67", "mongodb_id": "67", "alt": ["Blackburn Rovers"]},
    "Boavista": {"country": "Portugal", "statarea_id": "222", "mongodb_id": "222", "alt": []},
    "Bologna": {"country": "Italy", "statarea_id": "500", "mongodb_id": "500", "alt": []},
    "Dortmund": {"country": "Germany", "statarea_id": "165", "mongodb_id": "165", "alt": ["Borussia Dortmund", "Borussia_Dortmund"]}, # Snippet key "Dortmund"
    "M'gladbach": {"country": "Germany", "statarea_id": "163", "mongodb_id": "163", "alt": ["Borussia Monchengladbach", "Monchengladbach", "Borussia_Monchengladbach"]}, # Snippet key "M'gladbach"
    "Bournemouth": {"country": "England", "statarea_id": "35", "mongodb_id": "35", "alt": ["AFC Bournemouth"]},
    "Brentford": {"country": "England", "statarea_id": "55", "mongodb_id": "55", "alt": []},
    "Brescia": {"country": "Italy", "statarea_id": "518", "mongodb_id": "518", "alt": []},
    "Brighton": {"country": "England", "statarea_id": "51", "mongodb_id": "51", "alt": ["Brighton & Hove Albion"]},
    "Bristol City": {"country": "England", "statarea_id": "56", "mongodb_id": "56", "alt": ["Bristol_City"]}, # Snippet key "Bristol City"
    "Brondby": {"country": "Denmark", "statarea_id": "407", "mongodb_id": "407", "alt": ["Brondby IF"]},
    "Burgos": {"country": "Spain", "statarea_id": "9580", "mongodb_id": "9580", "alt": ["Burgos CF"]},
    "Burnley": {"country": "England", "statarea_id": "44", "mongodb_id": "44", "alt": []},
    "Eldense": {"country": "Spain", "statarea_id": "9692", "mongodb_id": "9692", "alt": ["CD Eldense"]}, # Snippet key "Eldense" (from CD Eldense)
    "Cadiz": {"country": "Spain", "statarea_id": "724", "mongodb_id": "724", "alt": ["Cadiz CF"]},
    "Caen": {"country": "France", "statarea_id": "88", "mongodb_id": "88", "alt": ["SM Caen"]},
    "Cagliari": {"country": "Italy", "statarea_id": "490", "mongodb_id": "490", "alt": []},
    "Cambuur": {"country": "Netherlands", "statarea_id": "420", "mongodb_id": "420", "alt": ["SC Cambuur"]},
    "Cardiff": {"country": "Wales", "statarea_id": "43", "mongodb_id": "43", "alt": ["Cardiff City"]}, # Plays in English league system
    "Carrarese": {"country": "Italy", "statarea_id": "1581", "mongodb_id": "1581", "alt": []},
    "Cartagena": {"country": "Spain", "statarea_id": "5262", "mongodb_id": "5262", "alt": ["FC Cartagena"]},
    "Casa Pia": {"country": "Portugal", "statarea_id": "4716", "mongodb_id": "4716", "alt": ["Casa_Pia"]}, # Snippet key "Casa Pia"
    "Castellon": {"country": "Spain", "statarea_id": "5254", "mongodb_id": "5254", "alt": ["CD Castellon"]},
    "Catanzaro": {"country": "Italy", "statarea_id": "1687", "mongodb_id": "1687", "alt": []},
    "Celta Vigo": {"country": "Spain", "statarea_id": "538", "mongodb_id": "538", "alt": ["Celta", "Celta_Vigo"]}, # My canonical "Celta Vigo", snippet used "Celta"
    "Celtic": {"country": "Scotland", "statarea_id": "247", "mongodb_id": "247", "alt": []},
    "Cercle Brugge": {"country": "Belgium", "statarea_id": "741", "mongodb_id": "741", "alt": ["Cercle_Brugge"]}, # Snippet key "Cercle Brugge"
    "Cesena": {"country": "Italy", "statarea_id": "509", "mongodb_id": "509", "alt": []},
    "Charleroi": {"country": "Belgium", "statarea_id": "736", "mongodb_id": "736", "alt": ["Sporting Charleroi"]},
    "Chaves": {"country": "Portugal", "statarea_id": "223", "mongodb_id": "223", "alt": ["GD Chaves"]},
    "Chelsea": {"country": "England", "statarea_id": "49", "mongodb_id": "49", "alt": []},
    "Cittadella": {"country": "Italy", "statarea_id": "510", "mongodb_id": "510", "alt": ["AS Cittadella"]},
    "Clermont": {"country": "France", "statarea_id": "99", "mongodb_id": "99", "alt": ["Clermont Foot", "Clermont_Foot"]}, # Snippet key "Clermont"
    "Club Brugge": {"country": "Belgium", "statarea_id": "569", "mongodb_id": "569", "alt": ["Club_Brugge", "ClubBruggeKV"]}, # Snippet key "Club Brugge"
    "Como": {"country": "Italy", "statarea_id": "895", "mongodb_id": "895", "alt": []},
    "Cordoba": {"country": "Spain", "statarea_id": "713", "mongodb_id": "713", "alt": ["Cordoba CF"]},
    "Cosenza": {"country": "Italy", "statarea_id": "10137", "mongodb_id": "10137", "alt": []},
    "Coventry": {"country": "England", "statarea_id": "1346", "mongodb_id": "1346", "alt": ["Coventry City"]},
    "Cracovia Krakow": {"country": "Poland", "statarea_id": "350", "mongodb_id": "350", "alt":["Cracovia_Krakow"]}, # Snippet key "Cracovia_Krakow"
    "Cremonese": {"country": "Italy", "statarea_id": "520", "mongodb_id": "520", "alt": []},
    "Crvena Zvezda": {"country": "Serbia", "statarea_id": "598", "mongodb_id": "598", "alt": ["Red Star Belgrade", "Crvena_Zvezda"]}, # Snippet key "Crvena_Zvezda"
    "Crystal Palace": {"country": "England", "statarea_id": "52", "mongodb_id": "52", "alt": []},
    "Graafschap": {"country": "Netherlands", "statarea_id": "199", "mongodb_id": "199", "alt": ["De Graafschap", "De_Graafschap"]}, # Snippet key "Graafschap" (from De Graafschap)
    "Den Bosch": {"country": "Netherlands", "statarea_id": "421", "mongodb_id": "421", "alt": ["FC Den Bosch", "Den_Bosch"]}, # Snippet key "Den_Bosch"
    "Alaves": {"country": "Spain", "statarea_id": "542", "mongodb_id": "542", "alt": ["Deportivo Alaves"]}, # Snippet key "Alaves" (from Deportivo Alaves)
    "La Coruna": {"country": "Spain", "statarea_id": "544", "mongodb_id": "544", "alt": ["Deportivo La Coruna", "Deportivo_La_Coruna", "Deportivo"]}, # Snippet key "La Coruna" (from Deportivo La Coruna)
    "Derby": {"country": "England", "statarea_id": "69", "mongodb_id": "69", "alt": ["Derby County"]},
    "Dinamo Bucuresti": {"country": "Romania", "statarea_id": "635", "mongodb_id": "635", "alt": []},
    "Dordrecht": {"country": "Netherlands", "statarea_id": "409", "mongodb_id": "409", "alt": ["FC Dordrecht"]},
    "Dunkerque": {"country": "France", "statarea_id": "1304", "mongodb_id": "1304", "alt": ["USL Dunkerque"]},
    "Eibar": {"country": "Spain", "statarea_id": "545", "mongodb_id": "545", "alt": ["SD Eibar"]},
    "Braunschweig": {"country": "Germany", "statarea_id": "744", "mongodb_id": "744", "alt": ["Eintracht Braunschweig"]}, # Snippet key "Braunschweig" (from Eintracht Braunschweig)
    "Ein Frankfurt": {"country": "Germany", "statarea_id": "169", "mongodb_id": "169", "alt": ["Eintracht Frankfurt", "EintrachtFrankfurt"]}, # Snippet key "Ein Frankfurt" (from Eintracht Frankfurt)
    "Elche": {"country": "Spain", "statarea_id": "797", "mongodb_id": "797", "alt": ["Elche CF"]},
    "FC Emmen": {"country": "Netherlands", "statarea_id": "208", "mongodb_id": "208", "alt": ["Emmen"]}, # Snippet key "FC Emmen" (from Emmen)
    "Empoli": {"country": "Italy", "statarea_id": "511", "mongodb_id": "511", "alt": []},
    "Espanol": {"country": "Spain", "statarea_id": "540", "mongodb_id": "540", "alt": ["Espanyol", "RCD Espanyol"]}, # Snippet key "Espanol" (from Espanyol)
    "Troyes": {"country": "France", "statarea_id": "110", "mongodb_id": "110", "alt": ["Estac Troyes", "ESTAC Troyes"]}, # Snippet key "Troyes" (from Estac Troyes)
    "Estoril": {"country": "Portugal", "statarea_id": "230", "mongodb_id": "230", "alt": ["Estoril Praia"]},
    "Estrela": {"country": "Portugal", "statarea_id": "15130", "mongodb_id": "15130", "alt": ["Estrela Da Amadora", "CF Estrela Amadora"]}, # Snippet key "Estrela" (from Estrela Da Amadora)
    "Everton": {"country": "England", "statarea_id": "45", "mongodb_id": "45", "alt": []},
    "Excelsior": {"country": "Netherlands", "statarea_id": "196", "mongodb_id": "196", "alt": []},
    "Eyupspor": {"country": "Turkey", "statarea_id": "3588", "mongodb_id": "3588", "alt": []},
    "Augsburg": {"country": "Germany", "statarea_id": "170", "mongodb_id": "170", "alt": ["FC Augsburg"]}, # Snippet key "Augsburg" (from FC Augsburg)
    "FC Botosani": {"country": "Romania", "statarea_id": "2581", "mongodb_id": "2581", "alt": []},
    "FC Copenhagen": {"country": "Denmark", "statarea_id": "400", "mongodb_id": "400", "alt": []},
    "FC Dender": {"country": "Belgium", "statarea_id": "6215", "mongodb_id": "6215", "alt": []},
    "FC Eindhoven": {"country": "Netherlands", "statarea_id": "197", "mongodb_id": "422", "alt": []}, # Note: statarea_id 197 in snippet, mongodb_id 422
    "FC Koln": {"country": "Germany", "statarea_id": "192", "mongodb_id": "192", "alt": ["1. FC Koln", "Koln", "1FCKöln", "1.FCKoln"]}, # My canonical for "1FCKöln"
    "FC Lugano": {"country": "Switzerland", "statarea_id": "606", "mongodb_id": "606", "alt": []},
    "FC Midtjylland": {"country": "Denmark", "statarea_id": "397", "mongodb_id": "397", "alt": []},
    "FC Nordsjaelland": {"country": "Denmark", "statarea_id": "398", "mongodb_id": "398", "alt": []},
    "Porto": {"country": "Portugal", "statarea_id": "212", "mongodb_id": "212", "alt": ["FC Porto"]}, # Snippet key "Porto" (from FC Porto)
    "Porto B": {"country": "Portugal", "statarea_id": "212", "mongodb_id": "243", "alt": ["FC Porto B"]}, # Snippet key "Porto B" (from FC Porto B)
    "Schalke 04": {"country": "Germany", "statarea_id": "174", "mongodb_id": "174", "alt": ["FC Schalke 04", "FCSchalke04"]}, # Snippet had 174, my old was 161. Using snippet's. My original "Schalke 04" has "FCSchalke04" as alt.
    "St Pauli": {"country": "Germany", "statarea_id": "186", "mongodb_id": "186", "alt": ["FC St. Pauli", "FCStPauli"]}, # Snippet key "St Pauli" (from FC St. Pauli)
    "FCSB": {"country": "Romania", "statarea_id": "559", "mongodb_id": "559", "alt": []},
    "Mainz 05": {"country": "Germany", "statarea_id": "164", "mongodb_id": "164", "alt": ["FSV Mainz 05", "FSVMainz05"]}, # Snippet key "Mainz 05" (from FSV Mainz 05)
    "Famalicao": {"country": "Portugal", "statarea_id": "242", "mongodb_id": "242", "alt": []},
    "Farense": {"country": "Portugal", "statarea_id": "231", "mongodb_id": "231", "alt": []},
    "Farul Constanta": {"country": "Romania", "statarea_id": "2596", "mongodb_id": "2596", "alt": []},
    "Feirense": {"country": "Portugal", "statarea_id": "213", "mongodb_id": "213", "alt": []},
    "Felgueiras": {"country": "Portugal", "statarea_id": "4744", "mongodb_id": "4744", "alt": []},
    "Fenerbahce": {"country": "Turkey", "statarea_id": "611", "mongodb_id": "611", "alt": []},
    "Ferencvarosi TC": {"country": "Hungary", "statarea_id": "651", "mongodb_id": "651", "alt": []},
    "Feyenoord": {"country": "Netherlands", "statarea_id": "209", "mongodb_id": "209", "alt": []},
    "Fiorentina": {"country": "Italy", "statarea_id": "502", "mongodb_id": "502", "alt": []},
    "Fortuna Dusseldorf": {"country": "Germany", "statarea_id": "158", "mongodb_id": "158", "alt": []},
    "For Sittard": {"country": "Netherlands", "statarea_id": "205", "mongodb_id": "205", "alt": ["Fortuna Sittard"]}, # Snippet key "For Sittard" (from Fortuna Sittard)
    "Frosinone": {"country": "Italy", "statarea_id": "512", "mongodb_id": "512", "alt": []},
    "Fulham": {"country": "England", "statarea_id": "36", "mongodb_id": "36", "alt": []},
    "GKS Katowice": {"country": "Poland", "statarea_id": "3484", "mongodb_id": "3484", "alt": []},
    "Galatasaray": {"country": "Turkey", "statarea_id": "645", "mongodb_id": "645", "alt": []},
    "Genk": {"country": "Belgium", "statarea_id": "742", "mongodb_id": "742", "alt": []}, # Note: User snippet had 742, my old was 555. Using snippet.
    "Genoa": {"country": "Italy", "statarea_id": "495", "mongodb_id": "495", "alt": []},
    "Gent": {"country": "Belgium", "statarea_id": "631", "mongodb_id": "631", "alt": ["KAA Gent"]},
    "Getafe": {"country": "Spain", "statarea_id": "546", "mongodb_id": "546", "alt": []},
    "Gil Vicente": {"country": "Portugal", "statarea_id": "762", "mongodb_id": "762", "alt": []},
    "Girona": {"country": "Spain", "statarea_id": "547", "mongodb_id": "547", "alt": []},
    "Gloria Buzau": {"country": "Romania", "statarea_id": "6232", "mongodb_id": "6232", "alt": []},
    "Go Ahead Eagles": {"country": "Netherlands", "statarea_id": "410", "mongodb_id": "410", "alt": []},
    "Gornik Zabrze": {"country": "Poland", "statarea_id": "340", "mongodb_id": "340", "alt": []},
    "Goztep": {"country": "Turkey", "statarea_id": "994", "mongodb_id": "994", "alt": []}, # User snippet is "Goztep", my original was "Goztepe"
    "Granada": {"country": "Spain", "statarea_id": "715", "mongodb_id": "715", "alt": ["Granada CF"]}, # Snippet key "Granada" (from Granada CF)
    "Grenoble": {"country": "France", "statarea_id": "101", "mongodb_id": "101", "alt": ["Grenoble Foot 38"]}, # Snippet key "Grenoble" (from Grenoble Foot 38)
    "Groningen": {"country": "Netherlands", "statarea_id": "202", "mongodb_id": "202", "alt": []},
    "Guimaraes": {"country": "Portugal", "statarea_id": "224", "mongodb_id": "224", "alt": ["Vitoria Guimaraes"]},
    "Guingamp": {"country": "France", "statarea_id": "90", "mongodb_id": "90", "alt": []},
    "HNK Gorica": {"country": "Croatia", "statarea_id": "1068", "mongodb_id": "1068", "alt": []},
    "HNK Hajduk Split": {"country": "Croatia", "statarea_id": "608", "mongodb_id": "608", "alt": []},
    "HNK Rijeka": {"country": "Croatia", "statarea_id": "561", "mongodb_id": "561", "alt": []},
    "Hamburg": {"country": "Germany", "statarea_id": "175", "mongodb_id": "175", "alt": ["Hamburger SV", "HamburgerSV"]}, # Snippet key "Hamburg" (from Hamburger SV). My old "Hamburg" had 167. Using snippet's ID for Hamburg.
    "Hannover": {"country": "Germany", "statarea_id": "166", "mongodb_id": "166", "alt": ["Hannover 96", "Hannover96"]}, # Snippet key "Hannover" (from Hannover 96)
    "Hatayspor": {"country": "Turkey", "statarea_id": "3575", "mongodb_id": "3575", "alt": []},
    "Heerenveen": {"country": "Netherlands", "statarea_id": "210", "mongodb_id": "210", "alt": []},
    "Helmond Sport": {"country": "Netherlands", "statarea_id": "424", "mongodb_id": "424", "alt": []},
    "Heracles": {"country": "Netherlands", "statarea_id": "206", "mongodb_id": "206", "alt": []},
    "Holstein Kiel": {"country": "Germany", "statarea_id": "191", "mongodb_id": "191", "alt": []},
    "Huesca": {"country": "Spain", "statarea_id": "726", "mongodb_id": "726", "alt": []},
    "Hull": {"country": "England", "statarea_id": "64", "mongodb_id": "64", "alt": ["Hull City"]}, # Snippet key "Hull" (from Hull City)
    "Inter": {"country": "Italy", "statarea_id": "505", "mongodb_id": "505", "alt": ["Internazionale"]},
    "Ipswich": {"country": "England", "statarea_id": "57", "mongodb_id": "57", "alt": []},
    "Buyuksehyr": {"country": "Turkey", "statarea_id": "564", "mongodb_id": "564", "alt": ["Istanbul Basaksehir", "İstanbul Başakşehir"]}, # Snippet key "Buyuksehyr" (from Istanbul Basaksehir)
    "Jagiellonia": {"country": "Poland", "statarea_id": "336", "mongodb_id": "336", "alt": []},
    "Regensburg": {"country": "Germany", "statarea_id": "177", "mongodb_id": "177", "alt": ["Jahn Regensburg"]}, # Snippet key "Regensburg" (from Jahn Regensburg)
    "Jong AZ": {"country": "Netherlands", "statarea_id": "418", "mongodb_id": "418", "alt": []},
    "Jong Ajax": {"country": "Netherlands", "statarea_id": "425", "mongodb_id": "425", "alt": []},
    "Jong PSV": {"country": "Netherlands", "statarea_id": "411", "mongodb_id": "411", "alt": []},
    "Jong Utrecht": {"country": "Netherlands", "statarea_id": "428", "mongodb_id": "428", "alt": []},
    "Juve Stabia": {"country": "Italy", "statarea_id": "863", "mongodb_id": "863", "alt": []},
    "Juventus": {"country": "Italy", "statarea_id": "496", "mongodb_id": "496", "alt": []},
    "Mechelen": {"country": "Belgium", "statarea_id": "266", "mongodb_id": "266", "alt": ["KV Mechelen"]}, # Snippet key "Mechelen" (from KV Mechelen)
    "KVC Westerlo": {"country": "Belgium", "statarea_id": "261", "mongodb_id": "261", "alt": ["Westerlo", "KVCOosterlo"]}, # My canonical "KVC Westerlo", snippet used "Westerlo"
    "Karlsruhe": {"country": "Germany", "statarea_id": "785", "mongodb_id": "785", "alt": ["Karlsruher SC"]}, # Snippet key "Karlsruhe" (from Karlsruher SC)
    "Kasimpasa": {"country": "Turkey", "statarea_id": "1004", "mongodb_id": "1004", "alt": []},
    "Kayserispor": {"country": "Turkey", "statarea_id": "1001", "mongodb_id": "1001", "alt": []},
    "Konyaspor": {"country": "Turkey", "statarea_id": "607", "mongodb_id": "607", "alt": []},
    "Korona Kielce": {"country": "Poland", "statarea_id": "346", "mongodb_id": "346", "alt": []},
    "Kortrijk": {"country": "Belgium", "statarea_id": "734", "mongodb_id": "734", "alt": []},
    "Las Palmas": {"country": "Spain", "statarea_id": "534", "mongodb_id": "534", "alt": []},
    "Laval": {"country": "France", "statarea_id": "433", "mongodb_id": "433", "alt": []},
    "Lazio": {"country": "Italy", "statarea_id": "487", "mongodb_id": "487", "alt": []},
    "Le Havre": {"country": "France", "statarea_id": "111", "mongodb_id": "111", "alt": []},
    "Lecce": {"country": "Italy", "statarea_id": "867", "mongodb_id": "867", "alt": []},
    "Lech Poznan": {"country": "Poland", "statarea_id": "347", "mongodb_id": "347", "alt": []},
    "Lechia Gdansk": {"country": "Poland", "statarea_id": "343", "mongodb_id": "343", "alt": []},
    "Leeds": {"country": "England", "statarea_id": "63", "mongodb_id": "63", "alt": ["Leeds United"]},
    "Leganes": {"country": "Spain", "statarea_id": "537", "mongodb_id": "537", "alt": []},
    "Legia Warszawa": {"country": "Poland", "statarea_id": "339", "mongodb_id": "339", "alt": ["Legia Warsaw", "LegiaWarszawa"]},
    "Leicester": {"country": "England", "statarea_id": "46", "mongodb_id": "46", "alt": ["Leicester City"]},
    "Leiria": {"country": "Portugal", "statarea_id": "4662", "mongodb_id": "4662", "alt": ["União de Leiria", "UniãoLeiria", "UniãodeLeiria"]}, # Snippet key "Leiria", my old "Leiria" had 218. Using snippet's ID.
    "Leixoes": {"country": "Portugal", "statarea_id": "244", "mongodb_id": "244", "alt": []},
    "Lens": {"country": "France", "statarea_id": "116", "mongodb_id": "116", "alt": []},
    "Levante": {"country": "Spain", "statarea_id": "539", "mongodb_id": "539", "alt": []},
    "Lille": {"country": "France", "statarea_id": "79", "mongodb_id": "79", "alt": []},
    "Liverpool": {"country": "England", "statarea_id": "40", "mongodb_id": "40", "alt": []},
    "Lorient": {"country": "France", "statarea_id": "97", "mongodb_id": "97", "alt": []},
    "Luton": {"country": "England", "statarea_id": "1359", "mongodb_id": "1359", "alt": ["Luton Town"]},
    "Lyngby": {"country": "Denmark", "statarea_id": "625", "mongodb_id": "625", "alt": []},
    "Lyon": {"country": "France", "statarea_id": "80", "mongodb_id": "80", "alt": ["Olympique Lyonnais"]},
    "MVV": {"country": "Netherlands", "statarea_id": "412", "mongodb_id": "412", "alt": ["MVV Maastricht"]},
    "Mafra": {"country": "Portugal", "statarea_id": "245", "mongodb_id": "245", "alt": []},
    "Malaga": {"country": "Spain", "statarea_id": "535", "mongodb_id": "535", "alt": []},
    "Mallorca": {"country": "Spain", "statarea_id": "798", "mongodb_id": "798", "alt": ["RCD Mallorca"]},
    "Man City": {"country": "England", "statarea_id": "50", "mongodb_id": "50", "alt": ["Manchester City", "Manchester_City"]}, # Snippet key "Man City" (from Manchester City)
    "Man United": {"country": "England", "statarea_id": "33", "mongodb_id": "33", "alt": ["Manchester United", "ManchesterUtd", "ManchesterUnited"]}, # Snippet key "Man United" (from Manchester Utd)
    "Mantova": {"country": "Italy", "statarea_id": "1693", "mongodb_id": "1693", "alt": []},
    "Maritimo": {"country": "Portugal", "statarea_id": "214", "mongodb_id": "214", "alt": []},
    "Marseille": {"country": "France", "statarea_id": "81", "mongodb_id": "81", "alt": ["Olympique Marseille"]},
    "Martigues": {"country": "France", "statarea_id": "3200", "mongodb_id": "3200", "alt": []},
    "Metz": {"country": "France", "statarea_id": "112", "mongodb_id": "112", "alt": ["FC Metz"]}, # Snippet key "Metz", my old "Metz" had 83. Using snippet.
    "Middlesbrough": {"country": "England", "statarea_id": "70", "mongodb_id": "70", "alt": []}, # My "Stoke" also has ID 70. This is a conflict to resolve later. For now, Middlesbrough keeps it.
    "Millwall": {"country": "England", "statarea_id": "58", "mongodb_id": "58", "alt": []},
    "Mirandes": {"country": "Spain", "statarea_id": "799", "mongodb_id": "799", "alt": ["CD Mirandés"]},
    "Modena": {"country": "Italy", "statarea_id": "899", "mongodb_id": "899", "alt": []},
    "Monaco": {"country": "France", "statarea_id": "91", "mongodb_id": "91", "alt": ["AS Monaco"]},
    "Montpellier": {"country": "France", "statarea_id": "82", "mongodb_id": "82", "alt": []},
    "Monza": {"country": "Italy", "statarea_id": "1579", "mongodb_id": "1579", "alt": []},
    "Moreirense": {"country": "Portugal", "statarea_id": "215", "mongodb_id": "215", "alt": []},
    "Motor Lublin": {"country": "Poland", "statarea_id": "14562", "mongodb_id": "14562", "alt": []},
    "NAC Breda": {"country": "Netherlands", "statarea_id": "203", "mongodb_id": "203", "alt": []},
    "Nijmegen": {"country": "Netherlands", "statarea_id": "413", "mongodb_id": "413", "alt": ["NEC Nijmegen"]}, # Snippet key "Nijmegen" (from NEC Nijmegen)
    "NK Dinamo Zagreb": {"country": "Croatia", "statarea_id": "620", "mongodb_id": "620", "alt": ["Dinamo Zagreb"]},
    "NK Lokomotiva Zagreb": {"country": "Croatia", "statarea_id": "1017", "mongodb_id": "1017", "alt": ["Lokomotiva Zagreb"]},
    "NK Osijek": {"country": "Croatia", "statarea_id": "616", "mongodb_id": "616", "alt": ["Osijek"]},
    "NK Slaven Belupo": {"country": "Croatia", "statarea_id": "1018", "mongodb_id": "1018", "alt": ["Slaven Belupo"]},
    "NK Varazdin": {"country": "Croatia", "statarea_id": "1483", "mongodb_id": "1483", "alt": ["Varazdin"]},
    "Nacional": {"country": "Portugal", "statarea_id": "225", "mongodb_id": "225", "alt": ["CD Nacional"]},
    "Nantes": {"country": "France", "statarea_id": "83", "mongodb_id": "83", "alt": []},
    "Napoli": {"country": "Italy", "statarea_id": "492", "mongodb_id": "492", "alt": ["SSC Napoli"]},
    "Newcastle": {"country": "England", "statarea_id": "34", "mongodb_id": "34", "alt": ["Newcastle United"]},
    "Nice": {"country": "France", "statarea_id": "84", "mongodb_id": "84", "alt": ["OGC Nice"]},
    "Norwich": {"country": "England", "statarea_id": "71", "mongodb_id": "71", "alt": ["Norwich City"]},
    "Nott'm Forest": {"country": "England", "statarea_id": "65", "mongodb_id": "65", "alt": ["Nottingham Forest"]}, # Snippet key "Nott'm Forest" (from Nottingham Forest)
    "Oud-Heverlee Leuven": {"country": "Belgium", "statarea_id": "260", "mongodb_id": "260", "alt": ["OH Leuven", "Leuven"]}, # Snippet key "Oud-Heverlee Leuven" (from OH Leuven)
    "Oliveirense": {"country": "Portugal", "statarea_id": "233", "mongodb_id": "233", "alt": []},
    "Olympiakos": {"country": "Greece", "statarea_id": "553", "mongodb_id": "553", "alt": ["Olympiakos Piraeus"]}, # Snippet key "Olympiakos" (from Olympiakos Piraeus)
    "Osasuna": {"country": "Spain", "statarea_id": "727", "mongodb_id": "727", "alt": []},
    "Oviedo": {"country": "Spain", "statarea_id": "718", "mongodb_id": "718", "alt": ["Real Oviedo"]},
    "Oxford": {"country": "England", "statarea_id": "1338", "mongodb_id": "1338", "alt": ["Oxford United"]}, # Snippet key "Oxford" (from Oxford United)
    "PAOK": {"country": "Greece", "statarea_id": "619", "mongodb_id": "619", "alt": []},
    "Zwolle": {"country": "Netherlands", "statarea_id": "193", "mongodb_id": "193", "alt": ["PEC Zwolle"]}, # Snippet key "Zwolle" (from PEC Zwolle)
    "PSV Eindhoven": {"country": "Netherlands", "statarea_id": "197", "mongodb_id": "197", "alt": ["PSV"]}, # Snippet key "PSV Eindhoven"
    "Pacos Ferreira": {"country": "Portugal", "statarea_id": "234", "mongodb_id": "234", "alt": []},
    "Pafos FC": {"country": "Cyprus", "statarea_id": "3403", "mongodb_id": "3403", "alt": ["Pafos"]},
    "Palermo": {"country": "Italy", "statarea_id": "522", "mongodb_id": "522", "alt": []},
    "Panathinaikos": {"country": "Greece", "statarea_id": "617", "mongodb_id": "617", "alt": []},
    "Paris FC": {"country": "France", "statarea_id": "114", "mongodb_id": "114", "alt": []},
    "Paris Saint Germain": {"country": "France", "statarea_id": "85", "mongodb_id": "85", "alt": ["PSG", "Paris SG", "ParisSaintGermain"]}, # My canonical, snippet used "Paris SG"
    "Parma": {"country": "Italy", "statarea_id": "523", "mongodb_id": "523", "alt": []},
    "Pau FC": {"country": "France", "statarea_id": "1297", "mongodb_id": "1297", "alt": ["Pau"]}, # Snippet key "Pau FC" (from Pau)
    "Penafiel": {"country": "Portugal", "statarea_id": "235", "mongodb_id": "235", "alt": []},
    "Piast Gliwice": {"country": "Poland", "statarea_id": "349", "mongodb_id": "349", "alt": []},
    "Pisa": {"country": "Italy", "statarea_id": "801", "mongodb_id": "801", "alt": []},
    "Plymouth": {"country": "England", "statarea_id": "1357", "mongodb_id": "1357", "alt": ["Plymouth Argyle"]},
    "Pogon Szczecin": {"country": "Poland", "statarea_id": "348", "mongodb_id": "348", "alt": []},
    "Portimonense": {"country": "Portugal", "statarea_id": "216", "mongodb_id": "216", "alt": []},
    "Portsmouth": {"country": "England", "statarea_id": "1355", "mongodb_id": "1355", "alt": []},
    "Preston": {"country": "England", "statarea_id": "59", "mongodb_id": "59", "alt": ["Preston North End"]},
    "QPR": {"country": "England", "statarea_id": "72", "mongodb_id": "72", "alt": ["Queens Park Rangers"]},
    "RB Leipzig": {"country": "Germany", "statarea_id": "173", "mongodb_id": "173", "alt": []},
    "Santander": {"country": "Spain", "statarea_id": "4665", "mongodb_id": "4665", "alt": ["Racing Santander"]}, # Snippet key "Santander" (from Racing Santander)
    "Radomiak Radom": {"country": "Poland", "statarea_id": "4248", "mongodb_id": "4248", "alt": []},
    "Rakow Czestochowa": {"country": "Poland", "statarea_id": "3491", "mongodb_id": "3491", "alt": []},
    "Randers FC": {"country": "Denmark", "statarea_id": "799", "mongodb_id": "401", "alt": ["Randers"]}, # Snippet has 799/401, my original had 404 for Randers. Using snippet.
    "Rangers": {"country": "Scotland", "statarea_id": "77", "mongodb_id": "257", "alt": []}, # Snippet has 77/257, my original had 248. Using snippet.
    "Rapid Bucuresti": {"country": "Romania", "statarea_id": "636", "mongodb_id": "636", "alt": ["Rapid Bucharest"]}, # Added this, common Romanian team
    "Rapid Vienna": {"country": "Austria", "statarea_id": "781", "mongodb_id": "6231", "alt": []}, # User snippet has two entries for Rapid Vienna, using the one with mongodb_id 6231
    "Vallecano": {"country": "Spain", "statarea_id": "728", "mongodb_id": "728", "alt": ["Rayo Vallecano"]}, # Snippet key "Vallecano" (from Rayo Vallecano)
    "Real Madrid": {"country": "Spain", "statarea_id": "541", "mongodb_id": "541", "alt": []},
    "Sociedad": {"country": "Spain", "statarea_id": "548", "mongodb_id": "548", "alt": ["Real Sociedad"]}, # Snippet key "Sociedad" (from Real Sociedad)
    "Red Star": {"country": "France", "statarea_id": "104", "mongodb_id": "104", "alt": []}, # This is Red Star FC (Paris), not Crvena Zvezda
    "Reggiana": {"country": "Italy", "statarea_id": "880", "mongodb_id": "880", "alt": []},
    "Reims": {"country": "France", "statarea_id": "93", "mongodb_id": "93", "alt": ["Stade Reims"]},
    "Rennes": {"country": "France", "statarea_id": "94", "mongodb_id": "94", "alt": ["Stade Rennais"]},
    "Rio Ave": {"country": "Portugal", "statarea_id": "226", "mongodb_id": "226", "alt": []},
    "Rizespor": {"country": "Turkey", "statarea_id": "1007", "mongodb_id": "1007", "alt": ["Caykur Rizespor"]},
    "Roda": {"country": "Netherlands", "statarea_id": "414", "mongodb_id": "414", "alt": ["Roda JC Kerkrade"]},
    "Rodez": {"country": "France", "statarea_id": "1301", "mongodb_id": "1301", "alt": []},
    "Roma": {"country": "Italy", "statarea_id": "497", "mongodb_id": "497", "alt": ["AS Roma"]},
    "Sp Braga": {"country": "Portugal", "statarea_id": "217", "mongodb_id": "217", "alt": ["SC Braga", "Braga"]}, # Snippet key "Sp Braga" (from SC Braga)
    "Freiburg": {"country": "Germany", "statarea_id": "160", "mongodb_id": "160", "alt": ["SC Freiburg", "SCFreiburg"]}, # Snippet key "Freiburg" (from SC Freiburg). My old Freiburg had 174. Using snippet.
    "Paderborn": {"country": "Germany", "statarea_id": "185", "mongodb_id": "185", "alt": ["SC Paderborn 07"]}, # Snippet key "Paderborn" (from SC Paderborn 07)
    "SSV Ulm 1846": {"country": "Germany", "statarea_id": "1652", "mongodb_id": "1652", "alt": ["Ulm"]},
    "Darmstadt": {"country": "Germany", "statarea_id": "181", "mongodb_id": "181", "alt": ["SV Darmstadt 98"]}, # Snippet key "Darmstadt" (from SV Darmstadt 98)
    "Elversberg": {"country": "Germany", "statarea_id": "1660", "mongodb_id": "1660", "alt": ["SV Elversberg"]}, # Snippet key "Elversberg" (from SV Elversberg)
    "Salernitana": {"country": "Italy", "statarea_id": "514", "mongodb_id": "514", "alt": []},
    "Sampdoria": {"country": "Italy", "statarea_id": "498", "mongodb_id": "498", "alt": []},
    "Samsunspor": {"country": "Turkey", "statarea_id": "3603", "mongodb_id": "3603", "alt": []},
    "Santa Clara": {"country": "Portugal", "statarea_id": "227", "mongodb_id": "227", "alt": []},
    "Sassuolo": {"country": "Italy", "statarea_id": "488", "mongodb_id": "488", "alt": []},
    "Sepsi OSK": {"country": "Romania", "statarea_id": "2585", "mongodb_id": "2585", "alt": ["Sepsi"]},
    "Sevilla": {"country": "Spain", "statarea_id": "536", "mongodb_id": "536", "alt": []},
    "Sheffield United": {"country": "England", "statarea_id": "62", "mongodb_id": "62", "alt": ["Sheffield Utd"]}, # Snippet key "Sheffield United"
    "Sheffield Weds": {"country": "England", "statarea_id": "62", "mongodb_id": "74", "alt": ["Sheffield Wednesday"]}, # Snippet key "Sheffield Weds"
    "Sibenik": {"country": "Croatia", "statarea_id": "1475", "mongodb_id": "1475", "alt": []},
    "Silkeborg": {"country": "Denmark", "statarea_id": "2073", "mongodb_id": "2073", "alt": []},
    "Sivasspor": {"country": "Turkey", "statarea_id": "1002", "mongodb_id": "1002", "alt": []},
    "Slask Wroclaw": {"country": "Poland", "statarea_id": "337", "mongodb_id": "337", "alt": []},
    "Slavia Praha": {"country": "Czech Republic", "statarea_id": "560", "mongodb_id": "560", "alt": ["Slavia Prague"]},
    "SonderjyskE": {"country": "Denmark", "statarea_id": "396", "mongodb_id": "396", "alt": ["Sonderjyske"]},
    "Southampton": {"country": "England", "statarea_id": "41", "mongodb_id": "41", "alt": []},
    "Greuther Furth": {"country": "Germany", "statarea_id": "178", "mongodb_id": "178", "alt": ["SpVgg Greuther Furth"]}, # Snippet key "Greuther Furth"
    "Sparta Rotterdam": {"country": "Netherlands", "statarea_id": "426", "mongodb_id": "426", "alt": []},
    "Spezia": {"country": "Italy", "statarea_id": "515", "mongodb_id": "515", "alt": []},
    "Sp Lisbon": {"country": "Portugal", "statarea_id": "228", "mongodb_id": "228", "alt": ["Sporting CP", "Sporting Lisbon"]}, # Snippet key "Sp Lisbon"
    "Sp Gijon": {"country": "Spain", "statarea_id": "731", "mongodb_id": "731", "alt": ["Sporting Gijon", "Sporting_Gijon", "SpGijon"]}, # Snippet key "Sp Gijon", my original Sp Gijon had 533
    "St Truiden": {"country": "Belgium", "statarea_id": "735", "mongodb_id": "735", "alt": ["Sint-Truiden"]},
    "Brest": {"country": "France", "statarea_id": "106", "mongodb_id": "106", "alt": ["Stade Brestois 29", "Stade Brest"]}, # Snippet key "Brest"
    "Stal Mielec": {"country": "Poland", "statarea_id": "3493", "mongodb_id": "3493", "alt": []},
    "Standard Liege": {"country": "Belgium", "statarea_id": "733", "mongodb_id": "733", "alt": ["Standard_Liege", "Standard"]}, # Snippet key "Standard", my old had 552. Using Snippet.
    "Stoke": {"country": "England", "statarea_id": "75", "mongodb_id": "75", "alt": ["Stoke City", "StokeCity"]}, # Snippet key "Stoke", my old "Stoke" had 70. User snippet more specific.
    "Strasbourg": {"country": "France", "statarea_id": "95", "mongodb_id": "95", "alt": []},
    "Sturm Graz": {"country": "Austria", "statarea_id": "637", "mongodb_id": "637", "alt": []},
    "Stuttgart": {"country": "Germany", "statarea_id": "172", "mongodb_id": "172", "alt": ["VfB Stuttgart", "VfBStuttgart"]}, # Snippet key "Stuttgart", my old Stuttgart had 160. Using snippet.
    "Sudtirol": {"country": "Italy", "statarea_id": "1578", "mongodb_id": "1578", "alt": ["FC Südtirol"]},
    "Sunderland": {"country": "England", "statarea_id": "746", "mongodb_id": "746", "alt": []},
    "Swansea": {"country": "England", "statarea_id": "76", "mongodb_id": "76", "alt": ["Swansea City"]}, # Wales team in English system
    "TSC Backa Topola": {"country": "Serbia", "statarea_id": "2646", "mongodb_id": "2646", "alt": []},
    "Telstar": {"country": "Netherlands", "statarea_id": "427", "mongodb_id": "427", "alt": []},
    "Tenerife": {"country": "Spain", "statarea_id": "719", "mongodb_id": "719", "alt": []},
    "Tondela": {"country": "Portugal", "statarea_id": "218", "mongodb_id": "218", "alt": []},
    "Torino": {"country": "Italy", "statarea_id": "503", "mongodb_id": "503", "alt": []},
    "Torreense": {"country": "Portugal", "statarea_id": "4799", "mongodb_id": "4799", "alt": []},
    "Tottenham": {"country": "England", "statarea_id": "47", "mongodb_id": "47", "alt": ["Spurs"]},
    "Toulouse": {"country": "France", "statarea_id": "96", "mongodb_id": "96", "alt": []},
    "Trabzonspor": {"country": "Turkey", "statarea_id": "998", "mongodb_id": "998", "alt": []},
    "Twente": {"country": "Netherlands", "statarea_id": "415", "mongodb_id": "415", "alt": ["FC Twente"]},
    "UTA Arad": {"country": "Romania", "statarea_id": "2589", "mongodb_id": "2589", "alt": []},
    "Udinese": {"country": "Italy", "statarea_id": "494", "mongodb_id": "494", "alt": []},
    "Union Berlin": {"country": "Germany", "statarea_id": "182", "mongodb_id": "182", "alt": ["1. FC Union Berlin"]},
    "St. Gilloise": {"country": "Belgium", "statarea_id": "1393", "mongodb_id": "1393", "alt": ["Union St. Gilloise", "Union Saint-Gilloise"]}, # Snippet key "St. Gilloise"
    "Venezia": {"country": "Italy", "statarea_id": "517", "mongodb_id": "517", "alt": ["Unione Venezia"]}, # Snippet key "Venezia"
    "Universitatea Cluj": {"country": "Romania", "statarea_id": "2599", "mongodb_id": "2599", "alt": ["U Cluj"]},
    "Utrecht": {"country": "Netherlands", "statarea_id": "207", "mongodb_id": "207", "alt": ["FC Utrecht"]},
    "VVV Venlo": {"country": "Netherlands", "statarea_id": "204", "mongodb_id": "204", "alt": []},
    "Valencia": {"country": "Spain", "statarea_id": "532", "mongodb_id": "532", "alt": []},
    "Valladolid": {"country": "Spain", "statarea_id": "720", "mongodb_id": "720", "alt": ["Real Valladolid"]},
    "Vejle": {"country": "Denmark", "statarea_id": "395", "mongodb_id": "395", "alt": []},
    "Verona": {"country": "Italy", "statarea_id": "504", "mongodb_id": "504", "alt": ["Hellas Verona"]},
    "Bochum": {"country": "Germany", "statarea_id": "176", "mongodb_id": "176", "alt": ["VfL Bochum"]}, # Snippet key "Bochum"
    "Wolfsburg": {"country": "Germany", "statarea_id": "161", "mongodb_id": "161", "alt": ["VfL Wolfsburg"]}, # Snippet key "Wolfsburg"
    "Viborg": {"country": "Denmark", "statarea_id": "2070", "mongodb_id": "2070", "alt": []},
    "Viktoria Plzen": {"country": "Czech Republic", "statarea_id": "567", "mongodb_id": "567", "alt": []},
    "Villarreal": {"country": "Spain", "statarea_id": "533", "mongodb_id": "533", "alt": []},
    "Vitesse": {"country": "Netherlands", "statarea_id": "200", "mongodb_id": "200", "alt": []},
    "Vizela": {"country": "Portugal", "statarea_id": "810", "mongodb_id": "810", "alt": []},
    "Volendam": {"country": "Netherlands", "statarea_id": "416", "mongodb_id": "416", "alt": ["FC Volendam"]},
    "Waalwijk": {"country": "Netherlands", "statarea_id": "417", "mongodb_id": "417", "alt": ["RKC Waalwijk"]},
    "Watford": {"country": "England", "statarea_id": "38", "mongodb_id": "38", "alt": []},
    "Werder Bremen": {"country": "Germany", "statarea_id": "162", "mongodb_id": "162", "alt": ["Werder_Bremen"]},
    "West Brom": {"country": "England", "statarea_id": "60", "mongodb_id": "60", "alt": ["West Bromwich Albion", "WestBrom"]}, # Snippet key "West Brom"
    "West Ham": {"country": "England", "statarea_id": "48", "mongodb_id": "48", "alt": ["West Ham United"]},
    "Willem II": {"country": "Netherlands", "statarea_id": "195", "mongodb_id": "195", "alt": []},
    "Wolves": {"country": "England", "statarea_id": "39", "mongodb_id": "39", "alt": ["Wolverhampton Wanderers"]},
    "Zaglebie Lubin": {"country": "Poland", "statarea_id": "345", "mongodb_id": "345", "alt": []},
    "Zaragoza": {"country": "Spain", "statarea_id": "732", "mongodb_id": "732", "alt": ["Real Zaragoza"]},
    "Zorya Luhansk": {"country": "Ukraine", "statarea_id": "2234", "mongodb_id": "2234", "alt": ["Zorya", "ZoryaLuhansk"]}, # From my original list
    "Kaiserslautern": {"country": "Germany", "statarea_id": "172", "mongodb_id": "172", "alt": ["1. FC Kaiserslautern", "1FCKaiserslautern"]}, # from my list
    "Nurnberg": {"country": "Germany", "statarea_id": "171", "mongodb_id": "171", "alt": ["1. FC Nürnberg", "FCNurnberg"]}, # from my list
    "Evian TG": {"country": "France", "statarea_id": "100", "mongodb_id": "100", "alt": ["Evian Thonon Gaillard FC"]}, # From my list
    # Additional entries from my original default/example list, if not covered by snippet
    "St Etienne": {"country": "France", "statarea_id": "94", "mongodb_id": "94", "alt": ["Saint Etienne", "Saint-Étienne", "AS Saint-Étienne"]},
    "Standard Liege": {"country": "Belgium", "statarea_id": "733", "mongodb_id": "733", "alt": ["Standard_Liege"]}, # My old Standard had 552, snippet has 733
    "Zulte Waregem": {"country": "Belgium", "statarea_id": "742", "mongodb_id": "742", "alt": ["Zulte_Waregem"]},
    "Eupen": {"country": "Belgium", "statarea_id": "739", "mongodb_id": "739", "alt": ["AS Eupen", "KAS Eupen", "ASEupen"]},
    "Naval": {"country": "Portugal", "statarea_id": "232", "mongodb_id": "232", "alt": ["Naval 1 de Maio", "Naval 1º de Maio", "Naval1ºdeMaio"]},
    "Hercules": {"country": "Spain", "statarea_id": "720", "mongodb_id": "720", "alt": ["Hércules CF", "Hércules"]}, # My original had 720, snippet used Hercules as key.
}

INITIAL_TEAM_NAME_NORMALIZATION: TeamNameNormalizationSchema = {
    "1899Hoffenheim": "Hoffenheim",
    "AC Milan": "Milan", "AC_Milan": "Milan",
    "ADODenHaag": "Den Haag", "ADO Den Haag": "Den Haag",
    "AFC_Hermannstadt": "AFC Hermannstadt",
    "AZAlkmaar": "AZ Alkmaar",
    "AalborgBK": "Aalborg",
    "Academico_Viseu": "Academico Viseu",
    "GFC Ajaccio": "Ajaccio GFCO",
    "Almere_City_FC": "Almere City", "Almere City FC": "Almere City",
    "AmiensSC": "Amiens",
    "Angers SCO": "Angers",
    "Annecy FC": "Annecy",
    "Royal Antwerp": "Antwerp",
    "APOEL Nicosia": "Apoel",
    "Aston_Villa": "Aston Villa",
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao", "Athletic_Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid", "Atletico_Madrid": "Ath Madrid",
    "AJ Auxerre": "Auxerre",
    "Bodrumspor": "BB Bodrumspor", "BB_Bodrumspor": "BB Bodrumspor",
    "FC Barcelona": "Barcelona",
    "SSC Bari": "Bari",
    "SC Bastia": "Bastia",
    "Bayer Leverkusen": "Leverkusen", "Bayer_Leverkusen": "Leverkusen",
    "Beerschot Wilrijk": "Beerschot VA", "Beerschot": "Beerschot VA", "Beerschot_Wilrijk": "Beerschot VA",
    "SL Benfica": "Benfica",
    "Benfica_B": "Benfica B",
    "Real Betis": "Betis",
    "Blackburn Rovers": "Blackburn",
    "Borussia Dortmund": "Dortmund", "Borussia_Dortmund": "Dortmund",
    "Borussia Monchengladbach": "M'gladbach", "Monchengladbach": "M'gladbach", "Borussia_Monchengladbach": "M'gladbach",
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Bristol_City": "Bristol City",
    "Brondby IF": "Brondby",
    "Burgos CF": "Burgos",
    "CD Eldense": "Eldense",
    "Cadiz CF": "Cadiz",
    "SM Caen": "Caen",
    "SC Cambuur": "Cambuur",
    "Cardiff City": "Cardiff",
    "FC Cartagena": "Cartagena",
    "Casa_Pia": "Casa Pia",
    "CD Castellon": "Castellon",
    "Celta": "Celta Vigo", "Celta_Vigo": "Celta Vigo",
    "Cercle_Brugge": "Cercle Brugge",
    "Sporting Charleroi": "Charleroi",
    "GD Chaves": "Chaves",
    "AS Cittadella": "Cittadella",
    "Clermont Foot": "Clermont", "Clermont_Foot": "Clermont",
    "Club_Brugge": "Club Brugge", "ClubBruggeKV": "Club Brugge",
    "Cordoba CF": "Cordoba",
    "Cracovia_Krakow": "Cracovia Krakow",
    "Red Star Belgrade": "Crvena Zvezda", "Crvena_Zvezda": "Crvena Zvezda",
    "De Graafschap": "Graafschap", "De_Graafschap": "Graafschap",
    "Den_Bosch": "Den Bosch", "FC Den Bosch": "Den Bosch",
    "Deportivo Alaves": "Alaves",
    "Deportivo La Coruna": "La Coruna", "Deportivo_La_Coruna": "La Coruna", "Deportivo": "La Coruna",
    "Derby County": "Derby",
    "FC Dordrecht": "Dordrecht",
    "USL Dunkerque": "Dunkerque",
    "SD Eibar": "Eibar",
    "Eintracht Braunschweig": "Braunschweig",
    "Eintracht Frankfurt": "Ein Frankfurt", "EintrachtFrankfurt": "Ein Frankfurt",
    "Elche CF": "Elche",
    "Emmen": "FC Emmen",
    "Espanyol": "Espanol", "RCD Espanyol": "Espanol",
    "Estac Troyes": "Troyes", "ESTAC Troyes": "Troyes",
    "Estoril Praia": "Estoril",
    "Estrela Da Amadora": "Estrela", "CF Estrela Amadora": "Estrela",
    "Zorya": "Zorya Luhansk", "ZoryaLuhansk": "Zorya Luhansk",
    "KAA Gent": "Gent",
    "Legia Warsaw": "Legia Warszawa", "LegiaWarszawa": "Legia Warszawa",
    "Leicester City": "Leicester",
    "Manchester City": "Man City", "Manchester_City": "Man City",
    "Saint Etienne": "St Etienne", "Saint-Étienne": "St Etienne", "AS Saint-Étienne": "St Etienne", "SaintEtienne": "St Etienne",
    "Standard_Liege": "Standard Liege", "Standard": "Standard Liege", # Added Standard mapping
    "Zulte_Waregem": "Zulte Waregem",
    "Westerlo": "KVC Westerlo", "KVCOosterlo": "KVC Westerlo",
    "PSG": "Paris Saint Germain", "Paris SG": "Paris Saint Germain", "ParisSaintGermain": "Paris Saint Germain",
    "AS Eupen": "Eupen", "KAS Eupen": "Eupen", "ASEupen": "Eupen",
    "Sporting Gijon": "Sp Gijon", "Sporting_Gijon": "Sp Gijon", "SpGijon": "Sp Gijon",
    "Werder_Bremen": "Werder Bremen",
    "1. FC Koln": "FC Koln", "Koln": "FC Koln", "1FCKöln": "FC Koln", "1.FCKoln": "FC Koln",
    "1. FC Kaiserslautern": "Kaiserslautern", "1FCKaiserslautern": "Kaiserslautern",
    "1. FC Nürnberg": "Nurnberg", "FCNurnberg": "Nurnberg",
    "Hamburger SV": "Hamburg", "HamburgerSV": "Hamburg",
    "FC Schalke 04": "Schalke 04", "FCSchalke04": "Schalke 04",
    "Hannover 96": "Hannover", "Hannover96": "Hannover",
    "SC Freiburg": "Freiburg", "SCFreiburg": "Freiburg",
    "FC St. Pauli": "St Pauli", "FCStPauli": "St Pauli",
    "Stoke City": "Stoke", "StokeCity": "Stoke",
    "West Bromwich Albion": "West Brom", "WestBrom": "West Brom",
    "FSV Mainz 05": "Mainz 05", "FSVMainz05": "Mainz 05",
    "VfB Stuttgart": "Stuttgart", "VfBStuttgart": "Stuttgart",
    "Manchester United": "Man United", "ManchesterUtd": "Man United", "ManchesterUnited": "Man United",
    "Naval 1 de Maio": "Naval", "Naval 1º de Maio": "Naval", "Naval1ºdeMaio": "Naval",
    "União de Leiria": "Leiria", "UniãoLeiria": "Leiria", "UniãodeLeiria": "Leiria",
    "Hércules CF": "Hercules", "Hércules": "Hercules",
    "FC Metz": "Metz",
    "Evian Thonon Gaillard FC": "Evian TG",
    "FC Augsburg": "Augsburg",
    "FC Twente": "Twente",
    "FC Utrecht": "Utrecht",
    "FC Volendam": "Volendam",
    "Fortuna Sittard": "For Sittard",
    "Granada CF": "Granada",
    "Grenoble Foot 38": "Grenoble",
    "Hellas Verona": "Verona",
    "Hull City": "Hull",
    "Internazionale": "Inter",
    "İstanbul Başakşehir": "Buyuksehyr", "Istanbul Basaksehir": "Buyuksehyr",
    "Jahn Regensburg": "Regensburg",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    "MVV Maastricht": "MVV",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "PSV": "PSV Eindhoven",
    "Plymouth Argyle": "Plymouth",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "RKC Waalwijk": "Waalwijk",
    "RCD Mallorca": "Mallorca",
    "Racing Santander": "Santander",
    "Rapid Bucharest": "Rapid Bucuresti",
    "Rayo Vallecano": "Vallecano",
    "Real Oviedo": "Oviedo",
    "Real Valladolid": "Valladolid",
    "Real Zaragoza": "Zaragoza",
    "SC Paderborn 07": "Paderborn",
    "SSC Napoli": "Napoli",
    "SV Darmstadt 98": "Darmstadt",
    "SV Elversberg": "Elversberg",
    "Sint-Truiden": "St Truiden",
    "SpVgg Greuther Furth": "Greuther Furth",
    "Sporting CP": "Sp Lisbon", "Sporting Lisbon": "Sp Lisbon",
    "Spurs": "Tottenham",
    "Stade Brestois 29": "Brest", "Stade Brest": "Brest",
    "Stade Reims": "Reims",
    "Stade Rennais": "Rennes",
    "U Cluj": "Universitatea Cluj",
    "VfL Bochum": "Bochum",
    "VfL Wolfsburg": "Wolfsburg",
    "Vitoria Guimaraes": "Guimaraes", # Added from common knowledge
    "CD Nacional": "Nacional", # Added from common knowledge
    "Caykur Rizespor": "Rizespor", # Added from common knowledge
    "FC Südtirol": "Sudtirol", # Added from common knowledge
    "1. FC Union Berlin": "Union Berlin", # Added from common knowledge
    "Union Saint-Gilloise": "St. Gilloise", # Added from common knowledge
    "Unione Venezia": "Venezia", # Added from common knowledge
    "1FCHeidenheim": "Heidenheim", # Defaulting to a sensible canonical for now
    "1FCKaiserslautern": "Kaiserslautern", # Already covered
    "1FCKöln": "FC Koln", # Already covered
    "1FCMagdeburg": "Magdeburg",
    "1FCNürnberg": "Nurnberg", # Already covered as Nurnberg
}