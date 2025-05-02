# config.py
import os

# --- Paths ---
# Use absolute paths or paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up 3 levels from utils/config.py
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_CSV_DIR = os.path.join(PROJECT_ROOT, 'football_data_db')  # Directory containing CSV files
UNIFIED_DATA_PATH = os.path.join(DATA_DIR, 'unified_data', 'mongo.parquet')

# MongoDB Configuration
MONGO_URI = 'mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin'  
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

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'unified_data'), exist_ok=True)

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


# --- Feature Engineering ---
ROLLING_WINDOW_SIZE = 10 # Number of games for rolling stats/form
ELO_K_FACTOR = 25
ELO_HOME_ADVANTAGE = 65
ELO_DEFAULT_RATING = 1500
ODDS_COLUMNS = ['B365H', 'B365D', 'B365A'] # Example bookmaker odds columns

# List of features to generate and use in models (can be dynamically selected later)
# Example:
FEATURE_SET = [
    'ImpliedProbH', 'ImpliedProbD', 'ImpliedProbA', 'BookmakerMargin',
    f'Home_Avg_GoalsScored_L{ROLLING_WINDOW_SIZE}', f'Home_Avg_GoalsConceded_L{ROLLING_WINDOW_SIZE}',
    f'Away_Avg_GoalsScored_L{ROLLING_WINDOW_SIZE}', f'Away_Avg_GoalsConceded_L{ROLLING_WINDOW_SIZE}',
    # Add more rolling stats features...
    # f'HomeFormPts_L{ROLLING_WINDOW_SIZE}', f'AwayFormPts_L{ROLLING_WINDOW_SIZE}', # If form calc works
    'HomeEloBefore', 'AwayEloBefore', 'EloDiff'
]


# --- Model Hyperparameters (Defaults or options for tuning) ---
# Example for RandomForest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Example for XGBoost
XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    # Objective/eval_metric often set based on task type in the model class
}


# --- Simulation Settings ---
MC_N_SIMULATIONS = 10000 # For Monte Carlo model


# --- Evaluation / Backtesting ---
ROI_BET_THRESHOLD = 0.05 # Minimum value edge for ROI calculation
ROI_STAKE = 1.0 # Fixed stake per bet for simple ROI calculation


# --- Other Settings ---
RANDOM_SEED = 42 # Global random seed for reproducibility where applicable


print("Configuration loaded.")

# Example of accessing config values in another file:
# import config
# print(config.ROLLING_WINDOW_SIZE)
# rf = RandomForestModel(**config.RF_PARAMS)