## TODO (WIP)

### 1. Retrieve Historical Data
- **Task**: Pull as many historical games as possible from all available leagues in `football_data_db` using the defined endpoints.
- **Status**: WIP
- **Assignee**: Barroca

### 2. Refactor `extract_daily_games.py`
- **Task**: Clean and refactor `extract_daily_games.py`. Implement logic to handle cases where raw data is missing by calculating the engineered features dynamically.
- **Reason**: Improve clarity and maintainability.

(Basicamente o file funciona , mas convem defenir propiamente pois jka tem 2k lines , e so funciona se tivermos statera data , pra backtest e preciso calcular as metricas , a para ,mas assim da pra reverse engenier )

### 3. Enhance Plotting Utilities
- **Task**: Improve the existing visualizations in `plotting_utils.py` for better insights and debugging.
- **Status**: Planned

( O Plot comvem ser ter 
1- UID
2- easy Identification
3- Individual PLot e proper metrics appeling 
4- Cached ( WIP ))

### 4. Optimize Portfolio Generation
- **Task**: Improve the logic for generating paper portfolios in `paper_generator.py`, using proper calibration models.
- **Status**: Assigned

usar Proper kelly crietion model para papeis eficientes
