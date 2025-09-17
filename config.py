import os
from datetime import datetime
import json

# Funktion för att läsa in tickers från en fil
def load_tickers_from_file(filename):
    tickers = []
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, 'r') as file:
            for line in file:
                ticker = line.strip()
                if ticker:
                    tickers.append(ticker)
    except FileNotFoundError:
        print(f"Fel: Filen '{filename}' hittades inte.")
        return []
    return tickers

# Funktion för att läsa in optimala parametrar från en JSON-fil
def load_optimal_params_from_file(filename="optimal_params.json"):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Varning: Filen '{filename}' hittades inte. Laddar tomma parametrar.")
        return {}
    except json.JSONDecodeError:
        print(f"Varning: Fel vid avkodning av JSON från '{filename}'.")
        return {}

# =========================
# Data
# =========================
class DataConfig:
    TICKERS = load_tickers_from_file("tickers.txt")
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')

# =========================
# Databas
# =========================
class DatabaseConfig:
    DB_NAME = "stock_data.db"

# =========================
# Backtest
# =========================
class BacktestConfig:
    INITIAL_CAPITAL = 100000
    BROKERAGE_FIXED_FEE = 69.0
    BROKERAGE_PERCENTAGE = 0.00069
    TRADE_ALLOCATION = 0.8     # andel kapital per handelsdag
    STOP_LOSS_PCT = 0.05       # 5% stop-loss

    REGRESSION_THRESHOLD = 0.01   # köp endast om prognos > 1%
    BINARY_THRESHOLD = 0.55       # köp endast om sannolikhet > 55%

# =========================
# Targets
# =========================
class TargetConfig:
    REGRESSION_DAYS = 7     # Måldagar för regression (förutsäga % avkastning över 7 dagar)
    BINARY_DAYS = 10        # Måldagar för binär target (1 om aktien stiger inom 10 dagar, annars 0)
    RANKING_DAYS = 7        # Måldagar för ranking (t.ex. ranka avkastning 7 dagar framåt)

    RANK_TOP_N = 10         # Antal aktier att hålla i portföljen
    RANK_TOP_PCT = 0.1      # Alternativt: andel av aktier att hålla i portföljen (10%) om inte RANK_TOP_N är satt
    RANK_REBALANCE_DAYS = 5 # Hur ofta portföljen rebalanseras (i dagar)


# =========================
# Paths
# =========================
class PathsConfig:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    TARGETS_DIR = os.path.join(DATA_DIR, "targets")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TARGETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Training
# =========================
class TrainingConfig:
    RANDOM_STATE = 42
    N_JOBS = -1
    CV_SPLITS = 4
    RANDOM_SEARCH_ITERS = 20
    USE_GPU = True
