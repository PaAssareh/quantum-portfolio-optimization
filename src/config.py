TICKERS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "NVDA", "PG", "XOM"]

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

BUDGET = 4
RISK_AVERSION = 0.5
TRANSACTION_COST_PENALTY = 0.02

DATA_DIR = "data/processed"
PRICES_FILE = f"{DATA_DIR}/adj_close.csv"
RESULTS_FILE = f"{DATA_DIR}/baseline_results.csv"
COMPARISON_RESULTS_FILE = f"{DATA_DIR}/comparison_results.csv"
SUMMARY_RESULTS_FILE = f"{DATA_DIR}/summary_results.csv"
CHART_FILE = f"{DATA_DIR}/comparison_chart.png"

SECTOR_MAP = {
    "AAPL": "Technology",
    "AMZN": "Consumer",
    "GOOGL": "Technology",
    "JNJ": "Healthcare",
    "JPM": "Financials",
    "KO": "ConsumerDefensive",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "PG": "ConsumerDefensive",
    "XOM": "Energy",
}

SECTOR_MAX_WEIGHTS = {
    "Technology": 0.50,
    "Consumer": 0.25,
    "Healthcare": 0.25,
    "Financials": 0.25,
    "ConsumerDefensive": 0.25,
    "Energy": 0.20,
}

PREVIOUS_WEIGHTS = {
    "AAPL": 0.10,
    "AMZN": 0.10,
    "GOOGL": 0.10,
    "JNJ": 0.10,
    "JPM": 0.10,
    "KO": 0.10,
    "MSFT": 0.10,
    "NVDA": 0.10,
    "PG": 0.10,
    "XOM": 0.10,
}