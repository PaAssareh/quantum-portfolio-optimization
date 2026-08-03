from config import TICKERS, START_DATE, END_DATE, BUDGET, RISK_AVERSION
from data_loader import download_adjusted_close, save_prices_to_csv
from portfolio_math import compute_daily_returns, annualized_mean_returns, annualized_covariance
from classical_baseline import brute_force_cardinality
import os

def main():
    os.makedirs("data/processed", exist_ok=True)

    prices = download_adjusted_close(TICKERS, START_DATE, END_DATE)
    save_prices_to_csv(prices)

    returns = compute_daily_returns(prices)
    mean_returns = annualized_mean_returns(returns).values
    cov_matrix = annualized_covariance(returns).values

    best_selection, best_score = brute_force_cardinality(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        budget=BUDGET,
        risk_aversion=RISK_AVERSION
    )

    selected_assets = [ticker for ticker, flag in zip(prices.columns, best_selection) if flag == 1]

    print("Selected assets:", selected_assets)
    print("Objective score:", best_score)

if __name__ == "__main__":
    main()