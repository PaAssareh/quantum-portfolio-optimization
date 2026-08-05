from typing import Tuple
import pandas as pd


def detect_regime(
    prices: pd.DataFrame,
    window: int = 20,
    vol_quantile: float = 0.5,
) -> Tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna(how="all")
    rolling_return = returns.rolling(window).mean().mean(axis=1)
    rolling_vol = returns.rolling(window).std().mean(axis=1)
    vol_threshold = rolling_vol.quantile(vol_quantile)

    regimes = pd.Series(index=rolling_return.index, dtype="object")
    regimes[(rolling_return > 0) & (rolling_vol <= vol_threshold)] = "bull"
    regimes[(rolling_return <= 0) | (rolling_vol > vol_threshold)] = "bear"
    regimes = regimes.fillna("bear")

    features = pd.DataFrame(
        {
            "rolling_return": rolling_return,
            "rolling_vol": rolling_vol,
            "vol_threshold": vol_threshold,
        }
    )
    return regimes, features


def regime_parameters(regime: str) -> dict:
    regime = str(regime).lower()
    if regime == "bull":
        return {
            "lambda_risk": 0.8,
            "P_card": 10.0,
            "P_turn": 1.0,
            "P_turn_quad": 1.5,
            "P_sector": 20.0,
        }
    return {
        "lambda_risk": 1.2,
        "P_card": 12.0,
        "P_turn": 2.0,
        "P_turn_quad": 3.0,
        "P_sector": 25.0,
    }