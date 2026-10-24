"""Analisi e ottimizzazione di portafoglio per la gestione patrimoniale."""

from .backtest import make_strategy, run_backtest
from .data import download_prices, generate_sample_prices, load_csv
from .optimize import (
    Portfolio,
    PortfolioOptimizer,
    expected_returns,
    ledoit_wolf_covariance,
    sample_covariance,
)

__all__ = [
    "Portfolio",
    "PortfolioOptimizer",
    "download_prices",
    "expected_returns",
    "generate_sample_prices",
    "ledoit_wolf_covariance",
    "load_csv",
    "make_strategy",
    "run_backtest",
    "sample_covariance",
]

__version__ = "0.1.0"
