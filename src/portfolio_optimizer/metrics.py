"""Metriche di rendimento e di rischio per asset e portafogli.

Tutte le funzioni lavorano su rendimenti semplici giornalieri (pandas Series o
DataFrame) e annualizzano con ``periods_per_year`` (default 252 giorni di borsa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Converte una serie di prezzi in rendimenti semplici, eliminando il primo NaN."""
    return prices.sort_index().pct_change().dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | dict) -> pd.Series:
    """Rendimenti giornalieri di un portafoglio ribilanciato ogni giorno ai pesi dati."""
    w = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    return returns.mul(w, axis=1).sum(axis=1).rename("portafoglio")


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Tasso di crescita annuo composto (rendimento geometrico annualizzato)."""
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    growth = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe ratio annualizzato; ``risk_free`` è un tasso annuo (es. 0.03 = 3%)."""
    excess = returns - risk_free / periods_per_year
    vol = excess.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Come lo Sharpe, ma penalizza solo la volatilità dei rendimenti negativi."""
    excess = returns - risk_free / periods_per_year
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    if downside == 0:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def drawdown(returns: pd.Series) -> pd.Series:
    """Perdita percentuale dal massimo precedente, giorno per giorno."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """VaR storico giornaliero, espresso come perdita positiva."""
    return float(-np.quantile(returns.dropna(), 1 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """CVaR (Expected Shortfall) storico: perdita media oltre il VaR."""
    returns = returns.dropna()
    threshold = np.quantile(returns, 1 - level)
    return float(-returns[returns <= threshold].mean())


def summary(
    returns: pd.DataFrame | pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Tabella riassuntiva delle metriche, una riga per serie."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        rows[name] = {
            "Rendimento annuo (CAGR)": cagr(r, periods_per_year),
            "Volatilità annua": annualized_volatility(r, periods_per_year),
            "Sharpe": sharpe_ratio(r, risk_free, periods_per_year),
            "Sortino": sortino_ratio(r, risk_free, periods_per_year),
            "Max drawdown": max_drawdown(r),
            "VaR 95% giornaliero": value_at_risk(r),
            "CVaR 95% giornaliero": conditional_value_at_risk(r),
        }
    return pd.DataFrame(rows).T
