"""Backtest walk-forward: le strategie vengono ricalibrate solo con i dati passati.

A ogni data di ribilanciamento si stimano i parametri sulla finestra precedente
(``lookback`` giorni), si calcolano i nuovi pesi e li si mantiene fino al
ribilanciamento successivo lasciandoli variare con i prezzi (buy & hold).
I costi di transazione sono applicati sul turnover.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimize import PortfolioOptimizer, expected_returns, ledoit_wolf_covariance

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame  # pesi decisi a ogni ribilanciamento
    turnover: float  # turnover medio per ribilanciamento


def make_strategy(
    kind: str,
    risk_free: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Strategy:
    """Crea una strategia per il backtest: 'max_sharpe', 'min_volatility', 'risk_parity', 'equal_weight'."""

    def strategy(window: pd.DataFrame) -> pd.Series:
        cov, _ = ledoit_wolf_covariance(window)
        opt = PortfolioOptimizer(expected_returns(window), cov, risk_free, min_weight, max_weight)
        if kind == "max_sharpe":
            try:
                return opt.max_sharpe().weights
            except ValueError:  # nessun asset batte il risk free: ripiega sul minimo rischio
                return opt.min_volatility().weights
        if kind == "min_volatility":
            return opt.min_volatility().weights
        if kind == "risk_parity":
            return opt.risk_parity().weights
        if kind == "equal_weight":
            return opt.equal_weight().weights
        raise ValueError(f"Strategia sconosciuta: {kind}")

    return strategy


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    lookback: int = 756,
    rebalance_every: int = 63,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Esegue il backtest e restituisce i rendimenti giornalieri out-of-sample."""
    if len(returns) <= lookback:
        raise ValueError("Storico insufficiente per la finestra di stima richiesta")

    values = returns.to_numpy()
    current = np.zeros(returns.shape[1])
    out = np.empty(len(returns) - lookback)
    decisions, turnovers = {}, []

    for i, t in enumerate(range(lookback, len(returns))):
        if (t - lookback) % rebalance_every == 0:
            target = strategy(returns.iloc[t - lookback : t]).reindex(returns.columns).fillna(0).to_numpy()
            turnover = np.abs(target - current).sum()
            cost = turnover * cost_bps / 10_000
            current = target
            decisions[returns.index[t]] = target
            turnovers.append(turnover)
        else:
            cost = 0.0

        day = float(current @ values[t])
        out[i] = day - cost
        current = current * (1 + values[t]) / (1 + day)  # i pesi derivano con i prezzi

    weights = pd.DataFrame.from_dict(decisions, orient="index", columns=returns.columns)
    # il primo acquisto non è un ribilanciamento: escluso dal turnover medio
    avg_turnover = float(np.mean(turnovers[1:])) if len(turnovers) > 1 else 0.0
    return BacktestResult(pd.Series(out, index=returns.index[lookback:]), weights, avg_turnover)
