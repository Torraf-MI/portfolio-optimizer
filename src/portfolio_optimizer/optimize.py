"""Stima dei parametri e ottimizzazione media-varianza (Markowitz) e risk parity.

L'ottimizzazione usa ``scipy.optimize`` (SLSQP) con vincoli lineari: somma dei
pesi pari a 1 e limiti minimo/massimo per ciascun asset (default: solo
posizioni lunghe, nessuna leva).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .metrics import TRADING_DAYS


# --------------------------------------------------------------------------- #
# Stima dei parametri
# --------------------------------------------------------------------------- #
def expected_returns(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Rendimento atteso annuo come media storica dei rendimenti (aritmetica)."""
    return returns.mean() * periods_per_year


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    return returns.cov() * periods_per_year


def ledoit_wolf_covariance(
    returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS
) -> tuple[pd.DataFrame, float]:
    """Covarianza con shrinkage di Ledoit-Wolf verso una matrice identità scalata.

    Riduce l'errore di stima della covarianza campionaria, che altrimenti porta
    l'ottimizzatore a concentrare il portafoglio su pochi asset. Restituisce la
    matrice annualizzata e l'intensità di shrinkage (tra 0 e 1).
    """
    x = returns.dropna().to_numpy()
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    target_scale = np.trace(sample) / n
    target = target_scale * np.eye(n)

    d2 = np.sum((sample - target) ** 2)
    b2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in x) / t**2
    b2 = min(b2_bar, d2)
    shrinkage = 0.0 if d2 == 0 else b2 / d2

    cov = shrinkage * target + (1 - shrinkage) * sample
    cov_df = pd.DataFrame(cov * periods_per_year, index=returns.columns, columns=returns.columns)
    return cov_df, float(shrinkage)


# --------------------------------------------------------------------------- #
# Ottimizzazione
# --------------------------------------------------------------------------- #
@dataclass
class Portfolio:
    """Un portafoglio ottimizzato con le sue caratteristiche attese (annue)."""

    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float

    def as_dict(self) -> dict:
        return {
            "Rendimento atteso": self.expected_return,
            "Volatilità attesa": self.volatility,
            "Sharpe atteso": self.sharpe,
        }


class PortfolioOptimizer:
    """Ottimizzatore di portafoglio su rendimenti attesi e matrice di covarianza.

    Parametri
    ---------
    mu : rendimenti attesi annui per asset.
    cov : matrice di covarianza annua.
    risk_free : tasso privo di rischio annuo, usato per lo Sharpe.
    min_weight, max_weight : limiti per singolo asset (scalari o dict per asset).
    """

    def __init__(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        risk_free: float = 0.0,
        min_weight: float | dict = 0.0,
        max_weight: float | dict = 1.0,
    ) -> None:
        self.assets = list(mu.index)
        if list(cov.index) != self.assets or list(cov.columns) != self.assets:
            raise ValueError("mu e cov devono avere gli stessi asset nello stesso ordine")
        self.mu = mu.astype(float)
        self.cov = cov.astype(float)
        self.risk_free = float(risk_free)
        self.bounds = self._build_bounds(min_weight, max_weight)

        lows, highs = zip(*self.bounds)
        if sum(lows) > 1 + 1e-9 or sum(highs) < 1 - 1e-9:
            raise ValueError("I limiti sui pesi non permettono un portafoglio che sommi a 1")

    # ----------------------------------------------------------------- utils
    def _build_bounds(self, lo, hi) -> list[tuple[float, float]]:
        def get(value, asset, default):
            return float(value.get(asset, default)) if isinstance(value, dict) else float(value)

        return [(get(lo, a, 0.0), get(hi, a, 1.0)) for a in self.assets]

    def _stats(self, w: np.ndarray) -> tuple[float, float, float]:
        ret = float(w @ self.mu.to_numpy())
        vol = float(np.sqrt(w @ self.cov.to_numpy() @ w))
        sharpe = (ret - self.risk_free) / vol if vol > 0 else float("nan")
        return ret, vol, sharpe

    def _portfolio(self, name: str, w: np.ndarray) -> Portfolio:
        w = np.where(np.abs(w) < 1e-6, 0.0, w)
        w = w / w.sum()
        ret, vol, sharpe = self._stats(w)
        return Portfolio(name, pd.Series(w, index=self.assets, name=name), ret, vol, sharpe)

    def _solve(self, objective, extra_constraints=(), name="") -> np.ndarray:
        n = len(self.assets)
        x0 = np.clip(np.full(n, 1 / n), *np.array(self.bounds).T)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_constraints]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"Ottimizzazione '{name}' non riuscita: {result.message}")
        return result.x

    # -------------------------------------------------------- portafogli
    def min_volatility(self) -> Portfolio:
        """Portafoglio a varianza minima globale."""
        cov = self.cov.to_numpy()
        w = self._solve(lambda w: w @ cov @ w, name="minima volatilità")
        return self._portfolio("Minima volatilità", w)

    def max_sharpe(self) -> Portfolio:
        """Portafoglio tangente: massimo rapporto rendimento in eccesso / volatilità."""
        if (self.mu <= self.risk_free).all():
            raise ValueError("Nessun asset ha rendimento atteso superiore al tasso privo di rischio")

        def neg_sharpe(w):
            ret, vol, _ = self._stats(w)
            return -(ret - self.risk_free) / vol

        w = self._solve(neg_sharpe, name="massimo Sharpe")
        return self._portfolio("Massimo Sharpe", w)

    def efficient_return(self, target_return: float) -> Portfolio:
        """Portafoglio a volatilità minima per un dato rendimento atteso annuo."""
        cov, mu = self.cov.to_numpy(), self.mu.to_numpy()
        lo, hi = self.return_range()
        if not lo - 1e-9 <= target_return <= hi + 1e-9:
            raise ValueError(
                f"Rendimento obiettivo {target_return:.2%} fuori dall'intervallo raggiungibile "
                f"[{lo:.2%}, {hi:.2%}]"
            )
        constraint = {"type": "eq", "fun": lambda w: w @ mu - target_return}
        w = self._solve(lambda w: w @ cov @ w, [constraint], name="rendimento obiettivo")
        return self._portfolio(f"Obiettivo {target_return:.1%}", w)

    def risk_parity(self) -> Portfolio:
        """Portafoglio a contributo di rischio uguale (Equal Risk Contribution).

        Risolve min 0.5·w'Σw − (1/n)·Σ log(w_i) e normalizza: alla soluzione ogni
        asset contribuisce in egual misura alla volatilità totale. I limiti
        min/max per asset non vengono applicati.
        """
        cov = self.cov.to_numpy()
        n = len(self.assets)
        result = minimize(
            lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n,
            np.full(n, 1 / n),
            jac=lambda w: cov @ w - 1 / (n * w),
            method="L-BFGS-B",
            bounds=[(1e-8, None)] * n,
        )
        if not result.success:
            raise RuntimeError(f"Ottimizzazione 'risk parity' non riuscita: {result.message}")
        return self._portfolio("Risk parity", result.x)

    def equal_weight(self) -> Portfolio:
        n = len(self.assets)
        return self._portfolio("Equipesato", np.full(n, 1 / n))

    # ----------------------------------------------------- frontiera
    def return_range(self) -> tuple[float, float]:
        """Rendimento atteso minimo e massimo ottenibili rispettando i vincoli."""
        mu = self.mu.to_numpy()
        n = len(mu)
        kwargs = dict(A_eq=np.ones((1, n)), b_eq=[1.0], bounds=self.bounds, method="highs")
        low = linprog(mu, **kwargs)
        high = linprog(-mu, **kwargs)
        return float(low.fun), float(-high.fun)

    def efficient_frontier(self, points: int = 40) -> pd.DataFrame:
        """Punti della frontiera efficiente: dal minimo rischio al massimo rendimento."""
        start = self.min_volatility().expected_return
        _, end = self.return_range()
        rows = []
        for target in np.linspace(start, end, points):
            try:
                p = self.efficient_return(target)
            except RuntimeError:
                continue
            rows.append({"Rendimento": p.expected_return, "Volatilità": p.volatility, "Sharpe": p.sharpe})
        return pd.DataFrame(rows)

    def risk_contributions(self, weights: pd.Series) -> pd.Series:
        """Quota di volatilità totale attribuibile a ciascun asset (somma = 1)."""
        w = weights.reindex(self.assets).fillna(0.0).to_numpy()
        cov = self.cov.to_numpy()
        contrib = w * (cov @ w)
        return pd.Series(contrib / contrib.sum(), index=self.assets, name="Contributo al rischio")
