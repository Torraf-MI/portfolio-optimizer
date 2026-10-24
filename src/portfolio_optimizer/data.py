"""Caricamento dei prezzi: da file CSV, da Yahoo Finance o dati dimostrativi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Universo dimostrativo: asset class tipiche di un portafoglio di un cliente privato.
# Parametri annui (rendimento, volatilità) indicativi, non previsioni.
SAMPLE_ASSETS = {
    "Azionario Globale": (0.075, 0.16),
    "Azionario Europa": (0.065, 0.18),
    "Azionario Emergenti": (0.070, 0.21),
    "Obbligazioni Gov. EUR": (0.020, 0.06),
    "Obbligazioni Corporate EUR": (0.030, 0.07),
    "Oro": (0.050, 0.15),
    "Immobiliare quotato (REIT)": (0.055, 0.19),
}

SAMPLE_CORRELATION = np.array(
    [
        [1.00, 0.88, 0.75, -0.05, 0.25, 0.10, 0.70],
        [0.88, 1.00, 0.70, 0.00, 0.30, 0.05, 0.68],
        [0.75, 0.70, 1.00, 0.00, 0.25, 0.20, 0.60],
        [-0.05, 0.00, 0.00, 1.00, 0.75, 0.25, 0.15],
        [0.25, 0.30, 0.25, 0.75, 1.00, 0.20, 0.35],
        [0.10, 0.05, 0.20, 0.25, 0.20, 1.00, 0.10],
        [0.70, 0.68, 0.60, 0.15, 0.35, 0.10, 1.00],
    ]
)


def load_csv(path: str | Path, date_column: str | None = None) -> pd.DataFrame:
    """Legge un CSV di prezzi: una colonna data e una colonna per ogni asset."""
    df = pd.read_csv(path)
    date_column = date_column or df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    return df.set_index(date_column).sort_index().astype(float)


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Scarica prezzi di chiusura rettificati da Yahoo Finance (richiede ``yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise ImportError("Installa yfinance: pip install 'portfolio-optimizer[yahoo]'") from exc

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("Nessun dato scaricato: verifica i ticker e la connessione")
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all").ffill().dropna()


def generate_sample_prices(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Genera prezzi sintetici ma realistici (moto browniano geometrico correlato).

    Servono a provare il progetto senza connessione: i numeri non rappresentano
    l'andamento reale di alcuno strumento.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    mu = np.array([m for m, _ in SAMPLE_ASSETS.values()])
    vol = np.array([v for _, v in SAMPLE_ASSETS.values()])
    cov_daily = np.outer(vol, vol) * SAMPLE_CORRELATION / 252

    shocks = rng.multivariate_normal(np.zeros(len(mu)), cov_daily, size=len(dates) - 1)
    # Code grasse: qualche giornata di stress di mercato
    shocks[rng.random(len(shocks)) < 0.01] *= 2.5
    # Riallinea media e volatilità del campione ai parametri scelti, così l'esempio
    # resta realistico qualunque sia il seme casuale
    shocks = (shocks - shocks.mean(axis=0)) / shocks.std(axis=0) * vol / np.sqrt(252)
    log_returns = shocks + (mu - 0.5 * vol**2) / 252

    paths = 100 * np.exp(np.vstack([np.zeros(len(mu)), np.cumsum(log_returns, axis=0)]))
    return pd.DataFrame(paths, index=dates, columns=list(SAMPLE_ASSETS)).round(4)
