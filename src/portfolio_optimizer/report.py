"""Grafici e report HTML per il cliente."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, PackageLoader, select_autoescape  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from . import metrics  # noqa: E402
from .backtest import make_strategy, run_backtest  # noqa: E402
from .optimize import (  # noqa: E402
    Portfolio,
    PortfolioOptimizer,
    expected_returns,
    ledoit_wolf_covariance,
)

PALETTE = ["#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0", "#2e75b6", "#a5a5a5"]
STRATEGIES = {
    "Massimo Sharpe": "max_sharpe",
    "Minima volatilità": "min_volatility",
    "Risk parity": "risk_parity",
    "Equipesato": "equal_weight",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


@dataclass
class Analysis:
    """Risultato completo dell'analisi, usato per grafici e report."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    optimizer: PortfolioOptimizer
    shrinkage: float
    portfolios: dict[str, Portfolio]
    frontier: pd.DataFrame
    backtests: dict[str, pd.Series] = field(default_factory=dict)
    backtest_turnover: dict[str, float] = field(default_factory=dict)


def analyze(
    prices: pd.DataFrame,
    risk_free: float = 0.025,
    min_weight: float = 0.0,
    max_weight: float = 0.35,
    lookback: int = 756,
    rebalance_every: int = 63,
    cost_bps: float = 10.0,
) -> Analysis:
    """Stima, ottimizza e (se lo storico basta) esegue il backtest walk-forward."""
    returns = metrics.to_returns(prices).dropna()
    cov, shrinkage = ledoit_wolf_covariance(returns)
    opt = PortfolioOptimizer(expected_returns(returns), cov, risk_free, min_weight, max_weight)

    portfolios = {
        "Massimo Sharpe": opt.max_sharpe(),
        "Minima volatilità": opt.min_volatility(),
        "Risk parity": opt.risk_parity(),
        "Equipesato": opt.equal_weight(),
    }
    result = Analysis(prices, returns, opt, shrinkage, portfolios, opt.efficient_frontier())

    if len(returns) > lookback + rebalance_every:
        for name, kind in STRATEGIES.items():
            bt = run_backtest(
                returns,
                make_strategy(kind, risk_free, min_weight, max_weight),
                lookback=lookback,
                rebalance_every=rebalance_every,
                cost_bps=cost_bps,
            )
            result.backtests[name] = bt.returns
            result.backtest_turnover[name] = bt.turnover
    return result


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def plot_frontier(a: Analysis, n_random: int = 3000, seed: int = 0) -> plt.Figure:
    opt = a.optimizer
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(opt.assets)), n_random)
    rets = w @ opt.mu.to_numpy()
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, opt.cov.to_numpy(), w))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(vols, rets, c=(rets - opt.risk_free) / vols, cmap="Blues", s=6, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Sharpe")
    ax.plot(
        a.frontier["Volatilità"],
        a.frontier["Rendimento"],
        color="#1f4e79",
        lw=2.5,
        label="Frontiera efficiente (con vincoli)",
    )

    tangent = a.portfolios["Massimo Sharpe"]
    x = np.linspace(0, a.frontier["Volatilità"].max(), 50)
    ax.plot(x, opt.risk_free + tangent.sharpe * x, ls="--", color="grey", lw=1, label="Capital market line")

    asset_vol = np.sqrt(np.diag(opt.cov))
    ax.scatter(asset_vol, opt.mu, color="black", marker="D", s=30, zorder=3)
    for name, v, r in zip(opt.assets, asset_vol, opt.mu):
        ax.annotate(name, (v, r), xytext=(5, 3), textcoords="offset points", fontsize=8)

    markers = {"Massimo Sharpe": "*", "Minima volatilità": "o", "Risk parity": "s", "Equipesato": "^"}
    for (name, p), color in zip(a.portfolios.items(), PALETTE[1:]):
        ax.scatter(
            p.volatility,
            p.expected_return,
            marker=markers[name],
            s=180,
            color=color,
            edgecolor="white",
            zorder=4,
            label=name,
        )

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Volatilità attesa annua")
    ax.set_ylabel("Rendimento atteso annuo")
    ax.set_title("Frontiera efficiente e portafogli ottimizzati")
    ax.set_xlim(0, max(asset_vol.max(), vols.max()) * 1.22)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_weights(a: Analysis) -> plt.Figure:
    weights = pd.DataFrame({n: p.weights for n, p in a.portfolios.items()})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    weights.T.plot(kind="barh", stacked=True, ax=ax, color=PALETTE, width=0.65)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Allocazione per strategia")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_backtest(a: Analysis, initial: float = 100_000) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for (name, r), color in zip(a.backtests.items(), PALETTE[1:]):
        ax1.plot(initial * (1 + r).cumprod(), label=name, color=color, lw=1.6)
        ax2.plot(metrics.drawdown(r), color=color, lw=1.1)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€ {v:,.0f}".replace(",", ".")))
    ax1.set_title(f"Backtest walk-forward: evoluzione di € {initial:,.0f}".replace(",", "."))
    ax1.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("Drawdown")
    fig.tight_layout()
    return fig


def plot_correlation(a: Analysis) -> plt.Figure:
    corr = a.returns.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(corr.iat[i, j]) > 0.6 else "black",
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlazioni dei rendimenti giornalieri")
    fig.tight_layout()
    return fig


def _to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1%}".replace(".", ",")


def _num(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.2f}".replace(".", ",")


def _eur(x: float) -> str:
    return f"€ {x:,.0f}".replace(",", ".")


def build_report(
    a: Analysis,
    output: str | Path,
    client_name: str = "Cliente",
    amount: float = 100_000,
    data_note: str = "",
) -> Path:
    """Scrive un report HTML autonomo (grafici inclusi) e ne restituisce il percorso."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    recommended = a.portfolios["Massimo Sharpe"]
    risk_contrib = a.optimizer.risk_contributions(recommended.weights)
    allocation = [
        {
            "asset": asset,
            "weight": _pct(w),
            "amount": _eur(w * amount),
            "risk": _pct(risk_contrib[asset]),
        }
        for asset, w in recommended.weights.sort_values(ascending=False).items()
        if w > 0
    ]

    expected = [
        {"name": n, "ret": _pct(p.expected_return), "vol": _pct(p.volatility), "sharpe": _num(p.sharpe)}
        for n, p in a.portfolios.items()
    ]

    backtest_rows, charts = (
        [],
        {
            "frontier": _to_base64(plot_frontier(a)),
            "weights": _to_base64(plot_weights(a)),
            "correlation": _to_base64(plot_correlation(a)),
        },
    )
    if a.backtests:
        table = metrics.summary(pd.DataFrame(a.backtests), a.optimizer.risk_free)
        for name, row in table.iterrows():
            backtest_rows.append(
                {
                    "name": name,
                    "cagr": _pct(row["Rendimento annuo (CAGR)"]),
                    "vol": _pct(row["Volatilità annua"]),
                    "sharpe": _num(row["Sharpe"]),
                    "mdd": _pct(row["Max drawdown"]),
                    "cvar": _pct(row["CVaR 95% giornaliero"]),
                    "turnover": _pct(a.backtest_turnover[name]),
                }
            )
        charts["backtest"] = _to_base64(plot_backtest(a, amount))
        first = next(iter(a.backtests.values()))
        backtest_period = f"{first.index[0]:%d/%m/%Y} – {first.index[-1]:%d/%m/%Y}"
    else:
        backtest_period = ""

    env = Environment(loader=PackageLoader("portfolio_optimizer"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(
        client=client_name,
        today=date.today().strftime("%d/%m/%Y"),
        amount=_eur(amount),
        period=f"{a.prices.index[0]:%d/%m/%Y} – {a.prices.index[-1]:%d/%m/%Y}",
        n_assets=len(a.optimizer.assets),
        risk_free=_pct(a.optimizer.risk_free),
        max_weight=_pct(max(hi for _, hi in a.optimizer.bounds)),
        shrinkage=f"{a.shrinkage:.3f}".replace(".", ","),
        rec=recommended,
        rec_ret=_pct(recommended.expected_return),
        rec_vol=_pct(recommended.volatility),
        rec_sharpe=_num(recommended.sharpe),
        allocation=allocation,
        expected=expected,
        backtest_rows=backtest_rows,
        backtest_period=backtest_period,
        charts=charts,
        data_note=data_note,
    )
    output.write_text(html, encoding="utf-8")
    return output


def save_charts(a: Analysis, folder: str | Path) -> list[Path]:
    """Salva i grafici come PNG (utile per README e presentazioni)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figures = {
        "frontiera.png": plot_frontier(a),
        "allocazione.png": plot_weights(a),
        "correlazioni.png": plot_correlation(a),
    }
    if a.backtests:
        figures["backtest.png"] = plot_backtest(a)
    paths = []
    for name, fig in figures.items():
        path = folder / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
