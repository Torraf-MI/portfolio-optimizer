import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import metrics


def test_cagr_of_constant_growth():
    # 252 giorni a rendimento costante che porta a +10% in un anno
    daily = 1.10 ** (1 / 252) - 1
    r = pd.Series([daily] * 252)
    assert metrics.cagr(r) == pytest.approx(0.10)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 90 -> 110: drawdown massimo = 90/120 - 1 = -25%
    prices = pd.Series([100, 120, 90, 110], dtype=float)
    r = metrics.to_returns(prices)
    assert metrics.max_drawdown(r) == pytest.approx(-0.25)


def test_volatility_annualization():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 100_000))
    assert metrics.annualized_volatility(r) == pytest.approx(0.01 * np.sqrt(252), rel=0.01)


def test_var_and_cvar_are_positive_losses_and_ordered():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, 50_000))
    var, cvar = metrics.value_at_risk(r), metrics.conditional_value_at_risk(r)
    assert var == pytest.approx(1.645 * 0.01, rel=0.03)
    assert cvar > var > 0


def test_portfolio_returns_weighted_sum():
    r = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.01]})
    out = metrics.portfolio_returns(r, {"A": 0.25, "B": 0.75})
    assert out.tolist() == pytest.approx([0.025, -0.0025])


def test_summary_has_one_row_per_series():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, (500, 3)), columns=list("XYZ"))
    table = metrics.summary(r, risk_free=0.02)
    assert list(table.index) == ["X", "Y", "Z"]
    assert "Sharpe" in table.columns
