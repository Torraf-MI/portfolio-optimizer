import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import PortfolioOptimizer, generate_sample_prices, ledoit_wolf_covariance
from portfolio_optimizer.metrics import to_returns

ASSETS = ["A", "B", "C"]


@pytest.fixture
def optimizer():
    mu = pd.Series([0.08, 0.05, 0.02], index=ASSETS)
    vol = np.array([0.20, 0.12, 0.05])
    corr = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 1.0]])
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=ASSETS, columns=ASSETS)
    return PortfolioOptimizer(mu, cov, risk_free=0.01)


def test_weights_sum_to_one_and_respect_bounds(optimizer):
    for p in (optimizer.max_sharpe(), optimizer.min_volatility(), optimizer.risk_parity()):
        assert p.weights.sum() == pytest.approx(1.0)
        assert (p.weights >= -1e-9).all()


def test_min_volatility_beats_every_single_asset(optimizer):
    p = optimizer.min_volatility()
    assert p.volatility <= np.sqrt(np.diag(optimizer.cov)).min() + 1e-9


def test_max_sharpe_is_best_on_frontier(optimizer):
    best = optimizer.max_sharpe().sharpe
    frontier = optimizer.efficient_frontier(points=25)
    assert best >= frontier["Sharpe"].max() - 1e-4


def test_two_asset_min_variance_closed_form():
    # con due asset incorrelati w_A = σB² / (σA² + σB²)
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    opt = PortfolioOptimizer(pd.Series([0.1, 0.05], index=["A", "B"]), cov)
    assert opt.min_volatility().weights["A"] == pytest.approx(0.2, abs=1e-4)


def test_max_weight_constraint(optimizer):
    capped = PortfolioOptimizer(optimizer.mu, optimizer.cov, 0.01, max_weight=0.4)
    for p in (capped.max_sharpe(), capped.min_volatility()):
        assert p.weights.max() <= 0.4 + 1e-6


def test_infeasible_bounds_raise(optimizer):
    with pytest.raises(ValueError):
        PortfolioOptimizer(optimizer.mu, optimizer.cov, max_weight=0.2)


def test_efficient_return_hits_target(optimizer):
    p = optimizer.efficient_return(0.05)
    assert p.expected_return == pytest.approx(0.05, abs=1e-6)


def test_efficient_return_out_of_range(optimizer):
    with pytest.raises(ValueError):
        optimizer.efficient_return(0.20)


def test_risk_parity_equal_contributions(optimizer):
    p = optimizer.risk_parity()
    contrib = optimizer.risk_contributions(p.weights)
    assert contrib.to_numpy() == pytest.approx(np.full(3, 1 / 3), abs=1e-4)


def test_ledoit_wolf_shrinkage_in_unit_interval():
    returns = to_returns(generate_sample_prices(start="2020-01-01", end="2021-12-31"))
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert 0 <= shrinkage <= 1
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert list(cov.columns) == list(returns.columns)
