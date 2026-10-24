import pandas as pd
import pytest

from portfolio_optimizer import generate_sample_prices, make_strategy, run_backtest
from portfolio_optimizer.cli import main
from portfolio_optimizer.metrics import to_returns
from portfolio_optimizer.report import analyze, build_report


@pytest.fixture(scope="module")
def prices():
    return generate_sample_prices(start="2018-01-01", end="2022-12-31")


def test_sample_prices_shape_and_positive(prices):
    assert prices.shape[1] == 7
    assert (prices > 0).all().all()


def test_backtest_is_out_of_sample(prices):
    returns = to_returns(prices)
    bt = run_backtest(returns, make_strategy("equal_weight"), lookback=252, rebalance_every=63, cost_bps=0)
    # i rendimenti partono solo dopo la finestra di stima
    assert bt.returns.index[0] == returns.index[252]
    assert len(bt.weights) == pytest.approx(len(bt.returns) / 63, abs=1)


def test_equal_weight_backtest_without_costs_matches_manual(prices):
    returns = to_returns(prices).iloc[:400]
    bt = run_backtest(returns, make_strategy("equal_weight"), lookback=100, rebalance_every=1, cost_bps=0)
    manual = returns.iloc[100:].mean(axis=1)
    pd.testing.assert_series_equal(bt.returns, manual, check_names=False)


def test_costs_reduce_performance(prices):
    returns = to_returns(prices)
    strategy = make_strategy("max_sharpe", max_weight=0.4)
    free = run_backtest(returns, strategy, lookback=252, cost_bps=0)
    costly = run_backtest(returns, strategy, lookback=252, cost_bps=50)
    assert (1 + costly.returns).prod() < (1 + free.returns).prod()


def test_report_is_written(prices, tmp_path):
    a = analyze(prices, lookback=252)
    path = build_report(a, tmp_path / "report.html", client_name="Mario Rossi", amount=50_000)
    html = path.read_text(encoding="utf-8")
    assert "Mario Rossi" in html
    assert "data:image/png;base64" in html
    assert "Backtest" in html


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["--demo", "--output", str(out)]) == 0
    assert out.exists()
    assert "Massimo Sharpe" in capsys.readouterr().out
