"""Esempio d'uso come libreria: dal CSV di prezzi al report per il cliente.

Esegui dalla cartella del progetto:  python examples/esempio.py
"""

from portfolio_optimizer import PortfolioOptimizer, expected_returns, ledoit_wolf_covariance, load_csv
from portfolio_optimizer.metrics import summary, to_returns
from portfolio_optimizer.report import analyze, build_report

prices = load_csv("data/prezzi_esempio.csv")
returns = to_returns(prices)

# 1. Metriche storiche di ogni asset class
print(summary(returns, risk_free=0.025).round(3))

# 2. Ottimizzazione con vincoli personalizzati: almeno il 20% in obbligazioni governative,
#    massimo il 10% in oro e il 30% su qualsiasi altro asset
cov, shrinkage = ledoit_wolf_covariance(returns)
opt = PortfolioOptimizer(
    expected_returns(returns),
    cov,
    risk_free=0.025,
    min_weight={"Obbligazioni Gov. EUR": 0.20},
    max_weight={asset: (0.10 if asset == "Oro" else 0.30) for asset in prices.columns},
)
p = opt.max_sharpe()
print(f"\nShrinkage Ledoit-Wolf: {shrinkage:.3f}")
print(p.weights.round(3))
print(f"Rendimento atteso {p.expected_return:.2%}, volatilità {p.volatility:.2%}")

# 3. Portafoglio a volatilità minima per un obiettivo di rendimento del 5%
print(opt.efficient_return(0.05).weights.round(3))

# 4. Report HTML completo con backtest
analysis = analyze(prices, risk_free=0.025, max_weight=0.35)
print(build_report(analysis, "output/report_cliente.html", client_name="Famiglia Bianchi", amount=500_000))
