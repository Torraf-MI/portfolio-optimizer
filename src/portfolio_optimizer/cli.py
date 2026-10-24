"""Riga di comando: genera il report di allocazione per un cliente.

Esempi::

    portfolio-optimizer --demo --cliente "Mario Rossi" --importo 250000
    portfolio-optimizer --csv data/prezzi.csv --max-peso 0.30
    portfolio-optimizer --ticker VWCE.DE XEON.DE IUSN.DE SGLD.MI --da 2015-01-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data, metrics
from .report import analyze, build_report, save_charts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="portfolio-optimizer", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV di prezzi: colonna data + una colonna per asset")
    src.add_argument("--ticker", nargs="+", help="Ticker Yahoo Finance da scaricare")
    src.add_argument("--demo", action="store_true", help="Usa dati sintetici dimostrativi")
    p.add_argument("--da", default="2015-01-01", help="Data iniziale per --ticker (AAAA-MM-GG)")
    p.add_argument("--risk-free", type=float, default=0.025, help="Tasso privo di rischio annuo")
    p.add_argument("--min-peso", type=float, default=0.0, help="Peso minimo per asset (default 0)")
    p.add_argument("--max-peso", type=float, default=0.35, help="Peso massimo per asset (default 0.35)")
    p.add_argument("--costi-bps", type=float, default=10.0, help="Costi di transazione in punti base")
    p.add_argument("--cliente", default="Cliente", help="Nome del cliente nel report")
    p.add_argument("--importo", type=float, default=100_000, help="Capitale da investire in euro")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    note = ""
    if args.csv:
        prices = data.load_csv(args.csv)
    elif args.ticker:
        prices = data.download_prices(args.ticker, start=args.da)
    else:
        prices = data.generate_sample_prices()
        note = "Analisi svolta su dati sintetici dimostrativi, non su strumenti reali."

    a = analyze(
        prices,
        risk_free=args.risk_free,
        min_weight=args.min_peso,
        max_weight=args.max_peso,
        cost_bps=args.costi_bps,
    )
    path = build_report(a, args.output, client_name=args.cliente, amount=args.importo, data_note=note)
    if args.grafici:
        save_charts(a, args.grafici)

    best = a.portfolios["Massimo Sharpe"]
    print("Portafoglio a massimo Sharpe")
    print(best.weights[best.weights > 0].sort_values(ascending=False).map("{:.1%}".format).to_string())
    print(
        f"\nRendimento atteso {best.expected_return:.2%} | "
        f"volatilità {best.volatility:.2%} | Sharpe {best.sharpe:.2f}"
    )
    if a.backtests:
        print("\nBacktest walk-forward")
        print(metrics.summary(pd.DataFrame(a.backtests), args.risk_free).round(3).to_string())
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
