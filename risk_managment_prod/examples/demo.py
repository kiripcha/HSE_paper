"""
Демонстрация модуля crypto_risk «end-to-end».

Запуск:
    .venv/bin/python examples/demo.py              # авто: биржа -> фолбэк
    .venv/bin/python examples/demo.py --synthetic  # детерминированный оффлайн

Сценарий повторяет работу риск-контура HFT-системы: сбор данных -> риск-отчёт
по портфелю (VaR/ES/LVaR) -> оптимизация весов -> предторговая проверка ->
бэктест модели VaR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_risk import RiskConfig, RiskEngine


def main(mode: str) -> None:
    cfg = RiskConfig(universe=("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
                               "AVAX", "LINK", "LTC"),
                     capital=1_000_000)
    eng = RiskEngine(cfg, data_mode=mode).load_data("2021-01-01", "2025-01-01")
    print(f"Источник данных: {eng.data_source}")
    print(f"Период: {eng.prices.index.min().date()}..{eng.prices.index.max().date()}, "
          f"наблюдений: {len(eng.prices)}\n")

    # 1) Оптимальные веса (max Sharpe, long-only)
    weights = eng.optimal_weights("max_sharpe", "long_only")
    print("=== Оптимальные веса (max Sharpe, long-only) ===")
    for a, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 1e-3:
            print(f"  {a:5} {w:6.1%}")

    # 2) Риск-отчёт
    rep = eng.risk_report(weights)
    print(f"\n=== Риск-отчёт портфеля ===")
    print(f"Годовая волатильность : {rep.annualized_vol:.1%}")
    print(f"Коэф. диверсификации  : {rep.diversification_ratio:.2f}")
    print(f"LVaR (1д, 99%)        : {rep.lvar['lvar']:.2%} "
          f"(ценовой {rep.lvar['price_var']:.2%} + ликв. {rep.lvar['liquidity_cost']:.2%})")
    print("\nVaR/ES портфеля разными методами:")
    print(rep.portfolio_risk.round(4).to_string(index=False))

    # 3) Предторговая проверка крупной заявки
    print("\n=== Предторговый риск-контроль (заявка BTC на $5M) ===")
    dec = eng.pre_trade_check("BTC", 5_000_000, "buy", adv_usdt=2e10,
                              current_drawdown=0.08)
    print(f"Одобрено: {dec.approved}, размер: ${dec.sized_notional:,.0f}")
    print("Причины:", "; ".join(dec.reasons))

    # 4) Бэктест VaR
    print("\n=== Бэктест VaR (historical, окно 252) ===")
    bt = eng.backtest_var(weights, method="historical", window=252)
    print(f"Пробои: {bt['n_violations']}/{bt['n_obs']} "
          f"= {bt['violation_rate']:.2%} (ожидаемо {bt['expected_rate']:.2%})")
    print(bt["backtest_table"][["test", "p_value", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="детерминированный оффлайн-режим")
    args = parser.parse_args()
    main("synthetic" if args.synthetic else "auto")
