"""
Дымовые тесты модуля crypto_risk. Запускаются в синтетическом режиме данных
(детерминированно, без сети), проверяют, что весь пайплайн исполняется и
выдаёт осмысленные (финитные, правильного знака) величины.

Запуск:  .venv/bin/python -m pytest tests/test_smoke.py -q
   или:  .venv/bin/python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_risk import RiskConfig, RiskEngine
from crypto_risk.data import synthetic_price_panel
from crypto_risk import var_es as ve
from crypto_risk import backtesting as bt
from crypto_risk import liquidity as liq
from crypto_risk import dependence as dep
from crypto_risk.portfolio import (PortfolioOptimizer, estimate_betas,
                                   beta_covariance, mean_cov,
                                   check_two_fund_theorem)
from crypto_risk.volatility import log_returns, ewma_volatility, GARCHModel


def _data():
    prices = synthetic_price_panel(
        ("BTC", "ETH", "SOL", "XRP", "ADA"), "2021-01-01", "2024-01-01")
    return prices, log_returns(prices)


def test_data_synthetic():
    prices, rets = _data()
    assert prices.shape[0] > 500 and prices.shape[1] == 5
    assert np.isfinite(rets.values).all()
    print("OK data: prices", prices.shape, "returns", rets.shape)


def test_volatility():
    _, rets = _data()
    v = ewma_volatility(rets["BTC"])
    assert (v > 0).all()
    g = GARCHModel(vol="GJR", dist="t").fit(rets["BTC"], horizon=10)
    assert g.nu and g.nu > 2 and len(g.forecast_var) == 10
    print(f"OK volatility: EWMA last={v.iloc[-1]:.4f}, GARCH nu={g.nu:.2f}")


def test_var_es_methods():
    _, rets = _data()
    r = rets["BTC"]
    g = GARCHModel(vol="GJR", dist="t").fit(r, horizon=10)
    tab = ve.compare_methods(r, garch_result=g)
    assert (tab.iloc[:, 2] > 0).all()  # все VaR положительны
    # ES >= VaR (ES толще хвоста)
    assert (tab.iloc[:, 3] >= tab.iloc[:, 2] - 1e-6).all()
    print("OK var_es:\n", tab.to_string(index=False))


def test_backtests():
    _, rets = _data()
    r = rets["BTC"].values
    # «корректный» VaR = параметрический скользящий
    var = np.full(len(r), np.nan)
    for t in range(252, len(r)):
        h = r[t-252:t]
        var[t] = ve.parametric_var_es(h.mean(), h.std(), 0.99, 0.975, 1, "t", 5).var
    m = ~np.isnan(var)
    table = bt.run_var_backtests(r[m], var[m], 0.99)
    assert {"Kupiec POF", "Engle-Manganelli DQ"}.issubset(set(table["test"]))
    print("OK backtests:\n", table[["test", "p_value", "reject_H0"]].to_string(index=False))


def test_portfolio():
    _, rets = _data()
    mu, cov = mean_cov(rets)
    opt = PortfolioOptimizer(mu, cov, names=list(rets.columns))
    for c in ("long_short", "short_limit", "long_only", "min_weight"):
        front = opt.efficient_frontier(n_points=20, constraint=c)
        assert len(front) >= 5
    be = estimate_betas(rets)
    cov_b = beta_covariance(be, use_adjusted=True)
    assert cov_b.shape == cov.shape
    tf = check_two_fund_theorem(mu, cov)
    print(f"OK portfolio: 4 frontiers, beta-cov ok, two-fund max_diff={tf['max_weight_diff']:.2e}")


def test_liquidity():
    lv = liq.bangia_lvar(0.05, 0.0003, 0.0002, 3.0)
    assert lv.lvar > lv.price_var
    ac = liq.AlmgrenChriss(sigma=0.02, eta=1e-6, gamma=1e-7, lam=1e-6)
    sched = ac.schedule(total_shares=1000, horizon=1.0, n_steps=10)
    assert abs(sched.holdings[0] - 1000) < 1e-6 and sched.holdings[-1] < 1e-6
    print(f"OK liquidity: {lv}, exec E_cost={sched.expected_cost:.4f}")


def test_dependence():
    _, rets = _data()
    t = dep.StudentTCopula.fit(rets)
    assert 2 < t.nu < 60
    cmp = dep.compare_dependence_models(rets)
    assert cmp.shape[0] == 10  # C(5,2) пар
    print(f"OK dependence: t-copula nu={t.nu:.2f}\n", cmp.round(3).to_string(index=False))


def test_evt():
    from crypto_risk import evt
    _, rets = _data()
    r = rets["BTC"]
    pot = evt.pot_var_es(r, 0.99, 0.975)
    assert pot.var > 0 and pot.es >= pot.var
    g = GARCHModel(vol="GJR", dist="t").fit(r, horizon=1)
    cevt = evt.conditional_evt(r, g, 0.99, 0.975)
    hill = evt.hill_estimator((-r[r < 0]).values)
    print(f"OK evt: POT VaR={pot.var:.4f} ES={pot.es:.4f} xi={pot.extra['xi']:.3f}; "
          f"cond-EVT VaR={cevt.var:.4f}; Hill alpha={hill:.2f}")


def test_covariance():
    from crypto_risk import covariance as cvm
    _, rets = _data()
    lw, d_lw = cvm.ledoit_wolf_cov(rets)
    cc, d_cc = cvm.constant_correlation_shrinkage(rets)
    rmt = cvm.rmt_denoise_cov(rets)
    dcc = cvm.dcc_garch(rets)
    cn_sample = cvm.condition_number(cvm.sample_cov(rets))
    cn_lw = cvm.condition_number(lw)
    assert lw.shape == (5, 5) and 0 <= d_lw <= 1
    assert dcc.cond_corr_last.shape == (5, 5) and 0 < dcc.a + dcc.b < 1
    print(f"OK covariance: LW δ={d_lw:.2f}, const-corr δ={d_cc:.2f}, "
          f"DCC(a={dcc.a:.3f},b={dcc.b:.3f}); cond# sample={cn_sample:.0f}->LW={cn_lw:.0f}")


def test_stress():
    from crypto_risk import stress as st
    from crypto_risk.portfolio import mean_cov
    _, rets = _data()
    w = np.ones(5) / 5
    mu, cov = mean_cov(rets, annualize=False)
    hist = st.historical_scenarios(rets, w, horizon=1, top=3)
    wc = st.worst_case_loss(w, mu, cov, plausibility_k=3.0)
    rev = st.reverse_stress_test(w, mu, cov, target_loss=0.40, names=list(rets.columns))
    cs = st.correlation_stress(rets, w, target_corr=0.95)
    assert len(hist) == 3 and wc["worst_loss"] > 0
    print(f"OK stress: worst 1d hist={hist.iloc[0,1]:.3f}, "
          f"worst-case(k=3)={wc['worst_loss']:.2%}, "
          f"reverse(-40%) k={rev['required_k']:.2f} driver={rev['main_driver']}, "
          f"corr-stress vol +{cs['vol_increase']:.0%}")


def test_volatility_range_har():
    from crypto_risk import volatility as v
    from crypto_risk.data import CryptoDataLoader
    ld = CryptoDataLoader(("BTC",), mode="synthetic")
    ohlc = ld.load_ohlcv("BTC", "2021-01-01", "2024-01-01")
    pk = v.parkinson_vol(ohlc); gk = v.garman_klass_vol(ohlc)
    yz = v.yang_zhang_vol(ohlc); rs = v.rogers_satchell_vol(ohlc)
    rv = (v.garman_klass_vol(ohlc) ** 2).dropna()
    har = v.har_rv(rv)
    assert (pk.dropna() > 0).all() and har.forecast_vol > 0
    print(f"OK volatility range/HAR: Parkinson last={pk.iloc[-1]:.4f}, "
          f"HAR-RV R2={har.r2:.2f}, forecast σ={har.forecast_vol:.4f}")


def test_risk_attribution():
    _, rets = _data()
    w = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    comp = ve.component_var_es(w, rets, method="gaussian")
    # сумма компонентных VaR ≈ VaR портфеля (свойство Эйлера)
    assert abs(comp["pct_contrib_VaR"].sum() - 1.0) < 1e-6
    comp_h = ve.component_var_es(w, rets, method="historical")
    print(f"OK attribution: top VaR-contributor = {comp['component_VaR'].idxmax()} "
          f"({comp['pct_contrib_VaR'].max():.0%})")


def test_es_backtests_and_model_risk():
    _, rets = _data()
    r = rets["BTC"].values
    var = np.full(len(r), np.nan); es = np.full(len(r), np.nan)
    for t in range(252, len(r)):
        h = r[t-252:t]
        e = ve.parametric_var_es(h.mean(), h.std(), 0.99, 0.975, 1, "t", 5)
        var[t] = e.var; es[t] = e.es
    m = ~np.isnan(var)
    t1 = bt.acerbi_szekely_test1(r[m], var[m], es[m])
    t2 = bt.acerbi_szekely_es(r[m], var[m], es[m])
    mr = bt.model_risk_metrics({"hist": 0.05, "garch": 0.06, "evt": 0.07},
                               n_violations=12, n_obs=1000)
    assert mr["basel_zone"] in ("GREEN", "YELLOW", "RED")
    print(f"OK ES backtests: AS-Test1 p={t1.p_value:.3f}, AS-Test2 p={t2.p_value:.3f}; "
          f"model-risk spread={mr['relative_spread']:.0%} zone={mr['basel_zone']}")


def test_multi_source_parser():
    from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR
    ld = MultiSourceCryptoLoader(universe=("BTC", "ETH"), mode="synthetic")
    panel = ld.load_close_panel("2015-01-01", "2024-01-01", use_cache=False)
    assert panel.shape[1] == 2 and len(panel) > 1000
    rep = ld.coverage_report()
    print(f"OK multi-parser (synthetic): shape={panel.shape}, "
          f"origin={ld.data_origin}, sources={ld.source_used}")


def test_engine_multi():
    from crypto_risk.data import TOP10_POPULAR
    cfg = RiskConfig(universe=TOP10_POPULAR)
    eng = RiskEngine(cfg, data_mode="auto")
    eng.load_data_multi("2015-01-01", "2024-01-01", synthetic=True)
    assert eng.prices.shape[1] == 10
    w = eng.optimal_weights("max_sharpe", "long_only", cov_method="shrinkage")
    assert abs(sum(w.values()) - 1.0) < 1e-3
    print(f"OK engine multi: 10 coins from {eng.prices.index.min().date()}, "
          f"max-Sharpe weights sum={sum(w.values()):.3f}")


def test_engine_end_to_end():
    cfg = RiskConfig(universe=("BTC", "ETH", "SOL", "XRP", "ADA"))
    eng = RiskEngine(cfg, data_mode="synthetic").load_data("2021-01-01", "2024-01-01")
    rep = eng.risk_report(use_copula=True)
    assert rep.annualized_vol > 0 and 0 < rep.diversification_ratio
    w = eng.optimal_weights("max_sharpe", "long_only")
    assert abs(sum(w.values()) - 1.0) < 1e-3
    dec = eng.pre_trade_check("BTC", 5_000_000, "buy", adv_usdt=2e9,
                              current_drawdown=0.05)
    bt_res = eng.backtest_var(method="historical", window=252)
    print(f"OK engine: source={eng.data_source}, ann_vol={rep.annualized_vol:.2%}, "
          f"DR={rep.diversification_ratio:.2f}")
    print(f"   pre-trade approved={dec.approved}, sized={dec.sized_notional:,.0f}")
    print(f"   VaR backtest: viol_rate={bt_res['violation_rate']:.3%} "
          f"(exp {bt_res['expected_rate']:.3%})")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n=== {name} ===")
            fn()
    print("\nВСЕ ДЫМОВЫЕ ТЕСТЫ ПРОЙДЕНЫ")
