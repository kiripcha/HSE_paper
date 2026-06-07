"""
RiskEngine — оркестратор модуля риск-менеджмента.

Единая точка входа для высокоскоростной торговой системы. Объединяет загрузку
данных, модели волатильности, VaR/ES, бэктестинг, оптимизацию портфеля, риск
ликвидности, зависимости (копулы) и адаптивные контроли в согласованный API:

    engine = RiskEngine(RiskConfig())
    engine.load_data()                      # сбор крипто-данных
    report = engine.risk_report(weights)    # VaR/ES/LVaR по портфелю
    ok = engine.pre_trade_check(...)        # предторговый риск-контроль (HFT)
    w  = engine.optimal_weights(...)        # веса с учётом риска
    bt = engine.backtest_var(...)           # валидация модели VaR

Дизайн: «толстый» движок поверх «тонких» модулей. Состояние (данные,
доходности, обученные модели) кэшируется внутри для скорости в реальном времени.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtesting as bt
from . import covariance as cvm
from . import liquidity as liq
from . import stress as st
from . import var_es as ve
from .config import RiskConfig
from .controls import (VolForecaster, drawdown_scale, kelly_weights,
                       vol_target_weights)
from .data import CryptoDataLoader, MultiSourceCryptoLoader, TOP10_POPULAR
from .dependence import StudentTCopula, copula_portfolio_returns
from .evt import conditional_evt, pot_var_es
from .portfolio import (PortfolioOptimizer, beta_covariance, estimate_betas,
                        mean_cov)
from .volatility import GARCHModel, log_returns


@dataclass
class RiskReport:
    asset_risk: pd.DataFrame          # VaR/ES по каждому активу
    portfolio_risk: pd.DataFrame      # VaR/ES портфеля разными методами
    lvar: dict                        # liquidity-adjusted VaR по портфелю
    diversification_ratio: float
    annualized_vol: float
    weights: dict
    horizons: tuple[int, ...]


@dataclass
class PreTradeDecision:
    approved: bool
    reasons: list[str]
    sized_notional: float
    metrics: dict


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None, data_mode: str = "auto"):
        self.cfg = config or RiskConfig()
        np.random.seed(self.cfg.seed)
        self.loader = CryptoDataLoader(universe=self.cfg.universe,
                                       quote=self.cfg.quote, mode=data_mode)
        self.prices: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self._garch_cache: dict = {}
        self.vol_forecaster = VolForecaster(method="ewma", lam=self.cfg.ewma_lambda,
                                            trading_days=self.cfg.trading_days_year)

    # --------------------------------------------------------------------- #
    # Данные
    # --------------------------------------------------------------------- #
    def load_data(self, start: str = "2021-01-01", end: str = "2025-01-01",
                  timeframe: str = "1d") -> "RiskEngine":
        self.prices = self.loader.load_close_panel(start=start, end=end,
                                                   timeframe=timeframe)
        self.returns = log_returns(self.prices)
        return self

    def load_data_multi(self, start: str = "2015-01-01", end: str = "2025-01-01",
                        universe: tuple | None = None,
                        synthetic: bool = False) -> "RiskEngine":
        """Загрузка котировок через мультиисточник (CryptoCompare/Yahoo/...).
        Длинная история (с 2015 г.), фолбэк и дозаполнение пропусков."""
        uni = universe or self.cfg.universe or TOP10_POPULAR
        mode = "synthetic" if synthetic else "auto"
        self.loader = MultiSourceCryptoLoader(universe=uni, mode=mode)
        self.prices = self.loader.load_close_panel(start=start, end=end)
        # ВНИМАНИЕ: у активов разные даты появления; log_returns сохранит NaN
        self.returns = log_returns(self.prices)
        return self

    def set_returns(self, returns: pd.DataFrame) -> "RiskEngine":
        """Позволяет подать готовые доходности (для тестов/бэктеста)."""
        self.returns = returns
        return self

    @property
    def data_source(self) -> str | None:
        return self.loader.source_used

    # --------------------------------------------------------------------- #
    # Модели волатильности (с кэшем)
    # --------------------------------------------------------------------- #
    def garch(self, asset: str, vol: str = "GJR", dist: str = "t", horizon: int = 10):
        key = (asset, vol, dist, horizon)
        if key not in self._garch_cache:
            self._garch_cache[key] = GARCHModel(vol=vol, dist=dist).fit(
                self.returns[asset], horizon=horizon)
        return self._garch_cache[key]

    # --------------------------------------------------------------------- #
    # Риск-отчёт по портфелю
    # --------------------------------------------------------------------- #
    def _weights_vector(self, weights: dict | np.ndarray | None) -> np.ndarray:
        cols = list(self.returns.columns)
        if weights is None:
            return np.ones(len(cols)) / len(cols)
        if isinstance(weights, dict):
            return np.array([weights.get(c, 0.0) for c in cols])
        return np.asarray(weights, dtype=float)

    def portfolio_returns(self, weights=None) -> pd.Series:
        w = self._weights_vector(weights)
        return pd.Series(self.returns.values @ w, index=self.returns.index,
                         name="portfolio")

    def risk_report(self, weights=None, methods=("historical", "garch", "fhs"),
                    use_copula: bool = True) -> RiskReport:
        """Полный риск-отчёт: VaR/ES по активам и портфелю на всех горизонтах."""
        w = self._weights_vector(weights)
        cols = list(self.returns.columns)
        port_ret = self.portfolio_returns(w)

        # --- риск по отдельным активам ---
        asset_rows = []
        for c in cols:
            est = ve.historical_var_es(self.returns[c], self.cfg.var_confidence,
                                       self.cfg.es_confidence, 1)
            asset_rows.append({"asset": c, "VaR_99": est.var, "ES_975": est.es,
                               "daily_vol": self.returns[c].std()})
        asset_risk = pd.DataFrame(asset_rows).set_index("asset")

        # --- риск портфеля разными методами и горизонтами ---
        port_rows = []
        garch_p = None
        if "garch" in methods or "fhs" in methods:
            try:
                garch_p = GARCHModel(vol="GJR", dist="t").fit(
                    port_ret, horizon=max(self.cfg.horizons))
            except Exception:
                garch_p = None
        for h in self.cfg.horizons:
            ests = [ve.historical_var_es(port_ret, self.cfg.var_confidence,
                                         self.cfg.es_confidence, h)]
            mu, sigma = float(port_ret.mean()), float(port_ret.std())
            ests.append(ve.parametric_var_es(mu, sigma, self.cfg.var_confidence,
                                             self.cfg.es_confidence, h, "t", nu=5))
            if garch_p is not None and "garch" in methods:
                ests.append(ve.garch_var_es(garch_p, self.cfg.var_confidence,
                                            self.cfg.es_confidence, h, mu))
            if garch_p is not None and "fhs" in methods:
                ests.append(ve.fhs_var_es(port_ret, garch_p, self.cfg.var_confidence,
                                          self.cfg.es_confidence, h))
            if use_copula and h == 1:
                try:
                    cop = StudentTCopula.fit(self.returns)
                    sim = copula_portfolio_returns(self.returns, w, cop, n_sims=20000,
                                                   seed=self.cfg.seed)
                    qv = np.quantile(sim, 1 - self.cfg.var_confidence)
                    qe = np.quantile(sim, 1 - self.cfg.es_confidence)
                    es = -sim[sim <= qe].mean()
                    ests.append(ve.RiskEstimate(-qv, es, "t_copula", h,
                                                self.cfg.var_confidence,
                                                self.cfg.es_confidence))
                except Exception:
                    pass
            for e in ests:
                port_rows.append({"horizon": h, "method": e.method,
                                  "VaR_99": e.var, "ES_975": e.es})
        portfolio_risk = pd.DataFrame(port_rows)

        # --- LVaR (по портфелю, агрегируя спреды активов как взвешенное среднее) ---
        price_var = ve.historical_var_es(port_ret, self.cfg.var_confidence,
                                         self.cfg.es_confidence, 1).var
        # дефолтные спреды (без стакана): крипта major ~ 1-3 б.п.
        mu_s, sd_s = 0.0003, 0.0002
        lvar = liq.bangia_lvar(price_var, mu_s, sd_s, self.cfg.spread_mult)

        # --- метрики диверсификации ---
        cov = self.returns.cov().values
        port_vol = float(np.sqrt(w @ cov @ w))
        weighted_vol = float(w @ np.sqrt(np.diag(cov)))
        dr = weighted_vol / port_vol if port_vol > 0 else 1.0

        return RiskReport(
            asset_risk=asset_risk,
            portfolio_risk=portfolio_risk,
            lvar={"price_var": lvar.price_var, "liquidity_cost": lvar.liquidity_cost,
                  "lvar": lvar.lvar, "liq_share": lvar.cost_share},
            diversification_ratio=dr,
            annualized_vol=port_vol * np.sqrt(self.cfg.trading_days_year),
            weights={c: float(wi) for c, wi in zip(cols, w)},
            horizons=self.cfg.horizons,
        )

    # --------------------------------------------------------------------- #
    # Оптимизация весов с учётом риска
    # --------------------------------------------------------------------- #
    def optimal_weights(self, objective: str = "max_sharpe",
                        constraint: str = "long_only",
                        cov_method: str = "sample",
                        use_adjusted_beta: bool = False) -> dict:
        """
        Возвращает оптимальные веса. cov_method: 'sample' | 'ewma' | 'beta'.
        objective: 'max_sharpe' | 'min_variance' | 'risk_parity'.
        """
        mu, cov = mean_cov(self.returns, annualize=True)
        if cov_method == "beta":
            be = estimate_betas(self.returns, self.cfg.market_proxy)
            cov = beta_covariance(be, use_adjusted=use_adjusted_beta, annualize=True)
        elif cov_method == "ewma":
            from .volatility import ewma_covariance
            cov = ewma_covariance(self.returns, self.cfg.ewma_lambda).values \
                * self.cfg.trading_days_year
        elif cov_method == "shrinkage":      # Ledoit-Wolf -> устойчивость весов
            cov, _ = cvm.ledoit_wolf_cov(self.returns, annualize=True)
        elif cov_method == "constant_corr":  # сжатие к постоянной корреляции
            cov, _ = cvm.constant_correlation_shrinkage(self.returns, annualize=True)
        elif cov_method == "rmt":            # очистка спектра (Marchenko-Pastur)
            cov = cvm.rmt_denoise_cov(self.returns, annualize=True)
        elif cov_method == "dcc":            # динамические корреляции на конец выборки
            cov = cvm.dcc_garch(self.returns).annualized_cov()

        opt = PortfolioOptimizer(mu, cov, names=list(self.returns.columns))
        if objective == "max_sharpe":
            pt = opt.max_sharpe(constraint=constraint)
        elif objective == "min_variance":
            pt = opt.min_variance(constraint=constraint)
        elif objective == "risk_parity":
            pt = opt.risk_parity()
        else:
            raise ValueError(objective)
        return {c: float(w) for c, w in zip(self.returns.columns, pt.weights)}

    # --------------------------------------------------------------------- #
    # Предторговый риск-контроль (HFT)
    # --------------------------------------------------------------------- #
    def pre_trade_check(self, asset: str, notional: float, side: str,
                        adv_usdt: float, current_drawdown: float = 0.0,
                        max_var_budget: float = 0.05,
                        max_participation: float = 0.10) -> PreTradeDecision:
        """
        Быстрая проверка заявки перед отправкой в стакан:
            1. лимит концентрации/ликвидности (участие в ADV);
            2. VaR-бюджет инструмента;
            3. деривингование по просадке;
            4. прогноз воздействия (square-root law).
        Возвращает решение и при необходимости урезанный размер.
        """
        reasons = []
        r = self.returns[asset]
        var = ve.historical_var_es(r, self.cfg.var_confidence,
                                   self.cfg.es_confidence, 1).var
        sigma = float(r.std())

        # 1. участие в обороте
        max_notional_liq = max_participation * adv_usdt
        sized = min(notional, max_notional_liq)
        if sized < notional:
            reasons.append(f"урезано по ликвидности до {sized:,.0f} (ADV-лимит)")

        # 2. VaR-бюджет
        if var > max_var_budget:
            reasons.append(f"VaR актива {var:.2%} > бюджета {max_var_budget:.2%}")

        # 3. drawdown
        dd_mult = drawdown_scale(current_drawdown, dd_limit=0.20)
        sized *= dd_mult
        if dd_mult < 1.0:
            reasons.append(f"деривингование по просадке x{dd_mult:.2f}")

        # 4. ожидаемое воздействие
        impact = liq.square_root_impact(sized / max(adv_usdt, 1e-9) * adv_usdt,
                                        adv_usdt, sigma)
        approved = (var <= max_var_budget) and sized > 0
        return PreTradeDecision(
            approved=bool(approved),
            reasons=reasons or ["ок"],
            sized_notional=float(sized),
            metrics={"asset_var": var, "sigma": sigma,
                     "expected_impact": float(impact) if np.isfinite(impact) else None,
                     "dd_mult": dd_mult},
        )

    def size_position_vol_target(self, weights=None, target_vol: float = 0.20,
                                 max_leverage: float = 3.0) -> dict:
        """Веса портфеля, отмасштабированные под целевую годовую волатильность."""
        w = self._weights_vector(weights)
        cov = self.returns.cov().values * self.cfg.trading_days_year
        w_scaled = vol_target_weights(w, cov, target_vol, max_leverage)
        return {c: float(x) for c, x in zip(self.returns.columns, w_scaled)}

    # --------------------------------------------------------------------- #
    # Хвостовой риск (EVT), декомпозиция риска, стресс-тесты, модельный риск
    # --------------------------------------------------------------------- #
    def tail_risk(self, weights=None, threshold_q: float = 0.90,
                  use_conditional: bool = True) -> dict:
        """EVT-оценка хвостового VaR/ES портфеля (POT + условный GARCH-EVT)."""
        port = self.portfolio_returns(weights)
        pot = pot_var_es(port, self.cfg.var_confidence, self.cfg.es_confidence,
                         threshold_q)
        out = {"pot": pot}
        if use_conditional:
            try:
                g = GARCHModel(vol="GJR", dist="t").fit(port, horizon=1)
                out["conditional_evt"] = conditional_evt(
                    port, g, self.cfg.var_confidence, self.cfg.es_confidence,
                    threshold_q)
            except Exception:
                pass
        return out

    def risk_attribution(self, weights=None, method: str = "gaussian") -> pd.DataFrame:
        """Декомпозиция VaR/ES портфеля на вклады активов (принцип Эйлера)."""
        w = self._weights_vector(weights)
        return ve.component_var_es(w, self.returns, self.cfg.var_confidence,
                                   self.cfg.es_confidence, method=method)

    def stress_test(self, weights=None, plausibility_k: float = 3.0) -> dict:
        """Набор стресс-тестов портфеля: исторические, корреляционный, worst-case."""
        w = self._weights_vector(weights)
        mu, cov = mean_cov(self.returns, annualize=False)
        return {
            "worst_historical": st.historical_scenarios(self.returns, w, horizon=1, top=5),
            "named_crashes": st.named_crypto_crashes(self.returns, w),
            "correlation_stress": st.correlation_stress(self.returns, w,
                                                        var_alpha=self.cfg.var_confidence),
            "worst_case": st.worst_case_loss(w, mu, cov, plausibility_k),
        }

    def reverse_stress(self, target_loss: float, weights=None) -> dict:
        """Обратный стресс-тест: сценарий, дающий заданный убыток target_loss."""
        w = self._weights_vector(weights)
        mu, cov = mean_cov(self.returns, annualize=False)
        return st.reverse_stress_test(w, mu, cov, target_loss,
                                      names=list(self.returns.columns))

    def model_risk(self, weights=None) -> dict:
        """Модельный риск: разброс оценок VaR между методами + зона Базеля."""
        port = self.portfolio_returns(weights)
        garch_p = GARCHModel(vol="GJR", dist="t").fit(port, horizon=1)
        ests = {
            "historical": ve.historical_var_es(port, self.cfg.var_confidence,
                                               self.cfg.es_confidence, 1).var,
            "parametric_t": ve.parametric_var_es(port.mean(), port.std(),
                                                 self.cfg.var_confidence,
                                                 self.cfg.es_confidence, 1, "t", 5).var,
            "garch": ve.garch_var_es(garch_p, self.cfg.var_confidence,
                                     self.cfg.es_confidence, 1).var,
            "evt": pot_var_es(port, self.cfg.var_confidence,
                              self.cfg.es_confidence).var,
        }
        bt_res = self.backtest_var(weights, method="historical", window=252)
        return bt.model_risk_metrics(ests, bt_res["n_violations"], bt_res["n_obs"])

    # --------------------------------------------------------------------- #
    # Бэктест VaR
    # --------------------------------------------------------------------- #
    def backtest_var(self, weights=None, method: str = "historical",
                     window: int = 252, var_alpha: float | None = None) -> dict:
        """
        Скользящий бэктест VaR портфеля: на каждый день строим VaR по окну
        прошлых данных и сравниваем с реализованной доходностью. Прогоняем
        полный набор тестов покрытия.
        """
        var_alpha = var_alpha or self.cfg.var_confidence
        port_ret = self.portfolio_returns(weights)
        r = port_ret.values
        n = len(r)
        var_series = np.full(n, np.nan)
        for t in range(window, n):
            hist = r[t - window:t]
            if method == "historical":
                est = ve.historical_var_es(hist, var_alpha, self.cfg.es_confidence, 1)
            elif method == "ewma":
                est = ve.ewma_var_es(pd.Series(hist), var_alpha,
                                     self.cfg.es_confidence, 1)
            else:
                mu, sigma = hist.mean(), hist.std()
                est = ve.parametric_var_es(mu, sigma, var_alpha,
                                           self.cfg.es_confidence, 1, "normal")
            var_series[t] = est.var

        valid = ~np.isnan(var_series)
        rr, vv = r[valid], var_series[valid]
        table = bt.run_var_backtests(rr, vv, var_alpha)
        viol = bt.get_violations(rr, vv)
        return {"backtest_table": table,
                "n_obs": int(valid.sum()),
                "n_violations": int(viol.sum()),
                "violation_rate": float(viol.mean()),
                "expected_rate": 1 - var_alpha,
                "var_series": pd.Series(var_series, index=port_ret.index),
                "realized": port_ret}
