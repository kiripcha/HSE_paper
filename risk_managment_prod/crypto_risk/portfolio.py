"""
Оптимизация и анализ криптовалютного портфеля (теория Марковица + расширения).

Закрывает задания проектной работы №1 (управление портфелем), адаптированные
под крипту:

    * границы эффективных портфелей при различных ограничениях:
        - короткие продажи без ограничений (перпы);
        - короткие продажи с лимитом (|w_i| <= 0.25 на короткую ногу);
        - short запрещён (спот, w_i >= 0);
        - минимальная доля 2% в каждый актив;
    * ковариация на основе исторических и скорректированных бет (market model),
      где рыночный индекс крипты — BTC;
    * портфель минимальной дисперсии, касательный (max Sharpe), риск-паритет (ERC);
    * наиболее рискованный портфель и Монте-Карло граница;
    * проверка two-fund theorem Блэка (1972).

Доходности подаются дневными; аннуализация — по 365 дням (крипта 24/7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MARKET_PROXY, RANDOM_SEED, TRADING_DAYS_YEAR

try:
    import cvxpy as cp  # type: ignore

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False


# --------------------------------------------------------------------------- #
# Оценка входных данных оптимизатора
# --------------------------------------------------------------------------- #
def mean_cov(returns: pd.DataFrame, annualize: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    """Вектор средних доходностей и ковариационная матрица (аннуализированные)."""
    mu = returns.mean().values
    cov = returns.cov().values
    if annualize:
        mu = mu * TRADING_DAYS_YEAR
        cov = cov * TRADING_DAYS_YEAR
    return mu, cov


@dataclass
class BetaEstimates:
    betas: pd.Series           # исторические бета к рынку
    adjusted_betas: pd.Series  # скорректированные (Blume): 0.67*b + 0.33
    alphas: pd.Series
    resid_var: pd.Series       # дисперсия идиосинкратических остатков
    market_var: float


def estimate_betas(returns: pd.DataFrame, market: str = MARKET_PROXY) -> BetaEstimates:
    """
    Оценка исторических β по рыночной модели  r_i = a_i + b_i * r_m + e_i.
    Скорректированные β по Блюму: β_adj = 0.67·β + 0.33 (регрессия к 1).
    """
    rm = returns[market].values
    var_m = rm.var(ddof=1)
    betas, alphas, resid = {}, {}, {}
    for col in returns.columns:
        ri = returns[col].values
        cov_im = np.cov(ri, rm, ddof=1)[0, 1]
        b = cov_im / var_m
        a = ri.mean() - b * rm.mean()
        e = ri - (a + b * rm)
        betas[col] = b
        alphas[col] = a
        resid[col] = e.var(ddof=1)
    betas = pd.Series(betas)
    return BetaEstimates(
        betas=betas,
        adjusted_betas=0.67 * betas + 0.33,
        alphas=pd.Series(alphas),
        resid_var=pd.Series(resid),
        market_var=float(var_m),
    )


def beta_covariance(beta_est: BetaEstimates, use_adjusted: bool = False,
                    annualize: bool = True) -> np.ndarray:
    """
    Ковариация по однофакторной модели:
        Σ = β β' σ_m²  +  diag(σ²_idiosyncratic).
    Это даёт более устойчивую (менее шумную) Σ, чем выборочная — задания 13-19.
    """
    b = (beta_est.adjusted_betas if use_adjusted else beta_est.betas).values
    sm2 = beta_est.market_var
    cov = np.outer(b, b) * sm2 + np.diag(beta_est.resid_var.values)
    if annualize:
        cov = cov * TRADING_DAYS_YEAR
    return cov


# --------------------------------------------------------------------------- #
# Оптимизатор портфеля
# --------------------------------------------------------------------------- #
@dataclass
class FrontierPoint:
    weights: np.ndarray
    ret: float
    vol: float
    sharpe: float


class PortfolioOptimizer:
    """
    Оптимизатор Марковица с настраиваемыми ограничениями.

    constraint:
        'long_short'   — короткие без ограничений;
        'short_limit'  — w_i >= -limit (доход от короткой ноги <= limit*капитала);
        'long_only'    — w_i >= 0;
        'min_weight'   — w_i >= min_w (>0), сумма = 1.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
                 names: list[str] | None = None):
        self.mu = np.asarray(mu, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.rf = rf
        self.n = len(mu)
        self.names = names or [f"A{i}" for i in range(self.n)]

    # ---- внутренняя постановка задачи ---------------------------------- #
    def _bounds(self, constraint: str, short_limit: float, min_w: float):
        if constraint == "long_short":
            return (-np.inf, np.inf)
        if constraint == "short_limit":
            return (-short_limit, np.inf)
        if constraint == "long_only":
            return (0.0, np.inf)
        if constraint == "min_weight":
            return (min_w, np.inf)
        raise ValueError(f"unknown constraint {constraint}")

    def min_variance(self, target_return: float | None = None,
                     constraint: str = "long_only", short_limit: float = 0.25,
                     min_w: float = 0.02) -> FrontierPoint:
        """Портфель минимальной дисперсии (опц. при целевой доходности)."""
        if not _HAS_CVXPY:
            return self._min_variance_analytic(target_return)
        w = cp.Variable(self.n)
        lo, hi = self._bounds(constraint, short_limit, min_w)
        cons = [cp.sum(w) == 1]
        if np.isfinite(lo):
            cons.append(w >= lo)
        if np.isfinite(hi):
            cons.append(w <= hi)
        if target_return is not None:
            cons.append(self.mu @ w >= target_return)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(self.cov))), cons)
        prob.solve()
        return self._point(np.asarray(w.value).ravel())

    def _min_variance_analytic(self, target_return=None) -> FrontierPoint:
        inv = np.linalg.pinv(self.cov)
        ones = np.ones(self.n)
        if target_return is None:
            w = inv @ ones / (ones @ inv @ ones)
        else:
            a = ones @ inv @ ones
            b = ones @ inv @ self.mu
            c = self.mu @ inv @ self.mu
            d = a * c - b * b
            lam = (c - b * target_return) / d
            gam = (a * target_return - b) / d
            w = inv @ (lam * ones + gam * self.mu)
        return self._point(w)

    def max_sharpe(self, constraint: str = "long_only", short_limit: float = 0.25,
                   min_w: float = 0.02) -> FrontierPoint:
        """Касательный портфель (максимальный коэффициент Шарпа)."""
        if not _HAS_CVXPY or constraint == "long_short":
            inv = np.linalg.pinv(self.cov)
            excess = self.mu - self.rf
            w = inv @ excess
            w = w / w.sum()
            return self._point(w)
        # для ограниченного случая — перебор по границе и выбор max Sharpe
        front = self.efficient_frontier(n_points=80, constraint=constraint,
                                        short_limit=short_limit, min_w=min_w)
        best = max(front, key=lambda p: p.sharpe)
        return best

    def max_risk(self, constraint: str = "long_short", short_limit: float = 0.25,
                 cap: float | None = None) -> FrontierPoint:
        """
        Наиболее рискованный портфель (задание 24*): максимизация дисперсии при
        sum(w)=1. Это НЕвыпуклая задача (максимум выпуклой функции), поэтому
        cvxpy неприменим. Используем свойство: максимум выпуклого квадратичного
        функционала над политопом достигается в ВЕРШИНЕ.

            * long-only, sum=1  -> симплекс, вершины = отдельные активы:
              максимум дисперсии = вложить всё в самый волатильный актив;
            * long-short с |w|<=cap, sum=1 -> приближаем направлением старшего
              собственного вектора Σ, проецируем на бокс и ренормируем.
        """
        if constraint == "long_only":
            vols = np.diag(self.cov)
            w = np.zeros(self.n)
            w[int(np.argmax(vols))] = 1.0
            return self._point(w)

        # long-short: старший собственный вектор Σ как направление макс. дисперсии
        cap = cap if cap is not None else (1.0 + short_limit)
        vals, vecs = np.linalg.eigh(self.cov)
        v = vecs[:, -1]
        if v.sum() < 0:
            v = -v
        w = np.clip(v, -cap, cap)
        s = w.sum()
        w = w / s if abs(s) > 1e-8 else np.full(self.n, 1.0 / self.n)
        # сравниваем с лучшей одиночной концентрацией и берём максимум риска
        single = self.max_risk("long_only")
        cand = self._point(w)
        return cand if cand.vol >= single.vol else single

    def risk_parity(self) -> FrontierPoint:
        """
        Портфель равного вклада в риск (Equal Risk Contribution, ERC).
        Решается итеративно (long-only), полезен как робастный бенчмарк для крипты.
        """
        n = self.n
        w = np.ones(n) / n
        for _ in range(500):
            mrc = self.cov @ w               # marginal risk contribution
            rc = w * mrc                     # risk contribution
            target = (w @ self.cov @ w) / n
            w = w * (target / (rc + 1e-12)) ** 0.5
            w = np.clip(w, 1e-6, None)
            w = w / w.sum()
        return self._point(w)

    def efficient_frontier(self, n_points: int = 50, constraint: str = "long_only",
                           short_limit: float = 0.25, min_w: float = 0.02
                           ) -> list[FrontierPoint]:
        """Граница эффективных портфелей: набор min-variance при целевых доходностях."""
        gmv = self.min_variance(None, constraint, short_limit, min_w)
        r_min = gmv.ret
        r_max = self.mu.max() if constraint != "min_weight" else self.mu @ np.full(self.n, 1/self.n) * 1.5
        targets = np.linspace(r_min, r_max * 0.999, n_points)
        pts = []
        for t in targets:
            try:
                p = self.min_variance(t, constraint, short_limit, min_w)
                if p.weights is not None and np.isfinite(p.vol):
                    pts.append(p)
            except Exception:  # noqa: BLE001
                continue
        return pts

    def monte_carlo_frontier(self, n_portfolios: int = 20_000,
                             constraint: str = "long_only",
                             seed: int = RANDOM_SEED) -> pd.DataFrame:
        """
        Граница методом статистических испытаний (задание 23***): случайные веса,
        облако (риск, доходность). Иллюстрирует выпуклую оболочку = границу.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_portfolios):
            if constraint == "long_only":
                w = rng.random(self.n)
                w = w / w.sum()
            else:
                w = rng.normal(0, 1, self.n)
                w = w / np.abs(w).sum()
            p = self._point(w)
            rows.append({"ret": p.ret, "vol": p.vol, "sharpe": p.sharpe})
        return pd.DataFrame(rows)

    # ---- утилиты -------------------------------------------------------- #
    def _point(self, w: np.ndarray) -> FrontierPoint:
        w = np.asarray(w, dtype=float)
        ret = float(self.mu @ w)
        vol = float(np.sqrt(max(w @ self.cov @ w, 0.0)))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
        return FrontierPoint(weights=w, ret=ret, vol=vol, sharpe=sharpe)


# --------------------------------------------------------------------------- #
# Two-fund theorem (Black 1972) — задание 22**
# --------------------------------------------------------------------------- #
def check_two_fund_theorem(mu: np.ndarray, cov: np.ndarray,
                           tol: float = 1e-6) -> dict:
    """
    Проверяет, что любой портфель на границе минимальной дисперсии (без
    ограничений) является линейной комбинацией двух опорных фронт-портфелей.
    Берём два фронтовых портфеля при доходностях r1, r2 и проверяем, что третий
    при r3 совпадает с их выпуклой комбинацией alpha*w1 + (1-alpha)*w2.
    """
    opt = PortfolioOptimizer(mu, cov)
    p1 = opt._min_variance_analytic(target_return=float(np.percentile(mu, 25)))
    p2 = opt._min_variance_analytic(target_return=float(np.percentile(mu, 75)))
    r3 = float(np.percentile(mu, 60))
    p3 = opt._min_variance_analytic(target_return=r3)
    # alpha из условия по доходности
    alpha = (r3 - p2.ret) / (p1.ret - p2.ret)
    combo = alpha * p1.weights + (1 - alpha) * p2.weights
    max_diff = float(np.max(np.abs(combo - p3.weights)))
    return {"alpha": alpha, "max_weight_diff": max_diff,
            "holds": max_diff < tol * 1e3,
            "note": "веса фронт-портфеля = линейная комбинация двух опорных"}
