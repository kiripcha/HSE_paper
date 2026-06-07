"""
Устойчивая оценка ковариационной матрицы.

Проблема (теория из RMGuide / Portfolio Concepts): выборочная ковариация при
N активов и T наблюдений содержит O(N²) параметров и сильно зашумлена, особенно
при T, сравнимом с N. Оптимизатор Марковица «выедает» этот шум (error
maximization) — веса нестабильны от окна к окну. Решения:

    * Ledoit-Wolf shrinkage — сжатие выборочной Σ к структурированной цели
      (масштабированная единичная или модель постоянной корреляции); оптимальная
      интенсивность δ оценивается аналитически => меньше шума, лучшая
      обусловленность;
    * RMT denoising (Marchenko-Pastur) — «срезание» собственных значений,
      попадающих в шумовую зону спектра случайной матрицы;
    * DCC-GARCH (Engle 2002) — ДИНАМИЧЕСКИЕ условные корреляции: ловят рост
      корреляций в стрессе (крипта «коррелирует к 1» на обвалах), что повышает
      точность портфельного риска.

Все функции возвращают numpy-матрицы (дневные, если не указано иное).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_YEAR

try:
    from sklearn.covariance import LedoitWolf, OAS  # type: ignore
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

try:
    from arch import arch_model  # type: ignore
    _HAS_ARCH = True
except Exception:  # pragma: no cover
    _HAS_ARCH = False


# --------------------------------------------------------------------------- #
# Shrinkage
# --------------------------------------------------------------------------- #
def sample_cov(returns: pd.DataFrame, annualize: bool = False) -> np.ndarray:
    cov = returns.cov().values
    return cov * TRADING_DAYS_YEAR if annualize else cov


def ledoit_wolf_cov(returns: pd.DataFrame, annualize: bool = False) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage к масштабированной единичной матрице (sklearn)."""
    if not _HAS_SK:
        return sample_cov(returns, annualize), 0.0
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_
    if annualize:
        cov = cov * TRADING_DAYS_YEAR
    return cov, float(lw.shrinkage_)


def oas_cov(returns: pd.DataFrame, annualize: bool = False) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage — часто точнее LW при гауссовости."""
    if not _HAS_SK:
        return sample_cov(returns, annualize), 0.0
    oas = OAS().fit(returns.values)
    cov = oas.covariance_
    if annualize:
        cov = cov * TRADING_DAYS_YEAR
    return cov, float(oas.shrinkage_)


def constant_correlation_shrinkage(returns: pd.DataFrame, annualize: bool = False
                                   ) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) «Honey, I Shrunk the Sample Covariance Matrix»: сжатие к
    модели ПОСТОЯННОЙ корреляции (все попарные корреляции = средняя r̄).
    Аналитическая интенсивность сжатия δ*. Особенно уместно для крипты, где
    корреляции высоки и близки.
    """
    X = returns.values
    t, n = X.shape
    X = X - X.mean(0)
    S = np.cov(X, rowvar=False, ddof=0)
    var = np.diag(S)
    std = np.sqrt(var)
    # цель F: постоянная корреляция
    corr = S / np.outer(std, std)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, var)

    # оптимальная интенсивность (упрощённая оценка pi/gamma)
    Xc = X
    pi_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi_mat[i, j] = np.mean((Xc[:, i] * Xc[:, j] - S[i, j]) ** 2)
    pi_hat = pi_mat.sum()
    gamma_hat = np.sum((F - S) ** 2)
    # rho (вклад off-diagonal) приближаем диагональю pi (консервативно)
    rho_hat = np.sum(np.diag(pi_mat))
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / t))
    cov = delta * F + (1 - delta) * S
    if annualize:
        cov = cov * TRADING_DAYS_YEAR
    return cov, float(delta)


# --------------------------------------------------------------------------- #
# RMT denoising (Marchenko-Pastur)
# --------------------------------------------------------------------------- #
def rmt_denoise_cov(returns: pd.DataFrame, annualize: bool = False) -> np.ndarray:
    """
    Очистка ковариации через теорию случайных матриц: собственные значения
    корреляционной матрицы ниже верхней границы спектра Марченко-Пастура
    считаются шумом и заменяются их средним; рыночная мода и реальные факторы
    сохраняются. Повышает устойчивость и обусловленность Σ.
    """
    X = returns.values
    t, n = X.shape
    std = X.std(0, ddof=1)
    corr = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(corr)
    q = t / n
    lam_max = (1 + np.sqrt(1 / q)) ** 2  # верхняя граница шумового спектра
    noise = vals < lam_max
    if noise.any():
        vals[noise] = vals[noise].mean()
    corr_clean = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)  # ренормировка диагонали к 1
    cov = corr_clean * np.outer(std, std)
    return cov * TRADING_DAYS_YEAR if annualize else cov


# --------------------------------------------------------------------------- #
# DCC-GARCH (динамические условные корреляции)
# --------------------------------------------------------------------------- #
@dataclass
class DCCResult:
    cond_cov_last: np.ndarray   # условная ковариация на конец выборки (дневная)
    cond_corr_last: np.ndarray
    a: float
    b: float
    avg_corr_path: pd.Series    # средняя попарная корреляция во времени
    names: list[str]

    def annualized_cov(self) -> np.ndarray:
        return self.cond_cov_last * TRADING_DAYS_YEAR


def dcc_garch(returns: pd.DataFrame, a0: float = 0.02, b0: float = 0.95
              ) -> DCCResult:
    """
    DCC-GARCH(1,1) (Engle 2002), двухшаговая оценка:
        1. на каждый ряд — univariate GARCH(1,1) -> σ_{i,t}, стандартизованные
           остатки z_{i,t} = r_{i,t}/σ_{i,t};
        2. динамика квази-корреляции:
              Q_t = (1-a-b)·Q̄ + a·z_{t-1}z_{t-1}' + b·Q_{t-1},
              R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}.
    Параметры (a,b) подбираются по сетке (макс. псевдо-LL корреляции).
    """
    cols = list(returns.columns)
    n = len(cols)
    Z = np.zeros_like(returns.values)
    sigmas = np.zeros_like(returns.values)

    for j, c in enumerate(cols):
        r = returns[c].dropna() * 100
        if _HAS_ARCH:
            res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
            s = (res.conditional_volatility / 100).reindex(returns.index).ffill().bfill()
        else:  # EWMA-фолбэк
            from .volatility import ewma_volatility
            s = ewma_volatility(returns[c]).reindex(returns.index).ffill().bfill()
        sigmas[:, j] = s.values
        Z[:, j] = (returns[c].values / s.values)
    Z = np.nan_to_num(Z)

    Qbar = np.cov(Z, rowvar=False)
    T = len(Z)

    def run_dcc(a, b):
        Q = Qbar.copy()
        ll = 0.0
        corr_path = np.empty(T)
        R_last = np.eye(n)
        for t in range(T):
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            R_last = R
            corr_path[t] = (R.sum() - n) / (n * (n - 1))
            z = Z[t].reshape(-1, 1)
            # псевдо-LL вклада (для подбора a,b)
            try:
                sign, logdet = np.linalg.slogdet(R)
                Rinv = np.linalg.inv(R)
                ll += -0.5 * (logdet + (z.T @ Rinv @ z - z.T @ z))
            except np.linalg.LinAlgError:
                ll += -1e6
            Q = (1 - a - b) * Qbar + a * (z @ z.T) + b * Q
        return ll, R_last, Q, corr_path

    best = (-np.inf, None)
    for a in np.linspace(0.005, 0.10, 8):
        for b in np.linspace(0.85, 0.98, 8):
            if a + b >= 0.999:
                continue
            ll, R, Q, cp = run_dcc(a, b)
            if ll > best[0]:
                best = (ll, (a, b, R, Q, cp))
    a, b, R_last, Q_last, cp = best[1]
    d_last = sigmas[-1]
    cov_last = R_last * np.outer(d_last, d_last)
    return DCCResult(cond_cov_last=cov_last, cond_corr_last=R_last, a=float(a),
                     b=float(b), avg_corr_path=pd.Series(cp, index=returns.index),
                     names=cols)


def condition_number(cov: np.ndarray) -> float:
    """Число обусловленности (мера устойчивости к инверсии): чем меньше, тем лучше."""
    vals = np.linalg.eigvalsh(cov)
    vals = vals[vals > 0]
    return float(vals.max() / vals.min()) if len(vals) else np.inf
