"""
Моделирование зависимостей между криптоактивами через копулы.

Линейная корреляция Пирсона недооценивает совместный риск: в стрессе крипта
«падает вся вместе» (высокая хвостовая зависимость), чего нормальная (гауссова)
модель не улавливает. Поэтому используем:

    * псевдонаблюдения (ранговое преобразование к равномерным маргиналам);
    * гауссову и t-копулы (t-копула имеет ненулевую хвостовую зависимость);
    * коэффициенты хвостовой зависимости (lower/upper) — эмпирические и
      теоретические (для t-копулы) — мера риска «совместного обвала»;
    * симуляцию из копулы для оценки портфельного VaR/ES с реалистичными хвостами.

Теоретическая база — материалы из theory/Copula (Embrechts et al.,
Demarta & McNeil «The t Copula», Genest & Favre и др.).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .config import RANDOM_SEED
from .var_es import _nearest_psd


# --------------------------------------------------------------------------- #
# Псевдонаблюдения и ранговые меры
# --------------------------------------------------------------------------- #
def pseudo_observations(returns: pd.DataFrame) -> np.ndarray:
    """Ранговое преобразование к (0,1): u_ij = rank_ij / (n+1)."""
    return returns.rank().values / (len(returns) + 1)


def kendall_tau_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    cols = returns.columns
    n = len(cols)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau, _ = stats.kendalltau(returns.iloc[:, i], returns.iloc[:, j])
            M[i, j] = M[j, i] = tau
    return pd.DataFrame(M, index=cols, columns=cols)


# --------------------------------------------------------------------------- #
# Эмпирическая хвостовая зависимость
# --------------------------------------------------------------------------- #
def empirical_tail_dependence(u: np.ndarray, v: np.ndarray, q: float = 0.05
                              ) -> tuple[float, float]:
    """
    Эмпирические коэффициенты нижней (lambda_L) и верхней (lambda_U) хвостовой
    зависимости на пороге q:
        lambda_L ≈ P(U<=q | V<=q),   lambda_U ≈ P(U>1-q | V>1-q).
    """
    lower = np.mean((u <= q) & (v <= q)) / q if q > 0 else np.nan
    upper = np.mean((u > 1 - q) & (v > 1 - q)) / q if q > 0 else np.nan
    return float(lower), float(upper)


def tail_dependence_matrix(returns: pd.DataFrame, q: float = 0.05,
                           which: str = "lower") -> pd.DataFrame:
    """Матрица эмпирической хвостовой зависимости всех пар активов."""
    u = pseudo_observations(returns)
    cols = returns.columns
    n = len(cols)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            l, up = empirical_tail_dependence(u[:, i], u[:, j], q)
            M[i, j] = M[j, i] = (l if which == "lower" else up)
    return pd.DataFrame(M, index=cols, columns=cols)


# --------------------------------------------------------------------------- #
# Гауссова копула
# --------------------------------------------------------------------------- #
@dataclass
class GaussianCopula:
    corr: np.ndarray
    names: list[str]

    @classmethod
    def fit(cls, returns: pd.DataFrame) -> "GaussianCopula":
        u = pseudo_observations(returns)
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        corr = np.corrcoef(z, rowvar=False)
        return cls(corr=corr, names=list(returns.columns))

    def simulate(self, n: int, seed: int = RANDOM_SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(_nearest_psd(self.corr))
        z = rng.standard_normal((n, len(self.names))) @ L.T
        return stats.norm.cdf(z)  # равномерные маргиналы

    @property
    def lower_tail_dependence(self) -> np.ndarray:
        """У гауссовой копулы хвостовая зависимость = 0 (кроме rho=1)."""
        return np.zeros_like(self.corr)


# --------------------------------------------------------------------------- #
# t-копула
# --------------------------------------------------------------------------- #
@dataclass
class StudentTCopula:
    corr: np.ndarray
    nu: float
    names: list[str]

    @classmethod
    def fit(cls, returns: pd.DataFrame, nu_grid=np.arange(3, 30)) -> "StudentTCopula":
        """
        Двухшаговая оценка: (1) корреляция из нормальных скоров (как прокси),
        (2) ν подбирается максимизацией псевдо-лог-правдоподобия t-копулы.
        """
        u = pseudo_observations(returns)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z = stats.norm.ppf(u)
        corr = np.corrcoef(z, rowvar=False)
        corr_psd = _nearest_psd(corr)

        def neg_ll(nu):
            return -_t_copula_loglik(u, corr_psd, nu)

        res = minimize_scalar(neg_ll, bounds=(2.5, 60), method="bounded")
        return cls(corr=corr_psd, nu=float(res.x), names=list(returns.columns))

    def simulate(self, n: int, seed: int = RANDOM_SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        d = len(self.names)
        L = np.linalg.cholesky(self.corr)
        z = rng.standard_normal((n, d)) @ L.T
        g = rng.chisquare(self.nu, size=(n, 1)) / self.nu
        t = z / np.sqrt(g)
        return stats.t.cdf(t, self.nu)

    def lower_tail_dependence_pair(self, rho: float) -> float:
        """
        Теоретический коэффициент хвостовой зависимости t-копулы:
            lambda = 2 · t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ).
        Симметричен (lower=upper).
        """
        arg = -np.sqrt((self.nu + 1) * (1 - rho) / (1 + rho))
        return float(2 * stats.t.cdf(arg, self.nu + 1))

    @property
    def lower_tail_dependence(self) -> np.ndarray:
        d = len(self.names)
        M = np.ones((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    M[i, j] = self.lower_tail_dependence_pair(self.corr[i, j])
        return M


def _t_copula_loglik(u: np.ndarray, corr: np.ndarray, nu: float) -> float:
    """Лог-правдоподобие t-копулы (с точностью до константы по маргиналам)."""
    d = corr.shape[0]
    t_inv = stats.t.ppf(u, nu)
    inv_corr = np.linalg.pinv(corr)
    sign, logdet = np.linalg.slogdet(corr)
    n = u.shape[0]
    # плотность t-копулы (Demarta & McNeil 2005)
    q = np.einsum("ij,jk,ik->i", t_inv, inv_corr, t_inv)
    log_num = (gammaln((nu + d) / 2) + (d - 1) * gammaln(nu / 2)
               - d * gammaln((nu + 1) / 2))
    ll = n * (log_num - 0.5 * logdet)
    ll += -((nu + d) / 2) * np.sum(np.log1p(q / nu))
    ll += ((nu + 1) / 2) * np.sum(np.log1p(t_inv ** 2 / nu))
    return float(ll)


# --------------------------------------------------------------------------- #
# Симуляция портфеля через копулу + эмпирические маргиналы
# --------------------------------------------------------------------------- #
def copula_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    copula,
    n_sims: int = 50_000,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Сэмплирует совместные доходности: зависимость берётся из копулы, а
    маргиналы — эмпирические (квантильное преобразование по историческим данным).
    Возвращает распределение доходностей портфеля для VaR/ES с учётом хвостов.
    """
    u = copula.simulate(n_sims, seed=seed)
    sim = np.empty_like(u)
    for j, col in enumerate(returns.columns):
        sim[:, j] = np.quantile(returns[col].values, u[:, j])
    return sim @ np.asarray(weights, dtype=float)


def compare_dependence_models(returns: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """
    Сравнивает эмпирическую нижнюю хвостовую зависимость со встроенной в
    гауссову (=0) и t-копулу — демонстрирует, почему для крипты нужна t-копула.
    """
    g = GaussianCopula.fit(returns)
    t = StudentTCopula.fit(returns)
    emp = tail_dependence_matrix(returns, q=q, which="lower").values
    t_theo = t.lower_tail_dependence
    iu = np.triu_indices(len(returns.columns), k=1)
    return pd.DataFrame({
        "empirical_lower_tail": emp[iu],
        "gaussian_copula": g.lower_tail_dependence[iu],
        "t_copula(nu=%.1f)" % t.nu: t_theo[iu],
    })
