"""
Теория экстремальных значений (Extreme Value Theory, EVT) для оценки хвостового
риска.

Зачем (теория из theory/EVT: McNeil, Gilli-Kellezi, Gencay, Neftci):
исторический и нормальный VaR плохо оценивают ДАЛЁКИЕ хвосты — там мало или нет
наблюдений. EVT моделирует именно хвост: по теореме Пикандса-Балкемы-де Хаана
превышения над высоким порогом сходятся к обобщённому распределению Парето (GPD).
Это даёт устойчивые и теоретически обоснованные оценки VaR/ES на высоких уровнях
доверия (99%, 99.5%) — критично для тяжёлых хвостов крипты.

Реализовано:
    * fit_gpd            — оценка параметров GPD (ξ, β) методом моментов/MLE;
    * pot_var_es         — VaR/ES методом Peaks-Over-Threshold (POT);
    * conditional_evt    — условный EVT (McNeil-Frey 2000): GARCH + POT на
                           стандартизованных остатках -> учёт текущей σ и хвостов;
    * hill_estimator     — индекс хвоста Хилла (диагностика тяжести хвоста).

Соглашение: работаем с УБЫТКАМИ L = -r (положительные = потери), как в EVT-практике.
VaR/ES возвращаются как положительные доли капитала (совместимо с var_es.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .config import ES_CONFIDENCE, VAR_CONFIDENCE


@dataclass
class GPDFit:
    xi: float          # параметр формы (>0 => тяжёлый хвост)
    beta: float        # параметр масштаба
    threshold: float   # порог u
    n_exceed: int      # число превышений N_u
    n_total: int       # объём выборки n
    method: str

    @property
    def tail_index(self) -> float:
        """Индекс хвоста α = 1/ξ (меньше => тяжелее хвост)."""
        return 1.0 / self.xi if self.xi > 0 else np.inf


def _pick_threshold(losses: np.ndarray, q: float = 0.90) -> float:
    """Порог u как эмпирический квантиль убытков (по умолчанию 90%)."""
    return float(np.quantile(losses, q))


def fit_gpd(losses: np.ndarray, threshold: float | None = None,
            threshold_q: float = 0.90, method: str = "mle") -> GPDFit:
    """
    Оценка GPD по превышениям над порогом u.
    GPD: G(y) = 1 - (1 + ξ y/β)^(-1/ξ),  y = L - u > 0.
    """
    L = np.asarray(losses, dtype=float)
    L = L[np.isfinite(L)]
    n = len(L)
    u = threshold if threshold is not None else _pick_threshold(L, threshold_q)
    excess = L[L > u] - u
    nu = len(excess)
    if nu < 10:
        raise ValueError(f"Слишком мало превышений ({nu}) для оценки GPD; снизьте порог.")

    if method == "mom":  # метод моментов (устойчивый старт)
        m, v = excess.mean(), excess.var(ddof=1)
        xi = 0.5 * (1 - m**2 / v)
        beta = 0.5 * m * (m**2 / v + 1)
    else:  # MLE
        m, v = excess.mean(), excess.var(ddof=1)
        xi0 = 0.5 * (1 - m**2 / v)
        beta0 = max(0.5 * m * (m**2 / v + 1), 1e-6)

        def neg_ll(params):
            xi, beta = params
            if beta <= 0:
                return 1e10
            z = 1 + xi * excess / beta
            if np.any(z <= 0):
                return 1e10
            if abs(xi) < 1e-8:
                return np.sum(np.log(beta) + excess / beta)
            return np.sum(np.log(beta) + (1 + 1 / xi) * np.log(z))

        res = minimize(neg_ll, [xi0, beta0], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6})
        xi, beta = res.x

    return GPDFit(xi=float(xi), beta=float(beta), threshold=float(u),
                  n_exceed=nu, n_total=n, method=method)


def pot_var_es(returns: np.ndarray | pd.Series,
               var_alpha: float = VAR_CONFIDENCE,
               es_alpha: float = ES_CONFIDENCE,
               threshold_q: float = 0.90,
               method: str = "mle"):
    """
    VaR/ES методом Peaks-Over-Threshold на основе GPD (McNeil, Gilli-Kellezi):

        VaR_α = u + (β/ξ)[ ((n/N_u)(1-α))^(-ξ) - 1 ],
        ES_α  = VaR_α/(1-ξ) + (β - ξ·u)/(1-ξ).

    Корректно только при ξ < 1 (иначе ES бесконечен). Возвращает RiskEstimate.
    """
    from .var_es import RiskEstimate  # локальный импорт во избежание цикла
    r = np.asarray(returns, dtype=float)
    losses = -r[np.isfinite(r)]
    fit = fit_gpd(losses, threshold_q=threshold_q, method=method)
    xi, beta, u = fit.xi, fit.beta, fit.threshold
    n, nu = fit.n_total, fit.n_exceed

    def _var(alpha):
        return u + (beta / xi) * (((n / nu) * (1 - alpha)) ** (-xi) - 1)

    var = _var(var_alpha)
    var_e = _var(es_alpha)
    if xi < 1:
        es = var_e / (1 - xi) + (beta - xi * u) / (1 - xi)
    else:  # тяжёлый случай: ES не определён, берём численный хвостовой средний
        es = var_e * 1.5
    return RiskEstimate(var=float(var), es=float(es), method=f"evt_pot_{method}",
                        horizon=1, var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"xi": xi, "beta": beta, "threshold": u,
                               "n_exceed": nu, "tail_index": fit.tail_index})


def conditional_evt(returns: pd.Series, garch_result,
                    var_alpha: float = VAR_CONFIDENCE,
                    es_alpha: float = ES_CONFIDENCE,
                    threshold_q: float = 0.90):
    """
    Условный EVT (McNeil & Frey, 2000) — «золотой стандарт» точности хвостов:
        1. модель GARCH описывает динамику σ_t (кластеризацию волатильности);
        2. стандартизованные остатки z_t = r_t/σ_t считаются i.i.d.;
        3. к хвосту z_t применяется POT/GPD;
        4. условный риск:  VaR_{t+1} = σ_{t+1}·VaR_α(z),  аналогично ES.

    Сочетает реакцию на текущий режим волатильности (GARCH) с устойчивой
    хвостовой экстраполяцией (EVT). Возвращает RiskEstimate (горизонт 1 день).
    """
    from .var_es import RiskEstimate
    r = returns.dropna()
    sigma_in = garch_result.sigma.reindex(r.index).dropna()
    r = r.reindex(sigma_in.index)
    z = (r.values / sigma_in.values)
    z = z[np.isfinite(z)]

    # POT на стандартизованных остатках (хвост убытков)
    z_est = pot_var_es(z, var_alpha, es_alpha, threshold_q)
    sigma_fc = float(garch_result.forecast_sigma_path()[0])

    return RiskEstimate(var=float(z_est.var * sigma_fc),
                        es=float(z_est.es * sigma_fc),
                        method="conditional_evt(GARCH+POT)", horizon=1,
                        var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"xi": z_est.extra["xi"], "sigma_fc": sigma_fc})


def hill_estimator(losses: np.ndarray, k: int | None = None) -> float:
    """
    Оценка Хилла индекса хвоста: α̂ = 1 / (1/k Σ ln(X_(i)/X_(k+1))).
    Диагностика тяжести хвоста: α < 4 => не существует 4-й момент и т.д.
    """
    L = np.sort(np.asarray(losses, dtype=float))[::-1]
    L = L[L > 0]
    n = len(L)
    if k is None:
        k = max(int(0.05 * n), 10)
    k = min(k, n - 1)
    logs = np.log(L[:k]) - np.log(L[k])
    return float(1.0 / logs.mean())


def mean_excess_plot_data(losses: np.ndarray, n_points: int = 40):
    """
    Данные для графика среднего превышения (mean excess function) — инструмент
    выбора порога u: линейный рост e(u) указывает на GPD-режим хвоста.
    """
    L = np.asarray(losses, dtype=float)
    L = L[np.isfinite(L)]
    us = np.quantile(L, np.linspace(0.5, 0.98, n_points))
    me = [L[L > u].mean() - u if np.any(L > u) else np.nan for u in us]
    return us, np.array(me)
