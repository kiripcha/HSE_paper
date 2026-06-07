"""
Оценка мер рыночного риска: Value-at-Risk (VaR) и Expected Shortfall (ES).

Согласно ТЗ:
    * VaR на уровне доверия 99%   (alpha = 0.99);
    * ES  на уровне доверия 97.5% (alpha = 0.975);
    * горизонты 1 и 10 торговых дней.

СОГЛАШЕНИЕ О ЗНАКЕ. И VaR, и ES возвращаются как ПОЛОЖИТЕЛЬНЫЕ числа,
выражающие убыток в долях капитала:
        VaR_alpha = -Q_{1-alpha}(r),
        ES_alpha  = -E[ r | r <= Q_{1-alpha}(r) ].
Значение 0.05 означает «убыток 5% капитала».

Реализованные методы (по нарастанию сложности):
    1. historical            — историческая симуляция (непараметрический квантиль);
    2. parametric (normal/t) — дельта-нормальный / Стьюдент;
    3. cornish_fisher        — поправка квантиля на асимметрию и эксцесс;
    4. ewma                  — RiskMetrics: σ из EWMA + нормальный/t квантиль;
    5. garch                 — условная σ из GARCH-семейства;
    6. fhs                   — filtered historical simulation (GARCH + бутстрэп остатков);
    7. monte_carlo           — параметрическая Монте-Карло (коррелированная) для портфеля.

Масштабирование на горизонт h:
    * параметрические/EWMA  — правило sqrt(h) для σ (и mu·h для среднего);
    * GARCH                 — корректная агрегация прогнозов дисперсии;
    * historical            — overlapping h-дневные доходности либо sqrt(h).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import ES_CONFIDENCE, RANDOM_SEED, VAR_CONFIDENCE
from .volatility import ewma_forecast, ewma_volatility


@dataclass
class RiskEstimate:
    var: float
    es: float
    method: str
    horizon: int
    var_alpha: float
    es_alpha: float
    extra: dict | None = None

    def as_money(self, capital: float) -> dict:
        return {"VaR": self.var * capital, "ES": self.es * capital}

    def __repr__(self) -> str:
        return (f"RiskEstimate(method={self.method}, h={self.horizon}, "
                f"VaR{self.var_alpha:.1%}={self.var:.4f}, "
                f"ES{self.es_alpha:.1%}={self.es:.4f})")


# --------------------------------------------------------------------------- #
# 1. Историческая симуляция
# --------------------------------------------------------------------------- #
def historical_var_es(
    returns: np.ndarray | pd.Series,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    overlap: bool = False,
) -> RiskEstimate:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if horizon > 1:
        if overlap:
            r = _overlapping_sum(r, horizon)
        else:
            return _scale_estimate(
                historical_var_es(r, var_alpha, es_alpha, 1), horizon
            )
    q_var = np.quantile(r, 1 - var_alpha)
    q_es = np.quantile(r, 1 - es_alpha)
    tail = r[r <= q_es]
    es = -tail.mean() if tail.size else -q_es
    return RiskEstimate(var=-q_var, es=es, method="historical", horizon=horizon,
                        var_alpha=var_alpha, es_alpha=es_alpha)


# --------------------------------------------------------------------------- #
# 2. Параметрический (нормальный / Стьюдент)
# --------------------------------------------------------------------------- #
def parametric_var_es(
    mu: float,
    sigma: float,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    dist: str = "normal",
    nu: float | None = None,
) -> RiskEstimate:
    """Дельта-нормальный или Стьюдент-VaR/ES по заданным mu, sigma (дневным)."""
    mu_h = mu * horizon
    sig_h = sigma * np.sqrt(horizon)

    if dist == "normal":
        z_v = stats.norm.ppf(1 - var_alpha)
        z_e = stats.norm.ppf(1 - es_alpha)
        var = -(mu_h + sig_h * z_v)
        # ES нормали: phi(z)/(1-alpha)
        es = -(mu_h - sig_h * stats.norm.pdf(z_e) / (1 - es_alpha))
    elif dist == "t":
        if nu is None or nu <= 2:
            nu = 5.0
        # стандартизуем t, чтобы Var=1, тогда масштабируем на sigma
        scale = np.sqrt((nu - 2) / nu)
        t_v = stats.t.ppf(1 - var_alpha, nu) * scale
        t_e = stats.t.ppf(1 - es_alpha, nu) * scale
        var = -(mu_h + sig_h * t_v)
        # ES для t (Acerbi/McNeil): (nu + t_e^2)/(nu-1) * pdf(t_e)/(1-alpha)
        x = stats.t.ppf(1 - es_alpha, nu)
        es_std = (stats.t.pdf(x, nu) / (1 - es_alpha)) * ((nu + x ** 2) / (nu - 1)) * scale
        es = -(mu_h - sig_h * es_std)
    else:
        raise ValueError("dist must be 'normal' or 't'")

    return RiskEstimate(var=float(var), es=float(es),
                        method=f"parametric_{dist}", horizon=horizon,
                        var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"mu": mu, "sigma": sigma, "nu": nu})


# --------------------------------------------------------------------------- #
# 3. Cornish-Fisher (модифицированный VaR)
# --------------------------------------------------------------------------- #
def cornish_fisher_var_es(
    returns: np.ndarray | pd.Series,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
) -> RiskEstimate:
    """
    Поправка квантиля нормали на асимметрию (S) и эксцесс (K) разложением
    Корниша-Фишера. Полезно для крипты с явной несимметрией и тяжёлыми хвостами.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    mu, sigma = r.mean(), r.std(ddof=1)
    S = stats.skew(r)
    K = stats.kurtosis(r, fisher=True)  # избыточный эксцесс

    def cf_quantile(alpha):
        z = stats.norm.ppf(1 - alpha)
        return (z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24
                - (2*z**3 - 5*z) * S**2 / 36)

    z_v = cf_quantile(var_alpha)
    z_e = cf_quantile(es_alpha)
    mu_h, sig_h = mu * horizon, sigma * np.sqrt(horizon)
    var = -(mu_h + sig_h * z_v)
    # ES приближаем средним по «хвостовым» CF-квантилям
    grid = np.linspace(es_alpha, 0.9999, 50)
    es_tail = np.mean([-(mu_h + sig_h * cf_quantile(a)) for a in grid])
    return RiskEstimate(var=float(var), es=float(es_tail),
                        method="cornish_fisher", horizon=horizon,
                        var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"skew": S, "kurtosis": K})


# --------------------------------------------------------------------------- #
# 4. EWMA / RiskMetrics
# --------------------------------------------------------------------------- #
def ewma_var_es(
    returns: pd.Series,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    lam: float = 0.94,
    dist: str = "normal",
    nu: float | None = None,
) -> RiskEstimate:
    sigma = ewma_forecast(returns, lam=lam)
    mu = float(np.asarray(returns).mean())
    est = parametric_var_es(mu, sigma, var_alpha, es_alpha, horizon, dist, nu)
    est.method = f"ewma_{dist}"
    return est


# --------------------------------------------------------------------------- #
# 5. GARCH-основанный VaR/ES
# --------------------------------------------------------------------------- #
def garch_var_es(
    garch_result,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    mu: float = 0.0,
) -> RiskEstimate:
    """
    VaR/ES по обученной GARCH-модели (см. volatility.GARCHResult).
    Берётся прогноз условной σ на горизонт h и квантиль соответствующего
    распределения инноваций (normal или t).
    """
    sig_h = garch_result.horizon_sigma(horizon)
    dist = "t" if garch_result.dist in ("t", "skewt") else "normal"
    nu = garch_result.nu
    # mu на горизонт берём из константы среднего (приблизительно 0 для крипты HFT)
    est = parametric_var_es(mu, sig_h / np.sqrt(max(horizon, 1)),
                            var_alpha, es_alpha, horizon, dist, nu)
    est.method = f"garch_{garch_result.vol}_{garch_result.dist}"
    est.extra = {"sigma_h": sig_h, "nu": nu}
    return est


# --------------------------------------------------------------------------- #
# 6. Filtered Historical Simulation (FHS)
# --------------------------------------------------------------------------- #
def fhs_var_es(
    returns: pd.Series,
    garch_result,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    n_paths: int = 20_000,
    seed: int = RANDOM_SEED,
) -> RiskEstimate:
    """
    Filtered Historical Simulation: стандартизуем доходности на условную σ
    GARCH, бутстрэпим стандартизованные остатки на h шагов вперёд, заново
    «надуваем» их прогнозом σ. Сочетает достоинства GARCH (динамика σ) и
    исторической симуляции (непараметрические хвосты). Хорошо подходит крипте.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna()
    sigma_in = garch_result.sigma.reindex(r.index).dropna()
    r = r.reindex(sigma_in.index)
    z = (r.values / sigma_in.values)
    z = z[np.isfinite(z)]

    sig_fc = garch_result.forecast_sigma_path()  # длины >= horizon
    if len(sig_fc) < horizon:
        sig_fc = np.concatenate([sig_fc, np.full(horizon - len(sig_fc), sig_fc[-1])])

    # h-дневная доходность как сумма масштабированных бутстрэп-остатков
    draws = rng.choice(z, size=(n_paths, horizon), replace=True)
    sim_h = (draws * sig_fc[:horizon]).sum(axis=1)

    q_var = np.quantile(sim_h, 1 - var_alpha)
    q_es = np.quantile(sim_h, 1 - es_alpha)
    es = -sim_h[sim_h <= q_es].mean()
    return RiskEstimate(var=float(-q_var), es=float(es), method="fhs",
                        horizon=horizon, var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"n_paths": n_paths})


# --------------------------------------------------------------------------- #
# 7. Монте-Карло для портфеля (коррелированные риск-факторы)
# --------------------------------------------------------------------------- #
def monte_carlo_var_es(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    n_sims: int = 50_000,
    dist: str = "normal",
    nu: float = 5.0,
    seed: int = RANDOM_SEED,
) -> RiskEstimate:
    """
    Параметрическая Монте-Карло симуляция доходностей портфеля.
    Корреляционная структура задаётся ковариационной матрицей cov (дневной),
    распределение инноваций — нормальное или многомерное t (тяжёлые хвосты).
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(_nearest_psd(cov))

    if dist == "normal":
        Z = rng.standard_normal((n_sims, len(w)))
    elif dist == "t":
        g = rng.chisquare(nu, size=(n_sims, 1)) / nu
        Z = rng.standard_normal((n_sims, len(w))) / np.sqrt(g)
        Z *= np.sqrt((nu - 2) / nu)  # стандартизация дисперсии к 1
    else:
        raise ValueError("dist must be 'normal' or 't'")

    daily_ret = mu + Z @ L.T
    # агрегируем на горизонт (приближённо: сумма h независимых дней)
    if horizon > 1:
        port_ret = np.zeros(n_sims)
        for _ in range(horizon):
            if dist == "normal":
                Zh = rng.standard_normal((n_sims, len(w)))
            else:
                g = rng.chisquare(nu, size=(n_sims, 1)) / nu
                Zh = rng.standard_normal((n_sims, len(w))) / np.sqrt(g)
                Zh *= np.sqrt((nu - 2) / nu)
            port_ret += (mu + Zh @ L.T) @ w
    else:
        port_ret = daily_ret @ w

    q_var = np.quantile(port_ret, 1 - var_alpha)
    q_es = np.quantile(port_ret, 1 - es_alpha)
    es = -port_ret[port_ret <= q_es].mean()
    return RiskEstimate(var=float(-q_var), es=float(es),
                        method=f"monte_carlo_{dist}", horizon=horizon,
                        var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"n_sims": n_sims})


# --------------------------------------------------------------------------- #
# Декомпозиция риска: marginal / component / incremental (принцип Эйлера)
# --------------------------------------------------------------------------- #
def component_var_es(
    weights: np.ndarray,
    returns: pd.DataFrame,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    method: str = "gaussian",
    n_sims: int = 100_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Раскладывает риск портфеля на вклады активов (Tasche, Euler allocation):

        * Marginal VaR_i   = ∂VaR/∂w_i          (чувствительность к позиции);
        * Component VaR_i  = w_i · Marginal_i   (сумма по i = VaR портфеля!);
        * % contribution   = Component_i / VaR.

    method='gaussian' — аналитически (эллиптическое допущение);
    method='historical'/'simulation' — через хвостовые сценарии (для ES тоже).
    Это отвечает на вопрос «какой актив реально создаёт риск», что точнее
    наивного взгляда на веса и нужно для риск-бюджетирования.
    """
    w = np.asarray(weights, dtype=float)
    cols = list(returns.columns)
    if method == "gaussian":
        cov = returns.cov().values
        sigma_p = np.sqrt(w @ cov @ w)
        z_v = -stats.norm.ppf(1 - var_alpha)
        z_e = stats.norm.pdf(stats.norm.ppf(1 - es_alpha)) / (1 - es_alpha)
        mvar = z_v * (cov @ w) / sigma_p          # marginal VaR
        mes = z_e * (cov @ w) / sigma_p           # marginal ES
        cvar = w * mvar
        ces = w * mes
    else:  # симуляционный/исторический: условный вклад в хвосте
        port = returns.values @ w
        q_v = np.quantile(port, 1 - var_alpha)
        q_e = np.quantile(port, 1 - es_alpha)
        tail_v = port <= q_v
        tail_e = port <= q_e
        # component VaR_i = -w_i · E[r_i | r_p ≈ VaR-квантиль]
        band = port <= np.quantile(port, 1 - var_alpha + 0.005)
        cvar = -w * returns.values[band].mean(axis=0)
        ces = -w * returns.values[tail_e].mean(axis=0)
        cvar = cvar / cvar.sum() * (-q_v) if cvar.sum() != 0 else cvar
        mvar = cvar / np.where(w == 0, np.nan, w)
        mes = ces / np.where(w == 0, np.nan, w)

    out = pd.DataFrame({
        "weight": w,
        "marginal_VaR": mvar,
        "component_VaR": cvar,
        "pct_contrib_VaR": cvar / cvar.sum() if cvar.sum() else cvar,
        "component_ES": ces,
        "pct_contrib_ES": ces / ces.sum() if ces.sum() else ces,
    }, index=cols)
    return out


def incremental_var(
    weights: np.ndarray,
    returns: pd.DataFrame,
    asset_idx: int,
    var_alpha: float = VAR_CONFIDENCE,
) -> float:
    """
    Incremental VaR актива: VaR(портфель с активом) − VaR(портфель без него).
    Показывает, сколько риска ДОБАВЛЯЕТ конкретная позиция.
    """
    w = np.asarray(weights, dtype=float)
    full = historical_var_es(returns.values @ w, var_alpha, var_alpha, 1).var
    w0 = w.copy(); w0[asset_idx] = 0.0
    if w0.sum() > 0:
        w0 = w0 / w0.sum()
    without = historical_var_es(returns.values @ w0, var_alpha, var_alpha, 1).var
    return float(full - without)


# --------------------------------------------------------------------------- #
# Агрегация на горизонт: бутстрэп вместо наивного sqrt-t
# --------------------------------------------------------------------------- #
def bootstrap_horizon_var_es(
    returns: pd.Series,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 10,
    block: int = 1,
    n_paths: int = 50_000,
    seed: int = RANDOM_SEED,
) -> RiskEstimate:
    """
    Оценка h-дневного VaR/ES бутстрэпом h-дневных доходностей.

    Правило sqrt(t) корректно лишь при i.i.d. нормальных доходностях; для крипты
    с тяжёлыми хвостами и кластеризацией σ оно ЗАНИЖАЕТ риск (Diebold et al.,
    «Scale models», theory/Time Aggregation). Блочный бутстрэп (block>1)
    сохраняет автокорреляцию волатильности.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if block <= 1:
        draws = rng.choice(r, size=(n_paths, horizon), replace=True)
        sim_h = draws.sum(axis=1)
    else:
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, n - block, size=(n_paths, n_blocks))
        sim_h = np.empty(n_paths)
        for p in range(n_paths):
            path = np.concatenate([r[s:s + block] for s in starts[p]])[:horizon]
            sim_h[p] = path.sum()
    q_v = np.quantile(sim_h, 1 - var_alpha)
    q_e = np.quantile(sim_h, 1 - es_alpha)
    es = -sim_h[sim_h <= q_e].mean()
    return RiskEstimate(var=float(-q_v), es=float(es),
                        method=f"bootstrap_h{horizon}", horizon=horizon,
                        var_alpha=var_alpha, es_alpha=es_alpha,
                        extra={"block": block})


# --------------------------------------------------------------------------- #
# Сравнительная таблица методов
# --------------------------------------------------------------------------- #
def compare_methods(
    returns: pd.Series,
    var_alpha: float = VAR_CONFIDENCE,
    es_alpha: float = ES_CONFIDENCE,
    horizon: int = 1,
    garch_result=None,
) -> pd.DataFrame:
    """Считает VaR/ES всеми доступными методами для одного ряда доходностей."""
    r = returns.dropna()
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    ests = [
        historical_var_es(r, var_alpha, es_alpha, horizon),
        parametric_var_es(mu, sigma, var_alpha, es_alpha, horizon, "normal"),
        parametric_var_es(mu, sigma, var_alpha, es_alpha, horizon, "t", nu=5),
        cornish_fisher_var_es(r, var_alpha, es_alpha, horizon),
        ewma_var_es(r, var_alpha, es_alpha, horizon, dist="normal"),
    ]
    if garch_result is not None:
        ests.append(garch_var_es(garch_result, var_alpha, es_alpha, horizon, mu))
        ests.append(fhs_var_es(r, garch_result, var_alpha, es_alpha, horizon))
    # EVT (POT/GPD) — точные хвосты; масштабируем sqrt(h) если horizon>1
    try:
        from .evt import pot_var_es
        evt = pot_var_es(r, var_alpha, es_alpha)
        if horizon > 1:
            evt = _scale_estimate(evt, horizon)
        ests.append(evt)
    except Exception:
        pass
    return pd.DataFrame([{
        "method": e.method, "horizon": e.horizon,
        f"VaR_{int(var_alpha*100)}": e.var,
        f"ES_{es_alpha*100:.1f}": e.es,
    } for e in ests])


# --------------------------------------------------------------------------- #
# Вспомогательные
# --------------------------------------------------------------------------- #
def _overlapping_sum(r: np.ndarray, h: int) -> np.ndarray:
    """Перекрывающиеся h-дневные доходности (сумма лог-доходностей)."""
    if len(r) < h:
        return r
    return np.convolve(r, np.ones(h), mode="valid")


def _scale_estimate(est: RiskEstimate, horizon: int) -> RiskEstimate:
    """Масштабирование VaR/ES на горизонт по правилу sqrt(h)."""
    f = np.sqrt(horizon)
    return RiskEstimate(var=est.var * f, es=est.es * f, method=est.method,
                        horizon=horizon, var_alpha=est.var_alpha,
                        es_alpha=est.es_alpha, extra=est.extra)


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Ближайшая положительно полуопределённая матрица (клиппинг собств. чисел)."""
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(vals) @ vecs.T
