"""
Количественная валидация (backtesting) моделей VaR и ES.

Соглашение: подаём РЕАЛИЗОВАННЫЕ доходности `returns` (со знаком: убыток < 0) и
ряд оценок `var` (ПОЛОЖИТЕЛЬНЫЕ, как в var_es). «Пробой» (violation/hit) =
{ returns_t < -var_t }, т.е. фактический убыток превысил VaR.

Реализованы тесты из списка ТЗ:
    * Kupiec (1995)            — безусловное покрытие (POF), LR ~ chi2(1);
    * Christoffersen (1998)    — независимость (Markov) и условное покрытие (CC);
    * Christoffersen-Pelletier / Haas — duration-based (Вейбулл vs экспонента);
    * Engle & Manganelli (2004)— Dynamic Quantile (DQ) тест;
    * Berkowitz (2001)         — LR на основе преобразования PIT (хвостовой);
    * Acerbi & Szekely (2014)  — backtesting Expected Shortfall (Test 2).

Каждый тест возвращает BacktestResult со статистикой, p-value и вердиктом.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


@dataclass
class BacktestResult:
    name: str
    statistic: float
    p_value: float
    reject_h0: bool
    detail: dict

    def verdict(self, level: float = 0.05) -> str:
        return ("ОТВЕРГАЕМ H0: модель некорректна"
                if self.p_value < level else
                "НЕ отвергаем H0: модель адекватна")

    def __repr__(self) -> str:
        return (f"[{self.name}] stat={self.statistic:.3f}, "
                f"p={self.p_value:.4f} -> {self.verdict()}")


def get_violations(returns, var) -> np.ndarray:
    """Бинарный ряд пробоев I_t = 1{ r_t < -VaR_t }."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    return (r < -v).astype(int)


# --------------------------------------------------------------------------- #
# Kupiec (1995) — Proportion of Failures (безусловное покрытие)
# --------------------------------------------------------------------------- #
def kupiec_pof(violations: np.ndarray, var_alpha: float = 0.99) -> BacktestResult:
    I = np.asarray(violations)
    n, x = len(I), int(I.sum())
    p = 1 - var_alpha            # ожидаемая частота пробоев
    pi = x / n if n else 0.0
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p)
                   - x * np.log(pi) - (n - x) * np.log(1 - pi))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return BacktestResult("Kupiec POF", float(lr), float(pval), pval < 0.05,
                          {"n": n, "violations": x, "expected": p * n,
                           "obs_rate": pi, "exp_rate": p})


# --------------------------------------------------------------------------- #
# Christoffersen (1998) — независимость и условное покрытие
# --------------------------------------------------------------------------- #
def christoffersen(violations: np.ndarray, var_alpha: float = 0.99) -> dict:
    """Возвращает три теста: independence, conditional coverage, и POF."""
    I = np.asarray(violations)
    # переходы марковской цепи
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(I)):
        a, b = I[i - 1], I[i]
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if len(I) > 1 else 0.0

    def _safe(p):  # защита от log(0)
        return min(max(p, 1e-12), 1 - 1e-12)

    ll_ind = ((n00 + n10) * np.log(1 - _safe(pi)) + (n01 + n11) * np.log(_safe(pi)))
    ll_dep = (n00 * np.log(1 - _safe(pi01)) + n01 * np.log(_safe(pi01))
              + n10 * np.log(1 - _safe(pi11)) + n11 * np.log(_safe(pi11)))
    lr_ind = -2 * (ll_ind - ll_dep)
    p_ind = 1 - stats.chi2.cdf(lr_ind, df=1)

    pof = kupiec_pof(I, var_alpha)
    lr_cc = pof.statistic + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

    return {
        "independence": BacktestResult(
            "Christoffersen Independence", float(lr_ind), float(p_ind),
            p_ind < 0.05, {"pi01": pi01, "pi11": pi11, "pi": pi}),
        "conditional_coverage": BacktestResult(
            "Christoffersen CC", float(lr_cc), float(p_cc), p_cc < 0.05,
            {"lr_pof": pof.statistic, "lr_ind": lr_ind}),
        "pof": pof,
    }


# --------------------------------------------------------------------------- #
# Duration-based (Christoffersen-Pelletier 2004 / Haas 2006) — Вейбулл
# --------------------------------------------------------------------------- #
def duration_test(violations: np.ndarray, var_alpha: float = 0.99) -> BacktestResult:
    """
    Тест на основе длительностей между пробоями. При корректной модели
    длительности D ~ геометрическому (дискретно) / экспоненциальному (непрерывно).
    H1: распределение Вейбулла с параметром формы b != 1 (зависимость).
    LR ~ chi2(1).
    """
    I = np.asarray(violations)
    idx = np.where(I == 1)[0]
    if len(idx) < 3:
        return BacktestResult("Duration (Weibull)", np.nan, np.nan, False,
                              {"note": "слишком мало пробоев"})
    durations = np.diff(idx).astype(float)

    # Лог-правдоподобие Вейбулла f(d)=a*b*(a*d)^(b-1)exp(-(a*d)^b)
    def neg_ll_weibull(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        return -np.sum(np.log(a * b) + (b - 1) * np.log(a * durations)
                       - (a * durations) ** b)

    # ограничение b=1 -> экспонента, a=1/mean
    a0 = 1.0 / durations.mean()
    ll_exp = -neg_ll_weibull([a0, 1.0])
    res = minimize(neg_ll_weibull, [a0, 1.0], method="Nelder-Mead")
    ll_w = -res.fun
    lr = -2 * (ll_exp - ll_w)
    pval = 1 - stats.chi2.cdf(max(lr, 0), df=1)
    return BacktestResult("Duration (Weibull)", float(max(lr, 0)), float(pval),
                          pval < 0.05, {"shape_b": res.x[1], "n_durations": len(durations)})


# --------------------------------------------------------------------------- #
# Engle & Manganelli (2004) — Dynamic Quantile (DQ) test
# --------------------------------------------------------------------------- #
def dq_test(returns, var, var_alpha: float = 0.99, lags: int = 4) -> BacktestResult:
    """
    Регрессия hit_t = I_t - (1-alpha) на константу, лаги hit и текущий VaR.
    При корректной модели все коэффициенты = 0. Статистика Вальда ~ chi2(q).
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    I = (r < -v).astype(float)
    p = 1 - var_alpha
    hit = I - p

    T = len(hit)
    start = lags
    y = hit[start:]
    X_cols = [np.ones(T - start)]
    for L in range(1, lags + 1):
        X_cols.append(hit[start - L:T - L])
    X_cols.append(v[start:])  # текущий VaR как регрессор
    X = np.column_stack(X_cols)

    # OLS бета и статистика DQ = beta' X'X beta / (p(1-p)) ~ chi2(k)
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (X.T @ y)
    dq = float(beta @ XtX @ beta / (p * (1 - p)))
    k = X.shape[1]
    pval = 1 - stats.chi2.cdf(dq, df=k)
    return BacktestResult("Engle-Manganelli DQ", dq, float(pval), pval < 0.05,
                          {"lags": lags, "k": k})


# --------------------------------------------------------------------------- #
# Berkowitz (2001) — хвостовой LR-тест на основе PIT
# --------------------------------------------------------------------------- #
def berkowitz_tail(returns, var, sigma=None, var_alpha: float = 0.99,
                   seed: int = 42) -> BacktestResult:
    """
    Берём преобразование z_t = Phi^{-1}(PIT). Под H0 z ~ N(0,1). Оцениваем
    (mu, sigma) MLE и тестируем (mu=0, sigma=1) через LR ~ chi2(2).
    Если ряд sigma не задан, используем приближённый PIT через индикатор пробоя.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    p = 1 - var_alpha

    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        u = stats.norm.cdf(r / s)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z = stats.norm.ppf(u)
    else:
        I = (r < -v).astype(int)
        u = np.where(I == 1, p * rng.uniform(size=len(I)),
                     p + (1 - p) * rng.uniform(size=len(I)))
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

    def neg_ll(params):
        mu, sd = params
        if sd <= 0:
            return 1e10
        return -np.sum(stats.norm.logpdf(z, mu, sd))

    ll_restr = -neg_ll([0.0, 1.0])
    res = minimize(neg_ll, [0.0, 1.0], method="Nelder-Mead")
    ll_unrestr = -res.fun
    lr = -2 * (ll_restr - ll_unrestr)
    pval = 1 - stats.chi2.cdf(max(lr, 0), df=2)
    return BacktestResult("Berkowitz", float(max(lr, 0)), float(pval), pval < 0.05,
                          {"mu_hat": res.x[0], "sigma_hat": res.x[1]})


# --------------------------------------------------------------------------- #
# Acerbi & Szekely (2014) — backtesting Expected Shortfall (Test 2)
# --------------------------------------------------------------------------- #
def acerbi_szekely_es(returns, var, es, es_alpha: float = 0.975,
                      n_boot: int = 5000, seed: int = 42) -> BacktestResult:
    """
    Test 2 Acerbi-Szekely: Z = mean( r_t * I_t / ((1-alpha) * ES_t) ) + 1,
    где I_t = 1{ r_t < -VaR_t }. Под корректной моделью E[Z]=0.
    p-value получаем бутстрэпом знаков (нулевое распределение симулируется
    перемешиванием/ресэмплингом). Z << 0 => ES недооценён.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    p = 1 - es_alpha
    I = (r < -v).astype(float)
    T = len(r)
    nv = I.sum()
    if nv == 0:
        return BacktestResult("Acerbi-Szekely ES (Test 2)", np.nan, np.nan, False,
                              {"note": "нет пробоев"})

    contrib = r * I / (p * e)
    Z = contrib.sum() / T + 1.0

    # Бутстрэп нулевого распределения: предполагаем r ~ -ES в хвосте (консервативно)
    # симулируем нормальные хвосты с теми же VaR/ES масштабами
    z_boot = np.empty(n_boot)
    for b in range(n_boot):
        rb = r.copy()
        # ресэмплинг доходностей хвоста под гипотезой корректности ES
        tail_idx = np.where(I == 1)[0]
        rb[tail_idx] = -e[tail_idx] * rng.uniform(0.5, 1.5, size=len(tail_idx))
        Ib = (rb < -v).astype(float)
        z_boot[b] = (rb * Ib / (p * e)).sum() / T + 1.0
    pval = float(np.mean(z_boot <= Z))  # одностороннее: недооценка риска
    return BacktestResult("Acerbi-Szekely ES (Test 2)", float(Z), pval, pval < 0.05,
                          {"violations": int(nv), "interpretation":
                           "Z<0 => ES недооценивает риск"})


# --------------------------------------------------------------------------- #
# Acerbi & Szekely (2014) — Test 1 (unconditional) и Test 3 (ranks)
# --------------------------------------------------------------------------- #
def acerbi_szekely_test1(returns, var, es, es_alpha: float = 0.975,
                         n_boot: int = 5000, seed: int = 42) -> BacktestResult:
    """
    Test 1 Acerbi-Szekely (условный на число пробоев):
        Z1 = (1/N_viol) Σ_{t: пробой} ( X_t / ES_t ) + 1,
    где X_t = -r_t (убыток). E[Z1]=0 при корректной ES. Z1>0 => ES занижен.
    p-value — бутстрэпом перестановки знаков превышений.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    I = (r < -v)
    nv = int(I.sum())
    if nv == 0:
        return BacktestResult("Acerbi-Szekely ES (Test 1)", np.nan, np.nan, False,
                              {"note": "нет пробоев"})
    Z1 = np.mean((-r[I]) / e[I]) - 1.0
    # бутстрэп нулевого распределения: ES корректен => убытки ~ e на хвосте
    zb = np.empty(n_boot)
    for b in range(n_boot):
        sim_loss = e[I] * rng.uniform(0.5, 1.5, size=nv)
        zb[b] = np.mean(sim_loss / e[I]) - 1.0
    pval = float(np.mean(zb >= Z1))
    return BacktestResult("Acerbi-Szekely ES (Test 1)", float(Z1), pval, pval < 0.05,
                          {"violations": nv, "interpretation": "Z>0 => ES занижен"})


def acerbi_szekely_test3(returns, dist_cdf, es_alpha: float = 0.975) -> BacktestResult:
    """
    Test 3 (ранговый): применяет PIT u_t = F_t(r_t) и проверяет, что ранги
    хвостовых наблюдений согласуются с прогнозным распределением. Здесь —
    упрощённая версия: статистика на основе среднего PIT-ранга в хвосте.
    dist_cdf(t) -> вектор U_t = F_t(r_t) в [0,1].
    """
    u = np.asarray(dist_cdf, dtype=float)
    u = u[np.isfinite(u)]
    p = 1 - es_alpha
    tail = u[u <= p]
    if len(tail) < 5:
        return BacktestResult("Acerbi-Szekely ES (Test 3)", np.nan, np.nan, False,
                              {"note": "мало хвостовых наблюдений"})
    # под H0 хвостовые U равномерны на [0,p]; среднее = p/2
    obs_mean = tail.mean()
    exp_mean = p / 2
    se = (p / np.sqrt(12)) / np.sqrt(len(tail))
    zstat = (obs_mean - exp_mean) / se
    pval = float(2 * (1 - stats.norm.cdf(abs(zstat))))
    return BacktestResult("Acerbi-Szekely ES (Test 3)", float(zstat), pval,
                          pval < 0.05, {"tail_n": len(tail)})


# --------------------------------------------------------------------------- #
# Модельный риск: разброс оценок VaR между методами + множитель Базеля
# --------------------------------------------------------------------------- #
def model_risk_metrics(var_estimates: dict, n_violations: int = None,
                       n_obs: int = None) -> dict:
    """
    Квантификация модельного риска (theory/Model Risk, Glasserman):
        * разброс оценок VaR между методами => неопределённость модели;
        * множитель Базеля по числу пробоев (зелёная/жёлтая/красная зона).
    var_estimates: {'historical': 0.05, 'garch': 0.06, ...}.
    """
    vals = np.array(list(var_estimates.values()), dtype=float)
    spread = (vals.max() - vals.min()) / vals.mean() if vals.mean() else np.nan
    out = {"var_estimates": var_estimates,
           "min": float(vals.min()), "max": float(vals.max()),
           "mean": float(vals.mean()),
           "relative_spread": float(spread),
           "conservative_var": float(vals.max())}
    if n_violations is not None and n_obs is not None:
        # множитель Базеля для 250 дней наблюдений
        scaled = n_violations * 250 / n_obs
        if scaled <= 4:
            mult, zone = 3.0, "GREEN"
        elif scaled <= 9:
            mult, zone = 3.0 + 0.2 * (scaled - 4), "YELLOW"
        else:
            mult, zone = 4.0, "RED"
        out.update({"basel_multiplier": round(mult, 2), "basel_zone": zone,
                    "scaled_violations_250d": round(scaled, 1)})
    return out


# --------------------------------------------------------------------------- #
# Сводный прогон всех тестов
# --------------------------------------------------------------------------- #
def run_var_backtests(returns, var, var_alpha: float = 0.99) -> pd.DataFrame:
    """Прогоняет все VaR-тесты и собирает результаты в таблицу."""
    I = get_violations(returns, var)
    chr_ = christoffersen(I, var_alpha)
    results = [
        kupiec_pof(I, var_alpha),
        chr_["independence"],
        chr_["conditional_coverage"],
        duration_test(I, var_alpha),
        dq_test(returns, var, var_alpha),
        berkowitz_tail(returns, var, var_alpha=var_alpha),
    ]
    return pd.DataFrame([{
        "test": x.name, "statistic": x.statistic, "p_value": x.p_value,
        "reject_H0": x.reject_h0, "verdict": x.verdict(),
    } for x in results])


def traffic_light(p_value: float) -> str:
    """Светофор Базеля по p-value теста покрытия."""
    if p_value > 0.05:
        return "GREEN"
    if p_value > 0.0001:
        return "YELLOW"
    return "RED"
