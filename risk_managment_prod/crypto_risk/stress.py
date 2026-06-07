"""
Стресс-тестирование портфеля.

VaR/ES описывают «нормальный» риск; стресс-тесты отвечают на вопрос «что будет в
кризис», когда статистические допущения ломаются (теория из theory/Stress Testing:
Breuer, Rebonato, Cont о кластеризации волатильности). Реализовано:

    * historical_scenarios   — реплей худших исторических окон (data-driven);
    * named_crypto_crashes   — известные крипто-обвалы (COVID-2020, LUNA/UST,
                               FTX, и т.п.) как именованные сценарии;
    * hypothetical_shock     — заданные пользователем шоки по активам;
    * correlation_stress     — стресс корреляций «к 1» (диверсификация исчезает);
    * volatility_stress      — мультипликативный стресс волатильностей;
    * worst_case_loss        — максимальный убыток на эллипсоиде правдоподобия
                               (Breuer «plausible worst-case scenario»);
    * reverse_stress_test    — обратный стресс-тест: найти сценарий заданной
                               тяжести убытка и факторы, которые его вызывают.

Все убытки — в долях капитала (положительные = потери), как в var_es.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StressResult:
    name: str
    pnl: float           # P&L портфеля (доля; <0 = убыток)
    detail: dict

    def __repr__(self) -> str:
        return f"[{self.name}] P&L = {self.pnl:+.2%}"


# --------------------------------------------------------------------------- #
# 1. Реплей худших исторических окон
# --------------------------------------------------------------------------- #
def historical_scenarios(returns: pd.DataFrame, weights: np.ndarray,
                         horizon: int = 1, top: int = 5) -> pd.DataFrame:
    """
    Худшие исторические периоды длиной `horizon` для текущих весов портфеля.
    Это самый честный стресс — реальные совместные движения активов.
    """
    w = np.asarray(weights, dtype=float)
    port = returns.values @ w
    if horizon > 1:
        port = np.convolve(port, np.ones(horizon), mode="valid")
        idx = returns.index[horizon - 1:]
    else:
        idx = returns.index
    s = pd.Series(port, index=idx[:len(port)])
    worst = s.nsmallest(top)
    return pd.DataFrame({"date": worst.index, f"{horizon}d_PnL": worst.values})


def named_crypto_crashes(returns: pd.DataFrame, weights: np.ndarray,
                         windows: dict | None = None) -> list[StressResult]:
    """
    Применяет известные крипто-кризисы к текущему портфелю (если периоды есть
    в данных). Считает кумулятивный P&L за окно события.
    """
    if windows is None:
        windows = {
            "COVID-крах (мар-2020)":      ("2020-03-08", "2020-03-13"),
            "Распродажа май-2021":        ("2021-05-12", "2021-05-23"),
            "Коллапс LUNA/UST (май-2022)":("2022-05-08", "2022-05-13"),
            "Банкротство FTX (ноя-2022)": ("2022-11-07", "2022-11-10"),
            "Кризис банков/USDC (мар-2023)":("2023-03-09","2023-03-11"),
        }
    w = np.asarray(weights, dtype=float)
    port = pd.Series(returns.values @ w, index=returns.index)
    out = []
    for name, (a, b) in windows.items():
        seg = port.loc[(port.index >= a) & (port.index <= b)]
        if len(seg) == 0:
            continue
        out.append(StressResult(name, float(seg.sum()),
                                {"days": len(seg), "from": a, "to": b}))
    return out


# --------------------------------------------------------------------------- #
# 2. Гипотетические сценарии
# --------------------------------------------------------------------------- #
def hypothetical_shock(weights: dict, shocks: dict) -> StressResult:
    """
    Заданные шоки доходностей по активам, напр. {'BTC': -0.30, 'ETH': -0.40}.
    Активы без шока считаются неизменными (0).
    """
    pnl = sum(weights.get(a, 0.0) * shocks.get(a, 0.0) for a in weights)
    return StressResult("Гипотетический шок", float(pnl), {"shocks": shocks})


def correlation_stress(returns: pd.DataFrame, weights: np.ndarray,
                       target_corr: float = 0.95, var_alpha: float = 0.99
                       ) -> dict:
    """
    Стресс корреляций «к 1»: в кризис диверсификация исчезает. Пересчитываем
    σ портфеля и нормальный VaR при стрессовой корреляционной матрице, сохраняя
    индивидуальные волатильности.
    """
    w = np.asarray(weights, dtype=float)
    std = returns.std().values
    base_corr = returns.corr().values
    n = len(std)
    stressed = np.full((n, n), target_corr)
    np.fill_diagonal(stressed, 1.0)
    cov_base = base_corr * np.outer(std, std)
    cov_str = stressed * np.outer(std, std)
    z = stats.norm.ppf(1 - var_alpha)
    vol_base = np.sqrt(w @ cov_base @ w)
    vol_str = np.sqrt(w @ cov_str @ w)
    return {"vol_base": float(vol_base), "vol_stressed": float(vol_str),
            "var_base": float(-z * vol_base), "var_stressed": float(-z * vol_str),
            "vol_increase": float(vol_str / vol_base - 1)}


def volatility_stress(returns: pd.DataFrame, weights: np.ndarray,
                      vol_mult: float = 2.0, var_alpha: float = 0.99) -> dict:
    """Мультипликативный стресс волатильностей (×vol_mult) при той же корреляции."""
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values
    z = stats.norm.ppf(1 - var_alpha)
    vol_base = np.sqrt(w @ cov @ w)
    vol_str = vol_base * vol_mult
    return {"var_base": float(-z * vol_base), "var_stressed": float(-z * vol_str),
            "vol_mult": vol_mult}


# --------------------------------------------------------------------------- #
# 3. Worst-case loss и обратный стресс-тест (Breuer)
# --------------------------------------------------------------------------- #
def worst_case_loss(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                    plausibility_k: float = 3.0) -> dict:
    """
    Максимальный убыток на эллипсоиде правдоподобия (Breuer et al.):
        макс по r убыток -w'r при (r-μ)'Σ^{-1}(r-μ) <= k².
    Решение замкнутое: r* = μ - k·Σw / sqrt(w'Σw),
        worst loss = -w'μ + k·sqrt(w'Σw).
    k — «радиус» в единицах махаланобисова расстояния (k=3 ≈ очень редкое событие).
    Возвращает убыток и сам сценарий (шоки по активам).
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    port_vol = np.sqrt(w @ cov @ w)
    scenario = mu - plausibility_k * (cov @ w) / port_vol
    loss = -(w @ mu) + plausibility_k * port_vol
    return {"worst_loss": float(loss), "plausibility_k": plausibility_k,
            "scenario_returns": scenario, "port_vol": float(port_vol)}


def reverse_stress_test(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                        target_loss: float, names: list[str] | None = None
                        ) -> dict:
    """
    Обратный стресс-тест: задан КАТАСТРОФИЧЕСКИЙ убыток (например, -50% капитала) —
    найти наиболее правдоподобный сценарий, который его вызывает, и факторы-драйверы.

        требуемое k = (target_loss + w'μ) / sqrt(w'Σw),
        сценарий r* = μ - k·Σw/sqrt(w'Σw).
    Меньшее k => более правдоподобный (и оттого более тревожный) сценарий.
    Возвращает k, P-значение события и ранжированный вклад активов в убыток.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(w)
    port_vol = np.sqrt(w @ cov @ w)
    k = (target_loss + (w @ mu)) / port_vol
    scenario = mu - k * (cov @ w) / port_vol
    contrib = w * scenario  # вклад каждого актива в P&L сценария
    # приблизительная вероятность события (χ² с n ст. свободы для махаланобиса)
    prob = float(1 - stats.chi2.cdf(k ** 2, df=n))
    names = names or [f"A{i}" for i in range(n)]
    contrib_s = pd.Series(contrib, index=names).sort_values()
    return {"required_k": float(k), "approx_probability": prob,
            "scenario_returns": pd.Series(scenario, index=names),
            "loss_contribution": contrib_s,
            "main_driver": contrib_s.index[0]}
