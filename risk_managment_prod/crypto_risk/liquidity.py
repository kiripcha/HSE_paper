"""
Риск ликвидности и транзакционные издержки для HFT-контура.

Для высокоскоростной торговой системы рыночный риск неотделим от риска
ликвидности: позицию нельзя закрыть «по середине стакана». Модуль реализует:

    * Liquidity-adjusted VaR (LVaR) по Bangia, Diebold, Schuermann, Stroughair (1999)
      — экзогенная компонента спреда добавляется к ценовому VaR;
    * модели рыночного воздействия (market impact): линейная (Kyle) и
      квадратного корня (Almgren) — эндогенная ликвидность крупной заявки;
    * Implementation Shortfall (Perold, 1988) — разрыв между «бумажной» и
      реализованной доходностью, встраиваемый в оптимизатор (задание 25*****);
    * оптимальное исполнение Almgren-Chriss — траектория ликвидации,
      минимизирующая E[издержки] + lambda·Var[издержки];
    * меры неликвидности Amihud и оценка Kyle's lambda по данным.

Все стоимости — в долях (relative) либо в денежных единицах котировки (USDT).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import LIQUIDITY_SPREAD_MULT, VAR_CONFIDENCE


# --------------------------------------------------------------------------- #
# 1. Exogenous-spread LVaR (Bangia et al. 1999)
# --------------------------------------------------------------------------- #
@dataclass
class LVaRResult:
    price_var: float          # ценовой VaR (доля)
    liquidity_cost: float     # компонента ликвидности (доля)
    lvar: float               # суммарный LVaR (доля)
    cost_share: float         # вклад ликвидности в LVaR
    detail: dict

    def __repr__(self) -> str:
        return (f"LVaR={self.lvar:.4f} (price={self.price_var:.4f} + "
                f"liq={self.liquidity_cost:.4f}, liq-share={self.cost_share:.1%})")


def bangia_lvar(
    price_var: float,
    rel_spread_mean: float,
    rel_spread_std: float,
    spread_mult: float = LIQUIDITY_SPREAD_MULT,
) -> LVaRResult:
    """
    Модель Bangia et al. (1999), экзогенная (стакан как данность):

        LVaR = P·VaR_price + 0.5·(mu_S + a·sigma_S),

    где S — относительный спред (ask-bid)/mid, a — множитель «худшего» спреда
    (для тяжёлых хвостов крипты a≈3). Компонента 0.5·(...) — половина спреда,
    которую теряем при немедленной ликвидации.

    price_var передаётся в долях (например 0.05 = 5%); spread — в долях.
    """
    liq = 0.5 * (rel_spread_mean + spread_mult * rel_spread_std)
    lvar = price_var + liq
    return LVaRResult(
        price_var=float(price_var),
        liquidity_cost=float(liq),
        lvar=float(lvar),
        cost_share=float(liq / lvar) if lvar > 0 else 0.0,
        detail={"mu_S": rel_spread_mean, "sigma_S": rel_spread_std,
                "a": spread_mult},
    )


def rel_spread_stats_from_book(order_book: dict) -> tuple[float, float]:
    """Грубая оценка mu_S, sigma_S из одного снимка стакана (для демо)."""
    rs = order_book["rel_spread"]
    # при отсутствии истории спреда берём sigma как долю от среднего
    return float(rs), float(rs * 0.5)


def rel_spread_stats_from_ohlc(ohlc: pd.DataFrame) -> tuple[float, float]:
    """
    Прокси относительного спреда по Corwin-Schultz (high-low estimator),
    когда данных стакана нет — только OHLCV. Возвращает (mean, std).
    """
    hl = np.log(ohlc["high"] / ohlc["low"]) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(ohlc[["high"]].rolling(2).max().squeeze()
                   / ohlc[["low"]].rolling(2).min().squeeze()) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0).dropna()
    if spread.empty:
        return 0.0005, 0.0002
    return float(spread.mean()), float(spread.std())


# --------------------------------------------------------------------------- #
# 2. Market impact (эндогенная ликвидность)
# --------------------------------------------------------------------------- #
def square_root_impact(order_size: float, adv: float, sigma: float,
                       y: float = 1.0) -> float:
    """
    Закон квадратного корня (Almgren et al. 2005; широко подтверждён на крипте):
        impact ≈ Y · sigma · sqrt(Q / ADV),
    где Q — объём заявки, ADV — средний дневной объём, sigma — дневная волатильность.
    Возвращает относительное проскальзывание (долю цены).
    """
    if adv <= 0:
        return np.nan
    return y * sigma * np.sqrt(order_size / adv)


def linear_impact(order_size: float, kyle_lambda: float) -> float:
    """Линейное воздействие Кайла: ΔP = lambda · Q."""
    return kyle_lambda * order_size


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series) -> float:
    """
    Мера неликвидности Amihud (2002): среднее |r_t| / DollarVolume_t.
    Чем выше, тем сильнее цена двигается на единицу оборота => тем менее ликвиден.
    """
    ratio = (returns.abs() / dollar_volume.replace(0, np.nan)).dropna()
    return float(ratio.mean())


def estimate_kyle_lambda(price_changes: pd.Series, signed_volume: pd.Series) -> float:
    """
    Оценка Kyle's lambda регрессией ΔP_t = lambda · (signed order flow) + e.
    signed_volume = знак сделки * объём (или приближение по правилу тика).
    """
    x = signed_volume.values
    y = price_changes.values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or x.std() == 0:
        return np.nan
    lam = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    return float(lam)


# --------------------------------------------------------------------------- #
# 3. Implementation Shortfall (Perold 1988)
# --------------------------------------------------------------------------- #
@dataclass
class ImplementationShortfall:
    total_bps: float
    spread_cost_bps: float
    impact_cost_bps: float
    timing_cost_bps: float
    fees_bps: float
    detail: dict


def implementation_shortfall(
    decision_price: float,
    executed_prices: np.ndarray,
    executed_sizes: np.ndarray,
    side: str = "buy",
    half_spread: float = 0.0,
    fee_rate: float = 0.0005,
    arrival_to_decision_drift: float = 0.0,
) -> ImplementationShortfall:
    """
    Implementation Shortfall = разница между стоимостью исполнения и «бумажной»
    стоимостью по цене принятия решения, разложенная на компоненты:
        spread + market impact + timing + комиссии.
    Возвращает издержки в базисных пунктах (б.п.) от номинала.
    """
    sign = 1.0 if side == "buy" else -1.0
    total_qty = executed_sizes.sum()
    avg_exec = np.average(executed_prices, weights=executed_sizes)
    notional = total_qty * decision_price

    # реализованный shortfall (для покупки: переплата относительно decision_price)
    is_cost = sign * (avg_exec - decision_price) * total_qty
    fees = fee_rate * total_qty * avg_exec
    spread_cost = half_spread * total_qty * decision_price
    timing = sign * arrival_to_decision_drift * total_qty * decision_price
    impact = is_cost - spread_cost - timing

    to_bps = lambda x: 1e4 * x / notional if notional else 0.0
    return ImplementationShortfall(
        total_bps=to_bps(is_cost + fees),
        spread_cost_bps=to_bps(spread_cost),
        impact_cost_bps=to_bps(impact),
        timing_cost_bps=to_bps(timing),
        fees_bps=to_bps(fees),
        detail={"avg_exec": avg_exec, "decision_price": decision_price,
                "notional": notional},
    )


# --------------------------------------------------------------------------- #
# 4. Almgren-Chriss — оптимальное исполнение
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionSchedule:
    times: np.ndarray
    holdings: np.ndarray      # x_j — остаток позиции
    trades: np.ndarray        # n_j — объём в каждом интервале
    expected_cost: float      # E[издержки]
    cost_variance: float      # Var[издержки]
    kappa: float


class AlmgrenChriss:
    """
    Модель оптимальной ликвидации Almgren-Chriss (2000).

    Параметры воздействия:
        eta   — коэффициент ВРЕМЕННОГО воздействия (на скорость торговли);
        gamma — коэффициент ПОСТОЯННОГО воздействия (на объём);
        sigma — волатильность цены (на единицу времени);
        lam   — неприятие риска (risk aversion).

    Оптимальная траектория:
        x_j = X · sinh(kappa·(T - t_j)) / sinh(kappa·T),
        kappa = arccosh(kappa_tilde^2/2 + 1)/tau,
        kappa_tilde^2 = lam·sigma^2·tau / eta_hat,  eta_hat = eta - 0.5·gamma·tau.
    """

    def __init__(self, sigma: float, eta: float, gamma: float, lam: float = 1e-6):
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = lam

    def schedule(self, total_shares: float, horizon: float, n_steps: int
                 ) -> ExecutionSchedule:
        X, T, N = total_shares, horizon, n_steps
        tau = T / N
        eta_hat = self.eta - 0.5 * self.gamma * tau
        kappa_tilde2 = self.lam * self.sigma ** 2 * tau / eta_hat
        kappa = np.arccosh(kappa_tilde2 / 2 + 1) / tau

        t = np.arange(N + 1) * tau
        if kappa * T < 1e-8:  # предел -> TWAP (равномерно)
            x = X * (1 - t / T)
        else:
            x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        n = -np.diff(x)  # объёмы торговли по интервалам (>0 = продаём)

        # E[издержки] (permanent + temporary) и Var
        perm = 0.5 * self.gamma * X ** 2
        temp = self.eta / tau * np.sum(n ** 2) if tau > 0 else 0.0
        e_cost = perm + temp
        v_cost = self.sigma ** 2 * np.sum(tau * x[:-1] ** 2)
        return ExecutionSchedule(times=t, holdings=x, trades=n,
                                 expected_cost=float(e_cost),
                                 cost_variance=float(v_cost), kappa=float(kappa))

    def efficient_frontier(self, total_shares: float, horizon: float,
                           n_steps: int, lambdas: np.ndarray) -> pd.DataFrame:
        """Граница 'издержки-риск' исполнения для разных lambda (аналог Марковица)."""
        rows = []
        for lam in lambdas:
            ac = AlmgrenChriss(self.sigma, self.eta, self.gamma, lam)
            s = ac.schedule(total_shares, horizon, n_steps)
            rows.append({"lambda": lam, "E_cost": s.expected_cost,
                         "Std_cost": np.sqrt(s.cost_variance), "kappa": s.kappa})
        return pd.DataFrame(rows)


def liquidity_adjusted_position_limit(
    capital: float, adv: float, price: float,
    max_participation: float = 0.10,
) -> float:
    """
    Лимит позиции с учётом ликвидности: не более max_participation от ADV,
    чтобы рыночное воздействие при выходе оставалось контролируемым.
    Возвращает максимальный нотионал (USDT).
    """
    max_qty = max_participation * adv
    return float(max_qty * price)
