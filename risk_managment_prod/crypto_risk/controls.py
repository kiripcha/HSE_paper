"""
Адаптивные риск-контроли торговой системы.

Связывает оценку риска с управлением позицией («адаптивные стратегии» из темы
проекта). Реализованы стандартные пруденциальные механизмы:

    * volatility targeting   — масштабирование экспозиции под целевую σ портфеля;
    * Kelly / fractional Kelly— оптимальный по росту капитала размер позиции;
    * drawdown control       — деривингование при просадке (risk-off режим);
    * stop-loss на основе VaR — динамический стоп = k·VaR;
    * risk budgeting         — распределение риск-бюджета между активами;
    * VolForecaster          — интерфейс прогноза волатильности с хуком под
                               DL-модель (LSTM/Temporal CNN); дефолт — EWMA/GARCH.

Эти контроли вызываются движком RiskEngine перед сайзингом и исполнением.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_YEAR
from .volatility import ewma_forecast


# --------------------------------------------------------------------------- #
# Volatility targeting
# --------------------------------------------------------------------------- #
def vol_target_leverage(forecast_vol_annual: float, target_vol_annual: float = 0.20,
                        max_leverage: float = 3.0) -> float:
    """
    Плечо для приведения волатильности к целевой:
        L = target_vol / forecast_vol,  с ограничением сверху.
    При росте прогнозной σ система автоматически снижает экспозицию.
    """
    if forecast_vol_annual <= 0:
        return 0.0
    return float(min(target_vol_annual / forecast_vol_annual, max_leverage))


def vol_target_weights(weights: np.ndarray, cov: np.ndarray,
                       target_vol_annual: float = 0.20,
                       max_leverage: float = 3.0) -> np.ndarray:
    """Масштабирует вектор весов так, чтобы σ портфеля = целевой."""
    port_vol = np.sqrt(weights @ cov @ weights)  # уже аннуализированная cov
    lev = vol_target_leverage(port_vol, target_vol_annual, max_leverage)
    return weights * lev


# --------------------------------------------------------------------------- #
# Kelly criterion
# --------------------------------------------------------------------------- #
def kelly_fraction(mu: float, sigma: float, rf: float = 0.0) -> float:
    """
    Доля капитала по критерию Келли для непрерывного случая:
        f* = (mu - rf) / sigma^2.
    Все величины — в одинаковом масштабе (например, дневные).
    """
    if sigma <= 0:
        return 0.0
    return float((mu - rf) / sigma ** 2)


def kelly_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
                  fraction: float = 0.5) -> np.ndarray:
    """
    Многомерный Келли: w* = Σ^{-1}(mu - rf). На практике берут дробного Келли
    (fraction≈0.25–0.5) ради робастности к ошибкам оценки — критично для крипты.
    """
    inv = np.linalg.pinv(cov)
    w = inv @ (np.asarray(mu) - rf)
    return fraction * w


# --------------------------------------------------------------------------- #
# Drawdown control
# --------------------------------------------------------------------------- #
def max_drawdown(equity: pd.Series) -> float:
    """Максимальная просадка кривой капитала (доля)."""
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def drawdown_scale(current_dd: float, dd_limit: float = 0.20,
                   floor: float = 0.0) -> float:
    """
    Множитель экспозиции в зависимости от текущей просадки:
        при dd=0 -> 1.0; при dd>=dd_limit -> floor (risk-off).
    Линейное деривингование между порогами.
    """
    cd = abs(current_dd)
    if cd >= dd_limit:
        return floor
    return float(1.0 - (1.0 - floor) * cd / dd_limit)


# --------------------------------------------------------------------------- #
# VaR-based stop-loss
# --------------------------------------------------------------------------- #
def var_stop_loss(entry_price: float, var_fraction: float, side: str = "long",
                  k: float = 1.0) -> float:
    """Уровень стоп-лосса = k·VaR от цены входа (VaR в долях)."""
    if side == "long":
        return entry_price * (1 - k * var_fraction)
    return entry_price * (1 + k * var_fraction)


# --------------------------------------------------------------------------- #
# Risk budgeting
# --------------------------------------------------------------------------- #
def risk_budget_weights(cov: np.ndarray, budget: np.ndarray | None = None,
                        iters: int = 500) -> np.ndarray:
    """
    Веса при заданном риск-бюджете b_i (вклад актива i в общий риск = b_i).
    budget=None -> равный вклад (ERC). Итеративный алгоритм Spinu.
    """
    n = cov.shape[0]
    b = np.ones(n) / n if budget is None else np.asarray(budget) / np.sum(budget)
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        w = w * (b / (rc + 1e-12)) ** 0.5
        w = np.clip(w, 1e-10, None)
        w = w / w.sum()
    return w


# --------------------------------------------------------------------------- #
# Прогноз волатильности с хуком под DL
# --------------------------------------------------------------------------- #
@dataclass
class VolForecast:
    sigma_daily: float
    sigma_annual: float
    source: str


class VolForecaster:
    """
    Интерфейс прогноза волатильности. По умолчанию — EWMA/GARCH. Метод
    `set_dl_model` позволяет подключить обученную DL-модель (LSTM, TCN,
    Transformer): любой объект с .predict(window)->sigma_daily. Так модуль
    риск-менеджмента интегрируется с DL-ядром торговой системы.
    """

    def __init__(self, method: str = "ewma", lam: float = 0.94,
                 trading_days: int = TRADING_DAYS_YEAR):
        self.method = method
        self.lam = lam
        self.trading_days = trading_days
        self._dl_model = None

    def set_dl_model(self, model) -> "VolForecaster":
        """Подключает DL-модель прогноза σ (duck-typing: .predict(returns))."""
        self._dl_model = model
        self.method = "dl"
        return self

    def forecast(self, returns: pd.Series, garch_result=None) -> VolForecast:
        if self.method == "dl" and self._dl_model is not None:
            sig = float(self._dl_model.predict(returns))
            src = "dl"
        elif self.method == "garch" and garch_result is not None:
            sig = garch_result.horizon_sigma(1)
            src = "garch"
        else:
            sig = ewma_forecast(returns, lam=self.lam)
            src = "ewma"
        return VolForecast(sigma_daily=sig,
                           sigma_annual=sig * np.sqrt(self.trading_days),
                           source=src)
