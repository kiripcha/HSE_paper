"""
Модели волатильности риск-факторов.

Стилизованные факты доходностей криптовалют, которые модель обязана учитывать:
    1. кластеризация волатильности  -> GARCH-семейство;
    2. тяжёлые хвосты               -> инновации Стьюдента (Student-t / skew-t);
    3. эффект рычага (асимметрия)   -> GJR-GARCH / EGARCH;
    4. высокая безусловная σ         -> аннуализация по 365 дням (крипта 24/7).

Реализованы:
    * EWMA / RiskMetrics            — быстрый рекурсивный прогноз σ²;
    * GARCHModel (обёртка над arch) — GARCH/GJR/EGARCH с norm/t/skewt;
    * утилиты прогноза дисперсии на горизонт h по правилу агрегации.

Все модели возвращают условную σ в терминах ДОЛЕЙ (не процентов).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EWMA_LAMBDA, TRADING_DAYS_YEAR

try:
    from arch import arch_model  # type: ignore

    _HAS_ARCH = True
except Exception:  # pragma: no cover
    _HAS_ARCH = False


# --------------------------------------------------------------------------- #
# Базовые преобразования
# --------------------------------------------------------------------------- #
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Логарифмические доходности r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().dropna()


def annualize_vol(daily_vol: float, periods: int = TRADING_DAYS_YEAR) -> float:
    return daily_vol * np.sqrt(periods)


# --------------------------------------------------------------------------- #
# EWMA / RiskMetrics
# --------------------------------------------------------------------------- #
def ewma_volatility(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """
    Экспоненциально взвешенная оценка σ_t (RiskMetrics):
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.

    Это и есть «схема взвешивания наблюдений с экспоненциальным забыванием»
    из исходного ТЗ, перенесённая на оценку волатильности.
    """
    r = returns.dropna().values
    n = len(r)
    var = np.empty(n)
    var[0] = r[0] ** 2 if n else 0.0
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.dropna().index, name="ewma_vol")


def ewma_forecast(returns: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """Однодневный прогноз σ_{T+1} по EWMA на конец выборки."""
    r = returns.dropna().values
    var = r[0] ** 2
    for t in range(1, len(r)):
        var = lam * var + (1 - lam) * r[t - 1] ** 2
    # шаг на T+1 использует последнее наблюдение
    var = lam * var + (1 - lam) * r[-1] ** 2
    return float(np.sqrt(var))


def ewma_covariance(returns: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """EWMA ковариационная матрица на конец выборки (для портфеля)."""
    X = returns.dropna().values
    n, k = X.shape
    cov = np.cov(X, rowvar=False)
    for t in range(1, n):
        x = X[t - 1].reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (x @ x.T)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


# --------------------------------------------------------------------------- #
# GARCH-семейство (обёртка над пакетом arch)
# --------------------------------------------------------------------------- #
@dataclass
class GARCHResult:
    sigma: pd.Series          # in-sample условная σ
    forecast_var: np.ndarray  # прогноз дисперсии на горизонт (длины h)
    params: dict
    dist: str
    vol: str
    nu: float | None          # степени свободы (для t)
    model_obj: object         # обученный объект arch для дальнейших прогнозов
    aic: float
    bic: float

    def forecast_sigma_path(self) -> np.ndarray:
        return np.sqrt(self.forecast_var)

    def horizon_sigma(self, h: int | None = None) -> float:
        """σ для совокупной доходности на горизонте h (sqrt от суммы дисперсий)."""
        v = self.forecast_var if h is None else self.forecast_var[:h]
        return float(np.sqrt(np.sum(v)))


class GARCHModel:
    """
    Обёртка над arch_model с человеческим интерфейсом.

    Parameters
    ----------
    vol  : 'GARCH' | 'EGARCH' | 'GJR'   (GJR = GARCH с o=1)
    dist : 'normal' | 't' | 'skewt'
    p, q, o : порядки модели.

    Доходности подаются в процентах внутри (arch так стабильнее), наружу всё
    возвращается в долях.
    """

    def __init__(self, vol: str = "GARCH", dist: str = "t", p: int = 1, q: int = 1, o: int = 0):
        if not _HAS_ARCH:
            raise RuntimeError("Пакет 'arch' не установлен — GARCH недоступен.")
        self.vol = "GARCH" if vol.upper() == "GJR" else vol
        self.o = 1 if vol.upper() == "GJR" else o
        self.dist = dist
        self.p, self.q = p, q
        self._scale = 100.0  # доли -> проценты

    def fit(self, returns: pd.Series, horizon: int = 1) -> GARCHResult:
        r = returns.dropna() * self._scale
        am = arch_model(
            r, mean="Constant", vol=self.vol, p=self.p, o=self.o, q=self.q, dist=self.dist
        )
        res = am.fit(disp="off")
        cond_sigma = res.conditional_volatility / self._scale
        cond_sigma.index = r.index

        fc = res.forecast(horizon=horizon, reindex=False)
        # дисперсия в долях^2
        fvar = fc.variance.values.ravel() / (self._scale ** 2)

        nu = res.params.get("nu", None)
        return GARCHResult(
            sigma=cond_sigma,
            forecast_var=fvar,
            params=dict(res.params),
            dist=self.dist,
            vol=("GJR" if self.o else self.vol),
            nu=float(nu) if nu is not None else None,
            model_obj=res,
            aic=float(res.aic),
            bic=float(res.bic),
        )


# --------------------------------------------------------------------------- #
# Range-based оценки волатильности (используют OHLC, не только close)
# --------------------------------------------------------------------------- #
# Эти оценки в 5–14 раз эффективнее close-to-close при том же числе наблюдений
# (теория из theory/Volatility): используют внутридневной размах H-L-O-C.
def parkinson_vol(ohlc: pd.DataFrame) -> pd.Series:
    """Паркинсон (1980): σ² = (1/4ln2)·(ln H/L)². Не учитывает гэпы/дрейф."""
    hl = np.log(ohlc["high"] / ohlc["low"]) ** 2
    return np.sqrt(hl / (4 * np.log(2)))


def garman_klass_vol(ohlc: pd.DataFrame) -> pd.Series:
    """Гарман-Класс (1980): эффективнее Паркинсона, учитывает open/close."""
    hl = 0.5 * np.log(ohlc["high"] / ohlc["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(ohlc["close"] / ohlc["open"]) ** 2
    return np.sqrt((hl - co).clip(lower=0))


def rogers_satchell_vol(ohlc: pd.DataFrame) -> pd.Series:
    """Роджерс-Сатчелл (1991): корректно работает при ненулевом дрейфе."""
    h, l, c, o = (np.log(ohlc[x]) for x in ["high", "low", "close", "open"])
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    return np.sqrt(rs.clip(lower=0))


def yang_zhang_vol(ohlc: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Янг-Чжанг (2000): минимальная дисперсия оценки, учитывает и гэпы, и дрейф.
    Комбинация overnight-, open-to-close- и Rogers-Satchell-компонент.
    Возвращает скользящую (window) дневную σ.
    """
    o, h, l, c = (np.log(ohlc[x]) for x in ["open", "high", "low", "close"])
    co = c - o
    oc_prev = o - c.shift(1)
    sigma_o2 = oc_prev.rolling(window).var()
    sigma_c2 = co.rolling(window).var()
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    sigma_rs2 = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt((sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2).clip(lower=0))


@dataclass
class HARResult:
    params: dict
    fitted: pd.Series
    forecast: float        # прогноз RV на следующий период (дисперсия)
    forecast_vol: float    # sqrt(forecast) — σ
    r2: float


def har_rv(realized_var: pd.Series) -> HARResult:
    """
    HAR-RV (Corsi 2009) — Heterogeneous AutoRegressive модель реализованной
    дисперсии. Каскад «горизонтов трейдеров»: день / неделя (5) / месяц (22):

        RV_{t+1} = c + β_d·RV_t + β_w·RV_t^{(5)} + β_m·RV_t^{(22)} + ε.

    Простая, но эмпирически очень точная для прогноза волатильности (точнее
    многих GARCH). На вход — ряд реализованной ДИСПЕРСИИ (например, из
    range-оценок), на выход — прогноз σ.
    """
    rv = realized_var.dropna()
    rv_d = rv
    rv_w = rv.rolling(5).mean()
    rv_m = rv.rolling(22).mean()
    df = pd.concat([rv.shift(-1), rv_d, rv_w, rv_m], axis=1).dropna()
    df.columns = ["y", "d", "w", "m"]
    X = np.column_stack([np.ones(len(df)), df["d"], df["w"], df["m"]])
    y = df["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # прогноз на следующий шаг по последним доступным значениям
    last = np.array([1.0, rv_d.iloc[-1], rv_w.iloc[-1], rv_m.iloc[-1]])
    fc = float(max(beta @ last, 1e-12))
    return HARResult(params={"c": beta[0], "beta_d": beta[1], "beta_w": beta[2],
                             "beta_m": beta[3]},
                     fitted=pd.Series(fitted, index=df.index),
                     forecast=fc, forecast_vol=float(np.sqrt(fc)), r2=float(r2))


def select_best_garch(
    returns: pd.Series,
    candidates: tuple[tuple[str, str], ...] = (
        ("GARCH", "normal"),
        ("GARCH", "t"),
        ("GJR", "t"),
        ("EGARCH", "t"),
        ("GJR", "skewt"),
    ),
    criterion: str = "bic",
    horizon: int = 1,
) -> tuple[GARCHResult, pd.DataFrame]:
    """
    Перебирает спецификации GARCH и выбирает лучшую по AIC/BIC.
    Возвращает (лучший результат, таблицу сравнения) — это закрывает требование
    ТЗ «обосновать и критически обсудить выбор стохастической модели».
    """
    rows = []
    best, best_score = None, np.inf
    for vol, dist in candidates:
        try:
            res = GARCHModel(vol=vol, dist=dist).fit(returns, horizon=horizon)
        except Exception as exc:  # noqa: BLE001
            rows.append({"vol": vol, "dist": dist, "aic": np.nan, "bic": np.nan,
                         "error": str(exc)[:40]})
            continue
        score = res.bic if criterion == "bic" else res.aic
        rows.append({"vol": res.vol, "dist": dist, "aic": res.aic, "bic": res.bic,
                     "nu": res.nu, "error": ""})
        if score < best_score:
            best, best_score = res, score
    table = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    return best, table
