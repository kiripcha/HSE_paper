"""
Модуль сбора рыночных данных для криптовалютного портфеля.

ПОЛНОСТЬЮ ПЕРЕПИСАН относительно исходного проекта (российские акции / MOEX).

Цепочка происхождения данных («откуда данные?»):
    * первоисточник  — биржевой матчинг-движок Binance (спот). Сделки и стакан
      формируются непосредственно на бирже;
    * транспорт      — публичный REST API Binance (iss-аналог), без ключей и
      аутентификации для публичных рыночных данных;
    * клиент         — библиотека `ccxt` (унифицированный интерфейс к 100+ бирж),
      что позволяет заменить Binance на Bybit/OKX/Kraken одной строкой.

Что собираем:
    * OHLCV свечи произвольного таймфрейма (1m … 1d) — для рыночного риска и
      оптимизации портфеля;
    * снимок стакана (order book L2) — для оценки риска ликвидности и
      транзакционных издержек в HFT-контуре.

Оффлайн-режим: если сети нет (или биржа недоступна), используется генератор
синтетических данных `synthetic_price_panel`, воспроизводящий стилизованные
факты крипторынка (тяжёлые хвосты, кластеризация волатильности, корреляции),
чтобы весь пайплайн оставался воспроизводимым и тестируемым.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import (
    DATA_CACHE,
    DEFAULT_UNIVERSE,
    QUOTE_ASSET,
    RANDOM_SEED,
    TRADING_DAYS_YEAR,
    to_symbol,
)

try:  # ccxt опционален — без него работает синтетический режим
    import ccxt  # type: ignore

    _HAS_CCXT = True
except Exception:  # pragma: no cover
    _HAS_CCXT = False


# --------------------------------------------------------------------------- #
# Низкоуровневая загрузка с биржи
# --------------------------------------------------------------------------- #
_TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
    "4h": 14_400_000, "1d": 86_400_000,
}


def _make_exchange(exchange_id: str = "binance"):
    if not _HAS_CCXT:
        raise RuntimeError("ccxt не установлен")
    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True, "timeout": 20_000})


def _fetch_ohlcv_paginated(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> list[list]:
    """Постранично выкачивает свечи в диапазоне [since, until]."""
    step = _TIMEFRAME_MS[timeframe]
    all_rows: list[list] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + step
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    # отрезаем хвост за пределами until
    all_rows = [r for r in all_rows if r[0] <= until_ms]
    return all_rows


def fetch_ohlcv(
    base: str,
    start: str,
    end: str,
    timeframe: str = "1d",
    quote: str = QUOTE_ASSET,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Свечи одного инструмента в виде DataFrame с индексом-датой."""
    exchange = _make_exchange(exchange_id)
    symbol = to_symbol(base, quote)
    since_ms = exchange.parse8601(f"{start}T00:00:00Z")
    until_ms = exchange.parse8601(f"{end}T00:00:00Z")
    rows = _fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, until_ms)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df.index.name = "date"
    return df


def fetch_order_book_snapshot(
    base: str,
    depth: int = 50,
    quote: str = QUOTE_ASSET,
    exchange_id: str = "binance",
) -> dict:
    """
    Снимок стакана для оценки ликвидности. Возвращает словарь с:
        bids/asks  : массивы [price, size]
        mid        : средняя цена
        spread     : абсолютный спред (ask-bid)
        rel_spread : относительный спред (spread / mid)
    """
    exchange = _make_exchange(exchange_id)
    ob = exchange.fetch_order_book(to_symbol(base, quote), limit=depth)
    bids = np.asarray(ob["bids"], dtype=float)
    asks = np.asarray(ob["asks"], dtype=float)
    best_bid, best_ask = bids[0, 0], asks[0, 0]
    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid
    return {
        "base": base,
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "rel_spread": spread / mid,
        "timestamp": pd.Timestamp.utcnow(),
    }


# --------------------------------------------------------------------------- #
# Синтетический генератор (оффлайн-режим)
# --------------------------------------------------------------------------- #
def synthetic_price_panel(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    start: str = "2021-01-01",
    end: str = "2025-01-01",
    freq: str = "D",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Генерирует панель цен закрытия со стилизованными фактами крипторынка:

        * кластеризация волатильности      — через GJR-GARCH(1,1)-подобный процесс;
        * тяжёлые хвосты                    — стандартизованное Стьюдент-t(ν≈4);
        * сонаправленность (системный риск) — общий рыночный фактор + бета;
        * различие активов                  — индивидуальные дрейф/волатильность.

    Возвращает DataFrame: индекс — даты, колонки — тикеры (цены закрытия).
    Это НЕ замена реальных данных, а детерминированный фолбэк для оффлайна/CI.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(dates)
    k = len(universe)

    # Параметры активов: дневной дрейф, идиосинкратическая σ и бета к рынку.
    # Идиосинкратическая компонента сознательно меньше рыночной, чтобы получить
    # реалистичные для крипты корреляции (~0.5–0.8) и хвостовую зависимость.
    idio_vol = rng.uniform(0.015, 0.035, size=k)        # 1.5–3.5% дневной σ (идио)
    drift = rng.uniform(-0.0002, 0.0010, size=k)        # дневной дрейф
    beta = rng.uniform(0.7, 1.3, size=k)                # бета к рынку
    beta[0] = 1.0                                       # первый актив = «рынок» (BTC)

    # Общий рыночный фактор с кластеризацией волатильности (GARCH), безусловная
    # дневная σ ≈ 3.5%, тяжёлые хвосты (t, ν=4) -> совместные обвалы.
    nu = 4.0
    t_scale = np.sqrt((nu - 2) / nu)
    sig_uncond = 0.035
    omega, alpha, beta_g = sig_uncond ** 2 * (1 - 0.10 - 0.88), 0.10, 0.88
    market_var = np.empty(n)
    market_var[0] = sig_uncond ** 2
    market_shock = np.empty(n)
    for t in range(n):
        z = rng.standard_t(nu) * t_scale
        market_shock[t] = np.sqrt(market_var[t]) * z
        if t + 1 < n:
            market_var[t + 1] = omega + alpha * market_shock[t] ** 2 + beta_g * market_var[t]

    # Идиосинкратические шоки + рыночный фактор
    log_prices = np.zeros((n, k))
    p0 = rng.uniform(np.log(1), np.log(50000), size=k)  # стартовые лог-цены
    log_prices[0] = p0
    for j in range(k):
        idio = rng.standard_t(nu, size=n) * t_scale * idio_vol[j]
        ret = drift[j] + beta[j] * market_shock + idio
        log_prices[1:, j] = p0[j] + np.cumsum(ret[1:])

    prices = pd.DataFrame(np.exp(log_prices), index=dates, columns=list(universe))
    prices.index.name = "date"
    return prices


# --------------------------------------------------------------------------- #
# Высокоуровневый загрузчик с кэшем и фолбэком
# --------------------------------------------------------------------------- #
class CryptoDataLoader:
    """
    Унифицированный загрузчик котировок с тремя режимами:

        mode="auto"      — пытается биржу, при ошибке -> кэш -> синтетика;
        mode="exchange"  — только биржа (ошибка пробрасывается);
        mode="synthetic" — только синтетика (детерминированный оффлайн).

    Кэширует панель цен в parquet, чтобы не дёргать API повторно.
    """

    def __init__(
        self,
        universe: Sequence[str] = DEFAULT_UNIVERSE,
        quote: str = QUOTE_ASSET,
        exchange_id: str = "binance",
        cache_dir: Path = DATA_CACHE,
        mode: str = "auto",
    ):
        self.universe = list(universe)
        self.quote = quote
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.source_used: str | None = None
        self.data_origin: str | None = None   # 'exchange' | 'synthetic'

    def _cache_path(self, start: str, end: str, timeframe: str,
                    origin: str) -> Path:
        # origin ('exchange'/'synthetic') в имени файла, чтобы реальные и
        # синтетические данные НЕ перезаписывали друг друга.
        assets = "-".join(self.universe)
        tag = f"{origin}_{self.exchange_id}_{assets}_{timeframe}_{start}_{end}"
        if len(tag) > 120:
            tag = (f"{origin}_{self.exchange_id}_{len(self.universe)}assets_"
                   f"{timeframe}_{start}_{end}")
        return self.cache_dir / f"prices_{tag}.csv"

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    def load_close_panel(
        self,
        start: str = "2021-01-01",
        end: str = "2025-01-01",
        timeframe: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Панель цен закрытия (колонки — тикеры). Главная точка входа.

        Атрибуты после вызова:
            self.data_origin  : 'exchange' | 'synthetic' — РЕАЛЬНОСТЬ данных;
            self.source_used  : подробный источник (cache/exchange/synthetic).
        """
        exch_cache = self._cache_path(start, end, timeframe, "exchange")
        synth_cache = self._cache_path(start, end, timeframe, "synthetic")

        # 1) синтетический режим — детерминированный, отдельный кэш
        if self.mode == "synthetic":
            if use_cache and synth_cache.exists():
                self.data_origin = "synthetic"
                self.source_used = f"cache:{synth_cache.name}"
                return self._read_cache(synth_cache)
            panel = synthetic_price_panel(self.universe, start, end)
            self.data_origin = "synthetic"
            self.source_used = "synthetic"
            if use_cache:
                panel.to_csv(synth_cache)
            return panel

        # 2) auto/exchange — сначала кэш реальных данных
        if use_cache and exch_cache.exists():
            self.data_origin = "exchange"
            self.source_used = f"cache:{exch_cache.name}"
            return self._read_cache(exch_cache)

        # 3) тянем с биржи
        try:
            panel = self._load_from_exchange(start, end, timeframe)
            self.data_origin = "exchange"
            self.source_used = f"exchange:{self.exchange_id}"
            if use_cache:
                panel.to_csv(exch_cache)
            return panel
        except Exception as exc:  # noqa: BLE001
            if self.mode == "exchange":
                raise
            warnings.warn(
                f"Биржа недоступна ({exc!r}); переключаюсь на синтетические данные.",
                RuntimeWarning,
            )
            # фолбэк НЕ кэшируем под именем exchange, чтобы при следующем запуске
            # повторить попытку получить реальные данные
            panel = synthetic_price_panel(self.universe, start, end)
            self.data_origin = "synthetic"
            self.source_used = "synthetic(fallback)"
            return panel

    def _load_from_exchange(self, start: str, end: str, timeframe: str) -> pd.DataFrame:
        cols = {}
        for base in self.universe:
            df = fetch_ohlcv(base, start, end, timeframe, self.quote, self.exchange_id)
            cols[base] = df["close"]
        panel = pd.DataFrame(cols).dropna(how="all").ffill().dropna()
        return panel

    def load_ohlcv(
        self,
        base: str,
        start: str = "2021-01-01",
        end: str = "2025-01-01",
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Полные OHLCV одного инструмента (для волатильности High-Low и т.п.)."""
        if self.mode == "synthetic":
            close = synthetic_price_panel([base], start, end)[base]
            return _ohlcv_from_close(close)
        try:
            return fetch_ohlcv(base, start, end, timeframe, self.quote, self.exchange_id)
        except Exception:
            if self.mode == "exchange":
                raise
            close = synthetic_price_panel([base], start, end)[base]
            return _ohlcv_from_close(close)


def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    """Достраивает грубые OHLCV из ряда цен закрытия (для синтетики)."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = np.abs(rng.normal(0, 0.01, size=len(close)))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.lognormal(10, 1, size=len(close)), index=close.index)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


def load_prices(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    start: str = "2021-01-01",
    end: str = "2025-01-01",
    timeframe: str = "1d",
    mode: str = "auto",
) -> pd.DataFrame:
    """Функциональная обёртка над CryptoDataLoader для быстрого использования."""
    loader = CryptoDataLoader(universe=universe, mode=mode)
    return loader.load_close_panel(start=start, end=end, timeframe=timeframe)
