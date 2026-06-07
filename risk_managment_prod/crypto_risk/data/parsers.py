"""
Многоисточниковый парсер криптовалютных котировок с авто-фолбэком.

Зачем несколько источников. Ни один бесплатный источник не идеален: Binance
начал торги лишь в 2017 г. (нет истории с 2015), CoinGecko на free-тарифе режет
длинную историю, у бирж бывают пропуски/делистинги. Поэтому парсер опрашивает
источники по приоритету и ДОЗАПОЛНЯЕТ пропуски из следующего источника
(combine_first), что даёт максимально полный и длинный ряд.

Цепочка происхождения данных:
    1. CryptoCompare  — агрегатор сделок десятков бирж, дневная история с 2010 г.
       (первоисточник — биржевой матчинг; CryptoCompare усредняет VWAP по биржам);
    2. Yahoo Finance  — индексные крипто-котировки (агрегатор CoinMarketCap),
       история с 2014–2015 гг.;
    3. CoinGecko      — рыночные данные (best-effort, free-тариф ограничен);
    4. Binance (ccxt) — биржевые свечи высокого качества с 2017 г.;
    5. синтетика      — детерминированный оффлайн-фолбэк.

Все источники возвращают дневную цену закрытия в USD (для ccxt — в USDT≈USD).
"""
from __future__ import annotations

import datetime as dt
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from ..config import DATA_CACHE, RANDOM_SEED
from .sources import synthetic_price_panel

_HEADERS = {"User-Agent": "Mozilla/5.0 (crypto-risk-research)"}

# Сопоставление тикеров с идентификаторами CoinGecko
_COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "XRP": "ripple",
    "SOL": "solana", "ADA": "cardano", "DOGE": "dogecoin", "TRX": "tron",
    "LINK": "chainlink", "LTC": "litecoin", "AVAX": "avalanche-2",
    "DOT": "polkadot", "MATIC": "matic-network", "XLM": "stellar",
    "BCH": "bitcoin-cash", "XMR": "monero",
}

# 10 самых популярных монет (по капитализации, без стейблкоинов)
TOP10_POPULAR: tuple[str, ...] = (
    "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "TRX", "LINK", "LTC",
)


def _to_utc_dates(index_like) -> pd.DatetimeIndex:
    """Конвертирует Series/массив timestamp'ов в DatetimeIndex с днём-разрешением (UTC)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index_like, utc=True))
    return idx.normalize()


# --------------------------------------------------------------------------- #
# Источники: каждый реализует fetch_close(base, start, end) -> pd.Series
# --------------------------------------------------------------------------- #
class CryptoCompareSource:
    name = "cryptocompare"
    URL = "https://min-api.cryptocompare.com/data/v2/histoday"

    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        r = requests.get(self.URL, params={"fsym": base, "tsym": "USD",
                                           "allData": "true"},
                         headers=_HEADERS, timeout=25).json()
        data = r.get("Data", {}).get("Data", [])
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        df = df[df["close"] > 0]
        s = pd.Series(df["close"].values, index=_to_utc_dates(df["time"] * 1e9),
                      name=base)
        return s.loc[(s.index >= pd.Timestamp(start, tz="UTC")) &
                     (s.index <= pd.Timestamp(end, tz="UTC"))]


class YahooSource:
    name = "yahoo"
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}-USD"

    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        p1 = int(pd.Timestamp(start).timestamp())
        p2 = int(pd.Timestamp(end).timestamp())
        r = requests.get(self.URL.format(sym=base),
                         params={"period1": p1, "period2": p2, "interval": "1d"},
                         headers=_HEADERS, timeout=25).json()
        res = r.get("chart", {}).get("result")
        if not res:
            return pd.Series(dtype=float)
        res = res[0]
        ts = res.get("timestamp")
        closes = res["indicators"]["quote"][0].get("close")
        if not ts or not closes:
            return pd.Series(dtype=float)
        s = pd.Series(closes, index=_to_utc_dates(np.array(ts) * 1e9), name=base)
        return s[s > 0].dropna()


class CoinGeckoSource:
    name = "coingecko"
    URL = "https://api.coingecko.com/api/v3/coins/{cid}/market_chart"

    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        cid = _COINGECKO_IDS.get(base)
        if cid is None:
            return pd.Series(dtype=float)
        r = requests.get(self.URL.format(cid=cid),
                         params={"vs_currency": "usd", "days": "max",
                                 "interval": "daily"},
                         headers=_HEADERS, timeout=25).json()
        prices = r.get("prices", [])
        if not prices:
            return pd.Series(dtype=float)
        arr = np.array(prices)
        s = pd.Series(arr[:, 1], index=_to_utc_dates(arr[:, 0] * 1e6), name=base)
        s = s[~s.index.duplicated(keep="last")]
        return s.loc[(s.index >= pd.Timestamp(start, tz="UTC")) &
                     (s.index <= pd.Timestamp(end, tz="UTC"))]


class BinanceCcxtSource:
    name = "binance"

    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        from .sources import fetch_ohlcv
        try:
            df = fetch_ohlcv(base, start, end, "1d")
            return df["close"].rename(base)
        except Exception:
            return pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# Многоисточниковый загрузчик с фолбэком и дозаполнением
# --------------------------------------------------------------------------- #
class MultiSourceCryptoLoader:
    """
    Загружает панель цен закрытия, опрашивая источники по приоритету и
    дозаполняя пропуски из следующих источников.

    Parameters
    ----------
    universe : тикеры (по умолчанию 10 популярных монет).
    sources  : список источников в порядке приоритета.
    mode     : 'auto' (сеть+фолбэк) | 'synthetic' (детерминированный оффлайн).
    """

    def __init__(self, universe: Sequence[str] = TOP10_POPULAR,
                 sources: list | None = None, mode: str = "auto",
                 cache_dir: Path = DATA_CACHE):
        self.universe = list(universe)
        self.sources = sources or [CryptoCompareSource(), YahooSource(),
                                   CoinGeckoSource(), BinanceCcxtSource()]
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.source_report: dict[str, dict] = {}   # отчёт: кто что дал
        self.data_origin: str | None = None
        self.source_used: str | None = None        # для совместимости с RiskEngine

    def _cache_path(self, start, end) -> Path:
        return self.cache_dir / (f"multi_{len(self.universe)}assets_"
                                 f"{start}_{end}.csv")

    def _fetch_asset(self, base: str, start: str, end: str
                     ) -> tuple[pd.Series, dict]:
        """Собирает один ряд: первый непустой источник = база, остальные —
        дозаполнение пропусков (combine_first)."""
        series = None
        report = {"primary": None, "filled_by": [], "n_filled": 0}
        for src in self.sources:
            try:
                s = src.fetch_close(base, start, end)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{src.name}:{base} ошибка {exc!r}", RuntimeWarning)
                s = pd.Series(dtype=float)
            time.sleep(0.05)
            if s is None or s.empty:
                continue
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if series is None:
                series = s
                report["primary"] = src.name
            else:
                missing_before = series.isna().sum() + 0
                merged = series.combine_first(s)  # заполняем пропуски из s
                added = int(len(merged) - len(series)) + int(
                    series.isna().sum() - merged.reindex(series.index).isna().sum())
                if added > 0 or len(merged) > len(series):
                    report["filled_by"].append(src.name)
                    report["n_filled"] += max(added, 0)
                series = merged
        if series is None:
            return pd.Series(dtype=float), report
        return series.sort_index(), report

    def load_close_panel(self, start: str = "2015-01-01", end: str = "2025-01-01",
                         use_cache: bool = True) -> pd.DataFrame:
        cache = self._cache_path(start, end)
        if self.mode != "synthetic" and use_cache and cache.exists():
            self.data_origin = "exchange"
            self.source_used = f"cache:{cache.name}"
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "date"
            return df

        if self.mode == "synthetic":
            self.data_origin = "synthetic"
            self.source_used = "synthetic"
            return synthetic_price_panel(self.universe, start, end)

        cols = {}
        for base in self.universe:
            s, rep = self._fetch_asset(base, start, end)
            self.source_report[base] = rep
            if not s.empty:
                cols[base] = s
        if not cols:
            warnings.warn("Все источники недоступны -> синтетика.", RuntimeWarning)
            self.data_origin = "synthetic"
            self.source_used = "synthetic(fallback)"
            return synthetic_price_panel(self.universe, start, end)

        panel = pd.DataFrame(cols).sort_index()
        full_idx = pd.date_range(panel.index.min(), panel.index.max(),
                                 freq="D", tz="UTC")
        panel = panel.reindex(full_idx)
        panel.index.name = "date"
        self.data_origin = "exchange"
        primaries = sorted({rep["primary"] for rep in self.source_report.values()
                            if rep["primary"]})
        self.source_used = "multi:" + "+".join(primaries)
        if use_cache:
            panel.to_csv(cache)
        return panel

    def coverage_report(self) -> pd.DataFrame:
        """Таблица: по каждому активу — основной источник, кто дозаполнял."""
        if not self.source_report:  # synthetic-режим или кэш-хит
            return pd.DataFrame(columns=["primary", "filled_by", "n_filled"])
        rows = []
        for base, rep in self.source_report.items():
            rows.append({"asset": base, "primary": rep["primary"],
                         "filled_by": ", ".join(rep["filled_by"]) or "—",
                         "n_filled": rep["n_filled"]})
        return pd.DataFrame(rows).set_index("asset")
