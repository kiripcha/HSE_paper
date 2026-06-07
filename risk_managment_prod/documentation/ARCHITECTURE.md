# ARCHITECTURE.md — логика модулей `crypto_risk`

Подробный разбор пакета `crypto_risk/`. Для каждого модуля: назначение, публичный
API, внутренняя логика, зависимости, минимальный пример. Документ построен как
справочник — нужен раздел, открываешь его и работаешь.

---

## 0. Карта пакета и принципы

```
crypto_risk/
├── __init__.py            экспорт RiskEngine, RiskConfig
├── config.py              ┐
├── data/                  │  «тонкие» модули — атомарные функции
│   ├── sources.py         │
│   └── parsers.py         │
├── volatility.py          │
├── var_es.py              │
├── evt.py                 │
├── covariance.py          │
├── backtesting.py         │
├── portfolio.py           │
├── liquidity.py           │
├── dependence.py          │
├── stress.py              │
├── controls.py            │
└── engine.py              ┘  «толстый» оркестратор поверх тонких модулей
```

### Сквозные принципы

1. **Тонкие модули — толстый оркестратор.** Каждый модуль работает с numpy/pandas
   и ничего не знает о других модулях. `engine.py` — единственное место, где они
   собираются вместе.
2. **Стандартные dtypes.** Цены — `pd.DataFrame` (колонки = тикеры, индекс =
   `DatetimeIndex` в UTC). Доходности — то же. Веса портфеля — `np.ndarray` или
   `dict[str, float]`. Ковариация — `np.ndarray (n,n)`.
3. **Соглашение о знаке.** `VaR_α` и `ES_α` возвращаются как ПОЛОЖИТЕЛЬНЫЕ числа,
   обозначающие убыток в долях капитала. `0.05` ≡ убыток 5%.
4. **Аннуализация.** Крипта торгуется 24/7, поэтому везде множитель **365**, не
   252. Константа: `config.TRADING_DAYS_YEAR`.
5. **Воспроизводимость.** `config.RANDOM_SEED = 42`. Все стохастические функции
   принимают `seed=`.
6. **Невидимая деградация запрещена.** Если источник данных недоступен или
   модель плохо сходится — возвращается явный фолбэк с пометкой источника
   (`loader.data_origin`, `loader.source_used`).

### Конфигурация (`config.py`)

```python
@dataclass
class RiskConfig:
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    quote: str = "USDT"
    market_proxy: str = "BTC"          # рыночный индекс для бет
    var_confidence: float = 0.99
    es_confidence: float = 0.975
    horizons: tuple[int, ...] = (1, 10)
    ewma_lambda: float = 0.94
    rolling_window: int = 252
    trading_days_year: int = 365
    seed: int = 42
    maker_fee: float = 0.0002          # 2 б.п.
    taker_fee: float = 0.0005          # 5 б.п.
    spread_mult: float = 3.0           # множитель «худшего» спреда в LVaR
    capital: float = 1_000_000.0
```

Также экспортируются: `RANDOM_SEED`, `DATA_CACHE` (Path), `DEFAULT_UNIVERSE`,
`TOP10_POPULAR` (через `data`), вспомогательная `to_symbol(base, quote)`.

---

## 1. `data/sources.py` — базовая загрузка (ccxt + синтетика)

### Назначение
Простой загрузчик дневных цен/OHLCV с **Binance через ccxt** и
**детерминированный синтетический генератор** для оффлайн-режима.

### Публичный API
| объект | назначение |
|---|---|
| `fetch_ohlcv(base, start, end, timeframe='1d', quote='USDT', exchange_id='binance')` | OHLCV одного инструмента, постранично через ccxt |
| `fetch_order_book_snapshot(base, depth=50, ...)` | снимок стакана: `{bids, asks, mid, spread, rel_spread}` |
| `synthetic_price_panel(universe, start, end, freq='D', seed=42)` | панель цен с реалистичными стилизованными фактами (GARCH-рыночный фактор + Student-t инновации + индивидуальные беты) |
| `CryptoDataLoader(universe, quote='USDT', exchange_id='binance', mode='auto'\|'exchange'\|'synthetic')` | загрузчик с кэшем CSV; `load_close_panel(start, end, timeframe)`, `load_ohlcv(base, ...)` |
| `load_prices(universe, start, end, timeframe, mode)` | функциональная обёртка |

### Внутренняя логика
- **Пагинация ccxt.** `_fetch_ohlcv_paginated` тянет свечи окнами `limit=1000` от
  `since` до `until` с уважением `rateLimit`.
- **Кэш.** `_cache_path` включает тег источника (`exchange`/`synthetic`), чтобы
  синтетический и реальный кэши никогда не пересекались. Формат CSV (без
  зависимостей от pyarrow).
- **Атрибуты состояния:** `loader.data_origin` (`exchange`/`synthetic`),
  `loader.source_used` (например, `cache:prices_exchange_binance_...csv`).
- **Синтетика.** Общий рыночный фактор моделируется как GARCH(1,1) с инновациями
  Стьюдента ν=4 и безусловной σ=3.5%; отдельные активы — `β·market + idio`,
  где `idio ~ t-Student(ν=4, σ=1.5–3.5%)`. Корреляции получаются ~0.5–0.8.

### Пример
```python
from crypto_risk.data import CryptoDataLoader, fetch_order_book_snapshot

ld = CryptoDataLoader(("BTC","ETH","SOL"), mode="auto")
panel = ld.load_close_panel("2021-01-01", "2025-01-01", "1d")   # реальные данные с кэшем
ohlc  = ld.load_ohlcv("BTC", "2024-01-01", "2025-01-01", "1d")  # OHLC для range-σ
book  = fetch_order_book_snapshot("BTC", depth=20)              # стакан сейчас
```

---

## 2. `data/parsers.py` — мультиисточниковый парсер с фолбэком

### Назначение
Длинная история (с 2015 г.), цепочка источников с **дозаполнением пропусков**.

### Публичный API
| объект | что делает |
|---|---|
| `TOP10_POPULAR = ("BTC","ETH","BNB","XRP","SOL","ADA","DOGE","TRX","LINK","LTC")` | каноничная вселенная |
| `CryptoCompareSource()` / `YahooSource()` / `CoinGeckoSource()` / `BinanceCcxtSource()` | источники с методом `.fetch_close(base, start, end) -> pd.Series` |
| `MultiSourceCryptoLoader(universe, sources=None, mode='auto'\|'synthetic', cache_dir=...)` | главный загрузчик; `load_close_panel(start, end, use_cache=True)`, `coverage_report()` |

### Внутренняя логика
1. **`_fetch_asset(base, start, end)`** — итерирует по списку источников. Первый
   непустой результат становится «основным», следующие источники только
   *дозаполняют* пропуски через `Series.combine_first`. Результат: максимально
   полный ряд + словарь `{primary, filled_by, n_filled}`.
2. **Кэш.** `_cache_path` создаёт файл `multi_{N}assets_{start}_{end}.csv`;
   читается с `parse_dates=True` и `tz='UTC'`.
3. **Полная дневная сетка.** Панель реиндексируется на `pd.date_range(min,
   max, freq='D', tz='UTC')` — даты, которых нет ни у одного источника,
   получают `NaN`. Это важно: разные монеты имеют разные даты «рождения».
4. **`coverage_report()`** возвращает DataFrame:
   `asset → primary, filled_by, n_filled`.
5. **`data_origin` / `source_used`** — для совместимости с `RiskEngine`.

### Источники в деталях

| Источник | URL / клиент | История | Особенности |
|---|---|---|---|
| CryptoCompare | `min-api.cryptocompare.com/data/v2/histoday?allData=true` | с 2010 г. | без ключа, **главный** для длинной истории |
| Yahoo Finance | `query1.finance.yahoo.com/v8/finance/chart/{sym}-USD` | с 2014–2015 г. | стабильный фолбэк для майоров |
| CoinGecko | `api.coingecko.com/api/v3/coins/{id}/market_chart` | best-effort | free-тариф режет историю; маппинг `BTC→bitcoin` и т.п. |
| Binance ccxt | `fetch_ohlcv` через `ccxt` | с 2017 г. | высокое качество, недавняя история |

### Пример
```python
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR

ld = MultiSourceCryptoLoader(TOP10_POPULAR, mode="auto")
prices = ld.load_close_panel("2015-01-01", "2025-01-01")
print(ld.coverage_report())     # таблица: кто дал, кто дозаполнил
print(ld.data_origin, ld.source_used)
```

---

## 3. `volatility.py` — оценка и прогноз σ

### Назначение
Всё, что считает условную/безусловную/реализованную волатильность.

### Публичный API
| функция/класс | назначение |
|---|---|
| `log_returns(prices)`, `simple_returns(prices)` | базовые преобразования |
| `annualize_vol(daily_vol, periods=365)` | масштаб σ |
| `ewma_volatility(returns, lam=0.94)` | ряд EWMA-σ (RiskMetrics) |
| `ewma_forecast(returns, lam=0.94)` | σ_{T+1} (скаляр) |
| `ewma_covariance(returns, lam=0.94)` | EWMA-Σ на конец выборки |
| `GARCHModel(vol='GARCH'\|'GJR'\|'EGARCH', dist='normal'\|'t'\|'skewt', p=1, q=1, o=0)` | обёртка над `arch.arch_model` |
| `GARCHResult(sigma, forecast_var, params, dist, vol, nu, aic, bic, ...)` | результат fit'a |
| `select_best_garch(returns, candidates, criterion='bic')` | перебор спецификаций |
| `parkinson_vol(ohlc)`, `garman_klass_vol(ohlc)`, `rogers_satchell_vol(ohlc)`, `yang_zhang_vol(ohlc, window=30)` | range-оценки |
| `har_rv(realized_var)` → `HARResult(params, fitted, forecast, forecast_vol, r2)` | HAR-RV модель Corsi |

### Внутренняя логика
- **EWMA.** Рекурсия σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}, σ²_0 = r²_0.
- **GARCHModel.** Доходности умножаются на 100 (так `arch` стабильнее), σ
  возвращается обратно в долях. `fit(returns, horizon)` сохраняет:
  - `sigma` — in-sample условная σ (доли),
  - `forecast_var` — массив прогнозов дисперсии длины `horizon`,
  - `params` — словарь параметров MLE,
  - `dist`, `vol`, `nu` (для t/skewt).
- **`horizon_sigma(h)`** = `sqrt(sum(forecast_var[:h]))` — корректная агрегация
  GARCH-дисперсии на горизонт.
- **Range-σ.** Все четыре формулы реализованы в чистом виде; работают только с
  OHLC (нужны `open/high/low/close`).
- **HAR-RV.** Регрессия OLS методом `np.linalg.lstsq`, признаки = текущая RV,
  средняя RV за 5 и 22 дня, цель = RV_{t+1}. R² и прогноз на 1 шаг сохраняются.

### Пример
```python
from crypto_risk import volatility as vol

r = vol.log_returns(prices["BTC"])
ewma_path = vol.ewma_volatility(r)
g = vol.GARCHModel(vol="GJR", dist="t").fit(r, horizon=10)
print(g.nu, g.bic, g.horizon_sigma(10))     # ν, BIC, 10-дн. σ

# Сравнение спецификаций
best, tbl = vol.select_best_garch(r, criterion="bic", horizon=10)

# Range-оценки на OHLC
yz = vol.yang_zhang_vol(ohlc, window=30)
har = vol.har_rv((vol.garman_klass_vol(ohlc) ** 2).dropna())
```

---

## 4. `var_es.py` — оценка VaR/ES (8 методов) + декомпозиция

### Назначение
Оценка хвостовых мер риска всеми ходовыми методами + аналитическая
декомпозиция риска портфеля по принципу Эйлера.

### Публичный API
| функция | возвращает |
|---|---|
| `historical_var_es(returns, var_alpha, es_alpha, horizon=1, overlap=False)` | `RiskEstimate` |
| `parametric_var_es(mu, sigma, ..., dist='normal'\|'t', nu=None)` | `RiskEstimate` |
| `cornish_fisher_var_es(returns, ...)` | `RiskEstimate` |
| `ewma_var_es(returns, ..., lam=0.94, dist='normal')` | `RiskEstimate` |
| `garch_var_es(garch_result, ..., mu=0.0)` | `RiskEstimate` |
| `fhs_var_es(returns, garch_result, ..., n_paths=20000)` | `RiskEstimate` (Filtered HS) |
| `monte_carlo_var_es(weights, mu, cov, ..., dist='normal'\|'t', nu=5)` | `RiskEstimate` (портфельный) |
| `component_var_es(weights, returns, ..., method='gaussian'\|'historical')` | `pd.DataFrame` (weight, marginal/component VaR/ES) |
| `incremental_var(weights, returns, asset_idx, ...)` | `float` |
| `bootstrap_horizon_var_es(returns, ..., horizon=10, block=1, n_paths=50000)` | `RiskEstimate` (без √t) |
| `compare_methods(returns, ..., horizon=1, garch_result=None)` | таблица всех методов |

### `RiskEstimate` (dataclass)
```python
@dataclass
class RiskEstimate:
    var: float; es: float
    method: str; horizon: int
    var_alpha: float; es_alpha: float
    extra: dict | None = None
    def as_money(self, capital): return {"VaR": self.var*capital, "ES": self.es*capital}
```

### Внутренняя логика (ключевое)
- **Historical**: квантиль через `np.quantile(r, 1-α)`. При `horizon>1` либо
  overlapping h-дневная сумма, либо масштаб √h (определяет `_scale_estimate`).
- **Parametric-t**: стандартизация t-распределения к Var=1 множителем
  `sqrt((ν-2)/ν)`, ES для t по формуле Acerbi/McNeil.
- **Cornish-Fisher**: квантиль нормали поправляется на skew/kurtosis;
  ES оценивается средним по CF-квантилям выше уровня (численно).
- **FHS**: GARCH-σ убирается из истории → стандартизованные остатки z →
  бутстрэпом z'ы и обратно «надуваются» прогнозом σ. Хорошо для крипты.
- **Monte Carlo**: коррелированные симуляции через Choleski `_nearest_psd(cov)`;
  для multivariate-t — `z / sqrt(chi2/ν)` нормализованное.
- **Component VaR (Эйлер)**: гауссов вариант — аналитический; исторический —
  `-E[r_i | r_p ∈ tail]`, нормированный к VaR портфеля.
- **Bootstrap horizon**: блочный бутстрэп длины `block` сохраняет σ-кластеризацию;
  по умолчанию iid-бутстрэп. Корректнее √t при тяжёлых хвостах.
- **EVT включена** в `compare_methods` через ленивый импорт `evt.pot_var_es`.

### Пример
```python
from crypto_risk import var_es as ve

# одна модель
est = ve.historical_var_es(returns["BTC"], 0.99, 0.975, horizon=1)
print(est.var, est.es)                # положительные доли убытка

# все методы рядом
tbl = ve.compare_methods(returns["BTC"], horizon=1, garch_result=garch_fit)

# декомпозиция риска (сумма по i = VaR портфеля)
attr = ve.component_var_es(weights, returns, method="gaussian")
```

---

## 5. `evt.py` — теория экстремальных значений (POT/GPD)

### Назначение
Точная оценка ДАЛЁКИХ хвостов через метод Peaks-Over-Threshold + условный EVT.

### Публичный API
| функция | назначение |
|---|---|
| `fit_gpd(losses, threshold=None, threshold_q=0.90, method='mle'\|'mom')` → `GPDFit(xi, beta, threshold, n_exceed, n_total, method)` | оценка GPD на превышениях |
| `pot_var_es(returns, var_alpha, es_alpha, threshold_q=0.90, method='mle')` → `RiskEstimate` | POT-VaR и POT-ES |
| `conditional_evt(returns, garch_result, ...)` → `RiskEstimate` | McNeil-Frey: GARCH + POT на стандартизованных остатках |
| `hill_estimator(losses, k=None)` → `float` | индекс хвоста α (диагностика) |
| `mean_excess_plot_data(losses, n_points=40)` → `(us, me)` | данные для mean-excess plot |

### Внутренняя логика
- **Убытки.** Внутри работаем с `L = -r` (положительные значения).
- **Выбор порога u.** По умолчанию — эмпирический квантиль 90%. Mean-excess plot
  помогает обосновать выбор: линейный рост ⇒ GPD-режим.
- **MLE GPD.** `neg_ll(xi, beta) = sum[ log β + (1+1/ξ) log(1 + ξ y/β) ]`,
  стартовая точка — метод моментов `xi₀ = 0.5(1 - μ²/v)`, `β₀ = 0.5μ(μ²/v+1)`.
- **VaR/ES формулы (Pickands):**
  - `VaR_α = u + (β/ξ)[((n/N_u)(1-α))^(-ξ) - 1]`
  - `ES_α  = VaR_α/(1-ξ) + (β - ξu)/(1-ξ)` при `ξ < 1`.
- **Conditional EVT.** Стандартизуем `z = r/σ`, фитим POT на `z`, получаем
  `VaR_α(z)`, умножаем на прогнозный σ: `VaR_{T+1} = σ_fc · VaR_α(z)`.
- **Hill.** `α̂ = 1 / (1/k · Σ ln(X_(i)/X_(k+1)))` по k самым большим убыткам.

### Пример
```python
from crypto_risk import evt

fit = evt.fit_gpd(losses, threshold_q=0.90)        # ξ, β, u
est = evt.pot_var_es(returns_port, 0.99, 0.975)    # хвостовые VaR/ES
cevt = evt.conditional_evt(returns_port, garch)    # с учётом текущей σ
us, me = evt.mean_excess_plot_data(losses)         # для plot
```

---

## 6. `covariance.py` — устойчивая Σ + DCC

### Назначение
Снижение шума ковариационной матрицы и динамические корреляции.

### Публичный API
| функция | назначение |
|---|---|
| `sample_cov(returns, annualize=False)` | выборочная Σ (numpy) |
| `ledoit_wolf_cov(returns, annualize)` → `(cov, δ)` | shrinkage к scaled identity (sklearn) |
| `oas_cov(returns, annualize)` → `(cov, δ)` | Oracle Approximating Shrinkage |
| `constant_correlation_shrinkage(returns, annualize)` → `(cov, δ)` | LW 2004 к const-corr цели |
| `rmt_denoise_cov(returns, annualize)` | очистка спектра по Marchenko-Pastur |
| `dcc_garch(returns, a0=0.02, b0=0.95)` → `DCCResult` | DCC-GARCH (Engle 2002) |
| `condition_number(cov)` | мера обусловленности |

### `DCCResult`
```python
@dataclass
class DCCResult:
    cond_cov_last: np.ndarray       # Σ_T (дневная)
    cond_corr_last: np.ndarray      # R_T
    a: float; b: float              # параметры DCC
    avg_corr_path: pd.Series        # средняя попарная ρ во времени
    names: list[str]
    def annualized_cov(self): return self.cond_cov_last * 365
```

### Внутренняя логика
- **Ledoit-Wolf / OAS** — через `sklearn.covariance` (стабильнее, чем велосипед).
- **Const-corr shrinkage** реализована вручную: цель F = `r̄ · σσ'`, диагональ —
  сами σ². Оптимальная интенсивность δ через упрощённую формулу
  `(π - ρ)/γ / T`.
- **RMT denoise.** Спектр correlation-матрицы; верхняя граница шума
  `(1 + √(N/T))²`; шумные собственные значения заменяются на их среднее, диагональ
  ренормируется к 1.
- **DCC-GARCH.** Двухшаговая оценка:
  1. univariate GARCH(1,1) для каждого ряда (`arch`), стандартизованные z;
  2. рекурсия `Q_t = (1-a-b)Q̄ + a z_{t-1}z_{t-1}' + b Q_{t-1}`;
  3. `(a,b)` подбираются перебором по сетке 8×8 по pseudo-LL.
- Если `arch` нет — fallback на EWMA-σ для шага 1.

### Пример
```python
from crypto_risk import covariance as cvm

cov, delta = cvm.ledoit_wolf_cov(rets, annualize=True)
clean = cvm.rmt_denoise_cov(rets, annualize=True)
dcc = cvm.dcc_garch(rets)
print(dcc.a, dcc.b, dcc.avg_corr_path.iloc[-1])
```

---

## 7. `backtesting.py` — валидация VaR/ES + модельный риск

### Назначение
Полная батарея тестов покрытия и качества хвостовых оценок.

### Публичный API
| функция | формула / тест |
|---|---|
| `get_violations(returns, var)` | `I_t = 1{r_t < -VaR_t}` |
| `kupiec_pof(violations, var_alpha)` | LR безусловного покрытия, χ²(1) |
| `christoffersen(violations, var_alpha)` → dict | independence + CC + POF |
| `duration_test(violations, var_alpha)` | Вейбулл vs экспонента, χ²(1) |
| `dq_test(returns, var, var_alpha, lags=4)` | Engle-Manganelli DQ, χ²(k) |
| `berkowitz_tail(returns, var, sigma=None, ...)` | LR на PIT, χ²(2) |
| `acerbi_szekely_es(returns, var, es, ...)` | ES Test 2 (бутстрэп) |
| `acerbi_szekely_test1(returns, var, es, ...)` | ES Test 1 (условный на пробои) |
| `acerbi_szekely_test3(returns, dist_cdf, ...)` | ES Test 3 (rank-based) |
| `model_risk_metrics(var_estimates: dict, n_violations, n_obs)` | разброс VaR + зона Базеля |
| `run_var_backtests(returns, var, var_alpha)` → DataFrame | прогон всех тестов |
| `traffic_light(p_value)` → `'GREEN'/'YELLOW'/'RED'` | светофор Базеля |

### `BacktestResult`
```python
@dataclass
class BacktestResult:
    name: str
    statistic: float
    p_value: float
    reject_h0: bool
    detail: dict
    def verdict(self, level=0.05): ...
```

### Внутренняя логика (ключевое)
- **Kupiec POF** — стандартный likelihood-ratio с защитой `x=0` и `x=n`.
- **Christoffersen** — переходы марковской цепи 2×2 (`n_ij`), `lr_ind` через
  pseudo-LL; CC = POF + ind, χ²(2).
- **Duration test** — выводит длительности между пробоями, MLE Weibull (Nelder-Mead).
- **DQ test** — OLS-регрессия `hit_t - (1-α)` на константу, лаги хитов и текущий
  VaR; статистика Вальда `β'X'Xβ / (p(1-p)) ~ χ²(k)`.
- **Berkowitz** — преобразование PIT (через σ если есть, иначе через индикатор
  пробоя с равномерной примесью), MLE нормальной (μ, σ), LR на (0, 1).
- **Acerbi-Szekely T1/T2** — Z-статистики; p-value бутстрэпом нулевого
  распределения, где хвостовые убытки симулируются под H0 (`r ~ -ES`).
- **Model risk** — разброс `(max-min)/mean` оценок VaR между методами +
  скалирование `n_viol * 250 / n_obs` для зоны Базеля.

### Пример
```python
from crypto_risk import backtesting as bt

I = bt.get_violations(realized, var_series)
print(bt.kupiec_pof(I))                         # Kupiec
print(bt.christoffersen(I)["conditional_coverage"])
tbl = bt.run_var_backtests(realized, var_series, var_alpha=0.99)
mr = bt.model_risk_metrics({"hist":0.05,"garch":0.06}, n_violations=12, n_obs=1000)
```

---

## 8. `portfolio.py` — оптимизация Марковица + расширения

### Назначение
Граница эффективных портфелей, ковариация на основе бет, риск-паритет,
two-fund theorem, Монте-Карло граница, максимально рискованный портфель.

### Публичный API
| функция/класс | назначение |
|---|---|
| `mean_cov(returns, annualize=True)` → `(mu, cov)` | базовые входы оптимизатора |
| `estimate_betas(returns, market='BTC')` → `BetaEstimates` | β и β_adj (Blume) |
| `beta_covariance(beta_est, use_adjusted=False, annualize=True)` | Σ = β β' σ²_m + diag(σ²_idio) |
| `PortfolioOptimizer(mu, cov, rf=0.0, names=None)` | главный объект |
| `check_two_fund_theorem(mu, cov)` | проверка Блэка (1972) |

### `PortfolioOptimizer` методы
| метод | constraint | примечание |
|---|---|---|
| `.min_variance(target_return=None, constraint='long_only', short_limit=0.25, min_w=0.02)` | все | global MV если target=None |
| `.max_sharpe(constraint=...)` | все | для long-short — closed-form; для constrained — перебор по границе |
| `.max_risk(constraint=..., cap=None)` | long-only / long-short | **аналитика по вершинам** (квадратичная задача невыпукла) |
| `.risk_parity()` | long-only | итерации Spinu, ERC |
| `.efficient_frontier(n_points=50, constraint=..., short_limit, min_w)` → `list[FrontierPoint]` | все | сетка по целевой доходности |
| `.monte_carlo_frontier(n_portfolios=20000, constraint='long_only', seed=42)` → DataFrame | long_only / long_short | случайные веса |

`FrontierPoint(weights, ret, vol, sharpe)`.

### Внутренняя логика
- **CVXPY используется** для constrained min-variance (`cp.quad_form(w,
  cp.psd_wrap(cov))`). Без cvxpy — аналитика по Lagrange.
- **`max_sharpe` long-only** делает фронт и берёт максимум по Шарпу (выпуклая
  задача через QP — корректно).
- **`max_risk`** — намеренно НЕ использует cvxpy (максимизация выпуклой функции
  невыпукла). Решение: вершины симплекса (long-only ⇒ одна монета) или старший
  собственный вектор Σ, проецированный на бокс (long-short).
- **`risk_parity`** — рекуррентная итерация
  `w_i ← w_i · sqrt(b_i / RC_i)` с нормализацией.
- **Two-fund theorem.** Берутся два опорных фронт-портфеля (при r₁, r₂),
  проверяется, что фронт-портфель при r₃ равен `α·w₁ + (1-α)·w₂`,
  `α = (r₃-r₂)/(r₁-r₂)`. На несингулярной Σ выполняется до 1e-15.

### Пример
```python
from crypto_risk import portfolio as pf

mu, cov = pf.mean_cov(rets, annualize=True)
opt = pf.PortfolioOptimizer(mu, cov, names=list(UNIVERSE))
gmv = opt.min_variance(constraint="long_only")
msr = opt.max_sharpe(constraint="long_only")
rp  = opt.risk_parity()
front = opt.efficient_frontier(40, "short_limit", short_limit=0.25)

# Beta-based Σ
be = pf.estimate_betas(rets, market="BTC")
cov_b = pf.beta_covariance(be, use_adjusted=True, annualize=True)
```

---

## 9. `liquidity.py` — риск ликвидности и исполнение

### Назначение
LVaR + рыночное воздействие + Implementation Shortfall + Almgren-Chriss.

### Публичный API
| функция / класс | формула / цель |
|---|---|
| `bangia_lvar(price_var, rel_spread_mean, rel_spread_std, spread_mult=3.0)` → `LVaRResult(price_var, liquidity_cost, lvar, cost_share)` | LVaR = ценовой VaR + 0.5(μ_S + a·σ_S) |
| `rel_spread_stats_from_book(order_book)` | (μ_S, σ_S) из одного снимка |
| `rel_spread_stats_from_ohlc(ohlc)` | Corwin-Schultz из OHLC |
| `square_root_impact(order_size, adv, sigma, y=1.0)` | impact ≈ y·σ·√(Q/ADV) |
| `linear_impact(order_size, kyle_lambda)` | ΔP = λ·Q |
| `amihud_illiquidity(returns, dollar_volume)` | mean(\|r\|/$V) |
| `estimate_kyle_lambda(price_changes, signed_volume)` | OLS-оценка |
| `implementation_shortfall(decision_price, executed_prices, executed_sizes, side, half_spread, fee_rate, ...)` → `ImplementationShortfall(total_bps, spread_cost_bps, impact_cost_bps, timing_cost_bps, fees_bps)` | разложение |
| `AlmgrenChriss(sigma, eta, gamma, lam=1e-6).schedule(total_shares, horizon, n_steps)` → `ExecutionSchedule(times, holdings, trades, expected_cost, cost_variance, kappa)` | оптимальная ликвидация |
| `AlmgrenChriss.efficient_frontier(...)` | граница издержки-риск |
| `liquidity_adjusted_position_limit(capital, adv, price, max_participation=0.10)` | абсолютный лимит позиции |

### Внутренняя логика
- **Bangia LVaR** — экзогенная компонента: `0.5 · (μ_S + a · σ_S)` добавляется
  к ценовому VaR; для тяжёлых хвостов крипты `a=3`.
- **Corwin-Schultz** — оценка спреда по двум подряд идущим барам через
  отношение high/low и квадрат log(H/L).
- **Almgren-Chriss.** Траектория `x_j = X · sinh(κ(T-t_j)) / sinh(κT)`,
  `κ = arccosh(κ̃²/2+1)/τ`, `κ̃² = λ σ² τ / η̂`, `η̂ = η - 0.5 γ τ`. Предельный
  случай `κT→0` ⇒ TWAP. E[cost] = `0.5 γ X² + (η/τ)·Σnⱼ²`, Var = `σ²·Σ(τ·xⱼ²)`.

### Пример
```python
from crypto_risk import liquidity as liq

lvar = liq.bangia_lvar(price_var=0.05, rel_spread_mean=3e-4, rel_spread_std=2e-4)
impact_bps = liq.square_root_impact(order_size=1e7, adv=2e10, sigma=0.04) * 1e4

ish = liq.implementation_shortfall(
    decision_price=75000, executed_prices=fills_px, executed_sizes=fills_sz,
    side="buy", half_spread=0.0002, fee_rate=0.0005)

sched = liq.AlmgrenChriss(sigma=0.02, eta=2.5e-6, gamma=2.5e-7, lam=1e-6
                          ).schedule(total_shares=1_000_000, horizon=1.0, n_steps=20)
```

---

## 10. `dependence.py` — копулы и хвостовая зависимость

### Назначение
Описание зависимости активов через копулы, оценка вероятности совместных обвалов.

### Публичный API
| функция / класс | назначение |
|---|---|
| `pseudo_observations(returns)` | ранги / (n+1) |
| `kendall_tau_matrix(returns)` | попарный τ |
| `empirical_tail_dependence(u, v, q=0.05)` → `(lower, upper)` | условные вероятности хвостов |
| `tail_dependence_matrix(returns, q, which='lower'\|'upper')` | матрица эмпирических λ |
| `GaussianCopula.fit(returns)`, `.simulate(n, seed)`, `.lower_tail_dependence` (=0) | гауссова копула |
| `StudentTCopula.fit(returns, nu_grid)`, `.simulate(n, seed)`, `.lower_tail_dependence_pair(rho)`, `.lower_tail_dependence` | t-копула |
| `copula_portfolio_returns(returns, weights, copula, n_sims, seed)` | симуляция доходностей портфеля с копульной зависимостью + эмпирические маргиналы |
| `compare_dependence_models(returns, q=0.05)` | таблица: эмпирическая / гауссова / t-копула |

### Внутренняя логика
- **Псевдонаблюдения.** `u = rank(r) / (n+1)` ∈ (0,1).
- **Gaussian copula.** Корреляция нормальных скоров `Φ⁻¹(u)`.
- **t-copula.** Корреляция тех же скоров + ν по MLE (одномерная оптимизация
  через `minimize_scalar(bounded)`); pseudo-LL t-копулы реализована в
  `_t_copula_loglik` через `scipy.special.gammaln` (важно: НЕ
  `scipy.stats.loggamma` — это распределение, а не функция).
- **Tail dependence (t-copula теоретическая):**
  `λ = 2 · t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))`. Гауссова даёт 0.
- **`copula_portfolio_returns`** — `u = copula.simulate(n)`, `sim[:, j] =
  np.quantile(returns[col], u[:, j])`, портфель = `sim @ weights`. Реалистичные
  хвосты при гибкой зависимости.

### Пример
```python
from crypto_risk import dependence as dep

t = dep.StudentTCopula.fit(rets)            # ν оценивается MLE
print(t.nu, t.lower_tail_dependence)        # маленькое ν ⇒ толстые совместные хвосты
sim = dep.copula_portfolio_returns(rets, weights, t, n_sims=40000)
```

---

## 11. `stress.py` — стресс-тесты

### Назначение
Сценарии, которые ломают допущения VaR.

### Публичный API
| функция | назначение |
|---|---|
| `historical_scenarios(returns, weights, horizon=1, top=5)` | худшие реализованные периоды |
| `named_crypto_crashes(returns, weights, windows=None)` → `list[StressResult]` | реплей COVID/LUNA/FTX/SVB |
| `hypothetical_shock(weights: dict, shocks: dict)` | заданные шоки |
| `correlation_stress(returns, weights, target_corr=0.95, var_alpha=0.99)` | подъём всех ρ → 0.95 |
| `volatility_stress(returns, weights, vol_mult=2.0, var_alpha=0.99)` | масштабирование σ |
| `worst_case_loss(weights, mu, cov, plausibility_k=3.0)` | Breuer ellipsoid |
| `reverse_stress_test(weights, mu, cov, target_loss, names)` | сценарий заданной тяжести |

### `StressResult`
```python
@dataclass
class StressResult:
    name: str; pnl: float; detail: dict
```

### Внутренняя логика
- **Worst-case.** Замкнутое решение: `r* = μ - k·Σw/σ_p`,
  `loss* = -w'μ + k·σ_p`.
- **Reverse stress.** `k* = (target_loss + w'μ)/σ_p`, сценарий из той же формулы.
  Аппроксимация вероятности — `1 - F_{χ²(n)}(k²)`.
- **Named crashes** — заранее заданные временные окна (если попадают в данные).

### Пример
```python
from crypto_risk import stress as stm

worst = stm.historical_scenarios(rets, w_eq, horizon=1, top=5)
crash = stm.named_crypto_crashes(rets, w_eq)
wc    = stm.worst_case_loss(w_eq, mu_d, cov_d, plausibility_k=3)
rev   = stm.reverse_stress_test(w_eq, mu_d, cov_d, target_loss=0.50,
                                names=list(UNIVERSE))
```

---

## 12. `controls.py` — адаптивный сайзинг

### Назначение
Переводит оценку риска в размер позиции; DL-хук волатильности.

### Публичный API
| функция / класс | назначение |
|---|---|
| `vol_target_leverage(forecast_vol_annual, target_vol_annual=0.20, max_leverage=3.0)` | плечо |
| `vol_target_weights(weights, cov, target_vol_annual, max_leverage)` | веса под целевую σ |
| `kelly_fraction(mu, sigma, rf=0.0)` | скалярный Келли |
| `kelly_weights(mu, cov, rf=0.0, fraction=0.5)` | многомерный (½-Келли по умолчанию) |
| `max_drawdown(equity)` | численная MDD |
| `drawdown_scale(current_dd, dd_limit=0.20, floor=0.0)` | множитель экспозиции |
| `var_stop_loss(entry_price, var_fraction, side='long', k=1.0)` | стоп-лосс на основе VaR |
| `risk_budget_weights(cov, budget=None, iters=500)` | веса при заданном риск-бюджете |
| `VolForecaster(method='ewma', lam=0.94, trading_days=365).forecast(returns, garch_result=None)` → `VolForecast(sigma_daily, sigma_annual, source)` | прогноз σ; `.set_dl_model(model)` для DL |

### Хук под DL
```python
class DummyLSTMVol:
    def predict(self, returns) -> float: ...    # дневная σ

forecaster = VolForecaster().set_dl_model(DummyLSTMVol())
fc = forecaster.forecast(returns)               # fc.source == 'dl'
```

Duck-typing: достаточно метода `.predict(returns) -> float`. Так подключается
любая обученная LSTM/TCN/Transformer-модель без переделки риск-контура.

---

## 13. `engine.py` — оркестратор `RiskEngine`

### Назначение
Единая точка входа: данные → модели → отчёты → решения. Поверх «тонких» модулей.

### Конструкция
```python
class RiskEngine:
    def __init__(self, config: RiskConfig | None = None,
                 data_mode: str = "auto"):
        self.cfg = config or RiskConfig()
        np.random.seed(self.cfg.seed)
        self.loader = CryptoDataLoader(universe=self.cfg.universe, mode=data_mode)
        self.prices = None
        self.returns = None
        self._garch_cache = {}
        self.vol_forecaster = VolForecaster(method="ewma", lam=self.cfg.ewma_lambda)
```

### Жизненный цикл
| метод | что происходит |
|---|---|
| `.load_data(start, end, timeframe='1d')` | через `CryptoDataLoader` (ccxt/синтетика); `self.prices`, `self.returns = log_returns(prices)` |
| `.load_data_multi(start='2015-01-01', end, universe=None, synthetic=False)` | заменяет loader на `MultiSourceCryptoLoader` |
| `.set_returns(returns)` | подать готовые доходности (для тестов) |
| `.garch(asset, vol='GJR', dist='t', horizon=10)` | кэшированный fit |
| `.portfolio_returns(weights)` | ряд `returns @ w` |
| `.risk_report(weights, methods=('historical','garch','fhs'), use_copula=True)` → `RiskReport` | сводный отчёт |
| `.optimal_weights(objective='max_sharpe'/'min_variance'/'risk_parity', constraint=..., cov_method='sample'/'ewma'/'beta'/'shrinkage'/'constant_corr'/'rmt'/'dcc', use_adjusted_beta=False)` | веса |
| `.pre_trade_check(asset, notional, side, adv_usdt, current_drawdown=0.0, max_var_budget=0.05, max_participation=0.10)` → `PreTradeDecision` | предторговый контроль |
| `.size_position_vol_target(weights, target_vol=0.20, max_leverage=3.0)` | vol-target веса |
| `.backtest_var(weights, method='historical'/'ewma'/'parametric', window=252, var_alpha=None)` | скользящий бэктест + тесты |
| `.tail_risk(weights, threshold_q=0.90, use_conditional=True)` | EVT (POT + conditional) |
| `.risk_attribution(weights, method='gaussian')` → DataFrame | Эйлер |
| `.stress_test(weights, plausibility_k=3.0)` | набор сценариев |
| `.reverse_stress(target_loss, weights)` | обратный стресс |
| `.model_risk(weights)` → dict | разброс VaR между методами + зона Базеля |

### `RiskReport` (dataclass)
```python
@dataclass
class RiskReport:
    asset_risk: pd.DataFrame          # VaR/ES по каждому активу
    portfolio_risk: pd.DataFrame      # VaR/ES портфеля разными методами
    lvar: dict                        # {'price_var','liquidity_cost','lvar','liq_share'}
    diversification_ratio: float
    annualized_vol: float
    weights: dict
    horizons: tuple[int, ...]
```

### `PreTradeDecision`
```python
@dataclass
class PreTradeDecision:
    approved: bool
    reasons: list[str]
    sized_notional: float
    metrics: dict
```

Алгоритм решения: лимит участия в ADV → VaR-бюджет инструмента → деривингование
по просадке → ожидаемое рыночное воздействие.

### Минимальный production-цикл
```python
from crypto_risk import RiskEngine, RiskConfig
from crypto_risk.data import TOP10_POPULAR

eng = RiskEngine(RiskConfig(universe=TOP10_POPULAR, capital=1_000_000))
eng.load_data_multi("2015-01-01", "2025-01-01")
w = eng.optimal_weights("max_sharpe", "long_only", cov_method="shrinkage")
rep = eng.risk_report(w, use_copula=True)
ok = eng.pre_trade_check("BTC", 5_000_000, "buy", adv_usdt=2e10,
                          current_drawdown=0.06)
bt = eng.backtest_var(w, method="historical", window=252)
```

---

## 14. Сквозные вопросы

### Кэш
- `data_cache/prices_{origin}_{exchange}_{assets}_{tf}_{start}_{end}.csv` —
  `CryptoDataLoader`.
- `data_cache/multi_{N}assets_{start}_{end}.csv` — `MultiSourceCryptoLoader`.
- Удалить кэш: `rm -f data_cache/*.csv`. Парсер заново сходит во все источники.

### Воспроизводимость
Все стохастические функции принимают `seed=`. Глобальный сид — `config.RANDOM_SEED`.
`RiskEngine.__init__` зовёт `np.random.seed(cfg.seed)`.

### Обработка NaN
- `log_returns` отбрасывает первую строку.
- При `MultiSourceCryptoLoader` молодые монеты (SOL с 2020) дают NaN в ранние
  даты. Для портфельной аналитики используется `returns.dropna()` — общее окно
  (с 2020-04 для топ-10). Per-asset аналитика использует `returns[c].dropna()`.

### Производительность
- GARCH fit на 5000 наблюдений: ~0.2-0.5 с (`arch`).
- `select_best_garch` (5 спецификаций): ~1-3 с на актив.
- DCC-GARCH (8×8 сетка a/b) на 10 активах: ~30-60 с.
- Backtest скользящий 1000 дней (historical): ~0.2 с.
- Bootstrap horizon 50k путей: ~0.2 с.

### Тестирование
`tests/test_smoke.py` — 17 функций, каждая проверяет один модуль на
синтетических данных. Запуск: `.venv/bin/python tests/test_smoke.py`.
В тесте `test_engine_end_to_end` — полный пайплайн.

### Зависимости
Жёсткие: `numpy, pandas, scipy, statsmodels, scikit-learn, requests`.
Желательные: `arch` (GARCH), `cvxpy` (constrained optimization), `ccxt` (Binance),
`matplotlib, seaborn` (plot). При отсутствии — graceful fallback (см. `_HAS_*`
флаги в каждом модуле).
