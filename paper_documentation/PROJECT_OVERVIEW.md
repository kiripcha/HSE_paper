### PROJECT_OVERVIEW.md — Архитектура и логика проекта

этот документ — структурный обзор системы. Назначение каждого модуля, кто его
запускает, что он потребляет, что отдаёт, и к какому слою теории относится.
подойдёт для понимания как всё устроено в целом.

для формул и кода методов — см. [METHODOLOGY.md](METHODOLOGY.md).
для быстрого хэндоффа агенту — см. [AGENT_CONTEXT.md](AGENT_CONTEXT.md).

---

### что это за проект

цель: высокоскоростная торговая система для управления и оптимизации
криптовалютного портфеля на основе глубокого обучения и адаптивных стратегий.

состав: 12 production-модулей в `src/`, плюс отдельный пакет
`crypto_risk/` с глубокой риск-аналитикой (VaR/ES, EVT, copulas, stress,
оптимизатор Маркóвица).

три режима выполнения:
- backtest — `ReplayEngine` подаёт исторические события из Parquet, fill_sim моделирует исполнение.
- paper — реальные live-данные с биржи, но без размещения ордеров (fill_sim).
- live — реальное исполнение через биржевой `OMS`.

один и тот же runtime (`src/live.TradingApp`) работает во всех трёх режимах —
разница только в подключаемом `ExchangeConnector` и `FillSimulator`.

---

### архитектура одной картинкой

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ВНЕШНИЙ МИР                                                             │
│  Binance WS / OKX WS / CryptoCompare API / Yahoo / CoinGecko / Synthetic │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  МОДУЛЬ 01 — market_data                                                 │
│  • BinanceConnector / OkxConnector                  (live)               │
│  • SortedBook / ArrayBook (Order Book Engine)                            │
│  • ResyncManager (gap-detection)                                         │
│  Output: AsyncIterator[MarketEvent]                                      │
└────────────┬────────────────────────┬────────────────────────────────────┘
             │                        │
             │ записать               │ потребить онлайн
             ▼                        │
┌────────────────────────┐            │
│  МОДУЛЬ 02 — storage    │           │
│  • ParquetTickWriter    │           │
│  • HistoricalLoader     │           │
│  • ReplayEngine         │───────────┤ ReplayEngine реализует тот
│    (drop-in для live)   │           │ же ExchangeConnector
└────────────────────────┘            │
                                      ▼
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 03 — features              │
                  │  • FeaturePipeline (incremental)   │
                  │  • Micro-price, OFI, Imbalance,    │
                  │    RSI, MACD, RealizedVol, ...     │
                  │  Output: FeatureVector             │
                  └────────────┬───────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ МОДУЛЬ 04       │  │ МОДУЛЬ 05        │  │ МОДУЛЬ 06        │
│ labeling        │  │ predictive_models│  │ regime_detection │
│ • triple-barrier│  │ • Ridge          │  │ • HMM            │
│ • purged k-fold │  │ • LightGBM       │  │ • BOCPD          │
│ • CPCV          │  │ • MLP, TCN       │  │ • VolCluster     │
│ • deflated SR   │  │ • MC-dropout σ   │  │ Output: Regime   │
│ Output:         │  │ Output: list[    │  │   State          │
│   LabeledDataset│  │    Prediction]   │  │                  │
└─────────────────┘  └────────┬─────────┘  └────────┬─────────┘
                              │                      │
                              └──────┬───────────────┘
                                     ▼
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 07 — meta_allocator        │
                  │  • ε-greedy / UCB1                 │
                  │  • Thompson Sampling               │
                  │  • LinUCB (контекст = regime)      │
                  │  Output: StrategyWeights +         │
                  │           AggregatedSignal         │
                  └────────────┬───────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 08 — portfolio_optimization│
                  │  • Markowitz + Ledoit-Wolf         │
                  │  • Risk Parity (ERC)               │
                  │  • HRP (López de Prado)            │
                  │  Output: TargetWeights             │
                  └────────────┬───────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────┐
              │  МОДУЛЬ 10 — risk_management           │
              │  Два слоя:                             │
              │  (a) RiskGate / PortfolioMonitor /     │
              │      KillSwitch — fast, runtime        │
              │  (b) CryptoRiskAdapter →               │
              │      crypto_risk: VaR/ES/EVT/Copula/   │
              │      Stress (heavy, batch)             │
              │  Output: ValidationResult              │
              └────────────┬───────────────────────────┘
                           │  approved?
                           ▼
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 09 — execution             │
                  │  • OMS (state machine)             │
                  │  • Agents: TWAP, VWAP, Almgren-    │
                  │    Chriss, MarketOrder, DRL stub   │
                  │  Output: Fill events               │
                  └────────────┬───────────────────────┘
                               │
                               ▼ (in backtest: FillSimulator)
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 11 — backtest              │
                  │  • QueueAwareFillSimulator         │
                  │  • SquareRootImpact slippage       │
                  │  • StrategyBacktest (модули 03-10) │
                  │  Output: BacktestReport            │
                  └────────────┬───────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────────────┐
                  │  МОДУЛЬ 12 — live_monitoring       │
                  │  • TradingApp (orchestrator)       │
                  │  • MetricsExporter (Prometheus)    │
                  │  • Health endpoints                │
                  │  Один runtime → 3 режима           │
                  └────────────────────────────────────┘
```

---

### поток данных от события к ордеру

в backtest и live последовательность одинаковая. Каждое рыночное
событие проходит этот путь (упрощённый):

```
MarketEvent (модуль 01/02)
   │
   ├─► OrderBook.apply(event)                                  ─┐
   ├─► FeaturePipeline.update(event, book) ──► FeatureVector   ─┤ горячий
   ├─► RegimeDetector.update(returns)       ──► RegimeState    ─┤ путь
   │                                                            │ (per
   │   [periodic: каждые N событий или раз в минуту]            │ event,
   │                                                            │ p99 <
   ▼                                                            │ 1 ms)
   PredictiveModel.predict_one(fv) for m in pool                │
        ──► list[Prediction]                                   ─┘

   Allocator.allocate(predictions, AllocatorContext)
        ──► StrategyWeights + AggregatedSignal                  } Periodic
                                                                } (per
   PortfolioOptimizer.optimize(signal, cov, constraints)        } rebalance
        ──► TargetWeights                                       } cycle,
                                                                } e.g. 1
   RiskGate.validate(target)                                    } минута)
        ──► ValidationResult
   CryptoRiskAdapter.validate_target(target) — опционально

   EMS: diff с current_positions → child orders
        FillSimulator/OMS → Fill events
        ──► PortfolioMonitor.on_fill(fill)

   KillSwitch monitored continuously
```

латентные бюджеты (целевые для модуля 01):
- `OrderBook.apply` p99 < 50 μs
- `pipeline.update` (полный feature vector) p99 < 100 μs
- `predict_one` (LightGBM/Ridge) < 1 ms; (DL через ONNX) < 1 ms

частоты (зависят от стратегии):
- Per-event: ingestion, order book, features, regime, kill-switch.
- Per-rebalance (минута/час): predictions, allocator, portfolio_opt, risk_gate.
- Periodic batch (час/день): crypto_risk полный отчёт, backtest CV-paths.

---

### подробно по модулям

### модуль 01 — `src/market_data/` — Market Data Ingestion + Order Book

когда срабатывает: непрерывно, на каждое WS-сообщение биржи.

что делает:
1. Парсит сообщения Binance и OKX в унифицированный `MarketEvent`.
2. Поддерживает `OrderBook` в памяти, применяя `book_delta` и `book_snapshot`.
3. Детектирует sequence gaps (`ResyncManager`) и инициирует resync.

ключевые классы и методы:
- `BinanceConnector(ws_url, symbols, channels).connect()`, `.stream() -> AsyncIterator[MarketEvent]`.
- `OkxConnector` — то же для OKX.
- `SortedBook(symbol).apply(event)`, `.snapshot(depth)`, `.best_bid_ask`.
- `ArrayBook(symbol, max_depth)` — то же, но с numpy-снимками.
- `make_order_book(symbol, implementation="sorted"|"array", max_depth)`.
- `ResyncManager.observe(event) -> bool` (True = нужен resync).
- Производные: `derive_mid(snapshot)`, `derive_micro_price(snapshot)`, `derive_imbalance(snapshot, levels)`.

контракт наружу:
```python
class MarketEvent(BaseModel, frozen=True):
    exchange: Literal["binance", "okx"]
    symbol: str
    ts_exchange_ns: int       # биржевой timestamp
    ts_local_ns: int          # локальный при приёме
    event_type: Literal["book_snapshot", "book_delta", "trade", "bbo"]
    seq: int | None
    payload: BookDelta | BookSnapshot | Trade | BBO
```

чуть теории: order book — это набор пар (price, qty) по обеим сторонам.
поток incremental updates требует sequence-tracking: пропустили номер → стакан
рассыпан и надо снимать snapshot. Mid-price — `(bid+ask)/2`; micro-price
по Stoikov (2018) учитывает дисбаланс объёмов: $p_b q_a / (q_a + q_b) + p_a q_b / (q_a + q_b)$.

---

### модуль 02 — `src/storage/` — Historical Storage + Replay

когда срабатывает:
- Writer — в реальном времени или однократно, для накопления данных.
- Loader/ReplayEngine — в момент запуска бэктеста.

что делает:
1. Пишет `MarketEvent`-ы в партиционированный Parquet (Hive-style).
2. Загружает обратно с lazy-iteration и time-range pruning.
3. `ReplayEngine` имплементирует тот же `ExchangeConnector` Protocol — backtester его подключает вместо live-коннектора.

ключевые классы:
- `ParquetTickWriter(StorageConfig).write(event)`, `await .close()`.
- `ParquetHistoricalLoader(StorageConfig).load(exchange, symbol, from_ns, to_ns) -> Iterator[MarketEvent]`.
- `ReplayEngine(storage, replay_cfg, from_ns, to_ns, sources)` — async streamer с детерминистичным tie-breaking.

партицирование:
```
data/ticks/exchange=binance/symbol=BTCUSDT/event_type=book_delta/year=2024/month=05/day=27/part-{ts_min}-{ts_max}.parquet
```

главный инвариант: бэктест через `ReplayEngine` производит **байт-в-байт
тот же стакан**, что live (тест `test_replay_reconstructs_same_book_as_live`).

---

### модуль 03 — `src/features/` — Feature Engineering

когда срабатывает: на каждом `MarketEvent` после `OrderBook.apply`.

что делает: incremental вычисление набора фич + сборка их в `FeatureVector`.

реализованные фичи (`src/features/_internal/features.py`):
- `MicroPrice` — Stoikov (2018).
- `Imbalance(levels)` — top-N volume imbalance.
- `OFI(window)` — Order Flow Imbalance (Cont-Kukanov 2014).
- `LogReturn(lag)` — лог-доходность через `lag` событий.
- `RealizedVol(window)` — std лог-доходностей.
- `RSI(period)` — Wilder RSI (инкрементально).
- `MACD(fast, slow)` — EMA-разница.
- `BollingerWidth(window)` — нормализованная ширина полос.

главное требование — train/serve parity: пайплайн в streaming и batch
режимах даёт побитово равные значения (тест `test_train_serve_parity`).

контракт наружу:
```python
class FeatureVector(BaseModel):
    ts_ns: int
    symbol: str
    values: np.ndarray
    names: tuple[str, ...]
    config_hash: str          # стабильный хэш конфига для версионирования
```

чуть теории: микроструктурные фичи (OFI, micro-price, imbalance) ловят
дисбаланс заявок — это короткий predictor для следующей доходности.
технические индикаторы (RSI, MACD) — менее информативны в крипте, но
ансамблируются с микроструктурой.

---

### модуль 04 — `src/labeling/` — Labeling & Cross-Validation

когда срабатывает: в офлайне, при подготовке датасета для обучения.

что делает: квант-специфичная разметка таргетов и leakage-free валидация
по методологии Лопеса де Прадо (AFML, главы 3-7, 14).

ключевые алгоритмы:
- `triple_barrier_labels(prices, ts, config)` — events label as +1/-1/0 по
  тому, какой барьер сработал первым (take-profit / stop-loss / vertical).
- `sample_uniqueness_weights(t0, t1)` — поправка на overlapping labels.
- `PurgedKFold(n_splits, embargo_pct)` — k-fold с purging обучающих образцов,
  чья жизнь пересекается с тестовой.
- `CombinatorialPurgedCV(n_splits, n_test_splits)` — даёт C(N, k) backtest paths.
- `deflated_sharpe(...)` — поправка на multiple-testing (Bailey-LdP 2014).
- `probabilistic_sharpe(...)` — вероятность того, что истинный Sharpe > benchmark.

контракт наружу:
```python
class LabeledDataset(BaseModel):
    X: np.ndarray                # (N, n_features)
    y: np.ndarray                # (N,) — labels
    sample_weights: np.ndarray   # (N,)
    t0: np.ndarray               # event start, ns
    t1: np.ndarray               # vertical barrier hit, ns
    feature_names: tuple[str, ...]
```

чуть теории: финансовая ML страдает от look-ahead bias и overlapping
labels. Triple-barrier + purged CV — индустриальный стандарт борьбы с этим.
без deflated Sharpe нельзя честно сравнивать варианты стратегий — best-of-N
смотрится впечатляюще на наивном Sharpe.

---

### модуль 05 — `src/models/` — Predictive Models

когда срабатывает:
- `fit()` — один раз перед запуском бэктеста / периодически в production.
- `predict_one()` / `predict_batch()` — на каждой ребалансировке.

что делает: предсказывает направление/величину будущей доходности.

пул моделей:
- `RidgeModel(alpha)` — линейная baseline.
- `LightGBMModel(num_leaves, n_estimators, lr)` — gradient boosting.
- `MLPModel(hidden, dropout, max_epochs)` — PyTorch MLP с MC-Dropout uncertainty.
- `TCNModel(channels, kernel, dropout, max_epochs)` — Temporal Convolutional Network.

все модели имплементируют `PredictiveModel` Protocol:
- `.fit(LabeledDataset) -> FitReport`
- `.predict_one(FeatureVector | np.ndarray) -> Prediction`
- `.predict_batch(X) -> np.ndarray`
- `.save(path)`

контракт наружу:
```python
class Prediction(BaseModel, frozen=True):
    model_id: str
    ts_ns: int
    symbol: str
    horizon_ns: int
    mu: float                  # point estimate
    sigma: float               # uncertainty (MC-Dropout / Bayesian std)
    direction_proba: float | None
```

чуть теории: одна модель не покрывает все режимы. Поэтому держим пул,
а бандит (модуль 07) выбирает лучшую сейчас. MC-Dropout (Gal-Ghahramani 2016)
даёт честную uncertainty estimate для bandit-аллокатора.

---

### модуль 06 — `src/regime/` — Regime Detection

когда срабатывает: на каждом обновлении (например, на новой портфельной
доходности).

что делает: классифицирует текущее состояние рынка в один из K режимов
strictly online — никогда не пересматривает прошлые решения (lookahead-bias-free).

реализованные детекторы:
- `HMMDetector(n_regimes, learning_rate)` — Markov-Switching, online EM.
- `BOCPDDetector(hazard, mu0, kappa0, alpha0, beta0)` — Bayesian Online
  change-Point Detection (Adams-MacKay 2007).
- `VolClusterDetector(n_regimes, window, history)` — rolling vol percentile.

контракт наружу:
```python
class RegimeState(BaseModel, frozen=True):
    ts_ns: int
    symbol: str
    regime_id: int                   # argmax probabilities
    probabilities: np.ndarray        # (n_regimes,)
    steps_in_regime: int
    change_point_proba: float        # для BOCPD
```

подвох BOCPD: $P(r_t = 0 | x_{1:t}) = H$ всегда по построению. Сигнал
смены режима — это падение argmax run-length (`steps_in_regime`), а не
значение `probabilities[0]`.

---

### модуль 07 — `src/meta_allocator/` — Adaptive Meta-Allocator

когда срабатывает: на каждой ребалансировке (раз в минуту/час).

что делает: выбирает, какой модели из пула доверять прямо сейчас, на
основе realized PnL каждой стратегии за прошлые периоды. Опционально
использует контекст (regime).

реализованные алгоритмы:
- `EpsilonGreedyAllocator(arms, eps, discount)` — c discount для non-stationarity.
- `UCB1Allocator(arms)` — оптимизм в условиях неопределённости.
- `ThompsonSamplingAllocator(arms, prior_sigma, noise_sigma)` — Gaussian posterior sampling.
- `LinUCBAllocator(arms, context_dim, alpha)` — contextual bandit, контекст = `RegimeState.probabilities`.

контракт наружу:
```python
class StrategyWeights(BaseModel, frozen=True):
    ts_ns: int
    weights: dict[str, float]               # {model_id: weight}
    aggregated_signal: AggregatedSignal
    exploration_active: bool

class AggregatedSignal(BaseModel, frozen=True):
    ts_ns: int
    symbol: str
    mu: float                # weighted mean across models
    sigma: float             # weighted uncertainty + disagreement
    direction_proba: float | None
    contributing_models: tuple[str, ...]
```

чуть теории: в стационарной среде Thompson Sampling часто оптимален. На
non-stationary рынках нужен discount или sliding window — иначе бандит
застревает на устаревшем лучшем. Регрет UCB-семейства — sublinear по T.

---

### модуль 08 — `src/portfolio/` — Portfolio Optimization

когда срабатывает: на каждой ребалансировке после получения
`AggregatedSignal` от меgaаллокатора.

что делает: преобразует ожидаемые доходности и ковариацию в **целевые
веса портфеля** с учётом ограничений.

реализованные оптимизаторы:
- `MarkowitzOptimizer(risk_aversion, shrinkage)` — mean-variance + Ledoit-Wolf-like shrinkage.
- `RiskParityOptimizer(max_iter)` — Equal Risk Contribution.
- `HRPOptimizer()` — Hierarchical Risk Parity (López de Prado 2016).
- `LedoitWolfCovariance()` — отдельная shrinkage-оценка cov.

контракт наружу:
```python
class TargetWeights(BaseModel, frozen=True):
    ts_ns: int
    weights: dict[str, float]
    expected_return: float
    expected_vol: float
    optimizer_id: str
    solver_status: Literal["optimal", "suboptimal", "infeasible"]

class PortfolioConstraints(BaseModel, frozen=True):
    max_weight_per_asset: float = 0.3
    min_weight_per_asset: float = 0.0
    leverage_cap: float = 1.0
    turnover_cap: float | None
    transaction_cost_bps: float = 5.0
```

чуть теории: наивный Markowitz взрывается из-за ill-conditioned cov-матрицы
и шумного $\mu$. HRP не использует $\mu$ вовсе и устойчив к ошибкам в
ковариации — это его главное преимущество (доказано в `test_hrp_robustness.py`).

---

### модуль 09 — `src/execution/` — Execution + OMS

когда срабатывает: после approval от RiskGate.

что делает:
1. `OMS` — конечный автомат ордера (NEW → PENDING → ACK → PARTIAL → FILLED/CANCEL).
2. `ExecutionAgent` — стратегия исполнения родительского ордера: TWAP/Almgren-Chriss/market/random.
3. Опционально DRL-агент (placeholder под gymnasium-env).

реализованные агенты:
- `MarketOrderAgent` — выкидывает остаток рынком (baseline).
- `TWAPAgent(total_slices)` — равные слайсы во времени.
- `AlmgrenChrissAgent(eta, gamma, sigma, risk_aversion)` — closed-form optimal execution.
- `RandomAgent` — стохастический baseline для тестов.

контракт OMS:
```python
class OrderState(BaseModel):
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    requested_quantity: float
    status: Literal["NEW", "PENDING", "ACK", "PARTIAL", "FILLED", "CANCELED", "REJECTED"]
    filled_quantity: float
    avg_fill_price: float | None
    last_update_ns: int

class Fill(BaseModel, frozen=True):
    ts_ns: int
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float
    client_order_id: str
```

чуть теории: Almgren-Chriss (2000) минимизирует implementation shortfall
с штрафом за риск. Trade-off: быстрее = больше market impact, медленнее =
больше risk от drift в цене.

---

### модуль 10 — `src/risk/` + `crypto_risk/` — Risk Management

два слоя:

### (a) Runtime layer (`src/risk/`) — fast, event-loop

когда срабатывает:
- `RiskGate.validate(target)` — перед каждой ребалансировкой.
- `PortfolioMonitor.on_fill(fill)` — на каждом Fill events.
- `KillSwitch.is_active()` — периодически в TradingApp.

что делает:
- `RiskGate` — leverage/concentration/drawdown/kill-switch проверка.
- `PortfolioMonitor` — текущие позиции, equity, drawdown, rolling VaR/CVaR.
- `KillSwitch` — hard-stop триггер.

контракт наружу:
```python
class ValidationResult(BaseModel, frozen=True):
    approved: bool
    adjusted_weights: dict[str, float] | None
    violations: tuple[RiskViolation, ...]
```

### (b) Analytics layer (`crypto_risk/` через `CryptoRiskAdapter`) — heavy, batch

когда срабатывает:
- Периодический отчёт (раз в час).
- Pre-trade check на ребалансе (если включено в конфиге).

что даёт:
- 9 методов VaR/ES + EVT (POT/GPD) + Conditional EVT.
- GARCH-family волатильность с MC-Dropout / DL хуком.
- DCC-GARCH, Ledoit-Wolf, RMT для cov-матрицы.
- 6 backtest-тестов для VaR (Kupiec, Christoffersen, DQ, Berkowitz, Acerbi-Szekely).
- LVaR (Bangia), Almgren-Chriss optimal execution, square-root impact.
- Gaussian/Student-t copulas для хвостовой зависимости.
- Stress tests (historical, named crashes, worst-case Breuer, reverse stress).

адаптер:
```python
adapter = CryptoRiskAdapter.from_returns(returns_df)
adapter.attach_dl_vol_model(my_lstm_vol)
res = adapter.validate_target(target, current_drawdown=0.05)
rep = adapter.portfolio_report(weights)      # serializable dict
tail = adapter.tail_risk(weights, threshold_q=0.90, use_conditional=True)
```

чуть теории: VaR — квантиль убытков. ES = условное ожидание в хвосте.
EVT POT/GPD моделирует именно хвост (тяжёлый в крипте), нормальный VaR его
систематически недооценивает.

---

### модуль 11 — `src/backtest/` — Backtest Engine

два бэктестера:

### `run_simple_backtest()` — minimal

берёт пред-построенные `target_weights_series` и прогоняет их через
`FillSimulator`. Хорошо для sanity-check отдельных компонентов.

### `strategyBacktest` — full end-to-end (использует модули 03-10)

когда срабатывает: офлайн, перед защитой / при iteration над стратегией.

что делает:
1. Обучает пул моделей на первой части истории (`train_fraction=0.4`).
2. Прогоняет полный конвейер на out-of-sample части: features → models → bandit → optimizer → risk → execute.
3. Собирает богатый `StrategyReport` со всеми историями и метриками.

использование ([src/backtest/strategy.py](../src/backtest/strategy.py)):
```python
from src.backtest import quick_strategy
from src.models import ModelSpec

rep = quick_strategy(
    prices_df,
    rebalance_every_days=5,
    model_specs=(ModelSpec(id='ridge', type='Ridge'),
                 ModelSpec(id='lgbm',  type='LightGBM'),
                 ModelSpec(id='mlp',   type='MLP')),
    regime_detector='vol_cluster',     # vol_cluster | hmm | bocpd
    allocator='thompson',              # eps_greedy | ucb1 | thompson | linucb
    optimizer='hrp',                   # hrp | risk_parity | markowitz
    use_crypto_risk=True,
)
print(rep.metrics)                # Sharpe, deflated Sharpe, max DD, ...
rep.equity_curve.plot()
```

контракт `StrategyReport`:
```python
@dataclass
class StrategyReport:
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    predictions_history: pd.DataFrame
    regime_history: pd.Series
    arm_history: pd.Series              # выбор бандита по времени
    fills: list[Fill]
    rejections: list[str]
    drawdown_history: pd.Series
    metrics: BacktestMetrics
    deflated_sharpe: float
    crypto_risk_report: dict | None
```

fill simulator: `QueueAwareFillSimulator` walks the book для market-ордеров
и проверяет crossing для limit-ордеров. Slippage по square-root impact:
$\Delta P / P = c \cdot \sigma \cdot \sqrt{Q/V}$.

---

### модуль 12 — `src/live/` — Live Trading Loop

когда срабатывает: главный процесс runtime, поднимается единожды.

что делает: оркестрирует все 11 предыдущих модулей в единый асинхронный
event-loop. Один и тот же класс работает во всех трёх режимах:

```python
app = TradingApp(
    config=TradingAppConfig(mode='paper'),   # 'backtest' | 'paper' | 'live'
    connector=binance_or_replay_engine,
    risk_gate=RiskGate(),
    crypto_risk_adapter=adapter,             # опционально
)
await app.run_until_done()
```

что включает:
- Event-loop по `connector.stream()`.
- Per-event handle с метрикой latency.
- `MetricsExporter` для Prometheus.
- `validate_target_weights()` — две-слойная проверка (RiskGate + crypto_risk).
- Health-endpoints / kill-switch heartbeat (опционально).

---

### cross-module контракты (Pydantic schemas)

все межмодульные интерфейсы — Pydantic v2 frozen models. Это даёт:
- JSON round-trip из коробки.
- Валидацию на границе.
- Невозможность мутаций.
- Готовность к serialization для логов/метрик.

карта схем:

| Schema | Откуда | Куда |
|--------|--------|------|
| `MarketEvent` | 01 (ingest) или 02 (replay) | 03 features, 06 regime, 11 backtest |
| `BookSnapshot` | 01 (OrderBook.snapshot) | 03 features, 11 fill_sim |
| `FeatureVector` | 03 | 04 labeling (X), 05 models (inference), 06 regime |
| `LabeledDataset` | 04 | 05 models (fit) |
| `Prediction` | 05 | 07 meta_allocator |
| `RegimeState` | 06 | 07 meta_allocator (контекст) |
| `StrategyWeights` + `AggregatedSignal` | 07 | 08 portfolio_opt |
| `TargetWeights` | 08 | 10 risk (validate), 09 execution |
| `PortfolioConstraints` | конфиг | 08 portfolio_opt |
| `ValidationResult` | 10 | 09 execution (если approved) |
| `ChildOrder` | 09 EMS | 11 fill_sim или live exchange |
| `Fill` | 09 OMS или 11 fill_sim | 10 monitor, 11 report |
| `PortfolioState` | 10 monitor | 12 live (snapshot для UI/alerts) |

---

### режимы выполнения

### backtest

```python
from src.storage import ReplayEngine, StorageConfig, ReplayConfig
engine = ReplayEngine(storage=StorageConfig(...), replay=ReplayConfig(speed='max'),
                      from_ns=..., to_ns=..., sources=[('binance', 'BTCUSDT')])
app = TradingApp(TradingAppConfig(mode='backtest'), connector=engine)
await app.run_until_done()
```

или через `StrategyBacktest` напрямую на price DataFrame (без событий тиков):

```python
from src.backtest import quick_strategy
rep = quick_strategy(prices_df, ...)
```

### paper trading

```python
from src.market_data import BinanceConnector
conn = BinanceConnector('wss://stream.binance.com:9443/ws', symbols, channels)
app = TradingApp(TradingAppConfig(mode='paper'), connector=conn)
# Внутри TradingApp.execution → FillSimulator вместо реального exchange OMS
await app.run_until_done()
```

### live (с реальным OMS)

то же самое, что paper, но в `app` нужно передать реальный `OMS`, подключённый
к биржевому API. На данный момент не подключено — паттерн заложен, но
требует API-keys (которые пока не настроены, см. README).

---

### когда что срабатывает — итоговая таблица

| Событие | Что вызывается |
|---------|----------------|
| WS-сообщение от биржи | `Connector._parse` → `MarketEvent` → очередь |
| `MarketEvent` (depth_delta) | `OrderBook.apply` |
| `MarketEvent` любой | `ResyncManager.observe`, `FeaturePipeline.update` |
| Новый `FeatureVector` | `RegimeDetector.update` (если фича — return) |
| Ребаланс-таймер | `Model.predict_one` для каждой модели в пуле |
| После predictions | `Allocator.allocate(predictions, context)` |
| Новый `StrategyWeights` | `PortfolioOptimizer.optimize` |
| Новый `TargetWeights` | `RiskGate.validate` → опционально `CryptoRiskAdapter.validate_target` |
| `ValidationResult.approved` | `EMS` → `ChildOrder` → `FillSimulator` / real OMS |
| Новый `Fill` | `PortfolioMonitor.on_fill`, `Allocator.update` (с realized reward) |
| Каждые ~1 час | `CryptoRiskAdapter.portfolio_report` для отчёта |
| Drawdown > limit | `KillSwitch.trigger` → следующая ребаланс отказана |

---

### глоссарий

| Термин | Значение |
|--------|----------|
| BBO | Best Bid/Offer — лучшая bid и ask цена |
| CPCV | Combinatorial Purged Cross-Validation (López de Prado) |
| DCC | Dynamic Conditional Correlation (Engle 2002) |
| DSR | Deflated Sharpe Ratio (Bailey-LdP 2014) |
| EVT | Extreme Value Theory; POT/GPD моделирует хвост |
| EMS | Execution Management System |
| FHS | Filtered Historical Simulation для VaR |
| HRP | Hierarchical Risk Parity |
| IC | Information Coefficient — корреляция фичи с future return |
| IS | Implementation Shortfall = $\sum (P_{exec} - P_{arrival}) \cdot Q$ |
| LVaR | Liquidity-adjusted VaR (Bangia exogenous) |
| MBO | Market-By-Order (L3 данные) |
| OFI | Order Flow Imbalance |
| OMS | Order Management System |
| PSR | Probabilistic Sharpe Ratio |
| VPIN | Volume-synchronized Probability of Informed Trading |

---

### где что искать в коде

| Хочешь… | Иди в… |
|---|---|
| понять оркестрацию | `src/live/_internal/app.py` |
| тянуть live данные | `src/market_data/_internal/connectors.py` |
| писать/читать историю | `src/storage/_internal/` |
| строить фичи | `src/features/_internal/features.py` |
| разметить таргеты | `src/labeling/_internal/triple_barrier.py` |
| обучить модели | `src/models/_internal/{baselines,torch_models}.py` |
| детектировать режим | `src/regime/_internal/detectors.py` |
| распределить капитал | `src/portfolio/_internal/optimizers.py` |
| исполнить ордер | `src/execution/_internal/{oms,agents}.py` |
| риск-проверка | `src/risk/_internal/{risk_gate,monitor}.py` |
| глубокая аналитика VaR/ES/EVT | `crypto_risk/{var_es,evt,backtesting}.py` |
| полный strategy backtest | `src/backtest/strategy.py` |
| примеры/визуализация | `notebooks/01-13_*.ipynb` |
| промт-доку для агента | `promts/00_shared_context.md` + `promts/NN_<module>.md` |
| методология | этот doc → [METHODOLOGY.md](METHODOLOGY.md) |
| хэндофф агенту | [AGENT_CONTEXT.md](AGENT_CONTEXT.md) |
