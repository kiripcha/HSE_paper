### AGENT_CONTEXT.md — Handoff для LLM-агента

этот документ — самодостаточный брифинг для другого LLM-агента, который
должен продолжить работу над проектом без чтения всего кода. Содержит:
конвенции, полный публичный API с сигнатурами, рецепты типичных задач, сжатый
теоретический ликбез, известные подводные камни и smoke-сценарии для проверки.

если ты читаешь этот файл — у тебя достаточно контекста, чтобы открыть
любой модуль, понять что он делает, и расширить его, не разрушив инварианты
системы.

для детальной формулы → код любого алгоритма — открой
[METHODOLOGY.md](METHODOLOGY.md).
для общей архитектурной картины — [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

### что это за проект (TL;DR)

дипломная работа: высокоскоростная торговая система для оптимизации
криптопортфеля на основе DL и адаптивных стратегий. 12 production-модулей
в `src/` + отдельный пакет `crypto_risk/` для глубокой риск-аналитики.

один runtime, три режима: backtest / paper / live. Разница — только в
подключаемом `ExchangeConnector` и (для backtest) `FillSimulator` вместо
реального OMS.

ключевая мета-стратегия: пул из 3-4 ML-моделей + regime detector +
multi-armed bandit allocator + HRP portfolio optimizer. Эта связка —
заявленная научная новизна работы.

---

### архитектура в одной картинке

```
ingestion → orderbook → features → models ──┐
                                     regime ─┴► allocator → optimizer → risk → execution
                                                                                  ↓
                                            backtest ◄── replay ◄── storage ◄────fills
                                                                                  ↓
                                                            crypto_risk (analytics, batch)
```

12 модулей в `src/`:

| # | Папка | Назначение | Контрактный output |
|---|-------|-----------|---------------------|
| 01 | `market_data/` | WS-приём + OrderBook | `MarketEvent`, `BookSnapshot` |
| 02 | `storage/` | Parquet + replay | `MarketEvent` (через ReplayEngine) |
| 03 | `features/` | Online features | `FeatureVector` |
| 04 | `labeling/` | Triple-barrier + CV | `LabeledDataset` |
| 05 | `models/` | Pred. models pool | `Prediction` |
| 06 | `regime/` | Regime detector | `RegimeState` |
| 07 | `meta_allocator/` | Bandits | `StrategyWeights`, `AggregatedSignal` |
| 08 | `portfolio/` | Optimization | `TargetWeights` |
| 09 | `execution/` | OMS + agents | `Fill` events |
| 10 | `risk/` | Gate + monitor + crypto_risk adapter | `ValidationResult`, `PortfolioState` |
| 11 | `backtest/` | Event-driven + strategy | `BacktestReport`, `StrategyReport` |
| 12 | `live/` | Orchestrator | (runtime, не схема) |

плюс пакет `crypto_risk/` на root уровне: VaR/ES/EVT/copulas/stress.

---

### конвенции (запомни их раз)

| | |
|---|---|
| Время | `int64` наносекунды UTC ВЕЗДЕ. Никаких `datetime` в горячем пути |
| Деньги | `float64` для PnL-неchувств; `Decimal` если потребуется (пока не нужно) |
| Аннуализация | 365 дней для крипты (24/7). Не 252 |
| VaR / ES | ПОЛОЖИТЕЛЬНЫЕ числа = убыток в долях капитала. `0.05` ≡ 5% |
| Лог-доходности | `log(P_t / P_{t-1})`, не simple-returns |
| Seed | `42` в большинстве мест, явный в каждой стохастической функции |
| Веса | `dict[str, float]` или `np.ndarray`; сумма = `leverage_cap` |
| Cov | `np.ndarray (n, n)`, units указаны в сигнатуре (daily / annualized) |
| Партиции Parquet | Hive-style `key=value` в путях |
| Тестовый dir | `pytest` запускается из корня проекта; пути относительные |

жёсткие инварианты:
1. `ReplayEngine` имплементирует `ExchangeConnector` Protocol — backtest и
   live используют тот же runtime.
2. Train/serve parity для фич — streaming-результат побитово равен batch.
3. Strictly online все regime-детекторы и фичи: не пересматривают прошлое.
4. Покрытие на каждом критическом тесте: `test_no_lookahead`, `test_replay_book_consistency`, `test_train_serve_parity`.

---

### публичный API (то, что вызывать снаружи)

### market data ingest

```python
from src.market_data import (
    BinanceConnector, OkxConnector,
    SortedBook, ArrayBook, make_order_book,
    ResyncManager,
    MarketEvent, BookSnapshot, BookDelta, BookLevel, Trade, BBO,
    derive_mid, derive_micro_price, derive_imbalance,
    random_walk_deltas, synthetic_snapshot,
)

conn = BinanceConnector(
    ws_url="wss://stream.binance.com:9443/ws",
    symbols=["BTCUSDT"], channels=["depth", "trade", "bookTicker"],
)
await conn.connect()
async for ev in conn.stream():
    book.apply(ev)
```

### storage + replay

```python
from src.storage import (
    ParquetTickWriter, ParquetHistoricalLoader, ReplayEngine,
    StorageConfig, ReplayConfig,
)

cfg = StorageConfig(root_path="data/ticks", compression="zstd")
writer = ParquetTickWriter(cfg)
for ev in events:
    writer.write(ev)
await writer.close()

engine = ReplayEngine(
    storage=cfg, replay=ReplayConfig(speed="max"),
    from_ns=t0, to_ns=t1,
    sources=[("binance", "BTCUSDT"), ("okx", "BTC-USDT")],
)
async for ev in engine.stream(): ...
```

### features pipeline

```python
from src.features import FeaturePipeline, FeaturesConfig, FeatureSpec

cfg = FeaturesConfig(features=[
    FeatureSpec(type="MicroPrice"),
    FeatureSpec(type="Imbalance", levels=1),
    FeatureSpec(type="OFI", window=100),
    FeatureSpec(type="RealizedVol", window=60),
    FeatureSpec(type="RSI", period=14),
    FeatureSpec(type="MACD", fast=12, slow=26),
])
pipe = FeaturePipeline(cfg)
for ev in events:
    book.apply(ev)
    fv = pipe.update(ev, book)              # FeatureVector | None во время warmup
```

### labeling + CV

```python
from src.labeling import (
    TripleBarrierConfig, triple_barrier_labels, sample_uniqueness_weights,
    LabeledDataset, PurgedKFold, CombinatorialPurgedCV,
    sharpe_ratio, probabilistic_sharpe, deflated_sharpe, sample_stats,
)

labels, t0, t1, _ = triple_barrier_labels(prices, ts_ns, TripleBarrierConfig(
    pt=2.0, sl=1.0, max_holding_steps=60, vol_window=100,
))
w = sample_uniqueness_weights(t0, t1)
ds = LabeledDataset(X=X, y=labels, sample_weights=w, t0=t0, t1=t1, feature_names=...)

cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
for train_idx, test_idx in cv.split(ds): ...

cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
print(cpcv.n_paths())   # 15
```

### predictive models

```python
from src.models import (
    RidgeModel, LightGBMModel, MLPModel, TCNModel,
    ModelSpec, ModelsConfig,
    Prediction, FitReport,
)

models = [
    RidgeModel(model_id="ridge", alpha=1.0),
    LightGBMModel(model_id="lgbm", n_estimators=200),
    MLPModel(model_id="mlp", hidden=(64, 64), dropout=0.2, seed=42),
]
for m in models:
    m.fit(ds)             # ds: LabeledDataset
    p = m.predict_one(x)  # x: FeatureVector | np.ndarray → Prediction
```

### regime detection

```python
from src.regime import HMMDetector, BOCPDDetector, VolClusterDetector, RegimeState

det = VolClusterDetector(n_regimes=3, window=100, history=1000)
for r in returns:
    st = det.update(r, ts_ns=int(ts.value), symbol="PORT")
    # st.regime_id, st.probabilities, st.steps_in_regime, st.change_point_proba
```

### meta-allocator

```python
from src.meta_allocator import (
    EpsilonGreedyAllocator, UCB1Allocator, ThompsonSamplingAllocator,
    LinUCBAllocator,
    AllocatorContext, StrategyWeights, AggregatedSignal,
)

alloc = LinUCBAllocator(arms=["ridge", "lgbm", "tcn"], context_dim=3, alpha=0.5)
ctx = AllocatorContext(ts_ns=ts_ns, regime_state=regime_state)
sw = alloc.allocate(predictions, context=ctx)
# обновить после получения реализованного reward:
alloc.update("ridge", reward=0.012, context=ctx)
```

### portfolio optimization

```python
from src.portfolio import (
    MarkowitzOptimizer, HRPOptimizer, RiskParityOptimizer,
    LedoitWolfCovariance,
    PortfolioConstraints, TargetWeights,
)

opt = HRPOptimizer()
constraints = PortfolioConstraints(
    max_weight_per_asset=0.35, min_weight_per_asset=0.0,
    leverage_cap=1.0, transaction_cost_bps=5.0,
)
target = opt.optimize(symbols, mu, cov, constraints)
# target.weights: dict[symbol, float]
```

### execution

```python
from src.execution import (
    OMS, ChildOrder, OrderEvent, OrderState, Fill,
    MarketOrderAgent, TWAPAgent, AlmgrenChrissAgent,
    ExecutionState, ExecutionAction,
)

oms = OMS()
oms.place(ChildOrder(client_order_id="x1", exchange="sim", symbol="BTC",
                     side="buy", order_type="limit", quantity=1.0, price=50000.0))
oms.on_event(OrderEvent(ts_ns=..., client_order_id="x1", event="partial_fill",
                        fill_quantity=0.5, fill_price=50001.0))

agent = AlmgrenChrissAgent(eta=1e-4, sigma=0.02, risk_aversion=1e-6)
action = agent.act(ExecutionState(
    inventory_remaining=1000.0, time_remaining_ns=600_000_000_000,
    mid_price=50000.0, spread=10.0,
))
```

### risk management

```python
from src.risk import (
    RiskGate, RiskLimits, PortfolioMonitor, KillSwitch,
    PortfolioState, ValidationResult, RiskViolation,
    CryptoRiskAdapter, CryptoRiskAdapterConfig, TorchVolAdapter,
    historical_var, historical_cvar, parametric_var, kupiec_pof_test,
)

# runtime layer
gate = RiskGate(RiskLimits(max_leverage=1.0, max_concentration_pct=0.3, max_drawdown_pct=0.15))
result = gate.validate(target_weights, current_drawdown_pct=0.05)
monitor = PortfolioMonitor(starting_cash=100_000.0)
monitor.on_fill(fill)
snap = monitor.snapshot(ts_ns=ts_ns)

# crypto_risk adapter (heavy analytics)
adapter = CryptoRiskAdapter.from_returns(returns_df)
adapter.attach_dl_vol_model(my_lstm_vol)
res = adapter.validate_target(target_weights, current_drawdown=0.05)
rep = adapter.portfolio_report(weights_dict)        # serializable
tail = adapter.tail_risk(weights_dict, threshold_q=0.90)
```

### crypto_risk (offline analytics, прямой доступ)

```python
from crypto_risk import RiskEngine, RiskConfig
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR
from crypto_risk import var_es as ve, evt, volatility as vol, portfolio as pf

eng = RiskEngine(RiskConfig(universe=TOP10_POPULAR), data_mode="auto")
eng.load_data_multi("2015-01-01", "2025-01-01")
w = eng.optimal_weights("max_sharpe", "long_only", cov_method="shrinkage")
rep = eng.risk_report(weights=w, use_copula=True)
dec = eng.pre_trade_check("BTC", 1_000_000, "buy", adv_usdt=1e10)
bt = eng.backtest_var(weights=w, method="historical", window=252)
```

### backtest

```python
from src.backtest import (
    run_simple_backtest, BacktestConfig, BacktestReport, BacktestMetrics,
    SimpleFillSimulator, QueueAwareFillSimulator, square_root_impact_bps,
    StrategyBacktest, StrategyConfig, StrategyReport, quick_strategy,
)

# simple — для sanity checks
report = run_simple_backtest(timestamps_ns, prices, book_snapshots,
                             target_weights_series, BacktestConfig())

# full end-to-end — для диплома
rep = quick_strategy(
    prices_df,
    rebalance_every_days=5,
    model_specs=(ModelSpec(id="ridge", type="Ridge"),
                 ModelSpec(id="lgbm", type="LightGBM")),
    regime_detector="vol_cluster",
    allocator="thompson",
    optimizer="hrp",
    use_crypto_risk=True,
)
print(rep.metrics, rep.deflated_sharpe)
rep.equity_curve.plot()
```

### live trading app

```python
from src.live import TradingApp, TradingAppConfig, MetricsExporter
from src.market_data import BinanceConnector
from src.risk import CryptoRiskAdapter

app = TradingApp(
    config=TradingAppConfig(mode="paper", symbols=["BTCUSDT"]),
    connector=BinanceConnector(...),
    risk_gate=RiskGate(),
    crypto_risk_adapter=adapter,
)
await app.run_until_done()    # blocks until SIGTERM or kill_switch
```

---

### рецепты типичных задач

### подключить нового predictive-модель в систему

1. Создать класс в `src/models/_internal/`:
   ```python
   class MyModel:
       name = "MyModel"; model_id: str; version = "1.0"; target_horizon_ns = 0
       def fit(self, dataset: LabeledDataset) -> FitReport: ...
       def predict_one(self, x) -> Prediction: ...
       def predict_batch(self, X: np.ndarray) -> np.ndarray: ...
       def save(self, path: Path) -> None: ...
   ```
2. Добавить в `src/models/__init__.py` экспорт.
3. Добавить case в `src/backtest/strategy.py::_build_model` (если хочешь
   использовать в StrategyBacktest).
4. Добавить тест в `src/models/tests/test_models.py` (smoke).
5. Прогнать `pytest src/models -v`.

### добавить новую фичу

1. В `src/features/_internal/features.py` создать класс:
   ```python
   class MyFeature:
       def __init__(self, param: int) -> None:
           self.name = f"myfeat_{param}"
           self.config_hash = f"myfeat_{param}"
           self._value = math.nan
           self.warmup_complete = False
       def update(self, event: MarketEvent, book: OrderBookProtocol | None) -> None: ...
       def value(self) -> float: return self._value
   ```
2. В `src/features/config.py` добавить тип в `FeatureType` Literal.
3. В `src/features/_internal/pipeline.py::_build_feature` добавить branch.
4. Экспортнуть в `src/features/__init__.py`.
5. Обязательно: добавить в `tests/test_pipeline_parity.py` чтобы проверить
   train/serve parity. Это главный инвариант модуля.

### подключить новый bandit-алгоритм

1. В `src/meta_allocator/_internal/allocators.py` создать класс с методами
   `allocate(predictions, context) -> StrategyWeights` и `update(arm, reward)`.
2. Добавить в `__init__.py` экспорт.
3. В `src/backtest/strategy.py::_build_allocator` добавить case.
4. Smoke-тест в `src/meta_allocator/tests/test_allocators.py`.

### добавить новый источник данных в crypto_risk

```python
class MyExchangeSource:
    name = "my-exchange"
    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        # запрос к API; вернуть Series с tz-aware DatetimeIndex (UTC)
        return pd.Series(closes, index=pd.DatetimeIndex(dates, tz="UTC"), name=base)

ld = MultiSourceCryptoLoader(
    universe=TOP10_POPULAR,
    sources=[CryptoCompareSource(), MyExchangeSource(), YahooSource()],
    mode="auto",
)
```

### запустить полный strategy backtest на новых данных

```python
import pandas as pd
from src.backtest import quick_strategy
from src.models import ModelSpec

prices = pd.read_parquet("data/historical/...parquet")
rep = quick_strategy(
    prices,
    train_fraction=0.4,
    rebalance_every_days=5,
    model_specs=(
        ModelSpec(id="ridge", type="Ridge"),
        ModelSpec(id="lgbm",  type="LightGBM"),
        ModelSpec(id="mlp",   type="MLP"),
    ),
    use_crypto_risk=True,
)
```

### получить live-данные с Binance / OKX без аккаунта

public WS работает без авторизации:

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_live_binance.py
PYTHONPATH=. .venv/bin/python scripts/smoke_live_okx.py
```

для размещения реальных ордеров (live trading) нужны API-keys биржи.
сейчас не настроены — задача за пределами scope диплома.

### использовать crypto_risk изолированно

```python
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR
ld = MultiSourceCryptoLoader(TOP10_POPULAR, mode="auto")
prices = ld.load_close_panel("2015-01-01", "2025-01-01")
# prices: pd.DataFrame [date × ticker]
```

---

### сжатый теоретический ликбез

### лог-доходности и аннуализация

$r_t = \ln(P_t / P_{t-1})$. Аннуализация: $\sigma_{\text{year}} = \sigma_{\text{day}} \cdot \sqrt{365}$
для крипты (не 252 как для акций).

### GARCH(1,1) с t-инновациями

$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$.
GJR: $+ \gamma r_{t-1}^2 \mathbb{1}_{r_{t-1} < 0}$ — leverage.
инновации Student-t ловят heavy tails.

### vaR / ES под нормальностью

$\text{VaR}_\alpha = -(\mu + \sigma z_{1-\alpha})$,
$\text{ES}_\alpha = -(\mu - \sigma \phi(z_{1-\alpha})/(1-\alpha))$.

### EVT POT/GPD

$L - u \mid L > u \sim \text{GPD}(\xi, \beta)$.
$\text{VaR}_\alpha = u + (\beta/\xi)\!\left[(n/N_u)(1-\alpha))^{-\xi} - 1\right]$.
$\xi > 0$ → heavy tail. $\xi \ge 1$ → ES не существует.

### conditional EVT (McNeil-Frey)

$\text{VaR}_{t+1} = \sigma_{t+1}^{GARCH} \cdot \text{VaR}_\alpha(z)$, где
$\text{VaR}_\alpha(z)$ — POT-квантиль стандартизованных остатков. Best practice.

### triple-barrier labels

upper/Lower/Vertical → label ∈ {-1, 0, +1}. Реалистичная разметка для квант ML.

### purged k-fold

из train исключаются образцы, чья жизнь $(t_0, t_1)$ пересекается с test-окном
+ embargo $h$. Защита от label-overlap leakage.

### deflated Sharpe (LdP 2014)

PSR с benchmark = ожидаемый max over $N$ trials под H0. Защищает от
data-snooping bias. Обязательная финальная метрика.

### multi-armed bandits

UCB: $A_t = \arg\max \hat\mu_a + \sqrt{2 \ln t / n_a}$. Regret $O(\log T)$.
thompson: posterior sampling. LinUCB: контекст-зависимый.

### HRP

hierarchical clustering на correlation distance + recursive bisection с
inverse-variance weights. Без $\boldsymbol\mu$, без QP. Устойчив к шуму.

### almgren-Chriss

$x_j = X \cdot \sinh(\kappa(T-t_j)) / \sinh(\kappa T)$. Trade-off $E[\text{cost}]$ vs $\text{Var}[\text{cost}]$.

### vol targeting

$L = \min(\sigma^* / \hat\sigma, L_{\max})$. Leverage обратно пропорционален вола.

### fractional Kelly

$\mathbf{w}^* = \Sigma^{-1}(\boldsymbol\mu - r_f)$. На практике $0.25$-$0.5$ Kelly.

### basel traffic light

0-4 violations / 250 days = GREEN (×3). 5-9 = YELLOW (3→4). ≥10 = RED (×4).

для более полной математики — [METHODOLOGY.md](METHODOLOGY.md).

---

### подводные камни (что НЕ-очевидно)

### время — int64 ns UTC

никаких `datetime` объектов в горячем пути. Все поля `ts_*` это `int`. Если
видишь Pydantic-ошибку при validate — скорее всего где-то datetime просочился.

### vaR / ES — ПОЛОЖИТЕЛЬНЫЕ

из crypto_risk-пакета `est.var > 0` означает убыток. Если в callsite кто-то
проверяет `var < 0` — это ошибка.

### BOCPD `change_point_proba` всегда = hazard

$P(r_t = 0 \mid x_{1:t}) = H$ по построению. Сигнал — это **падение argmax
run-length** (`steps_in_regime`), не `probabilities[0]`. Если меняешь BOCPD-
логику, не правь это поведение — это математический факт.

### pydantic Union на payload

`MarketEvent.payload` имеет валидатор `_coerce_payload` на основе `event_type`.
без него Pydantic выбирает первый матчинг типа Union (BookDelta — супер-сет
bookSnapshot). Не упрощай, валидатор обязателен.

### pyArrow Hive partitioning

`pq.read_table(path)` в современном PyArrow инферит Hive-партиции и конфликтует
с одноименными колонками в данных. Использую `pq.ParquetFile(path).read()`. Не
переходи обратно — поломаешь loader.

### train/serve parity для фич

каждая фича пишется в двух режимах: streaming `update()` и batch (вызов
того же update через DataFrame). Они должны давать побитово равный результат.
это покрыто `test_train_serve_parity`. Если ты добавил фичу и тест падает —
скорее всего использовал pandas rolling вместо incremental.

### PSR требует per-period Sharpe

формула PSR использует per-period Sharpe (не annualized). С большой
annualized SR (например 10+) денominator уходит в отрицательное под корнем.
в тестах передавай per-period SR (`r.mean() / r.std(ddof=1)`).

### riskGate.kill_switch_active

этот флаг переписывается извне в `TradingApp.validate_target_weights()`
из текущего состояния KillSwitch. Если ты используешь RiskGate напрямую — не
забудь обновлять `gate.kill_switch_active = ks.is_active()` перед каждым
вызовом `validate()`.

### daily annualization = 365

крипта 24/7. Не 252. Если внешняя система отдаёт akции-стандарт — переводи:
$\sigma_{\text{year, 252}} = \sigma_{\text{year, 365}} \cdot \sqrt{252/365}$.

### EVT работает с УБЫТКАМИ (positive)

внутри функций `crypto_risk.evt.*` уже сделано $L = -r$. Снаружи передавай
обычные returns, не negate-нутые.

### HRP iterative clip

`_clip_to_constraints` в `src/portfolio/_internal/optimizers.py` итеративно
clipает + нормирует. Не делай однократный clip — некоторые веса вылезут за
`max_weight_per_asset` после re-normalization.

### DCC-GARCH медленный

30-60 секунд на 10 активов. Не вызывай в горячем цикле. Кэшируй результат
или используй reшание-проксы (LW shrinkage).

### ccxt и arch — опциональные deps

если не установлены, соответствующие куски deg degrade gracefully. Не делай
их жёсткими импортами на module-top.

### t-copula loglik с gammaln

используется `scipy.special.gammaln`, не `scipy.stats.loggamma` (это
распределение, не функция). Исторический баг.

### `pre_trade_check` возвращает approved=False ещё и при размер=0

после `dd_mult * adv-clip` может получиться `sized_notional == 0`. Тогда
approved=False даже если хард-violation не было. В callsite проверяй
`sized_notional > 0` отдельно.

### strict_online режим у HMM

HMM делает медленное online-обновление параметров $(\mu, \sigma)$ через
learning rate. Это не Viterbi smoothing — прошлые states не меняются.
сохраняй это поведение при extensions.

### async generators в connectors

`stream()` — async generator. `async for ev in conn.stream()`. Не пытайся
вызывать `next()` или `iter()` — будет ошибка.

### notebooks: top-level await

все ноутбуки используют top-level `await` (поддерживается в IPython 7+). НЕ
оборачивай в `asyncio.run()` — в Jupyter уже есть event loop.

---

### окружение и зависимости

python: 3.11+ (тестировано на 3.12). Совместимо с 3.10.

установка:
```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
# crypto_risk-deps если их нет:
.venv/bin/pip install arch statsmodels cvxpy ccxt requests tqdm pandas scipy
```

жёсткие зависимости: numpy, pandas, polars, pyarrow, scipy, scikit-learn,
pydantic, structlog, prometheus_client, sortedcontainers, websockets, orjson,
torch, lightgbm, matplotlib.

опциональные: arch (GARCH), cvxpy (constrained QP), ccxt (Binance), mlflow.

запуск тестов:
```bash
.venv/bin/python -m pytest -q                              # все 163
.venv/bin/python -m pytest src/features/tests -v           # один модуль
.venv/bin/python -m pytest tests/test_cross_module_integration.py -v
```

запуск ноутбуков:
```bash
.venv/bin/python notebooks/_build_13.py                     # пересобрать .ipynb
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 13_full_strategy_backtest.ipynb --output 13_full_strategy_backtest.ipynb
```

---

### где что искать

| Хочешь… | Иди в… |
|---------|--------|
| понять оркестрацию live/paper/backtest | `src/live/_internal/app.py` |
| подключиться к бирже | `src/market_data/_internal/connectors.py` |
| записать/прочитать историю | `src/storage/_internal/{writer,loader,replay}.py` |
| новую фичу | `src/features/_internal/features.py` |
| модель | `src/models/_internal/{baselines,torch_models}.py` |
| детектировать режим | `src/regime/_internal/detectors.py` |
| bandit | `src/meta_allocator/_internal/allocators.py` |
| оптимизировать портфель | `src/portfolio/_internal/optimizers.py` |
| исполнить ордер | `src/execution/_internal/{oms,agents}.py` |
| лёгкий risk | `src/risk/_internal/{risk_gate,monitor,var}.py` |
| тяжёлый risk (VaR/EVT/Copula) | `crypto_risk/{var_es,evt,backtesting,dependence}.py` |
| GARCH/EWMA | `crypto_risk/volatility.py` |
| stress | `crypto_risk/stress.py` |
| полный strategy backtest | `src/backtest/strategy.py` |
| simple backtest | `src/backtest/_internal/engine.py` |
| fill simulator | `src/backtest/_internal/fill_sim.py` |
| Hydra configs | `configs/*.yaml` |
| примеры со всеми визуализациями | `notebooks/13_full_strategy_backtest.ipynb` |
| промт по конкретному модулю | `promts/NN_<module>.md` |
| общий контекст для агентов | `promts/00_shared_context.md` |
| архитектура | `documentation/PROJECT_OVERVIEW.md` |
| методология | `documentation/METHODOLOGY.md` |
| live smoke-тесты | `scripts/smoke_live_{binance,okx}.py` |
| исторический fetcher | `scripts/fetch_historical_data.py` |
| docs crypto_risk-пакета | `risk_managment_inner/documentation/` |

---

### smoke-сценарии для проверки интеграции

после любой правки прогни эти 6 проверок. Все должны пройти за < 1 минуты.

```python
# 1. Все top-level пакеты импортятся
import src.market_data, src.storage, src.features, src.labeling
import src.models, src.regime, src.meta_allocator, src.portfolio
import src.execution, src.risk, src.backtest, src.live
import crypto_risk

# 2. Live exchange smoke (нужна сеть)
# .venv/bin/python scripts/smoke_live_binance.py

# 3. Replay = live (главный invariant)
# .venv/bin/python -m pytest src/storage/tests/test_replay.py::test_replay_reconstructs_same_book_as_live

# 4. Train/serve parity (главный invariant features)
# .venv/bin/python -m pytest src/features/tests/test_pipeline_parity.py::test_train_serve_parity

# 5. Strategy backtest end-to-end
import numpy as np, pandas as pd
from src.backtest import quick_strategy
from src.models import ModelSpec
rng = np.random.default_rng(0)
n = 400
prices = pd.DataFrame({
    'BTC': 100*np.exp(np.cumsum(rng.normal(0.001, 0.02, n))),
    'ETH': 100*np.exp(np.cumsum(rng.normal(0.0005, 0.025, n))),
}, index=pd.date_range('2022-01-01', periods=n, freq='D'))
rep = quick_strategy(prices, use_crypto_risk=False,
                     model_specs=(ModelSpec(id='ridge', type='Ridge'),))
assert rep.metrics.n_trades > 0
assert (rep.equity_curve > 0).all()

# 6. Полный pytest
# .venv/bin/python -m pytest -q
# Ожидаемый результат: 163 passed
```

если хотя бы один валится — интеграция нарушила инвариант. Ищи где.

---

### что не реализовано полностью (roadmap)

архитектурно заложено, но требует доработки:

- DRL execution PPO: есть placeholder, нужно полное обучение через
  gymnasium-env с FillSimulator. См. `src/execution/_internal/agents.py`.
- Дифференцируемая Markowitz через cvxpylayers: end-to-end обучение модели
  + оптимизации совместно. Заложено в `src/portfolio/` но не подключено.
- Multi-target alpha models: сейчас регрессия предсказывает кросс-секционный
  средний return. Можно расширить до per-asset alpha.
- KillSwitch daemon в отдельном процессе с heartbeat-файлом: сейчас
  killSwitch — in-process state. Для production надо отдельный процесс.
- Grafana dashboard JSON: есть Prometheus metrics, но pre-built dashboards
  не созданы.
- Live OMS с API-keys биржи: чтение public-data работает, написание
  ордеров — нет. См. `src/execution/_internal/oms.py` (текущий — in-memory
  simulator).
- Tick-level features (модуль 03) в live-pipe модуля 12: сейчас
  `TradingApp._handle_event` не вызывает FeaturePipeline. В backtest (модуль 11)
  работает на daily-уровне; intraday tick-level pipeline — задача расширения.
- Vine copulas для N > 20 (есть в roadmap crypto_risk).
- CAViaR (Engle-Manganelli) — direct conditional quantile model.
- Reinforcement Learning для allocator — поверх bandit, hierarchical RL.

если получишь задачу из этого списка — следуй паттерну: новый модуль +
расширение публичного API + тест + раздел в ноутбуке.

---

### полезные команды

```bash
# Все тесты
.venv/bin/python -m pytest -q

# Один модуль
.venv/bin/python -m pytest src/features/tests -v

# Только cross-module integration
.venv/bin/python -m pytest tests/test_cross_module_integration.py -v

# Coverage
.venv/bin/python -m pytest --cov=src --cov-report=html

# Линтер
.venv/bin/python -m ruff check src/

# Mypy
.venv/bin/python -m mypy src/

# Live smoke tests (no auth needed)
PYTHONPATH=. .venv/bin/python scripts/smoke_live_binance.py
PYTHONPATH=. .venv/bin/python scripts/smoke_live_okx.py

# Загрузка исторических данных
PYTHONPATH=. .venv/bin/python scripts/fetch_historical_data.py \
    --start 2022-01-01 --end 2024-12-31 \
    --universe BTC,ETH,BNB,SOL --mode auto

# Перестроить ноутбук
.venv/bin/python notebooks/_build_13.py
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 13_full_strategy_backtest.ipynb --output 13_full_strategy_backtest.ipynb

# crypto_risk smoke (свой test suite пакета)
.venv/bin/python risk_managment_inner/tests/test_smoke.py
```

---

### контрольная фраза для быстрого узнавания контекста

если кратко:

> Это торговая система-диплом для крипты. 12 модулей в `src/` плюс
> `crypto_risk/` для аналитики. Один runtime (`src/live.TradingApp`) — три
> режима: backtest (через ReplayEngine) / paper (live data + FillSim) /
> live (с реальным OMS, пока не подключён). Контракты модулей — Pydantic v2
> frozen schemas. Время везде int64 ns UTC. VaR/ES всегда положительные.
> Аннуализация по 365. Главный инвариант: ReplayEngine = драп-ин для
> ExchangeConnector, поэтому backtest и live используют один и тот же код.
> Strategy backtest end-to-end — `src/backtest/strategy.py::quick_strategy`,
> демо со всеми визуализациями — `notebooks/13_full_strategy_backtest.ipynb`.
> 163 теста, все зелёные. Live WS-данные работают без аккаунта.
