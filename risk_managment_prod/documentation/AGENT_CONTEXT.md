# AGENT_CONTEXT.md — Handoff для LLM-агента

Этот документ — самодостаточный брифинг для **другого LLM-агента**, который
должен понять проект и интегрировать его в более крупную систему без чтения
остального кода и теоретических работ. Содержит: краткое описание, конвенции,
полный публичный API с сигнатурами, рецепты интеграции, сжатый теоретический
ликбез и список «подводных камней».

Если читаешь этот файл — у тебя достаточно контекста, чтобы пользоваться
пакетом и расширять его, не открывая ничего, кроме самого исходного кода.

---

## 1. Что это и зачем

**Пакет `crypto_risk/`** — модуль рыночного риск-менеджмента, разрабатываемый
для торговой системы, которая:
- торгует **криптовалютами** на спот/перпах,
- использует **глубокое обучение** для предсказательного сигнала,
- работает в режиме **HFT** (быстрая ротация позиций),
- адаптирует стратегии в режиме онлайн.

Модуль отвечает за:
1. сбор честных данных (с 2015 г.) из нескольких источников с фолбэком;
2. оценку **VaR / ES** портфеля десятками методов и их валидацию;
3. **построение оптимальных портфелей** (Markowitz + расширения);
4. учёт **риска ликвидности** и **издержек исполнения**;
5. **стресс-тестирование** (включая обратное);
6. **адаптивный сайзинг** позиций с хуком под DL-модель волатильности.

Не входит: исполнение заявок, P&L-учёт, сигналы.

---

## 2. Архитектура одной картинкой

```
RiskEngine (engine.py)            ◄── единственная точка входа
  │
  ├── data/ (parsers.py + sources.py)  крипто-парсер 5 источников + кэш
  ├── volatility.py                    GARCH-family, range-σ, HAR-RV, EWMA
  ├── var_es.py                        VaR/ES 9 методов + Euler-декомпозиция
  ├── evt.py                           POT/GPD, conditional EVT, Hill
  ├── covariance.py                    Ledoit-Wolf, RMT, DCC-GARCH
  ├── backtesting.py                   Kupiec, Christoffersen, DQ, Berkowitz,
  │                                    Acerbi-Szekely ES, model risk
  ├── portfolio.py                     эфф. граница (4 огранич.), beta-cov,
  │                                    risk-parity, two-fund, max-risk
  ├── liquidity.py                     LVaR (Bangia), Corwin-Schultz, market
  │                                    impact, IS, Almgren-Chriss
  ├── dependence.py                    Gaussian/t copula, tail dependence
  ├── stress.py                        historical, named crashes, worst-case
  │                                    (Breuer), reverse stress
  ├── controls.py                      vol-target, Kelly, drawdown, DL-хук
  └── config.py                        RiskConfig (один dataclass)
```

«Тонкие» модули атомарны: каждый принимает numpy/pandas и возвращает numpy/pandas
или dataclass с результатами. `engine.py` — оркестратор поверх них.

---

## 3. Конвенции (запомни их раз)

| | |
|---|---|
| Котировка | USDT (квази-USD) |
| Рыночный прокси | **BTC** (для бет, market model) |
| Аннуализация | **365** дней (крипта 24/7) |
| Семя | `crypto_risk.config.RANDOM_SEED = 42` |
| **Знак VaR/ES** | **ПОЛОЖИТЕЛЬНЫЙ** = убыток в долях капитала. `0.05` ≡ 5%. |
| Доходности | `log_returns(prices) = log(P_t / P_{t-1})`. NaN отбрасываются. |
| `mode` загрузчиков | `'auto'` (сеть + фолбэк) / `'exchange'` (только сеть, ошибка пробрасывается) / `'synthetic'` (детерминированно оффлайн) |
| Кэш | `data_cache/*.csv`, разделяет `exchange/synthetic` в имени файла |
| Веса портфеля | `np.ndarray` или `dict[str, float]`; должны суммироваться к 1 для long-only |
| Ковариация | `np.ndarray (n,n)`, в дневных или аннуализированных единицах — см. сигнатуру каждой функции |

---

## 4. Публичный API (то, что вызывать снаружи)

### 4.1 RiskEngine — главный объект

```python
from crypto_risk import RiskEngine, RiskConfig
from crypto_risk.data import TOP10_POPULAR

cfg = RiskConfig(universe=TOP10_POPULAR, capital=1_000_000)
eng = RiskEngine(cfg, data_mode="auto")          # 'auto'|'exchange'|'synthetic'

# --- Загрузка данных ---
eng.load_data(start="2021-01-01", end="2025-01-01", timeframe="1d")
# либо: длинная история через мультиисточник
eng.load_data_multi(start="2015-01-01", end="2025-01-01")

# Состояние после загрузки:
eng.prices         # pd.DataFrame [date × asset]
eng.returns        # pd.DataFrame log-returns
eng.data_source    # человекочитаемая строка (cache:/exchange:/multi:...)

# --- Оптимизация весов ---
w = eng.optimal_weights(
    objective="max_sharpe",       # 'min_variance'|'max_sharpe'|'risk_parity'
    constraint="long_only",       # 'long_short'|'short_limit'|'long_only'|'min_weight'
    cov_method="shrinkage",       # 'sample'|'ewma'|'beta'|'shrinkage'|'constant_corr'|'rmt'|'dcc'
    use_adjusted_beta=True,       # для cov_method='beta'
)
# w: dict[ticker, weight] — сумма = 1

# --- Риск-отчёт ---
rep = eng.risk_report(weights=w, use_copula=True)
# RiskReport(asset_risk, portfolio_risk, lvar, diversification_ratio,
#            annualized_vol, weights, horizons)

# --- EVT (точные хвосты) ---
tail = eng.tail_risk(weights=w, threshold_q=0.90, use_conditional=True)
# {'pot': RiskEstimate, 'conditional_evt': RiskEstimate}

# --- Декомпозиция риска ---
attr = eng.risk_attribution(weights=w, method="gaussian")
# DataFrame[asset → weight, marginal_VaR, component_VaR, pct_contrib_VaR, ...]

# --- Стресс ---
stress = eng.stress_test(weights=w, plausibility_k=3.0)
rev    = eng.reverse_stress(target_loss=0.50, weights=w)

# --- Модельный риск ---
mr = eng.model_risk(weights=w)
# {'var_estimates': {...}, 'relative_spread', 'basel_zone', 'basel_multiplier', ...}

# --- Бэктест VaR ---
bt = eng.backtest_var(weights=w, method="historical", window=252)
# {'backtest_table': DataFrame, 'n_violations', 'n_obs',
#  'violation_rate', 'expected_rate', 'var_series', 'realized'}

# --- HFT: предторговый контроль ---
dec = eng.pre_trade_check(
    asset="BTC", notional=5_000_000, side="buy",
    adv_usdt=2e10, current_drawdown=0.05,
    max_var_budget=0.05, max_participation=0.10,
)
# PreTradeDecision(approved, reasons, sized_notional, metrics)
```

### 4.2 Парсер (когда нужен только сбор данных)

```python
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR

ld = MultiSourceCryptoLoader(TOP10_POPULAR, mode="auto")
panel = ld.load_close_panel("2015-01-01", "2025-01-01", use_cache=True)
# panel: pd.DataFrame [date_UTC × ticker], NaN до даты появления актива

print(ld.data_origin)       # 'exchange' | 'synthetic'
print(ld.source_used)       # 'multi:cryptocompare+yahoo' или 'cache:...'
print(ld.coverage_report()) # таблица «кто что отдал, кто дозаполнил»
```

Источники в порядке приоритета: **CryptoCompare** (с 2010 г., без ключа) →
Yahoo (с 2015) → CoinGecko (best-effort) → Binance ccxt (с 2017) →
синтетика (детерминированно). Каждый — класс с методом
`fetch_close(base, start, end) -> pd.Series`.

### 4.3 Прямые вызовы тонких модулей

```python
from crypto_risk import volatility as vol, var_es as ve, evt
from crypto_risk import portfolio as pf, liquidity as liq
from crypto_risk import covariance as cvm, stress as stm, dependence as dep
from crypto_risk import backtesting as bt, controls as ctl

# Волатильность
r = vol.log_returns(prices["BTC"])
g = vol.GARCHModel(vol="GJR", dist="t").fit(r, horizon=10)
yz = vol.yang_zhang_vol(ohlc_df, window=30)
har = vol.har_rv((vol.garman_klass_vol(ohlc_df)**2).dropna())

# VaR/ES
est_h  = ve.historical_var_es(r, 0.99, 0.975, horizon=1)
est_t  = ve.parametric_var_es(r.mean(), r.std(), 0.99, 0.975, 1, "t", nu=5)
est_g  = ve.garch_var_es(g, 0.99, 0.975, 1)
est_f  = ve.fhs_var_es(r, g, 0.99, 0.975, 10)
est_e  = evt.pot_var_es(r, 0.99, 0.975)
est_ce = evt.conditional_evt(r, g, 0.99, 0.975)
tbl    = ve.compare_methods(r, garch_result=g)              # все методы в таблице

# Портфель
mu, cov = pf.mean_cov(returns, annualize=True)
opt = pf.PortfolioOptimizer(mu, cov, names=list(UNIVERSE))
gmv = opt.min_variance(constraint="long_only")
msr = opt.max_sharpe(constraint="long_only")
rp  = opt.risk_parity()
front = opt.efficient_frontier(40, "short_limit", short_limit=0.25)

# Беты
be = pf.estimate_betas(returns, market="BTC")
cov_b = pf.beta_covariance(be, use_adjusted=True, annualize=True)

# Устойчивая ковариация
cov_lw, delta = cvm.ledoit_wolf_cov(returns, annualize=True)
cov_rmt = cvm.rmt_denoise_cov(returns, annualize=True)
dcc = cvm.dcc_garch(returns)

# Ликвидность
lvar = liq.bangia_lvar(price_var=0.05, rel_spread_mean=3e-4, rel_spread_std=2e-4)
sched = liq.AlmgrenChriss(sigma=0.02, eta=2.5e-6, gamma=2.5e-7, lam=1e-6)\
              .schedule(total_shares=1_000_000, horizon=1.0, n_steps=20)

# Копулы
t = dep.StudentTCopula.fit(returns)
sim = dep.copula_portfolio_returns(returns, weights, t, n_sims=40000)

# Стресс
hist = stm.historical_scenarios(returns, weights, horizon=1, top=5)
wc   = stm.worst_case_loss(weights, mu_daily, cov_daily, plausibility_k=3.0)
rev  = stm.reverse_stress_test(weights, mu_daily, cov_daily,
                                target_loss=0.50, names=list(UNIVERSE))

# Бэктесты
I = bt.get_violations(realized_returns, var_series)
tbl = bt.run_var_backtests(realized_returns, var_series, var_alpha=0.99)
t1 = bt.acerbi_szekely_test1(realized_returns, var_series, es_series)
mr = bt.model_risk_metrics({"hist":0.05,"garch":0.06}, n_violations=12, n_obs=1000)

# Контроли
lev = ctl.vol_target_leverage(forecast_vol_annual=0.50, target_vol_annual=0.20)
kelly = ctl.kelly_weights(mu, cov, fraction=0.5)
dd_m  = ctl.drawdown_scale(current_dd=0.08, dd_limit=0.20)
```

---

## 5. Рецепты интеграции

### 5.1 Подключить производственную DL-модель прогноза σ

DL-модель (LSTM/TCN/Transformer) должна иметь метод
`predict(returns) -> float` (дневная σ в долях).

```python
class MyLSTMVol:
    def predict(self, returns: pd.Series) -> float:
        # ... инференс по последним N значениям returns
        return float(prediction)

# В RiskEngine:
eng.vol_forecaster.set_dl_model(MyLSTMVol())
fc = eng.vol_forecaster.forecast(eng.returns["BTC"])
# fc.source == 'dl'; fc.sigma_daily — то, что вернул MyLSTMVol.predict
```

Контур риска и vol-target используют этот прогноз через `VolForecaster`.

### 5.2 Встроить риск в торговую систему (pre-trade)

```python
# В обработчике сигнала торговой системы:
def on_signal(asset: str, target_notional: float, side: str):
    dec = engine.pre_trade_check(
        asset=asset, notional=target_notional, side=side,
        adv_usdt=get_adv(asset),                    # из биржевого API
        current_drawdown=portfolio.current_drawdown(),
        max_var_budget=0.05,
        max_participation=0.10,
    )
    if not dec.approved:
        log.warning(f"Trade rejected: {dec.reasons}")
        return
    execute_order(asset, dec.sized_notional, side)  # размер уже урезан
```

### 5.3 Бэктест собственной VaR-модели через нашу батарею тестов

```python
# Ты построил произвольный VaR ряд var_series (массив длины T) на доходностях
# realized (массив длины T). Прогоняешь батарею тестов:
from crypto_risk import backtesting as bt
table = bt.run_var_backtests(realized, var_series, var_alpha=0.99)
# Для ES:
t1 = bt.acerbi_szekely_test1(realized, var_series, es_series, es_alpha=0.975)
t2 = bt.acerbi_szekely_es(realized, var_series, es_series, es_alpha=0.975)
```

### 5.4 Добавить свой источник данных в парсер

```python
class MyExchangeSource:
    name = "my-exchange"
    def fetch_close(self, base: str, start: str, end: str) -> pd.Series:
        # ... запрос к API
        return pd.Series(closes, index=pd.DatetimeIndex(dates, tz="UTC"), name=base)

ld = MultiSourceCryptoLoader(
    universe=TOP10_POPULAR,
    sources=[CryptoCompareSource(), MyExchangeSource(), YahooSource()],
    mode="auto",
)
panel = ld.load_close_panel("2015-01-01", "2025-01-01")
```

Источник может быть классом, функцией, замыканием — нужно только наличие
метода `fetch_close(base, start, end) -> pd.Series` с tz-aware индексом.

### 5.5 Добавить новый метод VaR в `compare_methods`

В `var_es.py` в `compare_methods` сделай блок:
```python
try:
    from .my_method import my_var_es
    ests.append(my_var_es(r, var_alpha, es_alpha, horizon))
except Exception:
    pass
```
Метод должен вернуть `RiskEstimate(var, es, method, horizon, var_alpha, es_alpha, extra=None)`.

### 5.6 Использовать только парсер (без остальной системы)

```python
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR
ld = MultiSourceCryptoLoader(TOP10_POPULAR, mode="auto")
panel = ld.load_close_panel("2015-01-01", "2025-01-01")
panel.to_csv("my_data.csv")
```

---

## 6. Условный теоретический ликбез

Сжатый справочник формул, достаточных, чтобы рассуждать о модуле без чтения
работ.

### 6.1 Лог-доходности и аннуализация
$r_t = \ln(P_t/P_{t-1})$, $\sigma_{\text{year}} = \sigma_{\text{day}} \cdot \sqrt{365}$.

### 6.2 GARCH-семейство
GARCH(1,1): $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$.
GJR: добавочный член $\gamma r_{t-1}^2 \mathbb{1}_{r_{t-1}<0}$ ловит асимметрию.
Инновации Стьюдента $r_t = \sigma_t z_t$, $z_t \sim t_\nu$ — для тяжёлых хвостов.

### 6.3 EWMA
$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$, $\lambda = 0.94$.

### 6.4 VaR / ES (нормальный случай)
$\text{VaR}_\alpha = -(\mu + \sigma z_{1-\alpha})$;
$\text{ES}_\alpha = -(\mu - \sigma \phi(z_{1-\alpha})/(1-\alpha))$.

Для t: множитель `sqrt((ν-2)/ν)` приводит к Var=1; ES имеет замкнутый вид
(см. METHODOLOGY §B3).

### 6.5 EVT POT/GPD
Превышения $L - u \mid L > u \sim \text{GPD}(\xi, \beta)$. Тогда:
$$
\text{VaR}_\alpha = u + \frac{\beta}{\xi}\!\left[\left(\frac{n}{N_u}(1-\alpha)\right)^{-\xi}-1\right],
\quad \text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\beta - \xi u}{1-\xi}.
$$
$\xi > 0$ — тяжёлый хвост. $\xi \ge 1$ — ES не определён.

### 6.6 Conditional EVT (McNeil-Frey)
$\text{VaR}_{t+1} = \sigma_{t+1}^{GARCH} \cdot \text{VaR}_\alpha(z)$, где
$\text{VaR}_\alpha(z)$ — POT-квантиль стандартизованных остатков.

### 6.7 Markowitz
$\min \mathbf{w}^\top \Sigma \mathbf{w}$, $\mathbf{1}^\top \mathbf{w} = 1$,
$\boldsymbol\mu^\top \mathbf{w} \ge \mu^*$, $\mathbf{w} \in \mathcal{C}$.

### 6.8 Beta-based Σ (single-index)
$\Sigma = \boldsymbol\beta \boldsymbol\beta^\top \sigma_m^2 + \text{diag}(\sigma^2_\varepsilon)$.
Adjusted (Blume): $\beta^{\text{adj}} = 0.67 \beta + 0.33$.

### 6.9 Ledoit-Wolf shrinkage
$\hat\Sigma = \delta F + (1-\delta) S$, $\delta^*$ из аналитической оценки
ошибки. Цель F: масштабированная единичная или const-corr.

### 6.10 DCC-GARCH (Engle 2002)
$Q_t = (1-a-b)\bar Q + a z_{t-1} z_{t-1}^\top + b Q_{t-1}$,
$R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$. $z$ —
стандартизованные остатки от univariate GARCH.

### 6.11 Euler decomposition (Tasche)
Marginal: $\partial \rho/\partial w_i$. Component: $w_i \cdot \partial \rho/\partial w_i$.
$\sum_i \text{Component}_i = \rho(\mathbf{w})$.

### 6.12 Backtests (краткий справочник)
- **Kupiec POF**: $LR = -2\ln[p^x(1-p)^{n-x} / \hat\pi^x(1-\hat\pi)^{n-x}] \sim \chi^2(1)$.
- **Christoffersen Independence**: марковская цепь 2×2 $\to \chi^2(1)$.
- **DQ Engle-Manganelli**: OLS `hit_t` на лаги hit + VaR; статистика Вальда $\chi^2(k)$.
- **Berkowitz**: $z_t = \Phi^{-1}(F_t(r_t))$, LR на $(\mu=0, \sigma=1)$ $\chi^2(2)$.
- **Acerbi-Szekely T1**: $Z_1 = \frac{1}{N_v}\sum \frac{X_t}{ES_t} + 1$ на пробоях.

Светофор Базеля: GREEN ≤ 4 пробоя/250 дней (мн. 3.0), YELLOW 5-9 (3.0→4.0),
RED ≥ 10 (4.0).

### 6.13 Copulas
$C(u_1,\dots,u_n)$ описывает зависимость отдельно от маргиналов (Sklar).
- Gaussian: tail dependence $\equiv 0$.
- Student-t: $\lambda = 2 t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)})$.

### 6.14 LVaR (Bangia exogenous)
$\text{LVaR} = P \cdot \text{VaR}_{\text{price}} + 0.5(\mu_S + a \sigma_S)$,
$a=3$ для крипты.

### 6.15 Market impact (square-root law)
$\text{impact} \approx Y \cdot \sigma \cdot \sqrt{Q/\text{ADV}}$.

### 6.16 Almgren-Chriss
$x_j = X \cdot \sinh(\kappa(T-t_j))/\sinh(\kappa T)$, $\kappa$ растёт с
риск-аверсией $\lambda$.

### 6.17 Worst-case Breuer
$\text{loss}^* = -\mathbf{w}^\top\boldsymbol\mu + k\sigma_p$ при ограничении
махаланобиса $\|r-\mu\|_{\Sigma^{-1}} \le k$.

### 6.18 Vol targeting / Kelly / Drawdown
- Vol target: $L = \min(\sigma^*/\hat\sigma, L_{\max})$.
- Многомерный Келли: $\mathbf{w}^* = \Sigma^{-1}(\boldsymbol\mu - r_f)$. На практике
  $0.25$-$0.5$ Келли.
- Drawdown scale: $m = \max(\text{floor}, 1 - \text{dd}/\text{dd}_{\max})$.

---

## 7. Подводные камни (что НЕ-очевидно)

1. **VaR/ES возвращаются положительными.** `est.var > 0` означает убыток. Если
   увидишь отрицательный VaR в коде вызывающей системы — ошибка.

2. **Кэш данных строго различает синтетику и реальность** через префикс файла
   (`prices_exchange_...` vs `prices_synthetic_...`). Никогда не подсовывает
   синтетику вместо реальных.

3. **`MultiSourceCryptoLoader` НЕ кэширует синтетический фолбэк** под именем
   exchange — чтобы при следующем запуске повторить попытку получить реальные
   данные.

4. **`_to_utc_dates` оборачивает в DatetimeIndex явно**, потому что
   `pd.to_datetime(...)` на Series возвращает Series без метода `.normalize()`.
   Не упрощай эту функцию обратно.

5. **`max_risk` НЕ использует cvxpy** — задача невыпукла. Аналитика по
   вершинам / собственному вектору. Не пытайся «починить» через `Maximize`.

6. **GARCH-fit принимает доходности в долях**, но внутри умножает на 100 (пакет
   `arch` так стабильнее), потом делит обратно. σ и forecast_var возвращаются в
   долях. Если меняешь — проверь оба направления.

7. **При универсуме TOP-10 с 2015 г.** `returns.dropna()` начинается с 2020-04
   (когда появился SOL). Per-asset аналитика через `returns[c].dropna()`.

8. **DCC-GARCH медленный** (30-60 с для 10 активов). Не вызывай его в горячем
   цикле; результат кэшируется в `engine._garch_cache` только для univariate.

9. **t-copula loglik** использует `scipy.special.gammaln`, а не
   `scipy.stats.loggamma` (то — распределение). Это был баг, который исправлен.

10. **`pre_trade_check` возвращает `approved=False` НЕ только при отказе**, но
    и когда требуется урезание (size = 0). Проверяй `sized_notional > 0` отдельно.

11. **Аннуализация = 365**, не 252. Если внешняя система использует 252 (типично
    для акций), нужен перевод: `sigma_year_252 = sigma_year_365 * sqrt(252/365)`.

12. **`ccxt` опционален.** Если не установлен, `CryptoDataLoader` работает в
    `synthetic`. Аналогично `arch` для GARCH, `cvxpy` для constrained
    оптимизации, `sklearn` для shrinkage.

13. **Все «time-aware» индексы — UTC.** Не передавай naive datetime в
    `start`/`end` — может вылезти timezone-warning.

14. **EVT работает с УБЫТКАМИ** (положительные). Внутри функций конвертация
    `L = -r` уже сделана; снаружи передавай обычные доходности.

15. **`risk_attribution(method='gaussian')` точно аддитивна** (Эйлер); для
    `method='historical'` — приближённо аддитивна за счёт нормировки.

---

## 8. Окружение и зависимости

**Python**: 3.12 (проверено), совместимо с 3.10+.

**Установка** (см. `requirements.txt`):
```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Жёсткие зависимости**: `numpy, pandas, scipy, statsmodels, scikit-learn, requests`.
**Желательные**: `arch` (GARCH), `cvxpy` (constrained QP), `ccxt` (Binance).
**Только для ноутбука**: `matplotlib, seaborn, jupyter, nbconvert, ipykernel`.

**Сеть нужна** только при первом сборе данных. После — всё работает с кэша.

**Запуск тестов:**
```bash
.venv/bin/python tests/test_smoke.py    # 17 функций, all-OK ≈ 30 c
```

---

## 9. Где что искать в коде

| Хочешь… | Иди в… |
|---|---|
| понять оркестрацию | `crypto_risk/engine.py` |
| тянуть котировки | `crypto_risk/data/parsers.py` (long history) или `data/sources.py` (Binance) |
| модель волатильности | `crypto_risk/volatility.py` |
| оценку VaR/ES | `crypto_risk/var_es.py` + `crypto_risk/evt.py` |
| бэктест VaR/ES | `crypto_risk/backtesting.py` |
| оптимизатор портфеля | `crypto_risk/portfolio.py` + `crypto_risk/covariance.py` |
| ликвидность / исполнение | `crypto_risk/liquidity.py` |
| копулы / зависимости | `crypto_risk/dependence.py` |
| стресс-тесты | `crypto_risk/stress.py` |
| сайзинг / drawdown / Kelly | `crypto_risk/controls.py` |
| как это всё использовать | `examples/demo.py`, `Crypto_Portfolio_Risk_Management.ipynb` |
| примеры всех вызовов с проверками | `tests/test_smoke.py` |
| полную методологию | `documentation/METHODOLOGY.md` |
| детальный обзор модулей | `documentation/ARCHITECTURE.md` |

---

## 10. Минимальные «smoke»-сценарии для проверки интеграции

После любого изменения / интеграции выполни эти проверки:

```python
# 1. Парсер работает
from crypto_risk.data import MultiSourceCryptoLoader, TOP10_POPULAR
ld = MultiSourceCryptoLoader(TOP10_POPULAR, mode="synthetic")
p = ld.load_close_panel("2015-01-01", "2024-01-01")
assert p.shape == (3288, 10)

# 2. Engine end-to-end
from crypto_risk import RiskEngine, RiskConfig
eng = RiskEngine(RiskConfig(universe=TOP10_POPULAR), data_mode="auto")
eng.load_data_multi("2015-01-01", "2024-01-01", synthetic=True)
w = eng.optimal_weights("max_sharpe", "long_only", cov_method="shrinkage")
assert abs(sum(w.values()) - 1.0) < 1e-3

# 3. Risk report
rep = eng.risk_report(weights=w, use_copula=False)
assert rep.annualized_vol > 0
assert (rep.portfolio_risk["VaR_99"] > 0).all()      # VaR положителен!

# 4. Pre-trade
dec = eng.pre_trade_check("BTC", 1_000_000, "buy", adv_usdt=1e10)
assert dec.approved is True
assert dec.sized_notional > 0

# 5. Backtest VaR
bt = eng.backtest_var(weights=w, method="historical", window=252)
assert 0 < bt["violation_rate"] < 0.05
```

Все 5 должны проходить меньше чем за минуту. Если хотя бы один валится —
интеграция нарушила инвариант системы; ищи где.

---

## 11. Куда расти (что не реализовано)

Уже разобрано, но не интегрировано в production:
- **CAViaR** (Engle-Manganelli) — direct conditional quantile model.
- **Vine copulas** для N > 20.
- **RL-execution** вместо/поверх Almgren-Chriss (theory/Reinforcement Learning).
- **Deep hedging** (theory/Deep Learning: Buehler).
- **GAN-генерация синтетических сценариев** для стресса.
- **Многоэкзогенный фактор-модельный риск** (Fama-French-аналог для крипты).
- **Intraday-данные стакана** для онлайн-LVaR и market making risk.

Если получаешь задачу «добавь X» из этого списка — следуй паттерну: новый
файл `crypto_risk/X.py` + расширение `engine.py` методом + тест в
`tests/test_smoke.py` + раздел в ноутбуке.

---

## 12. Контрольная фраза для быстрого узнавания контекста

Если кратко: **«Это модуль рыночного риска криптопортфеля. RiskEngine —
вход; tonкие модули по областям (VaR/ES, EVT, портфель, ковариация,
ликвидность, копулы, стресс, контроли). VaR/ES всегда положительные. Данные —
с 2015 г. через мультиисточниковый парсер с фолбэком и кэшем. Аннуализация по
365. BTC = market proxy.»**
