### METHODOLOGY.md — Алгоритмы и методы, формула → код

этот документ — справочник по каждому алгоритму в проекте. Для каждого метода:
- формула в LaTeX,
- обоснование выбора,
- код из системы (со ссылкой на файл),
- подводные камни при реализации.

темы сгруппированы по предметным областям, а не по модулям — так удобнее
для рассуждения о теоретическом фундаменте диплома.

### содержание

- [A. Microstructure features](#a-microstructure-features)
- [B. Volatility models](#b-volatility-models)
- [C. Labels for financial ML](#c-labels-for-financial-ml)
- [D. Cross-validation](#d-cross-validation)
- [E. Statistical performance](#e-statistical-performance)
- [F. Predictive models](#f-predictive-models)
- [G. Regime detection](#g-regime-detection)
- [H. Multi-armed bandits](#h-multi-armed-bandits)
- [I. Portfolio optimization](#i-portfolio-optimization)
- [J. VaR / ES (9 methods)](#j-var--es-9-methods)
- [K. Backtests for VaR / ES](#k-backtests-for-var--es)
- [L. Execution algorithms](#l-execution-algorithms)
- [M. Risk controls (sizing)](#m-risk-controls-sizing)
- [N. Fill simulation](#n-fill-simulation)
- [O. Dependence / copulas](#o-dependence--copulas)
- [P. Stress tests](#p-stress-tests)

---

### a. Microstructure features

### a.1 Mid-price

$$p_{mid} = \frac{p_b + p_a}{2}$$

самая простая оценка справедливой цены. Чувствительна к спреду.

```python
# src/market_data/io_schemas.py
def derive_mid(snapshot: BookSnapshot) -> float | None:
    if not snapshot.bids or not snapshot.asks:
        return None
    return (snapshot.bids[0].price + snapshot.asks[0].price) / 2.0
```

### a.2 Stoikov micro-price

$$p_{micro} = p_b \cdot \frac{q_a}{q_b + q_a} + p_a \cdot \frac{q_b}{q_b + q_a}$$

учитывает дисбаланс объёмов. Когда $q_b \gg q_a$, цена сдвигается к ask
(больше покупателей → ожидается рост). Опередительный сигнал к mid в краткосроке.

```python
# src/market_data/io_schemas.py
def derive_micro_price(snapshot: BookSnapshot) -> float | None:
    if not snapshot.bids or not snapshot.asks:
        return None
    bid, ask = snapshot.bids[0], snapshot.asks[0]
    total = bid.quantity + ask.quantity
    if total <= 0:
        return None
    return bid.price * ask.quantity / total + ask.price * bid.quantity / total
```

### a.3 Volume Imbalance

$$\text{Imb}_N = \frac{\sum_{i=1}^N q_b^i - \sum_{i=1}^N q_a^i}{\sum_{i=1}^N q_b^i + \sum_{i=1}^N q_a^i} \in [-1, +1]$$

при $N=1$ — top-of-book imbalance; при $N=5,10$ — глубинный.

```python
# src/market_data/io_schemas.py
def derive_imbalance(snapshot: BookSnapshot, levels: int = 1) -> float | None:
    if not snapshot.bids or not snapshot.asks:
        return None
    bid_v = sum(b.quantity for b in snapshot.bids[:levels])
    ask_v = sum(a.quantity for a in snapshot.asks[:levels])
    total = bid_v + ask_v
    if total <= 0:
        return None
    return (bid_v - ask_v) / total
```

### a.4 Order Flow Imbalance (OFI)

$$\text{OFI}_t = \sum_{k \in W} e_k, \quad
e_k = \begin{cases}
+q_b^k & \text{если } p_b^k > p_b^{k-1} \\
+q_b^k - q_b^{k-1} & \text{если } p_b^k = p_b^{k-1} \\
-q_b^{k-1} & \text{если } p_b^k < p_b^{k-1}
\end{cases} - \text{аналогично для ask}$$

OFI агрегирует давление buy/sell ордеров в окне $W$. Сильный predictor
short-term price движения.

```python
# src/features/_internal/features.py
class OFI:
    def update(self, event, book):
        top = _book_top(book, event)
        if top is None: return
        if self._prev is None: self._prev = top; return
        pb, qb, pa, qa = top
        ppb, pqb, ppa, pqa = self._prev
        e_bid = qb if pb > ppb else (qb - pqb) if pb == ppb else -pqb
        e_ask = qa if pa < ppa else (qa - pqa) if pa == ppa else -pqa
        self._buf.append(e_bid - e_ask)
        self._prev = top
```

подвох: OFI определён только когда есть обновление stанка. На trade-only
событиях не двигается.

---

### b. Volatility models

### b.1 Realized volatility (rolling)

$$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{i=t-W+1}^{t} (r_i - \bar{r})^2}$$

базовая оценка из лог-доходностей в окне $W$.

```python
# src/features/_internal/features.py — RealizedVol
mids = collections.deque(maxlen=window + 1)
def update(self, event, book):
    mid = (pb + pa) / 2
    mids.append(mid)
    rets = [math.log(mids[i+1]/mids[i]) for i in range(len(mids)-1)]
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / max(1, len(rets) - 1)
    self._value = math.sqrt(var)
```

### b.2 EWMA volatility (RiskMetrics)

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2, \quad \lambda = 0.94$$

стандарт JP Morgan RiskMetrics. Реагирует на свежие новости быстрее
равновесной rolling-σ.

```python
# crypto_risk/volatility.py — ewma_volatility
def ewma_volatility(returns, lam=0.94):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[0]**2
    for t in range(1, len(returns)):
        var.iloc[t] = lam * var.iloc[t-1] + (1-lam) * returns.iloc[t-1]**2
    return np.sqrt(var)
```

### b.3 GARCH(1,1)

$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

условная гетероскедастичность. Параметры $(\omega, \alpha, \beta)$ через MLE.
GJR: добавляет $\gamma r_{t-1}^2 \mathbb{1}_{r_{t-1} < 0}$ — leverage effect.

распределение инноваций: Normal / Student-t / skewed-t. Для крипты выбираем
t (тяжёлые хвосты).

```python
# crypto_risk/volatility.py — GARCHModel
import arch
am = arch.arch_model(returns * 100, vol='GARCH', p=1, q=1, o=0, dist='t')
res = am.fit(disp='off')
sigma_in_sample = res.conditional_volatility / 100
forecast_var = res.forecast(horizon=10).variance.values[-1] / 100**2
```

подвох: arch-пакет стабильнее на returns × 100 — потом результат делится
обратно.

### b.4 Range-based estimators

используют OHLC, не только close. Дают меньше шума на той же длине истории.

parkinson:
$$\hat{\sigma}^2 = \frac{1}{4 W \ln 2} \sum_t (\ln H_t / L_t)^2$$

garman-Klass:
$$\hat{\sigma}^2 = \frac{1}{W} \sum_t \left[ 0.5 (\ln H_t/L_t)^2 - (2\ln 2 - 1)(\ln C_t / O_t)^2 \right]$$

yang-Zhang — комбинация open-to-open, close-to-close и Rogers-Satchell.
устойчив к overnight jumps.

```python
# crypto_risk/volatility.py
def parkinson_vol(ohlc):
    log_hl = np.log(ohlc['high'] / ohlc['low'])**2
    return np.sqrt(log_hl.mean() / (4 * np.log(2)))
```

### b.5 HAR-RV (Corsi 2009)

heterogeneous Auto-Regressive model для realized volatility:

$$RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w \overline{RV}_t^{(5)} + \beta_m \overline{RV}_t^{(22)} + \epsilon_{t+1}$$

captures long-memory без полноценной long-memory модели. Линейная регрессия,
сходится быстро.

```python
# crypto_risk/volatility.py — har_rv
X = np.column_stack([rv_daily, rv_5d, rv_22d])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
```

---

### c. Labels for financial ML

### c.1 Triple-barrier method

для события в момент $t_0$ задаём:
- Upper: $p_{t_0} \cdot (1 + \text{pt} \cdot \sigma_{t_0})$
- Lower: $p_{t_0} \cdot (1 - \text{sl} \cdot \sigma_{t_0})$
- Vertical: $t_0 + h$

label:
$$y_i = \begin{cases}
+1 & \text{если upper hit first} \\
-1 & \text{если lower hit first} \\
0 & \text{vertical (timeout)}
\end{cases}$$

ref: López de Prado, *Advances in Financial Machine Learning* (AFML),
chapter 3.

```python
# src/labeling/_internal/triple_barrier.py
for k, i in enumerate(event_indices):
    p0 = prices[i]
    v = vol[i]
    upper = p0 * (1 + cfg.pt * v)
    lower = p0 * (1 - cfg.sl * v)
    end = min(i + cfg.max_holding_steps, n - 1)
    label = 0
    for j in range(i + 1, end + 1):
        if prices[j] >= upper: label = +1; break
        if prices[j] <= lower: label = -1; break
    labels[k] = label
```

подвох: vol должен быть adaptive (EWMA-vol на момент $t_0$), не
константа. Иначе на разных режимах barriers неоптимальны.

### c.2 Sample uniqueness weights

при пересекающихся labels эффективное число независимых наблюдений < N.

$$\tilde{w}_i = \overline{\frac{1}{c_t}} \quad \text{(over event lifetime)}$$

где $c_t$ — concurrent active labels at time $t$.

ref: AFML Chapter 4.

```python
# src/labeling/_internal/triple_barrier.py
for k in range(n):
    overlap = ((t0 <= t1[k]) & (t1 >= t0[k])).sum()
    weights[k] = 1.0 / max(1, overlap)
weights = weights * n / weights.sum()  # нормировка к среднему = 1
```

### c.3 Meta-labeling

двухстадийная модель:
1. Primary: classifies side (-1/+1) или генерирует signal.
2. Meta: бинарно решает брать сделку? с учётом фич primary-модели.

позволяет отдельно оптимизировать precision (от meta) и recall (primary).

ref: AFML Chapter 3 §3.5.

---

### d. Cross-validation

### d.1 Purged k-fold

в стандартном k-fold обучающие сэмплы могут пересекаться по времени с
тестовыми → leakage. Purged k-fold убирает обучающие сэмплы $(t_0^j, t_1^j)$,
которые пересекаются с тестовым окном $[\min t_0^{\text{test}} - h, \max t_1^{\text{test}} + h]$,
где $h$ — embargo.

ref: AFML Chapter 7.

```python
# src/labeling/_internal/cv.py — PurgedKFold
def split(self, dataset):
    n = dataset.n_samples
    order = np.argsort(dataset.t0)
    fold_size = n // self.n_splits
    embargo_ns = int((dataset.t1.max() - dataset.t0.min()) * self.embargo_pct)
    for k in range(self.n_splits):
        test_pos = order[k*fold_size : (k+1)*fold_size]
        purged = ((dataset.t1 >= dataset.t0[test_pos].min() - embargo_ns) &
                  (dataset.t0 <= dataset.t1[test_pos].max() + embargo_ns))
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_pos] = False
        train_mask &= ~purged
        yield np.where(train_mask)[0], test_pos
```

### d.2 Combinatorial Purged CV (CPCV)

из $N$ групп выбираем $k$ как тестовые → $\binom{N}{k}$ backtest paths.
каждый путь — независимая оценка performance. Распределение их Sharpe — честная
оценка стратегии.

ref: AFML Chapter 12 §12.4.

```python
# src/labeling/_internal/cv.py — CombinatorialPurgedCV
for combo in itertools.combinations(range(self.n_splits), self.n_test_splits):
    test_idx = np.concatenate([groups[k] for k in combo])
    ...
```

для $N=6, k=2$ получаем $\binom{6}{2} = 15$ paths.

---

### e. Statistical performance

### e.1 Sharpe ratio (annualized)

$$\text{SR} = \frac{\overline{r}}{\sigma_r} \sqrt{P}$$

где $P$ — periods per year (365 для крипты, 252 для акций).

```python
# src/labeling/_internal/metrics.py
def sharpe_ratio(returns, periods_per_year=252):
    r = np.asarray(returns)
    mean, sd = r.mean(), r.std(ddof=1)
    return float(mean / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
```

### e.2 Probabilistic Sharpe Ratio (PSR)

вероятность того, что истинный Sharpe > benchmark, given finite N:

$$\text{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4}{4}\widehat{SR}^2}}\right)$$

где $\widehat{SR}$ — per-period Sharpe (не annualized), $\gamma_3$ — skew,
$\gamma_4$ — excess kurtosis.

```python
# src/labeling/_internal/metrics.py
def probabilistic_sharpe(sharpe_obs, sharpe_bench, n, skew, kurt):
    denom = 1 - skew*sharpe_obs + (kurt/4)*sharpe_obs**2
    if denom <= 0:
        return float(stats.norm.cdf(np.sign(sharpe_obs - sharpe_bench) * 10))
    z = (sharpe_obs - sharpe_bench) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))
```

подвох: формула требует per-period Sharpe, не annualized. С большой
annualized SR денominator уходит в отрицательное.

### e.3 Deflated Sharpe Ratio (DSR)

PSR с benchmark, учитывающим max over n_trials под H0:

$$SR^*_{\text{deflated}} = \hat{\sigma}_{SR} \cdot \left[(1-\gamma) z_{1-1/N} + \gamma z_{1-1/(N e)}\right]$$

где $\gamma$ — Euler-Mascheroni constant, $\hat{\sigma}_{SR}$ — оценка σ
sharpe среди всех попыток.

```python
# src/labeling/_internal/metrics.py
def deflated_sharpe(sharpe_obs, sharpe_std_estimate, n_trials, n_obs, skew, kurt):
    gamma = np.euler_gamma
    z1 = stats.norm.ppf(1 - 1/max(1, n_trials))
    z2 = stats.norm.ppf(1 - 1/(max(1, n_trials) * np.e))
    sharpe_bench = sharpe_std_estimate * ((1-gamma)*z1 + gamma*z2)
    return probabilistic_sharpe(sharpe_obs, sharpe_bench, n_obs, skew, kurt)
```

DSR — обязательный финальный метрик диплома, защищающий от data-snooping.

---

### f. Predictive models

### f.1 Ridge regression

$$\min_{\boldsymbol\beta} \|y - X\boldsymbol\beta\|^2 + \lambda \|\boldsymbol\beta\|_2^2$$

closed-form: $\hat{\boldsymbol\beta} = (X^\top X + \lambda I)^{-1} X^\top y$.

устойчив на коррелированных фичах (диагональная регуляризация). Без него
наивный OLS взрывается на multicollinearity типичной для фин-данных.

```python
# src/models/_internal/baselines.py — RidgeModel
from sklearn.linear_model import Ridge
self._impl = Ridge(alpha=self.alpha)
self._impl.fit(X, y, sample_weight=sample_weights)
```

### f.2 LightGBM (gradient boosting trees)

leaf-wise growth (не depth-wise), histogram-based splits, поддержка sample
weights и categorical из коробки. Обычно сильный baseline на табличных фичах.

```python
# src/models/_internal/baselines.py — LightGBMModel
import lightgbm as lgb
ds = lgb.Dataset(X, label=y, weight=sample_weights)
self._impl = lgb.train(
    {'objective': 'regression', 'num_leaves': 31, 'learning_rate': 0.05,
     'min_data_in_leaf': 100, 'verbose': -1},
    ds, num_boost_round=200,
)
```

### f.3 MLP с MC-Dropout

bayesian approximation: dropout остаётся активным на inference. K стохастических
проходов → mean = $\mu$, std = uncertainty.

```python
# src/models/_internal/torch_models.py — MLPModel._mc_dropout_sigma
self.net.train()  # включить dropout даже в inference
with torch.no_grad():
    preds = [float(self.net(x)) for _ in range(n_passes)]
self.net.eval()
return float(np.std(preds, ddof=1))
```

подвох: BatchNorm-слои в `train()` режиме портят инференс — используй
только Dropout, или замени BN на LayerNorm.

### f.4 TCN (Temporal Convolutional Network)

causal 1D-свёртки с dilation $2^i$. Receptive field $\sim 2^L$ при $L$
слоях. Параллелизуемо во времени (быстрее, чем LSTM).

```python
# src/models/_internal/torch_models.py — _CausalConv1d
class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))  # asymmetric pad → causal
```

---

### g. Regime detection

### g.1 HMM (Hamilton 1989, online filtering)

hidden state $s_t \in \{1, ..., K\}$ с переходами $A_{ij}$.
observation: $r_t \mid s_t \sim \mathcal{N}(\mu_{s_t}, \sigma_{s_t}^2)$.

online forward filtering:

$$\alpha_t(j) \propto p(r_t \mid s_t = j) \sum_i \alpha_{t-1}(i) A_{ij}$$

параметры $(\mu_j, \sigma_j)$ обновляются медленно через online EM.

```python
# src/regime/_internal/detectors.py — HMMDetector.update
prior = self.belief @ self.trans
lik = np.exp(-0.5 * ((val - self.means) / self.stds)**2) / (self.stds * sqrt(2*pi))
posterior = prior * lik
self.belief = posterior / posterior.sum()
# light online EM: move winning regime's μ/σ toward observed
argmax = np.argmax(self.belief)
self.means[argmax] = (1-lr)*self.means[argmax] + lr*val
```

strictly online: никаких Viterbi smoothing — это lookahead.

### g.2 BOCPD (Adams-MacKay 2007)

run-length $r_t$ — длительность текущего режима.
posterior $P(r_t \mid x_{1:t})$ итеративно через conjugate prior NIG.

hazard function $H(\tau) = h$ (constant). Update:

$$P(r_t = r+1, x_{1:t}) \propto P(r_{t-1} = r, x_{1:t-1}) \cdot \pi^{(r)}(x_t) \cdot (1 - h)$$
$$P(r_t = 0, x_{1:t}) \propto h \sum_r P(r_{t-1} = r, x_{1:t-1}) \cdot \pi^{(r)}(x_t)$$

где $\pi^{(r)}(x_t)$ — predictive likelihood.

```python
# src/regime/_internal/detectors.py — BOCPDDetector.update
log_pred = self._predictive_logpdf(v)         # per-run-length Student-t
pred = np.exp(log_pred - log_pred.max())      # scaled, NOT normalized!
growth = self._run_length_proba * pred * (1 - self.hazard)
change = (self._run_length_proba * pred * self.hazard).sum()
new_rl = np.concatenate(([change], growth))
new_rl = new_rl / new_rl.sum()
```

главный подвох: $P(r_t = 0 \mid x_{1:t}) = H$ всегда, по построению.
сигнал change-point — это падение argmax run-length (`steps_in_regime`),
а не значение `probabilities[0]`.

### g.3 Vol-cluster (online percentile)

простой baseline: классификация на K режимов по percentile rolling-σ.

```python
# src/regime/_internal/detectors.py — VolClusterDetector
vol = np.std(self._returns[-self.window:], ddof=1)
thresholds = np.quantile(self._vols_history, [1/3, 2/3])
regime = np.searchsorted(thresholds, vol)
```

прозрачно, легко интерпретируется, отличный sanity-check для более сложных
моделей.

---

### h. Multi-armed bandits

### h.1 ε-greedy с discount

с вероятностью $\epsilon$ — exploration (random arm), иначе — argmax estimated reward.
discount: $\hat{\mu}_a \leftarrow \gamma \hat{\mu}_a + (1-\gamma) r$.

```python
# src/meta_allocator/_internal/allocators.py — EpsilonGreedyAllocator
if self.rng.random() < self.eps:
    choice = self.arms[self.rng.integers(0, self.n_arms)]
else:
    choice = max(self.values, key=self.values.get)
self.values[arm] = self.discount * self.values[arm] + (1-self.discount) * reward
```

### h.2 UCB1 (Auer 2002)

$$A_t = \arg\max_a \hat{\mu}_a + \sqrt{\frac{2 \ln t}{n_a}}$$

оптимизм в условиях неопределённости. Regret $O(\log T)$.

```python
# src/meta_allocator/_internal/allocators.py — UCB1Allocator
ln_t = math.log(self.t)
scores = {a: self.means[a] + math.sqrt(2*ln_t / self.counts[a]) for a in self.arms}
choice = max(scores, key=scores.get)
```

### h.3 Thompson Sampling (Gaussian)

posterior $\mathcal{N}(\hat{\mu}_a, \hat{\sigma}_a^2)$. Sample $\tilde{\mu}_a$
из каждого posterior, выбрать $\arg\max$.

posterior update (Gaussian-Gaussian conjugate):
$$\text{prec}_{\text{post}} = \text{prec}_{\text{prior}} + 1/\sigma_n^2$$
$$\mu_{\text{post}} = (\text{prec}_{\text{prior}} \mu_{\text{prior}} + r / \sigma_n^2) / \text{prec}_{\text{post}}$$

```python
# src/meta_allocator/_internal/allocators.py — ThompsonSamplingAllocator
samples = {a: rng.normal(self.posterior_mu[a], sqrt(self.posterior_var[a]))
           for a in self.arms}
choice = max(samples, key=samples.get)
```

$$A_t = \arg\max_a \hat{\boldsymbol\theta}_a^\top \mathbf{x}_t + \alpha \sqrt{\mathbf{x}_t^\top A_a^{-1} \mathbf{x}_t}$$

где $A_a = I + \sum X X^\top$, $\hat{\boldsymbol\theta}_a = A_a^{-1} \sum r X$.

в нашем случае контекст $\mathbf{x}_t$ = распределение режимов из модуля 06.
это позволяет бандиту учить отображение режим → лучшая модель.

```python
# src/meta_allocator/_internal/allocators.py — LinUCBAllocator
A_inv = np.linalg.inv(self.A[arm])
theta = A_inv @ self.b[arm]
ucb = x @ theta + self.alpha * sqrt(x @ A_inv @ x)
```

---

### i. Portfolio optimization

### i.1 Markowitz mean-variance

$$\max_w \boldsymbol\mu^\top \mathbf{w} - \frac{\lambda}{2} \mathbf{w}^\top \Sigma \mathbf{w}$$
s.t. $\mathbf{1}^\top \mathbf{w} = 1$, $\mathbf{w} \in \mathcal{C}$.

closed-form unconstrained: $\mathbf{w}^* \propto \Sigma^{-1} \boldsymbol\mu / \lambda$.

```python
# src/portfolio/_internal/optimizers.py — MarkowitzOptimizer
inv_cov = np.linalg.solve(cov_shrunk, np.eye(n))
w_raw = inv_cov @ mu / self.risk_aversion
w = _clip_to_constraints(w_raw, constraints)
```

подвох: наивный Markowitz взрывается из-за шумного $\boldsymbol\mu$ и
ill-conditioned $\Sigma$. Без shrinkage не используется.

### i.2 Ledoit-Wolf shrinkage

$$\hat{\Sigma} = (1-\delta) S + \delta F$$

где $S$ — sample cov, $F$ — целевая структура (масштабированная I), $\delta$
— оптимальный shrinkage intensity из аналитической оценки MSE.

```python
# src/portfolio/_internal/optimizers.py
avg_var = float(np.mean(np.diag(cov)))
target = np.eye(n) * avg_var
cov_s = (1 - self.shrinkage) * cov + self.shrinkage * target
```

полная LW также есть в `crypto_risk/covariance.py` через sklearn.

### i.3 Risk Parity / ERC

$$w_i \cdot (\Sigma \mathbf{w})_i = w_j \cdot (\Sigma \mathbf{w})_j \quad \forall i, j$$

каждый актив вкладывает одинаковый риск в портфель. Итеративный алгоритм:

```python
# src/portfolio/_internal/optimizers.py — RiskParityOptimizer
for _ in range(self.max_iter):
    sigma_w = cov @ w
    rc = w * sigma_w
    target_rc = rc.sum() / n
    grad = rc - target_rc
    w = w - 1e-2 * grad / (np.abs(sigma_w) + 1e-12)
    w = w.clip(1e-6) / w.sum()
```

### i.4 HRP (López de Prado 2016)

hierarchical Risk Parity:
1. Distance: $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$.
2. Hierarchical clustering (single linkage).
3. Quasi-diagonalization: переупорядочить $\Sigma$ по кластерам.
4. Recursive bisection: распределить вес по кластерам пропорционально
   $1/\text{ClusterVar}$.

```python
# src/portfolio/_internal/optimizers.py — HRPOptimizer
dist = np.sqrt(0.5 * (1 - corr))
link = scipy.cluster.hierarchy.linkage(squareform(dist), method='single')
sort_ix = self._quasi_diag(link)
w = self._recursive_bisect(cov, sort_ix)
```

главное достоинство: устойчив к шуму в $\boldsymbol\mu$ и $\Sigma$. Не
требует решения QP. Показано экспериментом в `test_hrp_robustness.py` и в
ноутбуке 08.

---

### j. VaR / ES (9 methods)

все методы в `crypto_risk/var_es.py` и `crypto_risk/evt.py`. Сигнатура:

```python
@dataclass
class RiskEstimate:
    var: float           # ПОЛОЖИТЕЛЬНЫЙ убыток (доли капитала)
    es: float
    method: str
    horizon: int
    var_alpha: float
    es_alpha: float
```

### j.1 Historical

$$\text{VaR}_\alpha = -\text{quantile}_{1-\alpha}(r)$$

эмпирический квантиль. Не делает предположений о распределении.

```python
# crypto_risk/var_es.py — historical_var_es
var = -np.quantile(returns, 1 - var_alpha)
es = -returns[returns <= -var].mean()
```

### j.2 Parametric (Normal / Student-t)

normal:
$$\text{VaR}_\alpha = -(\mu + \sigma z_{1-\alpha})$$
$$\text{ES}_\alpha = -(\mu - \sigma \phi(z_{1-\alpha})/(1-\alpha))$$

student-t (с поправкой Var=1):
$$\text{VaR}_\alpha = -\left[\mu + \sigma \sqrt{(\nu-2)/\nu} \cdot t_{\nu, 1-\alpha}\right]$$

### j.3 GARCH-conditional

параметрический VaR с $\sigma_{t+1}$ из GARCH-модели:

$$\text{VaR}_{t+1} = -\mu - \sigma_{t+1}^{GARCH} \cdot z_{1-\alpha}$$

reagирует на текущую волатильность, в отличие от unconditional.

### j.4 FHS (Filtered Historical Simulation)

1. Стандартизировать прошлые returns: $z_t = r_t / \sigma_t^{GARCH}$.
2. На прогноз $h$ строим $\sigma_{t+1}^{GARCH}$, $\sigma_{t+2}^{GARCH}$ итеративно.
3. Sample $z_{t+1}, ..., z_{t+h}$ из эмпирического распределения $\{z\}$.
4. Get $r_{t+i} = z_{t+i} \cdot \sigma_{t+i}^{GARCH}$. Aggregate over $h$.
5. VaR/ES из эмпирического распределения $\sum r_{t+i}$.

хвосты — историческое распределение (плотные tails), вола — текущая. Лучший
из практических вариантов.

### j.5 EVT POT/GPD

превышения порога $u$:
$$L - u \mid L > u \sim \text{GPD}(\xi, \beta)$$

vaR:
$$\text{VaR}_\alpha = u + \frac{\beta}{\xi}\!\left[\!\left(\frac{n}{N_u}(1-\alpha)\right)^{-\xi} - 1\right]$$

ES (при $\xi < 1$):
$$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\beta - \xi u}{1-\xi}$$

$\xi > 0$ → heavy tail. $\xi \ge 1$ → ES не существует.

```python
# crypto_risk/evt.py — pot_var_es
threshold = np.quantile(losses, threshold_q)
excesses = losses[losses > threshold] - threshold
xi, beta = scipy.stats.genpareto.fit(excesses, floc=0)
n, nu = len(losses), len(excesses)
var = threshold + beta/xi * ((n/nu * (1 - alpha))**(-xi) - 1)
es = var / (1 - xi) + (beta - xi*threshold) / (1 - xi)
```

### j.6 Conditional EVT (McNeil-Frey)

$$\text{VaR}_{t+1} = \sigma_{t+1}^{GARCH} \cdot \text{VaR}_\alpha(z)$$

где $\text{VaR}_\alpha(z)$ — POT-квантиль стандартизованных остатков GARCH.

объединяет волатильность (GARCH) и хвост (EVT). Считается state-of-the-art для
финансовых VaR/ES.

### j.7 Monte Carlo

sample $r \sim \mathcal{F}$ (любое распределение) → эмпирический квантиль.

---

### k. Backtests for VaR / ES

### k.1 Kupiec POF (Proportion of Failures)

h0: violation rate = $p = 1 - \alpha$.

$$LR = -2\ln\frac{p^x (1-p)^{n-x}}{\hat{\pi}^x (1-\hat{\pi})^{n-x}} \sim \chi^2(1)$$

```python
# crypto_risk/backtesting.py & src/risk/_internal/var.py — kupiec_pof_test
pi_hat = violations / n
lr = -2 * (np.log((1-p)**(n-violations) * p**violations) -
           np.log((1-pi_hat)**(n-violations) * pi_hat**violations))
pval = 1 - stats.chi2.cdf(lr, df=1)
```

### k.2 Christoffersen Independence

h0: violations происходят независимо. 2×2 marko-chain. $\chi^2(1)$.

### k.3 Dynamic Quantile (DQ) — Engle-Manganelli

OLS `hit_t = 1{r_t < -VaR_t}` на лаги hit + VaR. Wald-statistic $\chi^2(k)$.

### k.4 Berkowitz LR

$z_t = \Phi^{-1}(F_t(r_t))$ под H0 должны быть iid $\mathcal{N}(0,1)$.
LR-test на $(\mu = 0, \sigma = 1)$ → $\chi^2(2)$.

### k.5 Acerbi-Szekely ES tests

test 1 (требует и VaR и ES):
$$Z_1 = \frac{1}{N_v} \sum_{t: r_t < -\text{VaR}_t} \frac{r_t}{\text{ES}_t} + 1$$

под H0 $E[Z_1] = 0$. Bootstrap для p-value.

### k.6 Basel traffic light

| Violations / 250 days | Zone | Multiplier |
|---|---|---|
| ≤ 4 | GREEN | 3.0 |
| 5–9 | YELLOW | 3.0 → 4.0 |
| ≥ 10 | RED | 4.0 |

```python
# crypto_risk/backtesting.py — model_risk_metrics
def basel_zone(violations):
    if violations <= 4: return "GREEN", 3.0
    if violations <= 9: return "YELLOW", 3.0 + 0.2 * (violations - 4)
    return "RED", 4.0
```

---

### l. Execution algorithms

### l.1 TWAP

равные слайсы во времени: $v_t = Q / N$.

```python
# src/execution/_internal/agents.py — TWAPAgent
qty = abs(inventory_remaining) / self.total_slices
return ExecutionAction(order_type='limit', side=..., quantity=qty)
```

### l.2 VWAP

слайсы пропорционально историческому volume profile (U-shape):
$v_t \propto V_t^{\text{hist}}$.

минимизация $E[\text{cost}] + \lambda \text{Var}[\text{cost}]$.

discrete optimal schedule:

$$x_j = X \cdot \frac{\sinh(\kappa(T - t_j))}{\sinh(\kappa T)}$$

где $\kappa = \text{arccosh}\!\left(1 + \frac{\lambda \sigma^2}{2 \eta}\right)$.

```python
# src/execution/_internal/agents.py — AlmgrenChrissAgent
kappa = math.acosh(1 + (sigma**2 * lam) / (2 * eta))
qty = X * math.sinh(kappa) / math.sinh(kappa * steps_left)
```

---

### m. Risk controls (sizing)

### m.1 Volatility targeting

$$L_t = \min\!\left(\frac{\sigma^*}{\hat{\sigma}_t}, L_{\max}\right)$$

leverage обратно пропорционально текущей вола. В low-vol режимах увеличивает
размер, в high-vol — уменьшает. Стабилизирует Sharpe.

```python
# crypto_risk/controls.py — vol_target_leverage
return min(target_vol / max(forecast_vol, 1e-6), max_leverage)
```

### m.2 Fractional Kelly

$$\mathbf{w}^* = \Sigma^{-1}(\boldsymbol\mu - r_f), \quad f^* = \text{Kelly fraction}$$

в практике $0.25$-$0.5$ Kelly — полный Kelly близок к ruin при ошибках в
$\boldsymbol\mu$.

```python
# crypto_risk/controls.py — kelly_weights
w_kelly = np.linalg.solve(cov, mu)
return fraction * w_kelly  # обычно fraction=0.5
```

### m.3 Drawdown scaling

$$m = \max(\text{floor}, 1 - \text{dd}_t / \text{dd}_{\max})$$

размер позиции масштабируется вниз с ростом drawdown. Эмулирует поведение
risk-averse фундменеджера.

```python
# crypto_risk/controls.py — drawdown_scale
def drawdown_scale(current_dd, dd_limit=0.20, floor=0.0):
    if current_dd <= 0: return 1.0
    return max(floor, 1.0 - current_dd / dd_limit)
```

---

### n. Fill simulation

### n.1 Square-root market impact

$$\frac{\Delta P}{P} \approx c \cdot \sigma \cdot \sqrt{\frac{Q}{V}}$$

где $Q$ — order size, $V$ — ADV (average daily volume), $\sigma$ — daily vol.
$c$ обычно 0.3-0.6 для крипты.

```python
# src/backtest/_internal/fill_sim.py
def square_root_impact_bps(qty, ref_volume, c=0.5):
    return c * 100.0 * math.sqrt(qty / ref_volume)   # bps
```

### n.2 Queue-aware fill simulator

- Market order: walks the book until full, avg price = volume-weighted.
- Limit order crossing: fill at best opposite.
- Limit order non-crossing: assume no fill in this tick.

```python
# src/backtest/_internal/fill_sim.py — QueueAwareFillSimulator
def _fill_market(self, order, book, ts_ns):
    levels = book.asks if order.side == 'buy' else book.bids
    remaining = order.quantity
    total_cost, total_qty = 0.0, 0.0
    for lvl in levels:
        consume = min(remaining, lvl.quantity)
        total_cost += consume * lvl.price
        total_qty += consume
        remaining -= consume
        if remaining <= 0: break
    avg = total_cost / total_qty
    bps = square_root_impact_bps(total_qty, sum(l.quantity for l in levels[:10]), self.slippage_c)
    adj = avg * bps / 10_000 * (1 if order.side == 'buy' else -1)
    return Fill(ts_ns=ts_ns + self.latency_ns, ..., price=avg + adj)
```

---

### o. Dependence / copulas

sklar's theorem: любое многомерное распределение разлагается на маргиналы и
copula:

$$F(x_1, ..., x_n) = C(F_1(x_1), ..., F_n(x_n))$$

### o.1 Gaussian copula

$$C(u_1, ..., u_n) = \Phi_n(\Phi^{-1}(u_1), ..., \Phi^{-1}(u_n); \mathbf{R})$$

tail dependence $\lambda_U = \lambda_L = 0$ — не моделирует совместные крайности.

### o.2 Student-t copula

$$C(u_1, ..., u_n) = T_{n,\nu}(t_\nu^{-1}(u_1), ..., t_\nu^{-1}(u_n); \mathbf{R})$$

tail dependence:
$$\lambda_L = \lambda_U = 2 \cdot T_{\nu+1}\!\left(-\sqrt{(\nu+1)\frac{1-\rho}{1+\rho}}\right)$$

ловит все падает вместе — критично для риск-аналитики крипты.

```python
# crypto_risk/dependence.py — StudentTCopula.fit / .sample
u = stats.rankdata(returns, axis=0) / (len(returns) + 1)
z = stats.t.ppf(u, df=nu)
R = np.corrcoef(z.T)
# sampling: G ~ chi2(nu), Z ~ N(0, R), Y = sqrt(nu/G) * Z, U = t.cdf(Y, df=nu)
```

---

### p. Stress tests

### p.1 Historical scenarios

просто прогоняем прошлые плохие дни через текущий портфель:
$$L^{\text{stress}} = -\mathbf{w}^\top \mathbf{r}^{\text{worst-day}}$$

### p.2 Named crashes

bookmarked dates: 2018 BTC crash, COVID-2020, TerraLuna May-2022, FTX
nov-2022. Простой бенчмарк как стратегия пережила бы X.

### p.3 Worst-case (Breuer)

$$\text{loss}^* = -\mathbf{w}^\top \boldsymbol\mu + k \sigma_p$$

s.t. Mahalanobis-constraint $\|\mathbf{r} - \boldsymbol\mu\|_{\Sigma^{-1}} \le k$.

получаем худший правдоподобный исход в пределах $k$ ст. отклонений.

```python
# crypto_risk/stress.py — worst_case_loss
port_vol = np.sqrt(w @ cov @ w)
worst_loss = -(w @ mu) + k * port_vol
```

### p.4 Reverse stress

какие изменения в активах приведут к loss $L^*$? Решаем обратную задачу:

$$\min \|\Delta \mathbf{r}\|_{\Sigma^{-1}}^2 \text{ s.t. } -\mathbf{w}^\top (\boldsymbol\mu + \Delta \mathbf{r}) = L^*$$

closed-form через Lagrange.

```python
# crypto_risk/stress.py — reverse_stress_test
lam = (target_loss - (-w @ mu)) / (w @ inv_cov @ w)
delta_r = -lam * inv_cov @ w
```

---

| Алгоритм | Файл |
|----------|------|
| Micro-price, imbalance | [src/market_data/io_schemas.py](../src/market_data/io_schemas.py) |
| OFI, RSI, MACD, all features | [src/features/_internal/features.py](../src/features/_internal/features.py) |
| Triple-barrier, purged CV, deflated SR | [src/labeling/_internal/](../src/labeling/_internal/) |
| Ridge, LGBM, MLP, TCN | [src/models/_internal/](../src/models/_internal/) |
| HMM, BOCPD, VolCluster | [src/regime/_internal/detectors.py](../src/regime/_internal/detectors.py) |
| ε-greedy, UCB, TS, LinUCB | [src/meta_allocator/_internal/allocators.py](../src/meta_allocator/_internal/allocators.py) |
| Markowitz, HRP, RP, LW | [src/portfolio/_internal/optimizers.py](../src/portfolio/_internal/optimizers.py) |
| Almgren-Chriss, TWAP | [src/execution/_internal/agents.py](../src/execution/_internal/agents.py) |
| VaR/ES (9 methods) | [crypto_risk/var_es.py](../crypto_risk/var_es.py) + [crypto_risk/evt.py](../crypto_risk/evt.py) |
| VaR backtests | [crypto_risk/backtesting.py](../crypto_risk/backtesting.py) |
| GARCH, EWMA, HAR-RV | [crypto_risk/volatility.py](../crypto_risk/volatility.py) |
| Copulas | [crypto_risk/dependence.py](../crypto_risk/dependence.py) |
| Stress tests | [crypto_risk/stress.py](../crypto_risk/stress.py) |
| Fill simulator | [src/backtest/_internal/fill_sim.py](../src/backtest/_internal/fill_sim.py) |
| End-to-end strategy | [src/backtest/strategy.py](../src/backtest/strategy.py) |

---

