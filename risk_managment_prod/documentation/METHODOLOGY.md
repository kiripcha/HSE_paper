# METHODOLOGY.md — теория, реализация, обоснование

Документ строится по схеме «формула → код → почему так, а не иначе». Цель —
оправдать каждый выбор и связать его с источниками из `theory/`. Код приведён
сокращённо (без логирования и краевых случаев); полные версии — в указанных
файлах.

---

## A. Доходности и волатильность

### A1. Лог-доходности

**Теория.** Для цены $P_t$:
$$
r_t = \ln \frac{P_t}{P_{t-1}}, \qquad
r_t^{(h)} = \sum_{k=1}^{h} r_{t+k}.
$$
Аддитивность во времени делает их удобными при агрегации, а распределение
ближе к стационарному, чем у простых доходностей.

**Код (`volatility.log_returns`):**
```python
def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
```

**Почему.** Простые доходности нельзя складывать во времени; для риска на
горизонте $h$ это критично. Лог-доходности — стандарт в риск-менеджменте
(McNeil, Frey, Embrechts «Quantitative Risk Management», гл. 3).

---

### A2. EWMA / RiskMetrics

**Теория.** Экспоненциально взвешенная оценка дисперсии:
$$
\sigma_t^2 = \lambda\, \sigma_{t-1}^2 + (1-\lambda)\, r_{t-1}^2.
$$
Стандарт JP Morgan RiskMetrics 1996: $\lambda = 0.94$ для дневных данных.
Это GARCH(1,1) с зафиксированными ω=0 и α+β=1.

**Код (`volatility.ewma_volatility`):**
```python
def ewma_volatility(returns, lam=0.94):
    r = returns.dropna().values
    var = np.empty(len(r)); var[0] = r[0] ** 2
    for t in range(1, len(r)):
        var[t] = lam * var[t-1] + (1 - lam) * r[t-1] ** 2
    return pd.Series(np.sqrt(var), index=returns.dropna().index)
```

**Почему.** Один параметр, рекурсия, мгновенный пересчёт — идеально для
real-time контура HFT. EWMA — это первое приближение к GARCH(1,1), но
быстрое и без подгонки. Используется как fallback при отсутствии `arch`.

---

### A3. GARCH-семейство (GARCH, GJR, EGARCH) + Student-t

**Теория.**
- GARCH(1,1) (Bollerslev 1986):
$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$.
- **GJR-GARCH** (Glosten-Jagannathan-Runkle 1993) добавляет асимметрию:
$\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{r_{t-1}<0})r_{t-1}^2 + \beta \sigma_{t-1}^2$.
Захватывает «leverage effect» — рост σ сильнее на падениях.
- **EGARCH** (Nelson 1991) моделирует $\ln \sigma_t^2$ (всегда положительная σ).
- **Инновации Стьюдента** $\varepsilon_t = z_t \cdot \sigma_t$, $z_t \sim t_\nu$:
учитывают тяжёлые хвосты, которые нормаль не описывает.

Оценка — методом максимального правдоподобия.

**Код (`volatility.GARCHModel`):**
```python
class GARCHModel:
    def __init__(self, vol='GARCH', dist='t', p=1, q=1, o=0):
        self.vol = 'GARCH' if vol.upper() == 'GJR' else vol
        self.o = 1 if vol.upper() == 'GJR' else o
        self.dist = dist; self.p, self.q = p, q
    def fit(self, returns, horizon=1):
        r = returns.dropna() * 100   # arch стабильнее в процентах
        am = arch_model(r, mean='Constant', vol=self.vol,
                        p=self.p, o=self.o, q=self.q, dist=self.dist)
        res = am.fit(disp='off')
        fc = res.forecast(horizon=horizon, reindex=False)
        return GARCHResult(sigma=res.conditional_volatility/100,
                           forecast_var=fc.variance.values.ravel()/100**2,
                           nu=res.params.get('nu'), aic=res.aic, bic=res.bic, ...)
```

**Почему GJR-t.** На крипте:
1. Кластеризация σ → нужен GARCH (см. ACF квадратов в ноутбуке).
2. Падения резче подъёмов → GJR ловит асимметрию (γ обычно > 0).
3. Эксцесс 10-30 → инновации Стьюдента; нормальные занижают хвост.
4. По BIC GJR-t регулярно выигрывает у GARCH-normal на наших данных.

Выбор спецификации формализован в `select_best_garch`: перебор
`{GARCH, GJR, EGARCH} × {normal, t, skewt}`, лучшая по BIC.

---

### A4. Range-оценки волатильности (Parkinson / GK / RS / YZ)

**Теория.** Close-to-close σ использует только одну точку из бара. OHLC-оценки
включают внутридневной размах и в разы эффективнее:

| Оценка | Формула | Чувствительность |
|---|---|---|
| Parkinson (1980) | $\sigma^2 = \frac{1}{4\ln 2}(\ln H/L)^2$ | игнорирует гэп/дрейф |
| Garman-Klass (1980) | $0.5(\ln H/L)^2 - (2\ln 2 - 1)(\ln C/O)^2$ | учитывает open/close |
| Rogers-Satchell (1991) | $(H-C)(H-O) + (L-C)(L-O)$ (в логах) | корректно при дрейфе |
| Yang-Zhang (2000) | смесь overnight + open-close + RS | минимальная дисперсия |

**Код (`volatility.yang_zhang_vol`):**
```python
def yang_zhang_vol(ohlc, window=30):
    o, h, l, c = (np.log(ohlc[x]) for x in ['open','high','low','close'])
    sigma_o2 = (o - c.shift(1)).rolling(window).var()
    sigma_c2 = (c - o).rolling(window).var()
    rs = (h - c)*(h - o) + (l - c)*(l - o)
    sigma_rs2 = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window+1)/(window-1))
    return np.sqrt(sigma_o2 + k*sigma_c2 + (1-k)*sigma_rs2)
```

**Почему.** Точнее σ → точнее VaR и портфельная оптимизация. На практике
range-оценки в 5-14 раз эффективнее close-to-close (Yang & Zhang 2000), а
данных OHLC хватает у любого крипто-провайдера. Используем их как «честный»
дневной прокси реализованной волатильности для HAR.

---

### A5. HAR-RV (Corsi 2009)

**Теория.** Heterogeneous Autoregressive модель реализованной дисперсии:
$$
\text{RV}_{t+1} = c + \beta_d \text{RV}_t + \beta_w \text{RV}_t^{(5)} +
\beta_m \text{RV}_t^{(22)} + \varepsilon_t,
$$
где $\text{RV}^{(5)}$ и $\text{RV}^{(22)}$ — средние RV за неделю и месяц.
Каскад «горизонтов трейдеров» (high-freq vs swing vs позиционные).

**Код (`volatility.har_rv`):**
```python
def har_rv(realized_var):
    rv = realized_var.dropna()
    df = pd.concat([rv.shift(-1), rv, rv.rolling(5).mean(),
                    rv.rolling(22).mean()], axis=1).dropna()
    df.columns = ['y', 'd', 'w', 'm']
    X = np.column_stack([np.ones(len(df)), df['d'], df['w'], df['m']])
    beta, *_ = np.linalg.lstsq(X, df['y'].values, rcond=None)
    fc = beta @ np.array([1, rv.iloc[-1], rv.rolling(5).mean().iloc[-1],
                          rv.rolling(22).mean().iloc[-1]])
    return HARResult(params={'c':beta[0],'beta_d':beta[1],'beta_w':beta[2],
                             'beta_m':beta[3]}, forecast=fc,
                     forecast_vol=np.sqrt(fc), r2=...)
```

**Почему.** Простая OLS-регрессия часто бьёт GARCH по точности 1-шаг прогноза
σ (Corsi 2009, Andersen et al. 2007). На реальных данных BTC у нас β_d ≈ 0.37
доминирует, R² ≈ 0.22 на дневной RV из Garman-Klass — это уровень,
сопоставимый с GARCH без его сложности.

---

## B. VaR и ES

### B1. Соглашение о знаке

Везде:
$$
\text{VaR}_\alpha = -Q_{1-\alpha}(r), \qquad
\text{ES}_\alpha = -\mathbb{E}[r \mid r \le Q_{1-\alpha}(r)],
$$
где $r$ — лог-доходность за период. Обе меры — положительные числа
(убыток в долях капитала). Возвращаются в `RiskEstimate(var, es, method,
horizon, var_alpha, es_alpha, extra)`.

---

### B2. Historical Simulation

**Теория.** Непараметрический квантиль эмпирического распределения. Минимум
допущений, чувствителен к попавшим в окно событиям.

**Код (`var_es.historical_var_es`):**
```python
def historical_var_es(returns, var_alpha=0.99, es_alpha=0.975, horizon=1):
    r = np.asarray(returns)[np.isfinite(returns)]
    if horizon > 1:
        return _scale_estimate(historical_var_es(r,...,horizon=1), horizon)
    q_var = np.quantile(r, 1 - var_alpha)
    q_es  = np.quantile(r, 1 - es_alpha)
    tail  = r[r <= q_es]
    return RiskEstimate(var=-q_var, es=-tail.mean(), method='historical', ...)
```

**Почему.** Базовая модель «без допущений». Слабость — отсутствие реакции на
текущий режим σ (не отличает спокойный март от паникующего; нужна FHS/GARCH-VaR
для этого).

---

### B3. Parametric VaR (Normal / Student-t)

**Теория.**
- Normal: $\text{VaR}_\alpha = -(\mu + \sigma\,z_{1-\alpha})$,
  $\text{ES}_\alpha = -(\mu - \sigma\,\phi(z_{1-\alpha})/(1-\alpha))$.
- Student-t (со стандартизацией к Var=1):
  $\text{ES}_\alpha = -\mu + \sigma \cdot \frac{f_\nu(t_{1-\alpha})}{1-\alpha} \cdot
  \frac{\nu + t_{1-\alpha}^2}{\nu - 1} \cdot \sqrt{(\nu-2)/\nu}$ (Acerbi/McNeil).

**Код (`var_es.parametric_var_es`):**
```python
def parametric_var_es(mu, sigma, var_alpha, es_alpha, horizon=1, dist='normal', nu=None):
    mu_h, sig_h = mu*horizon, sigma*np.sqrt(horizon)
    if dist == 'normal':
        z_v = stats.norm.ppf(1-var_alpha); z_e = stats.norm.ppf(1-es_alpha)
        var = -(mu_h + sig_h*z_v)
        es  = -(mu_h - sig_h*stats.norm.pdf(z_e)/(1-es_alpha))
    elif dist == 't':
        scale = np.sqrt((nu-2)/nu)
        t_v = stats.t.ppf(1-var_alpha, nu)*scale
        var = -(mu_h + sig_h*t_v)
        x = stats.t.ppf(1-es_alpha, nu)
        es_std = (stats.t.pdf(x,nu)/(1-es_alpha)) * ((nu+x**2)/(nu-1)) * scale
        es = -(mu_h - sig_h*es_std)
    return RiskEstimate(...)
```

**Почему.** Быстро, аналитично, ясно. **Нормальная** — нужна как опорная точка,
но для крипты систематически занижает риск (тонкая хвост ≠ реальная толщина).
**Student-t** с ν из GARCH — основной параметрический выбор.

---

### B4. Cornish-Fisher

**Теория.** Поправка квантиля нормали на третий (S) и четвёртый (K) моменты:
$$
z^{CF}_\alpha = z_\alpha + \tfrac{z_\alpha^2-1}{6}S + \tfrac{z_\alpha^3-3z_\alpha}{24}K
- \tfrac{2z_\alpha^3-5z_\alpha}{36}S^2.
$$

**Код (`var_es.cornish_fisher_var_es`):**
```python
def cf_quantile(alpha):
    z = stats.norm.ppf(1-alpha)
    return (z + (z**2-1)*S/6 + (z**3-3*z)*K/24 - (2*z**3-5*z)*S**2/36)
```

**Почему.** Простая поправка для не-нормальности без полной оценки распределения.
Хороший fallback, когда не хочется фитить t/EVT. ES оцениваем средним по
CF-квантилям выше уровня.

---

### B5. EWMA / GARCH / FHS VaR

**Теория.** Параметрический VaR с *условной* σ_t (из EWMA или GARCH-прогноза).
**FHS** (Hull-White 1998) — фильтруем доходности на σ, бутстрэпим
стандартизованные остатки, обратно умножаем на прогноз σ. Получаем
динамический не-параметрический хвост.

**Код (`var_es.fhs_var_es`):**
```python
def fhs_var_es(returns, garch_result, var_alpha=0.99, es_alpha=0.975,
               horizon=1, n_paths=20000):
    z = (returns/garch_result.sigma).dropna().values
    sig_fc = garch_result.forecast_sigma_path()[:horizon]
    draws = rng.choice(z, size=(n_paths, horizon), replace=True)
    sim_h = (draws * sig_fc).sum(axis=1)
    q_var = np.quantile(sim_h, 1-var_alpha)
    q_es  = np.quantile(sim_h, 1-es_alpha)
    return RiskEstimate(var=-q_var, es=-sim_h[sim_h<=q_es].mean(),
                        method='fhs', ...)
```

**Почему FHS.** Сочетает: (1) GARCH-динамику σ (учёт режима) и (2) непараметрические
хвосты (никаких допущений о форме). На крипте — один из лучших методов VaR/ES
по бэктесту (см. ноутбук §7-§10).

---

### B6. EVT — Peaks-Over-Threshold (POT / GPD)

**Теория.** По теореме Пикандса-Балкемы-де Хаана: при $u\to\infty$ распределение
превышений $L - u \mid L > u$ сходится к обобщённому распределению Парето:
$$
G_{\xi,\beta}(y) = 1 - \left(1 + \xi\,y/\beta\right)^{-1/\xi},
\quad y \ge 0.
$$
Оттуда:
$$
\text{VaR}_\alpha = u + \frac{\beta}{\xi}\!\left[\!\left(\frac{n}{N_u}(1-\alpha)\right)^{-\xi}-1\right],
$$
$$
\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\beta - \xi u}{1-\xi},
\qquad \xi < 1.
$$
$\xi > 0$ ⇒ тяжёлый хвост (типично для крипты).

**Код (`evt.pot_var_es`):**
```python
def pot_var_es(returns, var_alpha=0.99, es_alpha=0.975, threshold_q=0.90, method='mle'):
    losses = -np.asarray(returns)
    fit = fit_gpd(losses, threshold_q=threshold_q, method=method)
    xi, beta, u, n, nu = fit.xi, fit.beta, fit.threshold, fit.n_total, fit.n_exceed
    var  = u + (beta/xi) * (((n/nu)*(1-var_alpha))**(-xi) - 1)
    var_e= u + (beta/xi) * (((n/nu)*(1-es_alpha))**(-xi) - 1)
    es   = var_e/(1-xi) + (beta - xi*u)/(1-xi)   if xi < 1 else var_e*1.5
    return RiskEstimate(var, es, 'evt_pot_'+method, ...)
```

**Почему.** Исторический метод плох в *далёком* хвосте (квантиль 99% часто
завышает или занижает на 20-50% в зависимости от выборки). EVT моделирует
именно хвост и даёт устойчивые оценки. Теоретическое обоснование —
универсальный предел, не зависящий от тела распределения (McNeil «EVT for Risk
Managers», Gilli & Kellezi).

**Выбор порога u.** Эмпирический квантиль 90% по умолчанию + диагностика
*mean-excess plot* — линейный рост сигнализирует, что GPD-режим достигнут.

---

### B7. Conditional EVT (McNeil & Frey 2000)

**Теория.** Двухшаговая модель:
1. GARCH описывает динамику σ_t.
2. Стандартизованные остатки $z_t = r_t/\sigma_t$ считаются i.i.d.
3. К хвосту $z$ применяем POT/GPD ⇒ $\text{VaR}_\alpha(z)$.
4. Условный риск: $\text{VaR}_{t+1} = \sigma_{t+1} \cdot \text{VaR}_\alpha(z)$.

**Код (`evt.conditional_evt`):**
```python
def conditional_evt(returns, garch_result, var_alpha=0.99, es_alpha=0.975,
                    threshold_q=0.90):
    z = (returns / garch_result.sigma).dropna().values
    z_est = pot_var_es(z, var_alpha, es_alpha, threshold_q)
    sigma_fc = float(garch_result.forecast_sigma_path()[0])
    return RiskEstimate(var=z_est.var*sigma_fc, es=z_est.es*sigma_fc,
                        method='conditional_evt(GARCH+POT)', ...)
```

**Почему.** «Золотой стандарт» точности хвостового риска: чистая EVT не реагирует
на режим σ; чистый GARCH-t хорош лишь в той мере, в которой ν оценено
правильно (на ограниченной выборке плохо). Conditional EVT решает оба: σ от
GARCH, форма хвоста от POT. McNeil & Frey (2000) на S&P показали лучший Kupiec
из всех конкурентов.

---

### B8. Hill estimator

**Теория.** Оценка индекса хвоста $\alpha = 1/\xi$ по $k$ самым большим значениям:
$$
\hat\alpha = \left( \frac{1}{k} \sum_{i=1}^{k} \ln \frac{X_{(i)}}{X_{(k+1)}} \right)^{-1}.
$$

**Код (`evt.hill_estimator`):**
```python
def hill_estimator(losses, k=None):
    L = np.sort(losses)[::-1]; L = L[L > 0]
    k = k or max(int(0.05*len(L)), 10)
    logs = np.log(L[:k]) - np.log(L[k])
    return 1.0 / logs.mean()
```

**Почему.** Не для оценки VaR, а для **диагностики**. На крипте получаем
$\hat\alpha \approx 3$-$5$: четвёртый момент существует условно, третий — нет.
Это объясняет, почему дисперсия и эксцесс численно «прыгают» от окна к окну.

---

### B9. Bootstrap horizon (вместо √t)

**Теория (Diebold et al. «Scale models»).** Правило $\sqrt{h}$ для VaR корректно
только при i.i.d. нормальности. При тяжёлых хвостах и σ-кластеризации оно
ЗАНИЖАЕТ риск. Корректнее — бутстрэп $h$-дневных сумм; блочный бутстрэп
($\text{block} > 1$) дополнительно сохраняет автокорреляцию.

**Код (`var_es.bootstrap_horizon_var_es`):**
```python
def bootstrap_horizon_var_es(returns, var_alpha, es_alpha, horizon=10,
                             block=1, n_paths=50000):
    r = np.asarray(returns.dropna())
    if block <= 1:
        sim_h = rng.choice(r, size=(n_paths, horizon)).sum(1)
    else:
        sim_h = np.array([np.concatenate([r[s:s+block]
                          for s in rng.integers(0, len(r)-block, n_blocks)])[:horizon].sum()
                          for _ in range(n_paths)])
    q_v, q_e = np.quantile(sim_h, [1-var_alpha, 1-es_alpha])
    return RiskEstimate(-q_v, -sim_h[sim_h<=q_e].mean(), f'bootstrap_h{horizon}', ...)
```

**Почему.** Регуляторы (FRTB) требуют 10-дневного ES. На крипте с эксцессом ≫ 0
√t занижает 10-дневный ES примерно на 20-40% относительно фактического
распределения 10-дневных сумм. Бутстрэп даёт честную оценку.

---

### B10. Component / Marginal / Incremental VaR (Эйлер)

**Теория (Tasche).** Для однородной первой степени меры риска ρ:
$$
\rho(\mathbf{w}) = \sum_i w_i \cdot \frac{\partial \rho}{\partial w_i} \quad (\text{Эйлер}).
$$
Это даёт **аддитивное разложение** риска портфеля на вклады активов:
$$
\rho_i^{\text{comp}} = w_i \cdot \rho_i^{\text{marg}},
\qquad \sum_i \rho_i^{\text{comp}} = \rho(\mathbf{w}).
$$
Для гауссова VaR: $\rho_i^{\text{marg}} = z_\alpha \cdot (\Sigma w)_i / \sigma_p$.
Для ES — заменяем $z_\alpha$ на $\phi(z_\alpha)/(1-\alpha)$.

**Код (`var_es.component_var_es`):**
```python
def component_var_es(weights, returns, var_alpha=0.99, es_alpha=0.975, method='gaussian'):
    if method == 'gaussian':
        cov = returns.cov().values
        sigma_p = np.sqrt(weights @ cov @ weights)
        z_v = -stats.norm.ppf(1-var_alpha)
        z_e = stats.norm.pdf(stats.norm.ppf(1-es_alpha))/(1-es_alpha)
        mvar = z_v*(cov@weights)/sigma_p
        cvar = weights*mvar
        # ...
    return pd.DataFrame({'weight':weights, 'marginal_VaR':mvar,
                         'component_VaR':cvar,
                         'pct_contrib_VaR':cvar/cvar.sum(), ...})
```

**Почему.** Вес ≠ риск. Маленький по весу, но высоковолатильный/хвостовой
актив может быть главным контрибьютором. Аддитивность критична для
**риск-бюджетирования**: «у DOGE доля 5%, но 25% VaR — режем».

---

## C. Бэктестинг

### C1. Kupiec POF (1995)

**Теория.** $H_0$: фактическая частота пробоев = $p = 1-\alpha$. LR-статистика:
$$
LR_{POF} = -2\ln\!\left[ \frac{p^x(1-p)^{n-x}}{\hat\pi^x(1-\hat\pi)^{n-x}} \right]
\sim \chi^2(1).
$$

**Код (`backtesting.kupiec_pof`):**
```python
def kupiec_pof(violations, var_alpha=0.99):
    n, x = len(violations), int(violations.sum())
    p, pi = 1-var_alpha, x/n
    lr = -2 * (x*np.log(p) + (n-x)*np.log(1-p)
               - x*np.log(pi) - (n-x)*np.log(1-pi))
    return BacktestResult('Kupiec POF', lr, 1 - stats.chi2.cdf(lr, df=1), ...)
```

**Почему.** Самый базовый и стандартный тест покрытия. Используется регулятором
(Базель). Слабость — не видит кластеризацию пробоев (для этого Christoffersen).

---

### C2. Christoffersen (1998): independence + conditional coverage

**Теория.** Если модель корректна, пробои должны быть *независимыми*.
Марковская цепь 2×2 над $I_t \in \{0,1\}$:
$$
LR_{ind} = -2 \ln \frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}
{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}
\sim \chi^2(1).
$$
$LR_{CC} = LR_{POF} + LR_{ind} \sim \chi^2(2)$.

**Код (`backtesting.christoffersen`):**
```python
# подсчёт n_00, n_01, n_10, n_11 ...
ll_ind = ((n00+n10)*log(1-pi) + (n01+n11)*log(pi))
ll_dep = (n00*log(1-pi01) + n01*log(pi01) + n10*log(1-pi11) + n11*log(pi11))
lr_ind = -2*(ll_ind - ll_dep)
lr_cc  = lr_pof + lr_ind                    # ~ chi2(2)
```

**Почему.** Кластеризация пробоев — признак, что модель не реагирует на режим
σ (типично для нормально-VaR в стрессе). Christoffersen ловит именно это.

---

### C3. Duration test (Christoffersen-Pelletier 2004 / Haas 2006)

**Теория.** При корректной модели длительности $D_i$ между пробоями
распределены геометрически (дискретно) или экспоненциально (непрерывно).
$H_1$: Weibull с параметром формы $b \ne 1$.

**Код (`backtesting.duration_test`):**
```python
def neg_ll_weibull(params):
    a, b = params
    return -np.sum(np.log(a*b) + (b-1)*np.log(a*d) - (a*d)**b)
res = minimize(neg_ll_weibull, [1/d.mean(), 1.0], method='Nelder-Mead')
lr = -2*(ll_exp - ll_weibull); pval = 1 - stats.chi2.cdf(lr, df=1)
```

**Почему.** Тонкая проверка независимости: если пробои кучкуются (Weibull с b<1),
тест это поймает там, где Christoffersen пропустит (он смотрит только лаг-1).

---

### C4. Engle-Manganelli DQ (2004)

**Теория.** Регрессия:
$\text{hit}_t = I_t - (1-\alpha) = \beta_0 + \sum_k \beta_k \text{hit}_{t-k}
+ \gamma \cdot \text{VaR}_t + u_t$.
При корректной модели все коэффициенты = 0. Статистика Вальда $\sim \chi^2(k)$.

**Код (`backtesting.dq_test`):**
```python
def dq_test(returns, var, var_alpha=0.99, lags=4):
    hit = (returns < -var).astype(float) - (1-var_alpha)
    X = np.column_stack([np.ones(...), *[hit[start-L:T-L] for L in range(1,lags+1)], var[start:]])
    beta = np.linalg.solve(X.T @ X, X.T @ hit[start:])
    dq = beta @ (X.T @ X) @ beta / (p*(1-p))
    return BacktestResult('Engle-Manganelli DQ', dq, 1-stats.chi2.cdf(dq, df=X.shape[1]), ...)
```

**Почему.** Самый мощный из «дешёвых» динамических тестов: ловит и кластеризацию,
и зависимость пробоев от уровня VaR (классическая «ленивая» модель, которая
бьёт планку каждый раз при росте σ).

---

### C5. Berkowitz LR (2001)

**Теория.** Преобразование PIT $u_t = F_t(r_t) \to z_t = \Phi^{-1}(u_t)$. Под
$H_0$: $z \sim N(0, 1)$. Тест LR на $(\mu = 0, \sigma = 1)$ ~ $\chi^2(2)$.

**Код (`backtesting.berkowitz_tail`):**
```python
if sigma is not None:
    z = stats.norm.ppf(np.clip(stats.norm.cdf(r/sigma), 1e-6, 1-1e-6))
else:
    # приблизительный PIT через индикатор пробоя + равномерные примеси
res = minimize(neg_ll_normal, [0,1], method='Nelder-Mead')
lr = -2 * (ll_restr - ll_unrestr); pval = 1 - chi2.cdf(lr, df=2)
```

**Почему.** Использует не только бинарные пробои, а *всё* распределение
прогноза. Сильнее Kupiec, требует прогноз σ.

---

### C6. Acerbi-Szekely ES backtests (Tests 1, 2, 3)

**Теория (Acerbi & Szekely 2014).** Бэктест **ES**, на который VaR-тесты не
смотрят:
- **Test 1** (unconditional на пробои):
  $Z_1 = \frac{1}{N_T} \sum_{t: \text{viol}} \frac{X_t}{ES_t} + 1$,
  под $H_0$ $\mathbb{E}[Z_1]=0$, $Z_1<0$ ⇒ ES занижен.
- **Test 2** (по числу пробоев):
  $Z_2 = \frac{1}{T} \sum_t \frac{X_t I_t}{p \cdot ES_t} + 1$.
- **Test 3** (rank-based, через PIT в хвосте).

**Код (`backtesting.acerbi_szekely_test1`):**
```python
def acerbi_szekely_test1(returns, var, es, es_alpha=0.975, n_boot=5000):
    I = (returns < -var); nv = I.sum()
    Z1 = np.mean((-returns[I]) / es[I]) - 1
    # bootstrap H0: убытки на пробое ~ -ES * Uniform(0.5,1.5)
    z_boot = [np.mean((es[I]*rng.uniform(0.5,1.5,nv))/es[I]) - 1 for _ in range(n_boot)]
    pval = np.mean(z_boot >= Z1)
    return BacktestResult('Acerbi-Szekely ES (Test 1)', Z1, pval, ...)
```

**Почему.** ES — основная регуляторная мера в FRTB (с 2023). Бэктест ES
сложнее VaR (нужна форма хвоста). Acerbi-Szekely — самые цитируемые и
практичные, p-value через бутстрэп нулевого распределения.

---

### C7. Model risk + Basel traffic light

**Теория.** Модельный риск = неопределённость самой модели. Простая мера —
разброс оценок VaR между методами:
$\text{spread} = (\max - \min)/\text{mean}$. Базель определяет три зоны по числу
пробоев на 250-дневном окне:
- **GREEN** ≤ 4, множитель капитала 3.0;
- **YELLOW** 5-9, множитель 3.0 → 4.0 линейно;
- **RED** ≥ 10, множитель 4.0.

**Код (`backtesting.model_risk_metrics`):**
```python
def model_risk_metrics(var_estimates, n_violations, n_obs):
    vals = np.array(list(var_estimates.values()))
    spread = (vals.max() - vals.min()) / vals.mean()
    scaled = n_violations * 250 / n_obs
    if scaled <= 4: mult, zone = 3.0, 'GREEN'
    elif scaled <= 9: mult, zone = 3.0 + 0.2*(scaled-4), 'YELLOW'
    else: mult, zone = 4.0, 'RED'
    return {'relative_spread': spread, 'conservative_var': vals.max(),
            'basel_zone': zone, 'basel_multiplier': mult, ...}
```

**Почему.** Никогда не доверяй одной модели. Разброс между historical /
GARCH-t / EVT / FHS / параметрическим даёт оценку, насколько мы вообще понимаем
риск. Брать `max` — консервативно (Glasserman, theory/Model Risk).

---

## D. Портфель

### D1. Markowitz mean-variance optimization

**Теория.** Задача:
$$
\min_{\mathbf{w}} \mathbf{w}^\top \Sigma \mathbf{w}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \;
\boldsymbol\mu^\top \mathbf{w} \ge \mu^*, \;
\mathbf{w} \in \mathcal{C}.
$$
Граница эффективных портфелей — sweep по $\mu^*$.

**Код (`portfolio.PortfolioOptimizer.min_variance` через cvxpy):**
```python
w = cp.Variable(n)
cons = [cp.sum(w) == 1]
if constraint == 'long_only': cons.append(w >= 0)
elif constraint == 'short_limit': cons.append(w >= -short_limit)
elif constraint == 'min_weight': cons.append(w >= min_w)
if target_return is not None: cons.append(mu @ w >= target_return)
cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), cons).solve()
```

**Почему cvxpy.** Аккуратно решает выпуклый QP с любыми линейными
ограничениями. Без ограничений — closed-form через `pinv(cov)` (быстрее).
4 режима ограничений напрямую покрывают исходное ТЗ.

---

### D2. Beta-based covariance (single-index) + adjusted betas

**Теория.** Рыночная модель: $r_i = \alpha_i + \beta_i r_m + \varepsilon_i$.
Тогда $\Sigma = \boldsymbol\beta \boldsymbol\beta^\top \sigma_m^2
+ \text{diag}(\sigma^2_{\varepsilon_i})$.
Скорректированные беты (Blume): $\beta^{\text{adj}}_i = 0.67\beta_i + 0.33$.

**Код (`portfolio.estimate_betas` + `beta_covariance`):**
```python
b = np.cov(ri, rm)[0,1] / rm.var()
cov = np.outer(b, b) * var_m + np.diag(resid_var)
```

**Почему.** Выборочная Σ при N=10 и T~1000 уже зашумлена; одна-факторная
модель сжимает её до 2N+1 параметров — гораздо устойчивее. Blume-correction
учитывает регрессию β к 1 во времени (эмпирически подтверждено).

---

### D3. Ledoit-Wolf и Constant-correlation shrinkage

**Теория.** $\hat\Sigma = \delta F + (1-\delta) S$, где $S$ — выборочная,
$F$ — структурированная цель (scaled identity или const-corr). Оптимальное
$\delta^*$ минимизирует ожидаемую Frobenius-ошибку (Ledoit & Wolf 2003, 2004).

**Код (`covariance.ledoit_wolf_cov`):**
```python
def ledoit_wolf_cov(returns):
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_, lw.shrinkage_
```

**Код (`covariance.constant_correlation_shrinkage`):**
```python
corr = S / np.outer(std, std)
r_bar = (corr.sum() - n) / (n*(n-1))
F = r_bar * np.outer(std, std); np.fill_diagonal(F, var)
pi_hat = sum_{i,j} mean((X_i*X_j - S_ij)^2)
gamma_hat = sum((F - S)^2)
delta = clip(kappa/T, 0, 1)
cov = delta*F + (1-delta)*S
```

**Почему.** На крипте корреляции близки и почти однородны → const-corr target
часто даёт $\delta \to 1$ (полное сжатие к усреднённой модели), что даёт
самые устойчивые out-of-sample веса. LW-shrinkage снижает число обусловленности
Σ, что напрямую снижает «error maximization» в Markowitz.

---

### D4. RMT denoising (Marchenko-Pastur)

**Теория.** Спектр выборочной корреляции при $q = T/N \to \infty$:
$\lambda \in [(1-\sqrt{1/q})^2, (1+\sqrt{1/q})^2]$ — шумовая зона. Реальные
факторы выходят за верхнюю границу. Очистка: шумовые λ заменяются на их
среднее, диагональ ренормируется.

**Код (`covariance.rmt_denoise_cov`):**
```python
vals, vecs = np.linalg.eigh(corr)
lam_max = (1 + np.sqrt(1/q))**2
vals[vals < lam_max] = vals[vals < lam_max].mean()
corr_clean = vecs @ np.diag(vals) @ vecs.T
corr_clean /= np.outer(d, d)        # ренормировка к 1 на диагонали
```

**Почему.** Аккуратнее LW: явно отделяет «рыночную моду» и крупные факторы от
шума. Стандарт в hedge-fund-индустрии (Bouchaud, Potters).

---

### D5. DCC-GARCH (Engle 2002)

**Теория.** Двухшаговая:
1. Univariate GARCH(1,1) ⇒ $\sigma_{i,t}$, $z_{i,t} = r_{i,t}/\sigma_{i,t}$.
2. $Q_t = (1-a-b) \bar Q + a\, z_{t-1} z_{t-1}^\top + b\, Q_{t-1}$;
   $R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$.

**Код (`covariance.dcc_garch`):**
```python
# шаг 1: univariate GARCH per asset -> sigma и z
# шаг 2: подбор (a, b) по сетке 8x8 по pseudo-LL
for a in grid: for b in grid:
    Q = Qbar.copy(); ll = 0
    for t in range(T):
        R = Q / np.outer(sqrt(diag(Q)), sqrt(diag(Q)))
        ll += log-likelihood R contribution
        Q = (1-a-b)*Qbar + a * z_t @ z_t.T + b * Q
```

**Почему.** На крипте корреляции **взрываются в стрессе** (см. ноутбук §5). DCC
ловит это в одну формулу. Статическая Σ за 2020-2024 сглаживает реальный
профиль риска: в COVID-марте корреляция десятков альткоинов с BTC шла к 1.

---

### D6. Risk parity (ERC) и risk budgeting

**Теория.** Equal Risk Contribution: каждый актив вносит одинаковый вклад в
σ портфеля. Итерация Spinu:
$w_i \leftarrow w_i \sqrt{b_i / RC_i}$, где $RC_i = w_i \cdot (\Sigma w)_i$.

**Код (`portfolio.risk_parity` / `controls.risk_budget_weights`):**
```python
for _ in range(500):
    mrc = cov @ w
    rc  = w * mrc
    w   = w * (b / (rc + 1e-12)) ** 0.5
    w   = w / w.sum()
```

**Почему.** Робастный бенчмарк, не требующий оценки μ (μ оценивается очень
шумно). Часто бьёт оптимальный max-Sharpe out-of-sample.

---

### D7. Two-fund theorem (Black 1972)

**Теория.** Любой портфель на границе minimum-variance (без ограничений) —
линейная комбинация любых двух фронт-портфелей.

**Код (`portfolio.check_two_fund_theorem`):**
```python
p1 = opt._min_variance_analytic(target_return=μ_25)
p2 = opt._min_variance_analytic(target_return=μ_75)
p3 = opt._min_variance_analytic(target_return=r3)
alpha = (r3 - p2.ret) / (p1.ret - p2.ret)
max_diff = max|alpha*p1.w + (1-alpha)*p2.w - p3.w|     # должно быть ~ 1e-15
```

**Почему.** Это не *инструмент*, а *проверка корректности* нашего фронт-вычислителя.
На реальных данных max_diff < 1e-14 — численная истина.

---

### D8. Most-risky portfolio (вершина симплекса)

**Теория.** $\max \mathbf{w}^\top \Sigma \mathbf{w}$ s.t. $\sum w = 1$ — задача
**невыпуклая** (max выпуклой). Максимум на политопе достигается в *вершине*.
- long-only с $\sum w = 1$: вершины — отдельные активы → концентрируемся в
  самом волатильном.
- long-short с $|w|\le \text{cap}$: приближаем направлением старшего
  собственного вектора Σ, проецируем на бокс.

**Код (`portfolio.PortfolioOptimizer.max_risk`):**
```python
if constraint == 'long_only':
    w = np.zeros(n); w[argmax(diag(cov))] = 1.0
else:
    vals, vecs = np.linalg.eigh(cov); v = vecs[:, -1]
    if v.sum() < 0: v = -v
    w = clip(v, -cap, cap); w /= w.sum()
```

**Почему cvxpy НЕ годится.** Maximize(quad_form) нарушает DCP. Аналитическое
решение через вершины математически корректно и не требует солвера.

---

## E. Ликвидность и микроструктура

### E1. Bangia LVaR (1999)

**Теория.** Liquidity-adjusted VaR — добавка к ценовому VaR компоненты спреда:
$$
\text{LVaR} = P_t \cdot \text{VaR}_{\text{price}} + 0.5\,(\mu_S + a\,\sigma_S).
$$
$S$ — относительный спред $(ask-bid)/mid$. Для тяжёлых хвостов крипты $a=3$
(вместо 2.33 при нормальности).

**Код (`liquidity.bangia_lvar`):**
```python
def bangia_lvar(price_var, rel_spread_mean, rel_spread_std, spread_mult=3.0):
    liq = 0.5 * (rel_spread_mean + spread_mult * rel_spread_std)
    return LVaRResult(price_var=price_var, liquidity_cost=liq,
                      lvar=price_var + liq, cost_share=liq/(price_var+liq))
```

**Почему.** «Бумажный» VaR игнорирует, что при ликвидации мы теряем половину
спреда — и спред расширяется в стрессе. Bangia — экзогенная (стакан как данный)
часть; есть и эндогенная (impact крупной заявки), см. E3.

---

### E2. Corwin-Schultz spread estimator

**Теория.** Спред из двух подряд идущих баров HLC:
$$
\beta = (\ln H_t/L_t)^2 + (\ln H_{t-1}/L_{t-1})^2,
\;\;\gamma = (\ln H^*/L^*)^2,\; H^*=\max(H_t,H_{t-1}),\, L^*=\min,
$$
$$
\alpha = \frac{\sqrt{2\beta}-\sqrt\beta}{3-2\sqrt 2} - \sqrt{\gamma/(3-2\sqrt 2)},
\quad S = 2(e^\alpha - 1)/(1 + e^\alpha).
$$

**Код (`liquidity.rel_spread_stats_from_ohlc`):**
```python
hl = log(H/L)**2
beta = hl + hl.shift(1)
gamma = log(rolling_max(H,2)/rolling_min(L,2))**2
alpha = (sqrt(2*beta) - sqrt(beta))/k - sqrt(gamma/k)    # k = 3 - 2*sqrt(2)
spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
```

**Почему.** Когда нет L2/тиков (только OHLCV) — это лучший доступный прокси
спреда. Не требует биржевой ленты, работает на дневных данных. Используется
для прокси $\mu_S$, $\sigma_S$ в Bangia LVaR.

---

### E3. Square-root market impact (Almgren et al. 2005)

**Теория.** Эмпирический закон, подтверждённый на множестве рынков:
$$
\text{impact}(Q) \approx Y \cdot \sigma \cdot \sqrt{Q/\text{ADV}},
$$
где Q — объём заявки, ADV — дневной оборот, σ — дневная σ цены.

**Код (`liquidity.square_root_impact`):**
```python
def square_root_impact(order_size, adv, sigma, y=1.0):
    return y * sigma * np.sqrt(order_size / adv)
```

**Почему.** На крипте подтверждено независимыми работами (Donier, Bouchaud).
Используется в `pre_trade_check` для оценки скрытой стоимости крупной заявки
и в Almgren-Chriss как функция воздействия.

---

### E4. Implementation Shortfall (Perold 1988)

**Теория.** Разница между «бумажной» и фактической стоимостью:
$$
IS = \underbrace{(P_{\text{exec}} - P_{\text{decision}}) Q}_{\text{realized}}
+ \underbrace{(P_{\text{end}} - P_{\text{decision}}) (Q - Q_{\text{exec}})}_{\text{opportunity}}.
$$
Раскладывается на: спред, рыночное воздействие, timing, комиссии.

**Код (`liquidity.implementation_shortfall`):**
```python
is_cost      = sign * (avg_exec - decision_price) * total_qty
spread_cost  = half_spread * total_qty * decision_price
timing       = sign * drift * total_qty * decision_price
impact       = is_cost - spread_cost - timing
fees         = fee_rate * total_qty * avg_exec
return ImplementationShortfall(total_bps=1e4*(is_cost+fees)/notional, ...)
```

**Почему.** Главная метрика качества исполнения у buy-side. Встраивается в
оптимизатор (требование ТЗ №25 проекта 1) как штраф за сделку.

---

### E5. Almgren-Chriss optimal execution (2000)

**Теория.** Оптимальная траектория ликвидации $X$ акций за $T$ при риск-аверсии $\lambda$:
$$
x_j = X \cdot \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)},
\quad \kappa = \frac{\operatorname{arccosh}(\tilde\kappa^2/2+1)}{\tau},
\quad \tilde\kappa^2 = \frac{\lambda \sigma^2 \tau}{\eta - 0.5 \gamma \tau}.
$$
Предельный случай $\kappa T \to 0$ ⇒ TWAP. $E[\text{cost}]$ и $\text{Var}[\text{cost}]$
имеют замкнутые выражения; перебор по λ даёт **границу издержки-риск**.

**Код (`liquidity.AlmgrenChriss.schedule`):**
```python
tau = horizon / n_steps
eta_hat = self.eta - 0.5 * self.gamma * tau
kappa_tilde2 = self.lam * self.sigma**2 * tau / eta_hat
kappa = np.arccosh(kappa_tilde2/2 + 1) / tau
t = arange(N+1) * tau
x = X * sinh(kappa*(T - t)) / sinh(kappa*T)
```

**Почему.** Стандарт оптимального исполнения; даёт количественный ответ
«быстрее или медленнее ликвидировать» как функцию toleranceа к риску. В
HFT-системе — это план исполнения большой заявки, не пускающей цену в импакт.

---

## F. Зависимости

### F1. Эмпирическая хвостовая зависимость

**Теория.** Нижний коэффициент:
$\lambda_L = \lim_{q\to 0} \mathbb{P}(U_2 \le q \mid U_1 \le q)$,
эмпирически: $\hat\lambda_L = \frac{1}{q}\mathbb{P}(U_1\le q, U_2\le q)$.

**Код (`dependence.empirical_tail_dependence`):**
```python
lower = np.mean((u <= q) & (v <= q)) / q
upper = np.mean((u > 1-q) & (v > 1-q)) / q
```

---

### F2. Gaussian copula

**Теория.** $C(u_1,\dots,u_n) = \Phi_R(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_n))$.
**Хвостовая зависимость = 0** для любого $\rho < 1$.

**Код (`dependence.GaussianCopula`):**
```python
corr = np.corrcoef(stats.norm.ppf(np.clip(u, 1e-6, 1-1e-6)), rowvar=False)
```

**Почему.** Бенчмарк, который показывает, *как сильно* реальный риск
недооценивался бы при гауссовом допущении.

---

### F3. Student-t copula + теоретическая λ

**Теория (Demarta & McNeil 2005).** $C(u) = t_{\nu, R}(t^{-1}_\nu(u))$;
хвостовая зависимость пары:
$$
\lambda = 2 t_{\nu+1}\!\left(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\right).
$$
$\nu$ оценивается MLE (pseudo-LL копулы) при заданной R.

**Код (`dependence.StudentTCopula`):**
```python
def fit(returns):
    u = pseudo_observations(returns); u = np.clip(u, 1e-6, 1-1e-6)
    corr = np.corrcoef(stats.norm.ppf(u), rowvar=False)
    res = minimize_scalar(lambda nu: -_t_copula_loglik(u, corr, nu),
                          bounds=(2.5, 60), method='bounded')
    return StudentTCopula(corr=corr, nu=res.x, names=cols)

def lower_tail_dependence_pair(rho):
    return 2 * stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)
```

**Почему.** На реальных данных $\hat\nu \approx 5$ ⇒ заметная хвостовая
зависимость; гауссова занижает её в **ноль**. Для совместного риска
(катастрофы вроде LUNA) разница принципиальна.

**Важно:** в коде использован `scipy.special.gammaln`, **не**
`scipy.stats.loggamma` (то — распределение, а не функция).

---

## G. Стресс-тесты

### G1. Worst-case loss (Breuer)

**Теория.** $\max -\mathbf{w}^\top \mathbf{r}$ s.t. $(\mathbf{r}-\boldsymbol\mu)^\top
\Sigma^{-1}(\mathbf{r}-\boldsymbol\mu) \le k^2$. Замкнутое решение:
$\mathbf{r}^* = \boldsymbol\mu - k\Sigma\mathbf{w}/\sigma_p$,
$\text{loss}^* = -\mathbf{w}^\top\boldsymbol\mu + k\sigma_p$.

**Код (`stress.worst_case_loss`):**
```python
port_vol = np.sqrt(w @ cov @ w)
scenario = mu - k * (cov @ w) / port_vol
loss = -(w @ mu) + k * port_vol
```

**Почему.** Параметрический «худший сценарий» с явной мерой правдоподобия (k —
радиус махаланобиса). При $k=3$ это примерно событие «вероятность 0.1-0.5%».

---

### G2. Reverse stress testing (Breuer)

**Теория.** Зафиксирован target_loss (например, -50% капитала). Найти
*минимально неправдоподобный* сценарий:
$$
k^* = (target\_loss + \mathbf{w}^\top\boldsymbol\mu)/\sigma_p.
$$
Сценарий и вклад каждого актива:
$\mathbf{r}^* = \boldsymbol\mu - k^* \Sigma\mathbf{w}/\sigma_p$,
$\text{contrib}_i = w_i r^*_i$.

**Код (`stress.reverse_stress_test`):**
```python
k = (target_loss + w @ mu) / np.sqrt(w @ cov @ w)
scenario = mu - k * (cov @ w) / np.sqrt(w @ cov @ w)
contrib = w * scenario
prob = 1 - stats.chi2.cdf(k**2, df=n)        # приблизительная вероятность
```

**Почему.** Обычный стресс-тест отвечает «что будет в кризис»; **обратный** —
«что должно случиться, чтобы наступила катастрофа», и показывает, какие активы
её драйвят. На наших данных reverse(-50%) указывает на SOL как главный
драйвер — самый волатильный из топ-10.

---

## H. Адаптивный сайзинг

### H1. Volatility targeting

**Теория.** Плечо $L = \min(\sigma^*/\hat\sigma, L_{\max})$. При росте $\hat\sigma$
автоматически режется экспозиция.

**Код (`controls.vol_target_leverage`):**
```python
def vol_target_leverage(forecast_vol_annual, target_vol_annual=0.20, max_leverage=3.0):
    return min(target_vol_annual / forecast_vol_annual, max_leverage)
```

**Почему.** Простейший адаптивный механизм; в long run даёт более стабильную σ
портфеля и часто более высокий out-of-sample Sharpe (Moskowitz, Asness).

---

### H2. Fractional Kelly

**Теория.** Доля капитала по критерию Келли (непрерывный случай):
$f^* = (\mu - r_f)/\sigma^2$. Многомерный: $\mathbf{w}^* = \Sigma^{-1}(\boldsymbol\mu - r_f)$.

**Код (`controls.kelly_weights`):**
```python
def kelly_weights(mu, cov, rf=0.0, fraction=0.5):
    return fraction * (np.linalg.pinv(cov) @ (mu - rf))
```

**Почему ½-Келли.** Полный Келли оптимален лишь при известных μ, Σ. При
оценочной ошибке (огромной для крипты — μ оценивается с большим стандартным
отклонением) полный Келли часто *разоряет*. ¼-½ Келли — стандартная практика.

---

### H3. Drawdown control

**Теория.** Множитель экспозиции $m(\text{dd}) = 1 - (1-\text{floor})\cdot
\text{dd}/\text{dd\_limit}$, $m \ge \text{floor}$.

**Код (`controls.drawdown_scale`):**
```python
def drawdown_scale(current_dd, dd_limit=0.20, floor=0.0):
    cd = abs(current_dd)
    return floor if cd >= dd_limit else 1 - (1-floor)*cd/dd_limit
```

**Почему.** Регуляризатор, который ловит «черные лебеди» сценарии — если что-то
пошло не так, система плавно сдувает плечо до нуля, не дожидаясь, пока σ
поймает это в EWMA.

---

### H4. Risk budgeting (Spinu)

**Теория.** Веса при заданном векторе риск-бюджета $\mathbf{b}$ ($\sum b_i = 1$):
$w_i \cdot (\Sigma\mathbf{w})_i = b_i \cdot \mathbf{w}^\top \Sigma \mathbf{w}$ для всех $i$.

**Код (`controls.risk_budget_weights`):**
```python
for _ in range(500):
    rc = w * (cov @ w)
    w  = w * (b / (rc + 1e-12)) ** 0.5
    w /= w.sum()
```

**Почему.** Обобщение risk parity (там $b_i = 1/n$). Позволяет задать
неравные риск-бюджеты («хочу, чтобы BTC давал 40% риска, ETH — 25%, остальные
поровну»). Удобно для регуляторных лимитов и стратегического распределения.

---

## I. Дальнейшие шаги (что НЕ реализовано и почему)

Эти методы были в `theory/`, но осознанно оставлены за рамками:

| Метод | Источник | Причина |
|---|---|---|
| Vine copulas | Brechmann-Czado 2013 | избыточно при N=10; t-копула достаточна |
| CAViaR | Engle-Manganelli 2004 | реализован DQ-тест, сам CAViaR-режим — для углубления |
| Deep hedging (Buehler) | theory/Deep Learning | требует ансамбль обученных моделей; хук `VolForecaster.set_dl_model` уже есть |
| RL execution | theory/Reinforcement Learning | замещает Almgren-Chriss в HFT, требует тренинга; вне модуля риска |
| Bayesian VaR (Contreras-Satchell) | theory/VaR&ES | требует MCMC; точечного значения достаточно для production |
| Vasicek/migration stress (Mager) | theory/Stress Testing | кредитный, не рыночный; нерелевантно крипте |

---

## Список источников (theory/)

- McNeil A. «Extreme Value Theory for Risk Managers»
- Gilli M., Këllezi E. «EVT for Tail-Related Risk Measures»
- McNeil A., Frey R. «Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series»
- Corsi F. «A Simple Approximate Long-Memory Model of Realized Volatility» (2009)
- Yang D., Zhang Q. «Drift-Independent Volatility Estimation Based on High, Low, Open, and Close» (2000)
- Acerbi C., Szekely B. «Backtesting Expected Shortfall» (2014)
- Kupiec P. (1995), Christoffersen P. (1998), Engle-Manganelli (2004), Berkowitz (2001)
- Bangia A., Diebold F., Schuermann T., Stroughair J. «Modeling Liquidity Risk» (1999)
- Almgren R., Chriss N. «Optimal Execution of Portfolio Transactions» (2000)
- Perold A. «The Implementation Shortfall» (1988)
- Embrechts P., McNeil A., Straumann D. «Correlation and Dependence in Risk Management»
- Demarta S., McNeil A. «The t Copula and Related Copulas»
- Ledoit O., Wolf M. «Honey, I Shrunk the Sample Covariance Matrix» (2004)
- Engle R. «Dynamic Conditional Correlation» (2002)
- Breuer T., Jandacka M., Rheinberger K., Summer M. «How to Find Plausible, Severe, and Useful Stress Scenarios» (2009)
- Tasche D. «Risk Contributions and Performance Measurement» (1999)
- Glosten, Jagannathan, Runkle (1993); Nelson (1991); Bollerslev (1986)
