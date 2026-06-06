### документация проекта

три самодостаточных файла, каждый со своей аудиторией:

| Файл | Аудитория | Что внутри |
|------|-----------|------------|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Инженер/исследователь, изучающий проект | Архитектура, поток данных, модуль-за-модулем (назначение / когда срабатывает / API / теория) |
| [METHODOLOGY.md](METHODOLOGY.md) | Квант-аналитик, ревьюер, защита диплома | Формула → код → обоснование → подводные камни. Сгруппировано по областям: features, vol, labels, CV, models, regime, bandits, optimization, VaR/ES, backtests, execution |
| [AGENT_CONTEXT.md](AGENT_CONTEXT.md) | LLM-агент при хэндоффе | Самодостаточный брифинг: конвенции, полный API, рецепты задач, ликбез, pitfalls, smoke-сценарии |

### с чего начать

- хочу понять как устроено в целом → `PROJECT_OVERVIEW.md` сверху вниз.
- хочу понять почему так, а не иначе → `METHODOLOGY.md` по интересующей теме.
- хочу быстро продолжить работу с проекта → `AGENT_CONTEXT.md`.
- хочу убедиться, что ничего не сломал → `AGENT_CONTEXT.md` раздел 10 (smoke-сценарии).

### дополнительные документы

- [../README.md](../README.md) — корневой README с quick-start.
- [../promts/](../promts/) — детальные промты по каждому модулю (для агентов, реализующих компоненты).
- [../risk_managment_inner/documentation/](../risk_managment_inner/documentation/) — собственная документация пакета `crypto_risk` (ARCHITECTURE, METHODOLOGY, AGENT_CONTEXT).
- [../notebooks/](../notebooks/) — 13 исполняемых ноутбуков с теорией и экспериментами.

### контрольные команды

```bash
# Запустить весь test suite (ожидается 163 passed)
.venv/bin/python -m pytest -q

# Быстрый strategy backtest на синтетике
.venv/bin/python -c "
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from src.backtest import quick_strategy
from src.models import ModelSpec
rng = np.random.default_rng(0); n = 400
prices = pd.DataFrame({
    'BTC': 100*np.exp(np.cumsum(rng.normal(0.001, 0.02, n))),
    'ETH': 100*np.exp(np.cumsum(rng.normal(0.0005, 0.025, n))),
}, index=pd.date_range('2022-01-01', periods=n, freq='D'))
rep = quick_strategy(prices, use_crypto_risk=False,
                     model_specs=(ModelSpec(id='ridge', type='Ridge'),))
print(f'Sharpe={rep.metrics.sharpe:.2f}, Total={rep.metrics.total_return:+.2%}, '
      f'MaxDD={rep.metrics.max_drawdown:.2%}, DSR={rep.deflated_sharpe:.3f}')
"
```
