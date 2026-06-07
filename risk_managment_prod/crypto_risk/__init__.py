"""
crypto_risk — модуль риск-менеджмента высокоскоростной торговой системы для
управления и оптимизации криптовалютного портфеля на основе глубокого обучения
и адаптивных стратегий.

Подмодули:
    config       — параметры и конфигурация (RiskConfig);
    data         — сбор крипто-данных (ccxt + синтетический фолбэк);
    volatility   — EWMA, GARCH-семейство;
    var_es       — VaR/ES (исторический, параметрический, GARCH, FHS, Монте-Карло);
    backtesting  — валидация моделей VaR/ES (Kupiec, Christoffersen, DQ, ES-тесты);
    portfolio    — оптимизация Марковица, беты, риск-паритет, граница эффект. портфелей;
    liquidity    — LVaR, market impact, Implementation Shortfall, Almgren-Chriss;
    dependence   — копулы (Gaussian/Student-t), хвостовая зависимость;
    controls     — адаптивные риск-контроли (vol targeting, Kelly, drawdown);
    engine       — RiskEngine (оркестратор всего пайплайна).
"""
from .config import RiskConfig
from .engine import RiskEngine

__version__ = "1.0.0"
__all__ = ["RiskEngine", "RiskConfig"]
