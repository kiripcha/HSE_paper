# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------- #
# Образ для запуска модуля управления рисками крипто-портфеля.
#
# Содержимое:
#   * Python 3.12 + полный научный стек (numpy/pandas/scipy/statsmodels/sklearn,
#     arch, ccxt, cvxpy, matplotlib, seaborn, jupyter, nbconvert).
#   * пакет crypto_risk + ноутбуки + дымовые тесты + примеры + документация.
#   * кэш реальных котировок Binance с 2015 по 2025 год (CSV).
#
# Сборка:
#   docker build -t crypto-risk:latest .
#
# Запуск (Jupyter, по умолчанию порт 8888):
#   docker run --rm -p 8888:8888 -v $(pwd)/data_cache:/app/data_cache crypto-risk
#
# Запуск дымовых тестов:
#   docker run --rm crypto-risk python tests/test_smoke.py
#
# Запуск демонстрации:
#   docker run --rm crypto-risk python examples/demo.py --synthetic
# ---------------------------------------------------------------------------- #

FROM python:3.12-slim AS base

# Системные зависимости, необходимые для сборки cvxpy/scipy/arch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
        ca-certificates \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь
ARG USER=risk
ARG UID=1000
RUN useradd --create-home --uid ${UID} ${USER}

WORKDIR /app

# Сначала ставим зависимости — отдельным слоем для быстрого пересбора
COPY --chown=${USER}:${USER} requirements-lock.txt requirements.txt ./

# Используем requirements-lock.txt (зафиксированные версии) для воспроизводимости.
# Если файл блокировки отсутствует, fallback на requirements.txt.
RUN pip install --no-cache-dir --upgrade pip wheel && \
    if [ -f requirements-lock.txt ]; then \
        pip install --no-cache-dir -r requirements-lock.txt ; \
    else \
        pip install --no-cache-dir -r requirements.txt ; \
    fi

# Копируем код проекта
COPY --chown=${USER}:${USER} crypto_risk/ ./crypto_risk/
COPY --chown=${USER}:${USER} notebooks/ ./notebooks/
COPY --chown=${USER}:${USER} tests/ ./tests/
COPY --chown=${USER}:${USER} examples/ ./examples/
COPY --chown=${USER}:${USER} documentation/ ./documentation/
COPY --chown=${USER}:${USER} data_cache/ ./data_cache/
COPY --chown=${USER}:${USER} README.md build_notebook.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    JUPYTER_DATA_DIR=/home/${USER}/.jupyter \
    MPLBACKEND=Agg

USER ${USER}

EXPOSE 8888

# По умолчанию запускаем Jupyter Lab без токена (предполагается изолированный контур).
# Для безопасного запуска снаружи токен следует задать через --NotebookApp.token=...
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/app"]
