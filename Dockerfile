# Dockerfile для контейнеризации программы распознавания глаголицы
# Используется Python 3.9 на основе slim-образа для минимизации размера

FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt watchdog

# Копируем скрипт и обученную модель
COPY src/. ./src/
COPY ocr/. ./ocr/
COPY database/. ./database/
COPY rabbit_mq/. ./rabbit_mq/
COPY glagolitic_model_full.pth .
COPY glagolitic_model_full_v2.pth .
COPY glagolitic_model_full_v3.pth .
COPY __main__.py .

RUN mkdir -p /app/data

# Точка входа: запуск скрипта распознавания
CMD ["watchmedo auto-restart --directory=. --pattern="*.py" --recursive --signal SIGTERM python __main__.py"]