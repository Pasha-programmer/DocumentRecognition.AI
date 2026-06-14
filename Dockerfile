# Dockerfile для контейнеризации программы распознавания глаголицы

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
COPY aiModels/glagolitic_model_full_v1_1.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v1_1_tuned.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_0.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_0_tuned.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_1.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_1_tuned.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_2.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v2_2_tuned.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v3_0.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v3_0_tuned.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v4_0.pth ./aiModels/
COPY aiModels/glagolitic_model_full_v4_0_tuned.pth ./aiModels/
COPY __main__.py .

RUN mkdir -p /app/data

# Точка входа: запуск скрипта распознавания
CMD ["watchmedo", "auto-restart", "--directory=.", "--pattern=*.py", "--recursive", "--signal", "SIGTERM", "python", "__main__.py"]