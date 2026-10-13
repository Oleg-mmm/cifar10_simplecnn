FROM python:3.10-slim

WORKDIR /app

# Копируем отдельно от кода для кэширования слоев Docker
COPY requirements.txt .
# --no-cache-dir уменьшает размер итогового образа
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]