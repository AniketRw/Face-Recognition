FROM python:3.10-slim-bullseye

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PERSISTENT_STORAGE=/app/data

RUN apt-get update && apt-get install -y curl gnupg2 unixodbc unixodbc-dev gcc g++ build-essential libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
