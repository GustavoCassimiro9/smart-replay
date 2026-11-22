FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEADLESS=1

# Bibliotecas de sistema exigidas pelo OpenCV, e o ffmpeg para gravar os clipes em H.264
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Baixa o modelo RTMO para dentro da imagem, para o app iniciar rápido e sem internet
COPY config.py ./
COPY camera/ camera/
RUN python -c "from camera.detector import _get_model; _get_model()"

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
