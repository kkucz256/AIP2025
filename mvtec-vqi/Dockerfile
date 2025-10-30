FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860 8000
CMD ["python", "-m", "src.app.ui_gradio", "--config", "configs/default.yaml"]
