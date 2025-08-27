FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY whisper_asr.py .
COPY app.py .
COPY utils/ ./utils/
EXPOSE 7860
CMD ["python", "app.py"]