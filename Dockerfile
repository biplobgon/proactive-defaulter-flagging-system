# Hugging Face Spaces — Docker + Streamlit
FROM python:3.14-slim

# HF Spaces runs as non-root user (uid 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-spaces.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Create dirs that are gitignored but required at runtime
RUN mkdir -p outputs/models outputs/reports outputs/figures \
             data/processed/feature_cache data/raw \
    && chown -R appuser:appuser outputs data

USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
