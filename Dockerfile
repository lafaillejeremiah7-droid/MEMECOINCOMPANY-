FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY memescanner/ ./memescanner/
COPY scripts/signal_state.py ./scripts/signal_state.py
RUN useradd --uid 10001 --create-home scanner && mkdir /data && chown scanner:scanner /data /app
USER scanner
ENV PYTHONUNBUFFERED=1 MEMESCANNER_DATABASE_PATH=/data/memescanner.db MEMESCANNER_ENABLE_PAPER_TRADING=false
STOPSIGNAL SIGINT
CMD ["sh", "-c", "python scripts/signal_state.py preflight && exec python -m memescanner"]
