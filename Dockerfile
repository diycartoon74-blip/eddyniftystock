FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Render (and most PaaS) inject PORT; bind on all interfaces there.
ENV HOST=0.0.0.0 \
    PORT=10000

EXPOSE 10000

CMD ["python", "run.py"]
