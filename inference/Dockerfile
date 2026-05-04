FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", "main:app"]
