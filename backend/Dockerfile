FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY wheels /wheels
COPY requirements.txt /wheels/

RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt

RUN mkdir -p /app/uploads /app/models

COPY . .

EXPOSE 5000

CMD ["python", "app.py"] 