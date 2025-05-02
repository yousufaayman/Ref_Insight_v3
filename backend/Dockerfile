FROM python:3.9-slim as builder

WORKDIR /wheels

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
    numpy>=1.19.0 \
    matplotlib>=3.3.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.50.0 \
    opencv-python>=4.5.0 \
    tensorboard>=2.5.0 \
    seaborn>=0.11.0 \
    Pillow>=8.0.0 \
    flask>=2.0.0 \
    flask-cors>=3.0.10 \
    werkzeug>=2.0.0

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch>=1.9.0 torchvision>=0.10.0

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir --find-links=/wheels \
    numpy>=1.19.0 \
    matplotlib>=3.3.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.50.0 \
    opencv-python>=4.5.0 \
    tensorboard>=2.5.0 \
    seaborn>=0.11.0 \
    Pillow>=8.0.0 \
    flask>=2.0.0 \
    flask-cors>=3.0.10 \
    werkzeug>=2.0.0

RUN mkdir -p /app/uploads /app/models

COPY . .

EXPOSE 5000

CMD ["python", "app.py"] 