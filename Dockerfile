FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatlas-base-dev \
    libglib2.0-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt 

COPY src/ /app/src/

CMD ["python"]
