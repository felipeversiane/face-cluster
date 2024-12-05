FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

COPY src/requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src/ .

COPY ./database /app/database
