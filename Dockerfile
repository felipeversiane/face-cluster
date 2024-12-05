FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY src/ /app/src/
RUN ls -la /app/src/
COPY ./database /app/database
RUN ls -la /app/database

CMD ["python"]
