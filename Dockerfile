FROM python:3.10-slim

WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    wget \
    fonts-noto-cjk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Korean font as backup
RUN mkdir -p /tmp && \
    wget -O /tmp/NanumMyeongjo.ttf https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf || true

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
