FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn8-devel-ubuntu20.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# 한글 폰트 다운로드
RUN apt-get update && apt-get install -y wget && \
    mkdir -p /tmp && \
    wget -O /tmp/NanumMyeongjo.ttf https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf

COPY handler.py .
CMD ["python", "-u", "handler.py"]
