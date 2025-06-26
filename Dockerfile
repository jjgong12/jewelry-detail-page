FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# 한글 폰트 미리 다운로드
RUN mkdir -p /tmp && \
    wget -O /tmp/NanumMyeongjo.ttf https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf

COPY handler.py .

CMD ["python", "-u", "handler.py"]
