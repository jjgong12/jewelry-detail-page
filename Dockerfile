# Multi-stage build 사용
FROM python:3.9-slim as builder

# Requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 최종 이미지
FROM python:3.9-slim

# 빌더에서 설치된 패키지 복사
COPY --from=builder /root/.local /root/.local

# 환경 변수 설정 (중요!)
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /

# 모든 파일 복사
COPY . .

# RunPod handler 실행
CMD ["python", "-u", "handler.py"]
