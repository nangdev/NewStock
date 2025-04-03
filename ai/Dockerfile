# Python 3.11 베이스 이미지
FROM python:3.11-slim

# 시스템 패키지 설치: Java, KoNLPy 의존성 등
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    curl \
    git \
    gcc \
    g++ \
    build-essential \
    fonts-nanum \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 환경변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# KoNLPy가 Java를 메모리 과하게 쓰지 않게 제한
ENV _JAVA_OPTIONS="-Xmx2g"

# pip 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK punkt 리소스 다운로드
RUN python -m nltk.downloader punkt punkt_tab

# 앱 코드 복사
COPY . /app
WORKDIR /app

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
