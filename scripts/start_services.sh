#!/bin/bash

# ai 컨테이너만 먼저 시작
if [ "$1" == "ai" ] || [ $# -eq 0 ]; then
  docker-compose up -d ai
  # 리소스 조정 스크립트 실행
  ./scripts/adjust_resources.sh &
  # AI 서비스 헬스체크 대기
  echo "AI 서비스 준비 대기 중..."
  while ! curl -s http://localhost:8000/health | grep -q "true"; do
    sleep 5
  done
  echo "AI 서비스 준비 완료"
fi

# 다른 서비스 시작
if [ $# -eq 0 ]; then
  # 모든 서비스 시작
  docker-compose up -d mysql redis zookeeper kafka selenium-hub chrome backend crawl nginx
elif [ "$1" != "ai" ]; then
  # 특정 서비스만 시작
  docker-compose up -d $@
fi

echo "서비스 시작 완료"