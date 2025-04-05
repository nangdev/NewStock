#!/bin/bash
# adjust_resources.sh

# AI 컨테이너 이름
AI_CONTAINER="s12p21a304-ai-1"

# 조정할 리소스 값
AI_AFTER_CPU="1.5"
AI_AFTER_MEM="4G"

# 모델 로드 완료 감지
echo "모델 로드 대기 중..."
docker exec $AI_CONTAINER bash -c "while [ ! -f /tmp/models_loaded ]; do sleep 5; done"
echo "모델 로드 완료 감지. 리소스 조정 중..."

# 리소스 제한 조정
docker update --cpus=$AI_AFTER_CPU $AI_CONTAINER

echo "AI 컨테이너 리소스 제한 업데이트 완료: CPU $AI_AFTER_CPU"