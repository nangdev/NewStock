#!/bin/bash
# scripts/start_and_adjust.sh

# 모델 로드 시작
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
APP_PID=$!

# 모델 로드 완료 대기
while true; do
  sleep 10
  if curl -s http://localhost:8000/health | grep -q "true"; then
    echo "Models loaded successfully"
    
    # Docker 외부에서 리소스 제한 조정 (컨테이너 ID 필요)
    # 이 부분은 컨테이너 외부에서 실행해야 하므로, 신호를 보내는 방식 사용
    touch /tmp/models_loaded
    
    break
  fi
done

# 프로세스가 종료되지 않도록 대기
wait $APP_PID