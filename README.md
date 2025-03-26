# 서버 실행 방법

## 1. 가상환경 세팅
```cmd
cd ai/
python -m venv venv
source venv/Scripts/activate
```

## 2. 패키지 설치
```cmd
pip install -r requirements.txt
```

## 3. 서버 실행
```cmd
uvicorn main:app --reload
```

## 4. 가상환경 비활성화
```cmd
deactivate
```

