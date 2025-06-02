프로젝트 구조를 분석한 결과를 바탕으로 리드미 파일을 작성하겠습니다.

# 📈 뉴스톡(NewStock) - 뉴스 기반 주식 정보 애플리케이션

뉴스톡은 주식 관련 뉴스를 AI로 분석하여 투자자에게 유용한 정보를 제공하는 모바일 애플리케이션입니다. 뉴스 분석을 통해 종목별 호재/악재 점수를 산출하고, 뉴스 요약과 키워드 분석을 제공합니다.

## 🛠️ 기술 스택

### 백엔드
- **프레임워크**: Spring Boot 3.3.9
- **언어**: Java 17
- **데이터베이스**: MySQL 8.0.33
- **캐시**: Redis 7.2.4
- **메시지 큐**: Kafka 7.4.0
- **보안**: Spring Security, JWT
- **문서화**: Swagger (SpringDoc)
- **ORM**: JPA, QueryDSL
- **웹소켓**: Spring WebSocket
- **메일**: Spring Mail, Thymeleaf

### 프론트엔드
- **프레임워크**: React Native (Expo)
- **언어**: TypeScript
- **상태 관리**: Zustand, React Query
- **스타일링**: TailwindCSS, NativeWind
- **차트**: Victory Native, Gifted Charts
- **웹소켓**: StompJS
- **네트워크**: Axios

### AI 서비스
- **프레임워크**: FastAPI
- **언어**: Python
- **모델**: Hugging Face Transformers, KoBERT
- **텍스트 처리**: KoNLPy, SentencePiece
- **분석 기능**: 뉴스 감성 분석, 요약, 키워드 추출

### 크롤링 서비스
- **프레임워크**: Spring Boot
- **웹 크롤링**: Selenium
- **브라우저 자동화**: Selenium Grid

### 인프라
- **컨테이너화**: Docker, Docker Compose
- **웹 서버**: Nginx
- **CI/CD**: GitLab CI

## 📋 주요 기능

### 1. 주식 뉴스 분석
- 종목별 뉴스 수집 및 분석
- AI 기반 호재/악재 점수 산출
- 뉴스 요약 및 키워드 추출

### 2. 종목 정보 조회
- 실시간 주가 정보
- 종목별 뉴스 목록
- 스크랩 기능

### 3. 개인화 서비스
- 관심 종목 등록
- 맞춤형 뉴스레터
- 알림 서비스

### 4. 사용자 관리
- 회원가입/로그인
- 프로필 관리
- 뉴스 스크랩 기능

## 🏗️ 시스템 아키텍처

```
┌─────────────┐    ┌────────────┐    ┌────────────────┐
│  Frontend   │    │   Nginx    │    │     Backend    │
│ React Native│◄───┼───Gateway──┼───►│  Spring Boot   │
└─────────────┘    └────────────┘    └────────────────┘
                                              ▲
                                              │
                                              ▼
                   ┌────────────┐    ┌────────────────┐
                   │    AI      │    │    Crawler     │
                   │  FastAPI   │◄───┼───Spring Boot  │
                   └────────────┘    └────────────────┘
                         ▲                   ▲
                         │                   │
                         ▼                   ▼
                   ┌────────────┐    ┌────────────────┐
                   │   Kafka    │    │  Selenium Grid │
                   └────────────┘    └────────────────┘
                         ▲
                         │
                         ▼
              ┌─────────────────────┐
              │      Database       │
              │  MySQL    Redis     │
              └─────────────────────┘
```

## 📝 프로젝트 구조

```
.
├── .git/
├── .gitlab-ci.yml
├── .gitmessage.txt
├── README.md
├── ai/                 # AI 서비스 (FastAPI)
│   ├── main.py         # FastAPI 애플리케이션
│   ├── models/         # AI 모델
│   ├── utils/          # 유틸리티 함수
│   └── requirements.txt
├── backend/            # 백엔드 서비스 (Spring Boot)
│   ├── src/            # 소스 코드
│   │   └── main/
│   │       ├── java/
│   │       │   └── newstock/
│   │       │       ├── controller/
│   │       │       ├── domain/
│   │       │       ├── common/
│   │       │       └── ...
│   │       └── resources/
│   ├── build.gradle    # Gradle 설정
│   └── Dockerfile      # 백엔드 도커파일
├── crawl/              # 크롤링 서비스
│   ├── src/
│   ├── build.gradle
│   └── Dockerfile
├── docker-compose.yml  # 도커 컴포즈 설정
├── frontend/           # 프론트엔드 (React Native)
│   ├── app/            # 앱 화면
│   ├── api/            # API 클라이언트
│   ├── components/     # 컴포넌트
│   ├── assets/         # 정적 자원
│   └── package.json    # NPM 패키지
└── nginx/              # Nginx 설정
```

## 🚀 설치 및 실행 방법

### 1. AI 서버 실행
```bash
# 가상환경 세팅
cd ai/
python -m venv venv
source venv/Scripts/activate

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload

# 가상환경 비활성화
deactivate
```

### 2. 전체 서비스 실행 (Docker Compose)
```bash
# 환경 변수 설정 (.env 파일 생성)
# MySQL, API 키 등의 설정을 .env 파일에 추가

# 도커 컴포즈 실행
docker-compose up -d
```

## 📊 시스템 요구사항

- Docker 및 Docker Compose
- Java 17 이상
- Python 3.9 이상
- Node.js 16 이상
- 최소 8GB RAM, 20GB 저장공간
