# 보험 비교 AI 프로토타입 (Hybrid RAG)

## 📋 개요

이 프로젝트는 보험 비교 서비스의 데이터를 분석하여 내부 실무자에게 인사이트를 제공하는 프로토타입입니다. Streamlit(Frontend)과 FastAPI(Backend)로 구성되며, Docker Compose로 배포됩니다.

## 🏗️ 기술 스택

- **Frontend:** Streamlit (Python 3.10+)
- **Backend:** FastAPI (Python 3.10+)
- **AI/LLM:** LangChain, Gemini 3, ChromaDB
- **Containerization:** Docker & Docker Compose
- **Ports:** Backend: 8000, Frontend: 8501

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경변수 파일 복사
cp .env.example .env

# .env 파일에 Google Gemini API 키 입력
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 2. Docker로 실행

```bash
# Docker Compose로 모든 서비스 실행
docker-compose up --build

# 백그라운드에서 실행
docker-compose up --build -d
```

### 3. 접속

- **Frontend (Streamlit):** http://localhost:8501
- **Backend (FastAPI):** http://localhost:8000
- **API 문서:** http://localhost:8000/docs

## 📁 프로젝트 구조

```
├── frontend/                 # Streamlit 프론트엔드
│   ├── app.py              # 메인 애플리케이션
│   ├── pages/              # 페이지 컴포넌트
│   ├── components/         # 재사용 컴포넌트
│   ├── .streamlit/         # Streamlit 설정
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                 # FastAPI 백엔드
│   ├── app.py              # FastAPI 메인 앱
│   ├── api/                # API 엔드포인트
│   ├── core/               # 핵심 비즈니스 로직 (RAG)
│   ├── data/               # 데이터 처리
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── .env.example
```

## 🔌 API 통합

### 외부 API
1. **인증 API:** JWT 토큰 유효성 검증 및 플랜 목록 조회
2. **보험료 API:** 플랜별 보험료 상세 데이터 조회

### 내부 API 엔드포인트
- `POST /fetch-plans` - 플랜 목록 조회
- `POST /load-data` - 보험료 데이터 로드 및 RAG 처리
- `POST /chat-stream` - Hybrid RAG 기반 스트리밍 질의응답

## 🤖 RAG 아키텍처

- **Pandas Agent:** 구조화된 보험료 데이터 분석
- **Vector Retriever:** 보장내용 텍스트 검색 (ChromaDB)
- **LLM:** Gemini 3을 활용한 응답 생성

## 🛠️ 개발

### 로컬 개발
```bash
# 프론트엔드
cd frontend
pip install -r requirements.txt
streamlit run app.py

# 백엔드
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### 테스트
```bash
# Docker 컨테이너 로그 확인
docker-compose logs -f

# 특정 서비스 재시작
docker-compose restart backend
```