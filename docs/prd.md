# Project: 보험 비교 AI 프로토타입 (Hybrid RAG)

## 1. 개요 (Overview)

이 프로젝트는 보험 비교 서비스의 데이터를 분석하여 내부 실무자에게 인사이트를 제공하는 프로토타입입니다. **Streamlit(Frontend)**과 **FastAPI(Backend)**로 구성되며, **Docker Compose**를 통해 배포됩니다.

외부 API를 연동하여 플랜 목록과 보험료 데이터를 가져오고, 선택된 플랜의 데이터를 **Hybrid RAG(Pandas Agent + Vector Store)** 방식으로 분석하여 LLM(Gemini 3) 기반의 질의응답 서비스를 제공합니다.

## 2. 기술 스택 및 환경 (Tech Stack)

- **Architecture:** Client-Server (Streamlit ↔ FastAPI)
- **Frontend:** Streamlit (Python 3.10+)
- **Backend:** FastAPI (Python 3.10+)
- **Database:** MSSQL Server (pyodbc 연동)
- **Containerization:** Docker, Docker Compose
- **AI/LLM:** LangChain, Google Gemini 3, Pandas Agent
- **API Communication:** Server-Sent Events (SSE) for streaming
- **Additional Libraries:** httpx, requests, python-dotenv
- **Ports:**
  - Backend: `8000`
  - Frontend: `8501`

## 3. 프로젝트 구조 (Project Structure)

```
├── frontend/                 # Streamlit 프론트엔드
│   ├── app.py                # 메인 애플리케이션
│   ├── .streamlit/           # Streamlit 설정
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                  # FastAPI 백엔드
│   ├── main.py               # FastAPI 메인 앱
│   ├── database.py           # MSSQL 데이터베이스 연동
│   ├── rag_system.py         # 하이브리드 RAG 시스템
│   ├── data_manager.py       # 데이터 처리 관리
│   ├── api_client.py         # 외부 API 연동 (사용하지 않음)
│   ├── Dockerfile
│   └── requirements.txt
├── docs/                    # 문서
│   ├── prd.md               # 제품 요구사항 문서
│   ├── db_info.md           # 데이터베이스 구조 명세
│   └── json_info.md         # API JSON 구조 명세
├── docker-compose.yml        # Docker Compose 설정
├── .env.example             # 환경변수 예제
└── AGENTS.md                # 에이전트 개발 가이드
```

## 4. 데이터 소스 (Database)

본 서비스는 MSSQL 데이터베이스에 직접 연동하여 실시간 데이터를 조회합니다.

### 데이터베이스 연동
- **Database:** MSSQL Server
- **Authentication:** SQL Server 인증
- **Connection:** pyodbc 드라이버를 통한 직접 연동
- **Environment Variables:** DB_HOST, DB_NAME, DB_USER, DB_PASSWORD 설정 필요

### 주요 테이블
- **TB_MMLFCP_PLAN**: 플랜 기본 정보 (플랜 ID, 플랜명, 가입 조건 등)
- **TB_TIC_PRDT_PRICE**: 보험료 데이터 (연령대별, 성별별 보험료 정보)
- **TB_MMLFCP_PLAN_COVERAGE**: 보장 정보 (플랜별 보장 항목 및 내용)

### 데이터 조회 방식
1. **플랜 목록 조회**: `TB_MMLFCP_PLAN` 테이블에서 활성화된 플랜 목록 조회
2. **보험료 조회**: `TB_TIC_PRDT_PRICE` 테이블에서 플랜별, 연령별, 성별별 보험료 조회
3. **보장 정보 조회**: `TB_MMLFCP_PLAN_COVERAGE` 테이블에서 플랜별 보장 항목 조회

## 5. 백엔드 로직 및 RAG 전략 (Backend)

### 5.1. 내부 API 엔드포인트
1. **`GET /`**: 서버 상태 확인 엔드포인트.
2. **`POST /fetch-plans`**: 데이터베이스에서 플랜 목록 조회.
3. **`POST /get-comparison-tables`**: 플랜별 보험료 비교표 생성 (사람용 + LLM용 데이터).
4. **`POST /chat-stream`**: 사용자의 질문을 RAG 에이전트로 처리하여 스트리밍 응답 생성 (Server-Sent Events).

### 5.2. RAG 아키텍처
데이터베이스에서 조회된 보험료 및 보장 정보를 LangChain Pandas Agent를 통해 분석합니다.
- **Pandas Agent:** LangChain을 활용하여 구조적 데이터(보험료, 보장항목)를 분석하고 가격 비교, 정렬, 필터링 등의 질의응답 처리.
- **Fallback 모드:** LangChain 실패 시 통계적 분석으로 기본적인 응답 생성.
- **데이터 처리:** 
  - 실시간 데이터베이스 조회 기반 분석
  - 전처리된 데이터를 LLM에 전달하여 보험 비교 및 분석 수행
  - 스트리밍 방식으로 실시간 진행 상태 전송

## 6. 사용자 인터페이스 (Frontend)

Streamlit을 사용하여 다음과 같은 UI를 구성합니다.

### 6.1. 사이드바 (Navigation & Settings)
1. **플랜 설정:**
   - 플랜 목록 자동 로딩 (데이터베이스에서 직접 조회)
   - 플랜 선택 (드롭다운)
   - 가입 조건 분석 (성별, 나이 제한 확인)
   - 동적 나이 입력 (플랜별 제한 범위 내)
2. **데이터 분석:**
   - "데이터 분석 시작" 버튼
   - 분석 진행 상태 표시

### 6.2. 메인 페이지
- **상태 표시:**
  - 현재 분석 중인 플랜 정보
  - 새로고침 및 비교표 버튼
  - 개발 모드 질문 버튼 (환경변수 설정 시)
- **AI 보험 상담사:**
  - 스트리밍 챗 인터페이스 (실시간 진행 상태 표시)
  - `st.chat_message`를 활용한 대화형 인터페이스
  - 에러 처리 및 재시도 로직
- **데이터 뷰:**
  - 분석 근거 데이터를 확장형(Expandable) 표로 시각화
  - LLM용 데이터 확인 옵션

### 6.3. 비교표 모달창
- **보험사별 비교표:** 보장 항목별 상세 비교 표
- **요약 정보:** 플랜별 주요 특징 및 가격 정보
- **분석 가이드:** 데이터 분석 방법 및 해석 가이드

## 7. 설치 및 실행 (Getting Started)

### 7.1. 환경 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 필요한 환경변수를 설정합니다.

```bash
cp .env.example .env
```

`.env` 파일에 필요한 환경변수 설정:
```env
# LLM 설정
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key  # 대안 LLM

# 데이터베이스 연결 정보
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# 개발 환경 설정
DEVELOPMENT_MODE=true
ENABLE_DEV_QUESTIONS=true
```

### 7.2. Docker로 실행 (권장)

```bash
# 서비스 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up --build -d

# 로그 확인
docker-compose logs -f

# 서비스 재시작
docker-compose restart
```
- **Frontend:** http://localhost:8501
- **Backend:** http://localhost:8000 (Docs: /docs)
- **Health Check:** http://localhost:8000/

### 7.3. 로컬 개발 환경 실행

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port=8501
```

### 7.4. 데이터베이스 설정
MSSQL Server에 다음 테이블이 필요합니다:
- `TB_MMLFCP_PLAN` (플랜 정보)
- `TB_TIC_PRDT_PRICE` (보험료 정보)
- `TB_MMLFCP_PLAN_COVERAGE` (보장 정보)

데이터베이스 스키마는 `docs/db_info.md`를 참조하세요.

## 8. 구현 단계 (Roadmap)

1. **Project Setup:** Dockerfile(Front/Back), docker-compose.yml 작성.
2. **Backend - API Client:** `httpx`를 사용한 외부 API 1, 2 연동 모듈 구현.
3. **Backend - Data Processing:** JSON 파싱 및 Pandas/VectorDB 적재 로직 구현.
4. **Frontend - Sidebar:** JWT 입력 → 플랜 조회 → 조건 입력 → 분석 시작 흐름 구현.
5. **Integration:** 프론트엔드 버튼 클릭 시 백엔드 API 호출 및 데이터 흐름 연결.