# Project: 보험 비교 AI 프로토타입 (Hybrid RAG)

## 1. 개요 (Overview)

이 프로젝트는 보험 비교 서비스의 데이터를 분석하여 내부 실무자에게 인사이트를 제공하는 프로토타입입니다. **Streamlit(Frontend)**과 **FastAPI(Backend)**로 구성되며, **Docker Compose**를 통해 배포됩니다.

외부 API를 연동하여 플랜 목록과 보험료 데이터를 가져오고, 선택된 플랜의 데이터를 **Hybrid RAG(Pandas Agent + Vector Store)** 방식으로 분석하여 LLM(Gemini 3) 기반의 질의응답 서비스를 제공합니다.

## 2. 기술 스택 및 환경 (Tech Stack)

- **Architecture:** Client-Server (Streamlit ↔ FastAPI)
- **Frontend:** Streamlit (Python 3.10+)
- **Backend:** FastAPI (Python 3.10+)
- **Containerization:** Docker, Docker Compose
- **AI/LLM:** LangChain, Google Gemini 3, ChromaDB (Vector Store), Pandas
- **Ports:**
  - Backend: `8000`
  - Frontend: `8501`

## 3. 프로젝트 구조 (Project Structure)

```
├── frontend/                 # Streamlit 프론트엔드
│   ├── app.py                # 메인 애플리케이션
│   ├── pages/                # 페이지 컴포넌트
│   ├── components/           # 재사용 컴포넌트
│   ├── .streamlit/           # Streamlit 설정
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                  # FastAPI 백엔드
│   ├── app.py                # FastAPI 메인 앱
│   ├── api/                  # API 엔드포인트
│   ├── core/                 # 핵심 비즈니스 로직 (RAG)
│   ├── data/                 # 데이터 처리
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── .env.example
```

## 4. 데이터 소스 (External APIs)

본 서비스는 외부 API를 통해 실시간 데이터를 조회합니다.

### API 1: 플랜 목록 조회 (Auth & Plan List)
- **Role:** JWT 토큰 유효성 검증 및 사용 가능한 '플랜 목록(Plan List)' 조회
- **Method:** `GET`
- **URL:** `https://mmlfcp.ohmymanager.com/api/Auth`
- **Parameters:**
  - `token`: {사용자 입력 JWT 토큰값}
  - `access_path`: Fixed value `MMLFCP_WEB`
- **Expected Output:** 플랜 ID(`plan_id`)와 플랜 이름이 포함된 JSON 리스트.

### API 2: 플랜별 보험료 상세 조회 (Product Premiums)
- **Role:** RAG 분석을 위한 핵심 데이터(보험료, 보장내용) 조회
- **Method:** `GET`
- **URL:** `https://mmlfcp.ohmymanager.com/api/ProductPremiums`
- **Query Parameters:**
  - `plan_id`: (API 1에서 선택된 값)
  - `age`: (사용자 입력, Integer)
  - `gender`: (사용자 입력, 'M' or 'F')
- **Headers:**
  - `Authorization`: `Bearer {사용자 입력 JWT 토큰값}`

## 5. 백엔드 로직 및 RAG 전략 (Backend)

### 5.1. 내부 API 엔드포인트
1. **`POST /fetch-plans`**: API 1을 호출하여 플랜 목록 반환.
2. **`POST /load-data`**: API 2를 호출하여 데이터를 가져온 뒤, Pandas DataFrame 및 VectorDB에 적재.
3. **`POST /chat`**: 사용자의 질문을 Hybrid RAG 에이전트로 처리하여 응답 생성.

### 5.2. Hybrid RAG 아키텍처
API 2의 응답 JSON(`data.json`)을 두 가지 방식으로 처리하여 분석 정확도를 높입니다.
- **Pandas Agent:** `coverage_premiums` 리스트를 DataFrame으로 변환하여 가격 비교, 정렬, 필터링 등 구조적 데이터 분석 수행.
- **Vector Retriever:** `product_insur_premiums` 리스트 내의 `insur_bojang`(보장설명) 텍스트를 ChromaDB에 임베딩하여 약관 및 보장 내용 관련 질문 해결.

## 6. 사용자 인터페이스 (Frontend)

Streamlit을 사용하여 다음과 같은 UI를 구성합니다.

### 6.1. 사이드바 (Navigation & Settings)
1. **메뉴 선택:** '생손보플랜 보험료' 등 기능 선택.
2. **인증 및 설정:**
   - JWT 토큰 입력 (Password 타입).
   - "플랜 조회" 버튼 (API 1 호출).
3. **시뮬레이션 입력:**
   - Plan 선택 (API 1 결과).
   - 나이(Default: 46) 및 성별(M/F) 입력.
   - "데이터 분석 시작" 버튼 (API 2 호출 및 RAG 데이터 로드).

### 6.2. 메인 페이지
- **Chat Interface:** `st.chat_message`를 활용한 대화형 인터페이스.
- **Data View:** 분석 근거가 된 데이터(DataFrame)를 확장형(Expandable) 표로 시각화.

## 7. 설치 및 실행 (Getting Started)

### 7.1. 환경 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 Google Gemini API 키를 입력합니다.

```bash
cp .env.example .env
# .env 파일 편집: GOOGLE_API_KEY=your_actual_key
```

### 7.2. Docker로 실행 (권장)

```bash
# 서비스 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up --build -d
```
- **Frontend:** http://localhost:8501
- **Backend:** http://localhost:8000 (Docs: /docs)

### 7.3. 로컬 개발 환경 실행

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 8. 구현 단계 (Roadmap)

1. **Project Setup:** Dockerfile(Front/Back), docker-compose.yml 작성.
2. **Backend - API Client:** `httpx`를 사용한 외부 API 1, 2 연동 모듈 구현.
3. **Backend - Data Processing:** JSON 파싱 및 Pandas/VectorDB 적재 로직 구현.
4. **Frontend - Sidebar:** JWT 입력 → 플랜 조회 → 조건 입력 → 분석 시작 흐름 구현.
5. **Integration:** 프론트엔드 버튼 클릭 시 백엔드 API 호출 및 데이터 흐름 연결.