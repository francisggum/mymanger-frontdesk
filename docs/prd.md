# Project: Insurance Comparison AI Prototype (Hybrid RAG)

## 1. Project Overview

이 프로젝트는 보험 비교 서비스의 데이터를 분석하여 내부 실무자에게 인사이트를 제공하는 프로토타입이다.
Streamlit(Frontend)과 FastAPI(Backend)로 구성되며, Docker Compose로 배포된다.
**외부 API 2개**를 연동하여 플랜 목록을 가져오고, 선택된 플랜의 보험료 데이터를 분석하여 RAG 기반 질의응답을 수행한다.

## 2. Tech Stack & Environment

- **Architecture:** Client-Server (Streamlit <-> FastAPI)
- **Frontend:** Streamlit (Python)
- **Backend:** FastAPI (Python 3.10+)
- **Containerization:** Docker Compose
- **AI/LLM:** LangChain, gemini3, ChromaDB (Vector Store), Pandas
- **Ports:**
  - Backend: `8000`
  - Frontend: `8501`

## 3. Data Sources (External APIs)

### API 1: 플랜 목록 조회 (Auth & Plan List)

- **Role:** JWT 토큰 유효성 검증 및 사용 가능한 '플랜 목록(Plan List)' 조회
- **Method:** GET
- **URL:** `https://mmlfcp.ohmymanager.com/api/Auth`
- **Parameters:**
  - `token`: {사용자 입력 JWT 토큰값}
  - `access_path`: Fixed value `MMLFCP_WEB`
- **Expected Output:** 플랜 ID(`plan_id`)와 플랜 이름이 포함된 JSON 리스트.

### API 2: 플랜별 보험료 상세 조회 (Product Premiums)

- **Role:** RAG 분석을 위한 핵심 데이터(보험료, 보장내용) 조회
- **Method:** GET
- **URL:** `https://mmlfcp.ohmymanager.com/api/ProductPremiums`
- **Query Parameters:**
  - `plan_id`: (Selected from API 1)
  - `age`: (User Input, Integer)
  - `gender`: (User Input, 'M' or 'F')
- **Headers:**
  - `Authorization`: `Bearer {사용자 입력 JWT 토큰값}` (Note: Check if 'Bearer ' prefix is needed, otherwise pass raw token)

## 4. User Interface (Streamlit)

### 4.1. Navigation (Sidebar)

Streamlit의 `st.sidebar`에 네비게이션과 설정 메뉴를 구성한다.

1. **Menu Selection (Radio/Selectbox):**
   - **Option 1: 생손보플랜 보험료** (현재 구현 대상)
2. **Auth & Settings:**
   - **JWT Token Input:** `st.text_input` (Password type recommended). 1일 유효기간이므로 매번 입력 가능해야 함.
   - **"플랜 조회" 버튼:** 클릭 시 API 1을 호출하여 `plan_id` 목록을 가져옴.
3. **Simulation Inputs (Visible after Plan List loaded):**
   - **Plan Selectbox:** API 1에서 가져온 플랜 이름 선택.
   - **Age Input:** `st.number_input` (Default: 46).
   - **Gender Input:** `st.radio` (Male: 'M', Female: 'F').
   - **"데이터 분석 시작" 버튼:** 클릭 시 API 2를 호출하고 Backend RAG 데이터 로드.

### 4.2. Main Page (생손보플랜 보험료)

- **Chat Interface:** 사용자와 AI의 대화창 (`st.chat_message`).
- **Data View (Expandable):** AI 분석의 근거가 된 `coverage_premiums` 데이터를 표(DataFrame)로 시각화.

## 5. Backend Logic (FastAPI + LangChain)

### 5.1. Endpoints

1. **`POST /fetch-plans`**
   - Input: `{"jwt_token": "..."}`
   - Logic: Call API 1.
   - Output: List of `plan_id` and `plan_name`.
2. **`POST /load-data`**
   - Input: `{"jwt_token": "...", "plan_id": "...", "age": 46, "gender": "M"}`
   - Logic: Call API 2 -> Process JSON -> Update Pandas DataFrame & VectorDB.
3. **`POST /chat`**
   - Input: `{"query": "..."}`
   - Logic: Run Hybrid RAG Agent.

### 5.2. RAG Strategy (Hybrid)

API 2의 응답 JSON(`data.json` 구조 참조)을 두 갈래로 처리한다.

- **Pandas Agent:** `coverage_premiums` 리스트를 DataFrame으로 변환. 가격 비교, 정렬, 필터링 수행.
- **Vector Retriever:** `product_insur_premiums` 리스트 내의 `insur_bojang`(보장설명) 텍스트를 ChromaDB에 임베딩. 약관/보장내용 질문 해결.

## 6. Implementation Steps

1. **Project Setup:** Dockerfile (Front/Back), docker-compose.yml 작성.
2. **Backend - API Client:** `httpx`를 사용하여 API 1, 2 연동 모듈 구현.
3. **Backend - Data Processing:** JSON 파싱 및 Pandas/VectorDB 적재 로직 구현.
4. **Frontend - Sidebar:** JWT 입력 -> 플랜 조회 -> 조건 입력 -> 분석 시작 흐름 구현.
5. **Integration:** 프론트엔드 버튼 클릭 시 백엔드 API 호출 및 데이터 흐름 연결.
