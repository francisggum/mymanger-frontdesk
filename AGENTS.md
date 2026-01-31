# AGENTS.md - 에이전트 개발 가이드

이 파일은 이 보험 비교 AI 프로젝트에서 작업하는 코딩 에이전트를 위한 가이드라인을 포함하고 있습니다.

## 커뮤니케이션 및 언어 가이드라인

*   **주요 언어:** 모든 커뮤니케이션, 코드 설명, 주석 및 문서는 **반드시 한국어**로 작성해야 합니다.
*   **UI/UX 텍스트:** 사용자에게 보여지는 모든 UI 텍스트는 한국어를 사용합니다.
*   **변수/함수명:** 코드는 영어로 작성하되, 의미를 명확히 해야 합니다.

## 핵심 참조 문서 (필독)

개발 및 수정 작업을 진행하기 전에 다음 문서들을 반드시 확인하여 기준을 잡으십시오.

*   **`/docs/prd.md` (제품 요구사항 문서):** 프로젝트의 목표, 핵심 기능 명세, 사용자 플로우가 정의되어 있습니다. 비즈니스 로직을 구현하거나 UI를 설계할 때 이 문서를 기준으로 삼으십시오.
*   **`/docs/json_info.md` (데이터 구조 명세):** API에서 사용되는 JSON 데이터 구조와 필드에 대한 설명이 포함되어 있습니다. 데이터 파싱, Pydantic 모델링, 데이터 전처리 로직 작성 시 이 문서를 참조하십시오.

## 프로젝트 개요

이 프로젝트는 다음을 사용하는 Python 기반의 보험 비교 프로토타입입니다.
- **프론트엔드:** Streamlit (포트 8501)
- **백엔드:** FastAPI (포트 8000) 
- **AI 스택:** LangChain, Gemini 3, 하이브리드 RAG를 위한 ChromaDB
- **배포:** Docker Compose

## 빌드 및 개발 명령어

### 로컬 개발
```bash
# 백엔드 개발
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프론트엔드 개발  
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501

# 도커(Docker) 개발
docker-compose up --build
docker-compose up --build -d  # 백그라운드 실행
docker-compose logs -f         # 로그 확인
docker-compose restart backend # 특정 서비스 재시작
```

### 테스트
현재 공식적인 테스트 스위트(Test Suite)는 없습니다. 다음을 통해 수동 테스트를 진행합니다:
- 프론트엔드: http://localhost:8501
- 백엔드 API 문서: http://localhost:8000/docs
- 헬스 체크: http://localhost:8000

### 환경 설정
```bash
# 환경변수 템플릿 복사
cp .env.example .env
# .env 파일을 수정하여 GOOGLE_API_KEY 추가
```

## 코드 스타일 가이드라인

### Python 컨벤션
- **Python 버전:** 3.10 이상
- **스타일:** PEP 8 준수
- **줄 길이:** 최대 88-100자
- **임포트(Imports):** 그룹화하여 임포트 (표준 라이브러리, 서드파티, 로컬 순)
- **타입 힌트:** 일관되게 사용

### 임포트 스타일 예시
```python
# 표준 라이브러리 임포트
import os
import logging
from typing import List, Dict, Any, Optional

# 서드파티 임포트
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 로컬 임포트
from api_client import api_client
from data_manager import data_manager
```

### 네이밍 컨벤션
- **변수:** snake_case (예: `jwt_token`, `plan_id`)
- **함수:** snake_case (예: `fetch_plans`, `process_premium_data`)
- **클래스:** PascalCase (예: `ExternalAPIClient`, `DataManager`)
- **상수:** UPPER_SNAKE_CASE (예: `GOOGLE_API_KEY`)
- **파일:** snake_case (예: `api_client.py`, `rag_system.py`)

### 에러 처리
- API 호출 및 데이터 처리 시 try-except 블록 사용
- `logger.error()`를 사용하여 적절한 컨텍스트와 함께 에러 로깅
- API 엔드포인트에서는 의미 있는 메시지와 함께 HTTPException 발생
- 구조화된 에러 응답 반환

```python
try:
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
except httpx.HTTPStatusError as e:
    logger.error(f"HTTP error fetching plans: {e}")
    raise Exception(f"Failed to fetch plans: HTTP {e.response.status_code}")
```

### API 디자인 패턴
- 요청/응답 검증에 Pydantic 모델 사용 (필드 정의 시 `/docs/json_info.md` 참조)
- 모든 HTTP 클라이언트 작업에 Async/await 사용
- 상태(status)와 데이터(data)가 포함된 구조화된 응답 포맷
- 개발 환경에서는 모든 출처(origins)에 대해 CORS 활성화

### 클래스 구조
- 가능한 경우 상속보다는 합성(Composition) 사용
- 모듈 레벨의 전역 인스턴스 사용 (예: `api_client = ExternalAPIClient()`)
- `__init__`에서 환경변수 설정을 통해 컴포넌트 초기화
- 서비스에 대한 의존성 주입 패턴 사용

### 로깅 (Logging)
- 모듈 레벨 로거 사용: `logger = logging.getLogger(__name__)`
- 성공적인 작업은 info 레벨로 로깅
- 예상치 못했지만 치명적이지 않은 상황은 warning 레벨
- 실패 및 예외 세부 정보는 error 레벨

### 환경 구성
- 환경 변수 관리에 python-dotenv 사용
- 필수: Gemini/LangChain을 위한 `GOOGLE_API_KEY`
- 선택: 다양한 타임아웃 및 설정 값
- `.env` 파일은 절대 커밋하지 말 것

### 데이터 처리
- 구조화된 데이터 작업에는 pandas 사용
- 처리 전 데이터 구조 검증 (`/docs/json_info.md` 기준)
- 비어있거나(Empty) None인 데이터는 우아하게(gracefully) 처리
- 처리된 결과에 대한 메타데이터 반환

### AI/LLM 통합
- 일관성을 위해 LangChain 추상화 사용
- 보험 설명 텍스트 저장을 위해 ChromaDB 사용
- 하이브리드 접근: 구조화된 데이터는 pandas 에이전트 + 텍스트는 벡터 검색
- 일관된 응답을 위해 Temperature는 0.1로 설정

## 아키텍처 가이드라인

### 백엔드 구조
- `main.py`: FastAPI 앱 설정 및 엔드포인트
- `api_client.py`: 외부 API 연동
- `data_manager.py`: 데이터 처리 및 세션 관리  
- `rag_system.py`: 하이브리드 RAG 구현

### 프론트엔드 구조
- `app.py`: 메인 Streamlit 애플리케이션
- 사용자 상호작용을 위한 세션 상태(Session state) 관리
- 설정 입력을 위한 사이드바
- AI 상호작용을 위한 채팅 인터페이스

### API 통합
- 외부 API를 통한 JWT 토큰 인증
- 두 가지 주요 외부 엔드포인트: Auth(인증) 및 ProductPremiums(상품보험료)
- 프론트엔드-백엔드 통신을 위한 내부 API
- 네트워크 실패에 대한 에러 처리

## 개발 참고사항

- **이 프로젝트는 한국어 주석과 UI 텍스트를 사용합니다.**
- 보험 도메인 용어는 한국어로 되어 있습니다.
- 기존 테스트 스위트가 없으므로 수동 테스트가 필요합니다.
- 서버 재시작 시 벡터 저장소(Vector stores)는 초기화됩니다.
- 세션 데이터는 메모리에만 저장됩니다.
- 개발 편의를 위해 CORS가 활성화되어 있습니다.

## 공통 작업

새로운 API 엔드포인트 추가 시:
1. 요청/응답을 위한 Pydantic 모델 추가 (`json_info.md` 참조)
2. 에러 처리가 포함된 비동기(async) 핸들러 구현
3. 적절한 로깅 및 검증 추가
4. /docs의 FastAPI 문서를 통해 테스트

RAG 시스템 수정 시:
1. 벡터 저장소 초기화 테스트
2. 문서 청킹(chunking)이 작동하는지 확인
3. 다양한 쿼리 유형으로 하이브리드 채팅 테스트
4. 토큰 사용량 및 API 비용 모니터링

프론트엔드 업데이트 시:
1. Streamlit 컴포넌트가 올바르게 작동하는지 테스트
2. 세션 상태 관리 확인
3. API 실패에 대한 에러 처리 테스트
4. 한국어 텍스트가 깨지지 않고 제대로 표시되는지 확인