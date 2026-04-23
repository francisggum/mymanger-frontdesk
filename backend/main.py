import asyncio
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from database import db_manager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import rag_system

logger = logging.getLogger(__name__)
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# 로깅 레벨 설정 (더 상세한 로그를 위해 INFO로 설정)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


load_dotenv()

app = FastAPI(title="Insurance Comparison AI API", version="1.0.0")

# 2. GZip 압축 추가 (CORS 위에 두셔도 되고 아래에 두셔도 무방합니다)
# 100KB(1024 * 100 bytes) 이상의 응답만 압축하여 CPU 자원을 효율적으로 사용합니다.
app.add_middleware(GZipMiddleware, minimum_size=1024 * 100)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadDataRequest(BaseModel):
    plan_id: str = "000000011011"  # 기본값 설정
    age: int = 46  # 기본값 설정
    gender: str = "M"  # 기본값 설정 (남성)


class PremiumAgeRequest(BaseModel):
    """
    연령별 보험료 조회 요청 모델

    Attributes:
        plan_id: 플랜 ID
        company_cd: 보험사 코드
        gender: 성별 (M/F)
        age: 시작 나이
    """

    plan_id: str = "000000011011"
    company_cd: str = "DB"
    gender: str = "M"
    age: int = 46


class ChatRequest(BaseModel):
    query: str
    llm_data: Optional[Dict[str, Any]] = None
    human_data: Optional[str] = None  # JSON 문자열 (orient='table' 형식)
    model: Optional[str] = None  # "gemini" 또는 "openai", None이면 기본값 사용
    plan_name: Optional[str] = None  # 플랜명
    gender: Optional[str] = None  # 성별 (남성/여성)
    age: Optional[int] = None  # 나이


class CoverageMapping(BaseModel):
    coverage_cd: str
    coverage_name: Optional[str] = None


class CoverageInsurMappingRecord(BaseModel):
    coverage_cd: str
    coverage_name: str
    insur_cd: str
    guide_insur_amount: float
    use_yn: str


class CoverageInsurMappingResponse(BaseModel):
    status: str
    data: List[CoverageInsurMappingRecord]


class PlanCoverage(BaseModel):
    plan_id: str
    coverage_cd: str
    guide_coverage_amount: Optional[float] = None
    is_selected_coverage: Optional[str] = None
    coverage_seq: Optional[int] = None


class PlanInfo(BaseModel):
    plan_id: str
    plan_name: str
    plan_category: Optional[str] = ""  # 플랜 카테고리
    plan_type: Optional[str] = ""
    plan_type_name: Optional[str] = ""
    insu_compy_type: Optional[str] = ""  # 보험사 유형 코드
    insu_compy_type_name: Optional[str] = ""  # 보험사 유형명
    refund_payment_type: Optional[str] = ""  # 환급금 지급 유형
    simplified_underwriting_yn: Optional[str] = ""  # 간편심사 여부
    renewal_yn: Optional[str] = ""  # 갱신 여부
    notice_type: Optional[str] = ""  # 고지 유형
    payment_due_type: Optional[str] = ""  # 납입 기간 유형 코드
    payment_due_type_name: Optional[str] = ""  # 납입 기간 유형명
    min_m_age: Optional[int] = 0  # 남성 최소 연령
    max_m_age: Optional[int] = 0  # 남성 최대 연령
    min_f_age: Optional[int] = 0  # 여성 최소 연령
    max_f_age: Optional[int] = 0  # 여성 최대 연령


class ProductInfo(BaseModel):
    """
    상품 정보 응답 모델

    Attributes:
        plan_id: 플랜 ID
        company_code: 보험사 코드
        product_code: 상품 코드
        company_nm: 보험사명
    """

    plan_id: str
    company_code: str
    product_code: str
    company_nm: Optional[str] = ""


@app.get("/")
async def root():
    return {"message": "Insurance Comparison AI API is running"}


@app.post("/fetch-plans", response_model=List[PlanInfo])
async def fetch_plans():
    """
    플랜 목록 조회 API

    데이터베이스에 저장된 보험 플랜 목록을 조회하여 반환합니다.
    이 엔드포인트는 프론트엔드에서 플랜 선택 UI를 구성할 때 사용됩니다.

    Returns:
        List[PlanInfo]: 보험 플랜 정보 목록
        - 각 플랜에는 plan_id, plan_name, plan_type 등의 기본 정보 포함

    Raises:
        HTTPException: 데이터 조회 중 오류 발생 시 500 에러 반환
    """
    try:
        plans = db_manager.fetch_plans()
        return [PlanInfo(**plan) for plan in plans]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-products", response_model=List[ProductInfo])
async def fetch_products():
    """
    상품 목록 조회 API

    TB_MMLFCP_PLAN_PRODUCT 테이블에서 사용 가능한 상품 목록을 조회하여 반환합니다.
    각 상품에는 플랜 ID, 보험사 코드, 상품 코드, 보험사명이 포함됩니다.

    Returns:
        List[ProductInfo]: 상품 정보 목록

    Raises:
        HTTPException: 데이터 조회 중 오류 발생 시 500 에러 반환
    """
    try:
        products = db_manager.fetch_products()
        return [ProductInfo(**product) for product in products]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-plan-standard-coverages", response_model=List[PlanCoverage])
async def get_plan_standard_coverages():
    """
    플랜별 표준 보장 항목 조회
    """
    try:
        coverages = db_manager.fetch_plan_standard_coverages()
        return [PlanCoverage(**cov) for cov in coverages]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-coverage-mapping", response_model=List[CoverageMapping])
async def get_coverage_mapping():
    """
    보장코드와 보장명 매핑 정보 조회
    """
    try:
        mappings = db_manager.fetch_coverage_mapping()
        return [CoverageMapping(**m) for m in mappings]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-coverage-insur-mapping", response_model=CoverageInsurMappingResponse)
async def get_coverage_insur_mapping():
    """
    보장코드-담보코드 매핑 정보 조회

    Returns:
        CoverageInsurMappingResponse: 보장코드-담보코드 매핑 정보
            - status: 처리 상태 ("success" 또는 "error")
            - data: 매핑 데이터 리스트
                - coverage_cd: 보장 코드
                - coverage_name: 보장명
                - insur_cd: 담보 코드
                - guide_insur_amount: 가이드 보험금액 (float)
                - use_yn: 사용 여부

    Raises:
        HTTPException: 데이터 조회 중 오류 발생 시 500 에러 반환
    """
    try:
        mapping_data = db_manager.get_coverage_insur_mapping()
        return {"status": "success", "data": mapping_data}
    except Exception as e:
        logger.error(f"보장코드-담보코드 매핑 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-comparison-tables")
async def get_comparison_tables(request: LoadDataRequest):
    """
    보험료 비교표 생성 API

    사람이 읽기 편한 비교표와 LLM이 읽기 편한 비교표를 반환
    """
    try:
        # 데이터베이스에서 비교표용 데이터 전처리
        comparison_data = db_manager.process_premium_data_for_comparison(
            request.plan_id, request.gender, request.age
        )

        # 로깅
        logger.info(f"=== 비교표 생성 결과 ===")
        logger.info(
            f"플랜 ID: {request.plan_id}, 성별: {request.gender}, 나이: {request.age}"
        )
        logger.info(f"총 회사 수: {comparison_data['summary']['total_companies']}")
        logger.info(f"총 보장 수: {comparison_data['summary']['total_coverages']}")
        logger.info(
            f"사람용 테이블 크기: {len(comparison_data['human_readable_table'])}"
        )
        logger.info(f"LLM용 데이터 크기: {len(comparison_data['llm_readable_data'])}")
        logger.info("=== 비교표 생성 끝 ===")

        return comparison_data

    except Exception as e:
        logger.error(f"비교표 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CoverageRecord(BaseModel):
    company_code: str
    company_name: str
    product_code: str
    product_name: str
    coverage_cd: str
    coverage_name: str
    sum_premium: float
    insur_item_name_list: str
    insur_item_name_coverage_list: str
    payment_due_list: str
    guide_premium_list: str
    guide_contract_amount_max: float


class CoverageResponse(BaseModel):
    status: str
    data: List[CoverageRecord]


class PremiumAgeRecord(BaseModel):
    """
    연령별 보장 보험료 응답 레코드

    Attributes:
        age: 나이
        coverage_cd: 보장 코드
        guide_contract_amount: 안내 계약 금액
        guide_premium: 안내 보험료
    """

    age: int
    coverage_cd: str
    guide_contract_amount: int
    guide_premium: int


class PremiumAgeResponse(BaseModel):
    """
    연령별 보장 보험료 응답 모델

    Attributes:
        status: 처리 상태 ("success" 또는 "error")
        data: 연령별 보장 보험료 데이터 리스트
    """

    status: str
    data: List[PremiumAgeRecord]


class PremiumCoverageRecord(BaseModel):
    """
    보장별 보험료 응답 레코드
    """

    company_code: str
    company_name: str
    product_code: str
    product_name: str
    product_detail_name: Optional[str] = None
    join_condition: Optional[str] = None
    coverage_cd: str
    coverage_name: str
    coverage_seq: Optional[int] = None
    is_selected_coverage: Optional[str] = None
    guide_coverage_amount: Optional[float] = None
    guide_coverage_premium: Optional[int] = None
    coverage_amount_ratio: Optional[float] = None


class PremiumCoverageResponse(BaseModel):
    status: str
    data: List[PremiumCoverageRecord]


@app.post("/get-premium-coverage_single", response_model=PremiumCoverageResponse)
async def get_premium_coverage_single(request: LoadDataRequest):
    """
    보장별 보험료 조회 API

    플랜별 보장 항목별로 합산된 보험료 정보를 반환합니다.

    Args:
        request: LoadDataRequest
            - plan_id: 플랜 ID
            - gender: 성별 (M/F)
            - age: 나이

    Returns:
        PremiumCoverageResponse: 보장별 보험료 데이터
            - status: 처리 상태
            - data: 보장별 보험료 리스트
                - company_code: 보험사 코드
                - company_name: 보험사명
                - product_code: 상품 코드
                - product_name: 상품명
                - product_detail_name: 상품 상세명
                - join_condition: 가입조건
                - coverage_cd: 보장 코드
                - coverage_name: 보장명
                - coverage_seq: 보장 순번
                - is_selected_coverage: 선택 보장 여부
                - guide_coverage_amount: 가이드 보장 금액
                - guide_coverage_premium: 가이드 보장 보험료
                - coverage_amount_ratio: 보장 금액 비율

    Raises:
        HTTPException: 데이터 조회 중 오류 발생 시 500 에러 반환
    """
    try:
        premium_data = db_manager.fetch_premium_data_coverage(
            request.plan_id, request.gender, request.age
        )

        logger.info(f"=== 보장별 보험료 조회 결과 ===")
        logger.info(
            f"플랜 ID: {request.plan_id}, 성별: {request.gender}, 나이: {request.age}"
        )
        logger.info(f"총 기록 수: {len(premium_data)}")
        logger.info("=== 보장별 보험료 조회 끝 ===")

        return {
            "status": "success",
            "data": premium_data,
        }

    except Exception as e:
        logger.error(f"보장별 보험료 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-premium-coverages", response_model=CoverageResponse)
async def get_premium_coverages(request: LoadDataRequest):
    """
    보험료 coverage 데이터 생성 API

    각 보험사의 보장 항목별-상세 주특약별 보험료 정보를 반환합니다.

    Returns:
        CoverageResponse: CoverageResponse 스키마参照
        - status: 처리 상태 ("success" 또는 "error")
        - data: 보장 항목별 보험료 데이터 리스트
            - company_code: 보험사 코드
            - company_name: 보험사명
            - product_code: 상품 코드
            - product_name: 상품명
            - coverage_cd: 보장 코드
            - coverage_name: 보장명
            - sum_premium: 총 보험료
            - insur_item_name_list: 보장 항목명 리스트 (| 구분)
            - insur_item_name_coverage_list: 보장 금액 리스트 (| 구분)
            - payment_due_list: 납입 기간 리스트 (| 구분)
            - guide_premium_list: 보험료 상세 리스트 (+ 구분)
            - guide_contract_amount_max: 최대 계약 보장액

    Raises:
        HTTPException: 데이터 조회 중 오류 발생 시 500 에러 반환
    """
    try:
        coverage_data = db_manager.process_premium_data_for_coverages(
            request.plan_id, request.gender, request.age
        )

        # 로깅
        logger.info(f"=== Coverage 데이터 생성 결과 ===")
        logger.info(
            f"플랜 ID: {request.plan_id}, 성별: {request.gender}, 나이: {request.age}"
        )
        logger.info(f"총 기록 수: {len(coverage_data)}")
        logger.info("=== Coverage 데이터 생성 끝 ===")

        return {
            "status": "success",
            "data": coverage_data,
        }

    except Exception as e:
        logger.error(f"Coverage 데이터 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-premium-age", response_model=PremiumAgeResponse)
async def get_premium_age(request: PremiumAgeRequest):
    """
    연령별 보장별 보험료 조회 API

    하나의 플랜에 대한 하나의 보험사상품에서 동일한 성별의
    입력된 나이 이후부터 최대가입연령까지의 보장별 보험료를 반환합니다.

    Args:
        request: PremiumAgeRequest
            - plan_id: 플랜 ID
            - company_cd: 보험사 코드
            - gender: 성별 (M/F)
            - age: 시작 나이

    Returns:
        PremiumAgeResponse: 연령별 보장별 보험료 데이터
            - status: 처리 상태
            - data: 연령별 보장 보험료 리스트
                - age: 나이
                - coverage_cd: 보장 코드
                - guide_contract_amount: 안내 계약 금액
                - guide_premium: 안내 보험료

    Raises:
        HTTPException: 입력 검증 오류 또는 데이터 조회 오류 발생 시 500 에러 반환
    """
    try:
        # 입력 검증
        if not request.plan_id:
            raise HTTPException(status_code=400, detail="plan_id는 필수 입력값입니다.")
        if not request.company_cd:
            raise HTTPException(
                status_code=400, detail="company_cd는 필수 입력값입니다."
            )
        if request.gender not in ["M", "F"]:
            raise HTTPException(
                status_code=400, detail="gender는 'M' 또는 'F'만 허용됩니다."
            )
        if request.age < 0 or request.age > 100:
            raise HTTPException(
                status_code=400, detail="age는 0에서 100 사이의 값이어야 합니다."
            )

        # 데이터 조회
        premium_data = db_manager.fetch_premium_by_age(
            plan_id=request.plan_id,
            gender=request.gender,
            age=request.age,
            company_cd=request.company_cd,
        )

        # 로깅
        logger.info(f"=== 연령별 보장별 보험료 조회 결과 ===")
        logger.info(
            f"플랜 ID: {request.plan_id}, 보험사: {request.company_cd}, 성별: {request.gender}, 시작 나이: {request.age}"
        )
        logger.info(f"총 기록 수: {len(premium_data)}")
        logger.info("=== 연령별 보장별 보험료 조회 끝 ===")

        return {
            "status": "success",
            "data": premium_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"연령별 보장별 보험료 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """
    스트리밍 방식의 질의응답 엔드포인트
    Server-Sent Events (SSE) 형식으로 실시간 진행 상태 전송
    """

    async def generate_stream():
        logger.info(f"[BACKEND] Stream generation 시작 - 쿼리: '{request.query}'")
        try:
            # 요청에 LLM 데이터가 있는지 확인
            if not request.llm_data:
                logger.error("[BACKEND] LLM 데이터 없음 오류")
                error_data = {
                    "status": "error",
                    "message": "No LLM data provided. Please load data first using /get-comparison-tables endpoint.",
                }
                error_json = json.dumps(error_data, ensure_ascii=False)
                logger.info(f"[BACKEND] 에러 청크 전송: {error_json}")
                yield f"data: {error_json}\n\n"
                return

            # RAG 시스템 스트리밍 실행
            logger.info("[BACKEND] RAG 시스템 스트리밍 호출 시작")

            # human_data가 JSON 문자열인 경우 파싱
            human_data_parsed = None
            if request.human_data:
                try:
                    human_data_parsed = json.loads(request.human_data)
                    logger.info("[BACKEND] human_data JSON 파싱 완료")
                except json.JSONDecodeError as e:
                    logger.error(f"[BACKEND] human_data JSON 파싱 오류: {e}")
                    human_data_parsed = {}

            chunk_count = 0
            async for chunk in rag_system.hybrid_chat_stream_with_data(
                request.query,
                request.llm_data,
                human_data_parsed,
                request.model,
                request.plan_name,
                request.gender,
                request.age,
            ):
                chunk_count += 1
                try:
                    # JSON 직렬화 가능한지 확인하고 변환
                    safe_chunk = _clean_for_json_serialization(chunk)
                    chunk_json = json.dumps(safe_chunk, ensure_ascii=False)

                    # SSE 형식으로 전송
                    sse_data = f"data: {chunk_json}\n\n"

                    yield sse_data

                except (TypeError, ValueError) as e:
                    logger.error(f"[BACKEND] JSON 직렬화 오류: {e}, chunk: {chunk}")
                    error_chunk = {
                        "status": "error",
                        "message": "Response serialization error",
                    }
                    error_json = json.dumps(error_chunk, ensure_ascii=False)
                    yield f"data: {error_json}\n\n"
                    break

            logger.info(f"[BACKEND] Stream generation 완료 - 총 청크 수: {chunk_count}")

        except Exception as e:
            logger.error(f"[BACKEND] Stream generation error: {type(e).__name__}: {e}")
            error_data = {"status": "error", "message": str(e)}
            error_json = json.dumps(error_data, ensure_ascii=False)
            yield f"data: {error_json}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
            "X-Accel-Buffering": "no",  # Nginx buffering 방지
        },
    )


# 2. 나머지 모든 경로를 'frontend' 폴더의 파일들로 연결합니다.
# html=True 옵션을 주면 '/' 접속 시 자동으로 'index.html'을 찾습니다.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


def _clean_for_json_serialization(data: Any) -> Any:
    """
    JSON 직렬화를 위해 무한대/NaN 값을 None으로 변환

    Args:
        data: 변환할 데이터

    Returns:
        JSON 직렬화 가능한 데이터
    """
    try:
        if isinstance(data, dict):
            return {
                key: _clean_for_json_serialization(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [_clean_for_json_serialization(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            # DataFrame은 먼저 fillna로 NaN을 0으로 변환 후 dict로 변환
            cleaned_df = data.fillna(0).replace([np.inf, -np.inf], 0)
            return _clean_for_json_serialization(cleaned_df.to_dict())
        elif isinstance(data, (float, np.floating)):
            if math.isnan(data) or math.isinf(data):
                return 0  # None 대신 0으로 변환
            return float(data)
        elif isinstance(data, (int, np.integer)):
            if math.isnan(data) or math.isinf(data):
                return 0
            return int(data)
        else:
            return data
    except Exception:
        return 0  # 예외 발생 시 0으로 반환


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
