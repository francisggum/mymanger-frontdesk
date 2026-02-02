from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from api_client import api_client
from data_manager import data_manager
from rag_system import rag_system
import pandas as pd
import numpy as np
import math

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JWTRequest(BaseModel):
    jwt_token: str


class LoadDataRequest(BaseModel):
    jwt_token: str
    plan_id: str
    age: int
    gender: str


class ChatRequest(BaseModel):
    query: str


class PlanInfo(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: Optional[str] = ""
    plan_type_name: Optional[str] = ""
    plan_payterm_type_name: Optional[str] = ""
    plan_min_m_age: Optional[int] = 0
    plan_max_m_age: Optional[int] = 0
    plan_min_f_age: Optional[int] = 0
    plan_max_f_age: Optional[int] = 0


@app.get("/")
async def root():
    return {"message": "Insurance Comparison AI API is running"}


@app.post("/fetch-plans", response_model=List[PlanInfo])
async def fetch_plans(request: JWTRequest):
    try:
        plans = await api_client.fetch_plans(request.jwt_token)
        return [
            PlanInfo(
                **{
                    **plan,
                    "plan_min_m_age": int(plan.get("plan_min_m_age", 0)),
                    "plan_max_m_age": int(plan.get("plan_max_m_age", 0)),
                    "plan_min_f_age": int(plan.get("plan_min_f_age", 0)),
                    "plan_max_f_age": int(plan.get("plan_max_f_age", 0)),
                }
            )
            for plan in plans
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-data")
async def load_data(request: LoadDataRequest):
    try:
        # 외부 API에서 데이터 가져오기
        premium_data = await api_client.fetch_product_premiums(
            request.jwt_token, request.plan_id, request.age, request.gender
        )

        # 데이터 처리 및 저장
        result = data_manager.process_premium_data(
            premium_data, request.plan_id, request.age, request.gender
        )

        # RAG 시스템 초기화 (VectorDB)
        insurance_data = data_manager.get_insurance_data()
        if insurance_data:
            vector_init_success = rag_system.initialize_vector_store(insurance_data)
            result["vector_store_initialized"] = vector_init_success
        else:
            result["vector_store_initialized"] = False

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-comparison-table")
async def get_comparison_table():
    """
    현재 로드된 데이터를 기반으로 보험사별 보장 항목 비교 표 생성
    """
    try:
        # 현재 세션 정보 확인
        session_info = data_manager.get_current_session_info()

        if not session_info["has_coverage_data"]:
            raise HTTPException(
                status_code=400,
                detail="No coverage data loaded. Please load data first using /load-data endpoint.",
            )

        # 현재 데이터 가져오기 (가장 최근에 로드된 데이터 재사용)
        if (
            data_manager.coverage_premiums_df is None
            or data_manager.coverage_premiums_df.empty
        ):
            raise HTTPException(
                status_code=400,
                detail="No coverage premiums data available. Please load data first.",
            )

        # 비교 표 생성
        normalized_df = data_manager.normalize_coverage_amounts(
            data_manager.coverage_premiums_df
        )
        aggregated_df = data_manager.aggregate_coverage_by_code(normalized_df)
        comparison_table = data_manager.create_comparison_table(aggregated_df)

        return {
            "status": "success",
            "session_info": session_info,
            "comparison_table": _clean_for_json_serialization(
                comparison_table.to_dict()
            ),
            "raw_aggregated_data": _clean_for_json_serialization(
                aggregated_df.to_dict()
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 현재 세션 정보 확인
        session_info = data_manager.get_current_session_info()

        if (
            not session_info["has_coverage_data"]
            and not session_info["has_insurance_data"]
        ):
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please load data first using /load-data endpoint.",
            )

        # 데이터 가져오기
        coverage_df = data_manager.get_coverage_dataframe()
        insurance_data = data_manager.get_insurance_data()

        # Hybrid RAG 시스템으로 질의응답
        result = rag_system.hybrid_chat(request.query, coverage_df, insurance_data)

        # JSON 직렬화를 위해 데이터 정리
        cleaned_result = _clean_for_json_serialization(result)

        return cleaned_result

    except HTTPException:
        raise
    except Exception as e:
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
            # 현재 세션 정보 확인
            session_info = data_manager.get_current_session_info()
            logger.info(f"[BACKEND] 세션 정보 확인: {session_info}")

            if (
                not session_info["has_coverage_data"]
                and not session_info["has_insurance_data"]
            ):
                logger.error("[BACKEND] 데이터 없음 오류")
                error_data = {
                    "status": "error",
                    "message": "No data loaded. Please load data first using /load-data endpoint.",
                }
                error_json = json.dumps(error_data, ensure_ascii=False)
                logger.info(f"[BACKEND] 에러 청크 전송: {error_json}")
                yield f"data: {error_json}\n\n"
                return

            # 데이터 가져오기
            logger.info("[BACKEND] 데이터 가져오기 시작")
            coverage_df = data_manager.get_coverage_dataframe()
            insurance_data = data_manager.get_insurance_data()
            logger.info(f"[BACKEND] 데이터 가져오기 완료 - df shape: {coverage_df.shape if coverage_df is not None else None}, insurance_data: {len(insurance_data) if insurance_data else 0}")

            # RAG 시스템 스트리밍 실행
            logger.info("[BACKEND] RAG 시스템 스트리밍 호출 시작")
            chunk_count = 0
            async for chunk in rag_system.hybrid_chat_stream(
                request.query, coverage_df, insurance_data
            ):
                chunk_count += 1
                try:
                    # JSON 직렬화 가능한지 확인하고 변환
                    safe_chunk = _clean_for_json_serialization(chunk)
                    chunk_json = json.dumps(safe_chunk, ensure_ascii=False)
                    logger.info(f"[BACKEND] 청크 {chunk_count} 직렬화 완료: {len(chunk_json)}바이트")
                    
                    # SSE 형식으로 전송
                    sse_data = f"data: {chunk_json}\n\n"
                    logger.info(f"[BACKEND] 청크 {chunk_count} 전송: {safe_chunk.get('status', 'unknown')}, {safe_chunk.get('message', '')}")
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
