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
from database import db_manager
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


class LoadDataRequest(BaseModel):
    plan_id: str
    age: int
    gender: str


class ChatRequest(BaseModel):
    query: str
    llm_data: Optional[Dict[str, Any]] = None
    human_data: Optional[str] = None  # JSON 문자열 (orient='table' 형식)
    model: Optional[str] = None  # "gemini" 또는 "openai", None이면 기본값 사용


class PlanInfo(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: Optional[str] = ""
    plan_type_name: Optional[str] = ""
    insu_compy_type_name: Optional[str] = ""
    plan_payterm_type_name: Optional[str] = ""
    plan_min_m_age: Optional[int] = 0
    plan_max_m_age: Optional[int] = 0
    plan_min_f_age: Optional[int] = 0
    plan_max_f_age: Optional[int] = 0


@app.get("/")
async def root():
    return {"message": "Insurance Comparison AI API is running"}


@app.post("/fetch-plans", response_model=List[PlanInfo])
async def fetch_plans():
    try:
        plans = db_manager.fetch_plans()
        return [
            PlanInfo(
                **{
                    **plan,
                    "plan_min_m_age": int(plan.get("min_m_age", 0)),
                    "plan_max_m_age": int(plan.get("max_m_age", 0)),
                    "plan_min_f_age": int(plan.get("min_f_age", 0)),
                    "plan_max_f_age": int(plan.get("max_f_age", 0)),
                }
            )
            for plan in plans
        ]
    except Exception as e:
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
                request.query, request.llm_data, human_data_parsed, request.model
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
