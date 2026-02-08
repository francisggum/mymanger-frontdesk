from openai import OpenAI, AsyncOpenAI
import google.genai as genai
from google.genai import types
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from pydantic import SecretStr
import logging
import os
import time
import asyncio
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# 로깅 레벨 설정 (더 상세한 로그를 위해 INFO로 설정)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Dummy 클래스 정의 (실제 LLM을 사용하지 않을 때)
class DummyChatGoogleGenerativeAI:
    def __init__(self, *args, **kwargs):
        pass


class DummyChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass


# LangChain imports for pandas agent (with fallback handling)
LANGCHAIN_AVAILABLE = False
ChatGoogleGenerativeAI = None
ChatOpenAI = None
create_pandas_dataframe_agent = None
ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
TOOL_CALLING_DESCRIPTION = "tool-calling"

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI

    # 최신 LangChain 버전에서 agent_types 경로 변경
    try:
        from langchain_classic.agents.agent_types import AgentType as LangChainAgentType

        ZERO_SHOT_REACT_DESCRIPTION = LangChainAgentType.ZERO_SHOT_REACT_DESCRIPTION
    except ImportError:
        # fallback: 직접 문자열 정의 (최신 버전에서는 문자열도 지원)
        ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"

    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain pandas agent imports 성공")
except ImportError as e:
    logger.warning(f"LangChain imports 실패: {e}")
    # 실패 시 Dummy 함수들로 설정
    ChatGoogleGenerativeAI = None
    ChatOpenAI = None
    create_pandas_dataframe_agent = None


# Google AI 초기화
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("Google AI client 초기화 성공")
except Exception as e:
    logger.error(f"Google AI client 초기화 실패: {e}")
    client = None

# OpenAI 초기화
try:
    openai_client = None
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = AsyncOpenAI(
            api_key=openai_api_key, base_url="https://openrouter.ai/api/v1"
        )
        logger.info("OpenAI client 초기화 성공")
    else:
        logger.info("OpenAI API 키가 없음")
except ImportError:
    logger.warning("OpenAI 라이브러리 설치 필요")
    openai_client = None
except Exception as e:
    logger.error(f"OpenAI client 초기화 실패: {e}")
    openai_client = None


class HybridRAGSystem:
    def __init__(self):
        self._pandas_llm = None
        self._human_data_llm = "openai"  # 기본 LLM 모델 설정
        self._init_pandas_agent()
        logger.info("HybridRAGSystem 초기화 완료")

    def _init_pandas_agent(self):
        """LangChain pandas agent 초기화"""
        try:
            # 환경 변수 설정
            google_api_key = os.getenv("GOOGLE_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")

            if LANGCHAIN_AVAILABLE and ChatGoogleGenerativeAI and google_api_key:
                logger.info("Google Generative AI로 pandas agent 초기화 시도")
                try:
                    # 최신 LangChain에서는 옵션이 변경될 수 있으므로 기본값으로 시도
                    self._pandas_llm = ChatGoogleGenerativeAI(
                        model="gemini-3-flash-preview",
                        google_api_key=SecretStr(google_api_key),
                        temperature=0.1,
                    )
                    logger.info("ChatGoogleGenerativeAI 성공적으로 초기화")
                except Exception as e:
                    logger.warning(f"ChatGoogleGenerativeAI 초기화 실패: {e}")
                    # 옵션 없이 시도
                    try:
                        self._pandas_llm = ChatGoogleGenerativeAI(
                            model="google/gemini-3-flash-preview",
                            google_api_key=SecretStr(google_api_key),
                        )
                        logger.info(
                            "ChatGoogleGenerativeAI 성공적으로 초기화 (기본 옵션)"
                        )
                    except Exception as e2:
                        logger.error(f"ChatGoogleGenerativeAI 완전 초기화 실패: {e2}")
                        self._pandas_llm = DummyChatGoogleGenerativeAI()

            elif LANGCHAIN_AVAILABLE and ChatOpenAI and openai_api_key:
                logger.info("OpenAI로 pandas agent 초기화 시도")
                try:
                    self._pandas_llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.1,
                        api_key=SecretStr(openai_api_key),
                    )
                    logger.info("ChatOpenAI 성공적으로 초기화")
                except Exception as e:
                    logger.error(f"ChatOpenAI 초기화 실패: {e}")
                    self._pandas_llm = DummyChatOpenAI()

            else:
                if not LANGCHAIN_AVAILABLE:
                    logger.warning("LangChain 없음 - Dummy LLM으로 초기화")
                else:
                    logger.warning("API 키 없음 - Dummy LLM으로 초기화")
                self._pandas_llm = DummyChatGoogleGenerativeAI()

        except Exception as e:
            logger.error(f"pandas agent 초기화 실패: {e}")
            self._pandas_llm = DummyChatGoogleGenerativeAI()

    def _convert_human_data_to_markdown(self, human_data: Dict[str, Any]) -> str:
        """
        피벗 테이블 형태의 human_data를 마크다운으로 변환
        (orient='table' 형식 지원)

        Args:
            human_data: 피벗 테이블 형태의 딕셔너리 또는 orient='table' JSON

        Returns:
            마크다운 형식의 테이블 문자열
        """
        try:
            if not human_data or not isinstance(human_data, dict):
                return "데이터가 없습니다."

            # orient='table' 형식 처리 (schema와 data 키가 있는 경우)
            if "data" in human_data and "schema" in human_data:
                logger.info("[RAG] orient='table' 형식의 데이터 감지")
                data = human_data.get("data", [])
                schema = human_data.get("schema", {})
                fields = schema.get("fields", [])

                if not data:
                    return "데이터가 없습니다."

                # DataFrame 생성
                df = pd.DataFrame(data)

                # 첫 번째 필드를 인덱스로 설정 (보통 보장명 컬럼)
                if fields and len(fields) > 0:
                    index_field = fields[0].get("name")
                    if index_field and index_field in df.columns:
                        df = df.set_index(index_field)
                        logger.info(f"[RAG] 인덱스 설정: {index_field}")

                # 마크다운으로 변환
                markdown_table = df.to_markdown(index=True)

                # if markdown_table is not None:
                #     with open("/app/human_df1.md", "w", encoding="utf-8") as f:
                #         f.write(markdown_table)
                # logger.info("사람용 비교표 markdown 파일 저장 완료: /app/human_df1.md")
                return markdown_table if markdown_table else "변환할 데이터가 없습니다."

            # 기존 딕셔너리 형식 처리 (하위 호환성)
            logger.info("[RAG] 기존 딕셔너리 형식의 데이터 감지")
            rows = []
            for coverage_key, company_data in human_data.items():
                if isinstance(company_data, dict):
                    for company_name, premium_info in company_data.items():
                        rows.append(
                            {
                                "coverage_name": coverage_key,
                                "company_name": company_name,
                                "premium_info": premium_info,
                            }
                        )

            if not rows:
                return "변환할 데이터가 없습니다."

            # DataFrame 생성 및 피벗
            df = pd.DataFrame(rows)
            pivot_df = df.pivot_table(
                index="coverage_name",
                columns="company_name",
                values="premium_info",
                fill_value="-",
                aggfunc="first",
            )

            # 마크다운으로 변환
            markdown_table = pivot_df.to_markdown(index=True)

            # None 체크
            if markdown_table is None:
                markdown_table = "데이터가 없거나 변환에 실패했습니다."

            return markdown_table

        except Exception as e:
            logger.error(f"Human 데이터 마크다운 변환 실패: {e}")
            return f"데이터 변환 중 오류가 발생했습니다: {str(e)}"

    async def _stream_llm_response(
        self, prompt: str, model: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        LLM을 통한 스트리밍 응답 생성 (모델 선택 기능)

        Args:
            prompt: LLM에 전달할 프롬프트
            model: 사용할 모델 ("gemini" 또는 "openai")

        Yields:
            스트리밍 응답 청크
        """
        try:
            # 상태 청크 전송
            yield {
                "status": "llm_generating",
                "message": "AI 답변 생성 중...",
                "progress": 85,
                "timestamp": time.time(),
            }

            response_text = ""
            chunk_count = 0
            usage_metadata = None

            if model == "gemini":
                # Google AI 스트리밍 호출
                if client:
                    response = client.models.generate_content_stream(
                        model="gemini-3-flash-preview", contents=[prompt]
                    )

                    for chunk in response:
                        if chunk.text:
                            response_text += chunk.text
                            chunk_count += 1

                            yield {
                                "status": "llm_streaming",
                                "message": "AI 답변 생성 중...",
                                "progress": 90,
                                "chunk": chunk.text,
                                "partial_response": response_text,
                                "timestamp": time.time(),
                            }

                        # Gemini: 마지막 청크에서 사용량 정보 추출
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            usage_metadata = chunk.usage_metadata
                else:
                    yield {
                        "status": "error",
                        "message": "Google AI client가 초기화되지 않았습니다.",
                        "progress": 0,
                        "timestamp": time.time(),
                    }
                    return

            elif model == "openai":
                # OpenAI 스트리밍 호출
                if openai_client:
                    response = await openai_client.chat.completions.create(
                        # model="deepseek/deepseek-v3.2",
                        model="x-ai/grok-4.1-fast",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        # extra_body={
                        #     "reasoning": {
                        #         "effect": "hight",
                        #         "exclude": False,
                        #         "enabled": True,
                        #     }
                        # },
                    )

                    # OpenAI 스트리밍은 async for 사용
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            chunk_text = chunk.choices[0].delta.content
                            response_text += chunk_text
                            chunk_count += 1

                            yield {
                                "status": "llm_streaming",
                                "message": "AI 답변 생성 중...",
                                "progress": 90,
                                "chunk": chunk_text,
                                "partial_response": response_text,
                                "timestamp": time.time(),
                            }

                        # OpenAI: 사용량 정보 추출 (마지막 청크)
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_metadata = chunk.usage
                else:
                    yield {
                        "status": "error",
                        "message": "OpenAI client가 초기화되지 않았습니다.",
                        "progress": 0,
                        "timestamp": time.time(),
                    }
                    return
            else:
                yield {
                    "status": "error",
                    "message": f"지원하지 않는 모델: {model}",
                    "progress": 0,
                    "timestamp": time.time(),
                }
                return

            # 사용량 로깅
            if usage_metadata:
                if model == "gemini":
                    prompt_tokens = usage_metadata.prompt_token_count
                    completion_tokens = usage_metadata.candidates_token_count
                    total_tokens = usage_metadata.total_token_count
                    cost = (
                        usage_metadata.cost
                        if hasattr(usage_metadata, "cost")
                        else "N/A"
                    )

                else:  # openai
                    prompt_tokens = usage_metadata.prompt_tokens
                    completion_tokens = usage_metadata.completion_tokens
                    total_tokens = usage_metadata.total_tokens
                    cost = (
                        usage_metadata.cost
                        if hasattr(usage_metadata, "cost")
                        else "N/A"
                    )

                logger.info("=" * 50)
                logger.info("=== LLM 토큰 사용량 ===")
                logger.info(f"모델: {model}")
                logger.info(f"프롬프트 토큰: {prompt_tokens:,}")
                logger.info(f"완성 토큰: {completion_tokens:,}")
                logger.info(f"총 토큰: {total_tokens:,}")
                logger.info(f"비용: {cost} 크레딧")

                # Gemini 비용估算 (예시 가격 - 실제 가격은 API 문서 확인 필요)
                if model == "gemini":
                    # gemini-3-flash-preview 가격 (예시: $0.075/1M 입력 토큰, $0.30/1M 출력 토큰)
                    input_cost = (prompt_tokens / 1_000_000) * 0.075
                    output_cost = (completion_tokens / 1_000_000) * 0.30
                    estimated_cost = input_cost + output_cost
                    logger.info(f"예상 비용: ${estimated_cost:.6f}")

                logger.info("=" * 50)

            # 최종 완료 청크
            yield {
                "status": "complete",
                "message": "분석 완료!",
                "progress": 100,
                "response": response_text,
                "usage": (
                    {
                        "prompt_tokens": prompt_tokens if usage_metadata else 0,
                        "completion_tokens": completion_tokens if usage_metadata else 0,
                        "total_tokens": total_tokens if usage_metadata else 0,
                        "cost": cost,
                    }
                    if usage_metadata
                    else None
                ),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"LLM 스트리밍 응답 생성 실패: {e}")
            yield {
                "status": "error",
                "message": f"AI 답변 생성 실패: {str(e)}",
                "progress": 0,
                "timestamp": time.time(),
            }

    async def hybrid_chat_stream_with_data(
        self,
        query: str,
        llm_data: Dict[str, Any],
        human_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        plan_name: Optional[str] = None,
        gender: Optional[str] = None,
        age: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        단순화된 스트리밍 질의응답 - LLM 직접 호출 방식

        Args:
            query: 사용자 질문
            llm_data: LLM 데이터
            human_data: Human 데이터 (선택사항)
            model: 사용할 LLM 모델 ("gemini" 또는 "openai"), None이면 기본값 사용
            plan_name: 플랜명
            gender: 성별 (남성/여성)
            age: 나이
        """
        start_time = time.time()
        logger.info(f"Hybrid Chat Stream with Data 시작 - 쿼리: '{query}'")

        logger.info(f"LLM 데이터 샘플 100자: {str(llm_data)[:100] if llm_data else 0}")

        try:
            # 1. 데이터 처리 상태 전송
            yield {
                "status": "processing_data",
                "message": "데이터 준비 중...",
                "progress": 30,
                "timestamp": time.time(),
            }

            # Human 데이터를 마크다운으로 변환
            markdown_table = ""
            if human_data:
                markdown_table = self._convert_human_data_to_markdown(human_data)
                logger.info(
                    f"Human 데이터 마크다운 변환 완료 (길이: {len(markdown_table)})"
                )
                logger.info(
                    f"Human 데이터 마크다운 내용: {markdown_table[:100]}..."  # 첫 100자만 로깅
                )

            # 2. LLM 호출 상태 전송
            yield {
                "status": "llm_generating",
                "message": "AI 답변 생성 중...",
                "progress": 60,
                "timestamp": time.time(),
            }

            # 3. 시스템 프롬프트 구성
            system_prompt = f"""
#### 개요 ####
너는 보험 비교 전문가야. 아래 제공된 표는 각 보험사별 답볍명과 보험료(가입금액) 데이터야. 
괄호 안의 숫자는 세부 산출 내역이니, 최종 보험료를 비교할 때는 괄호 밖의 숫자를 우선적으로 확인해줘.
제공된 보험료 비교표는 플랜명이 "{plan_name or '미지정'}"이고 성별은 "{gender or '미지정'}", 나이는 "{age or '미지정'}"세 기준으로 작성되어 있어
다른 나이, 다른 성별, 다른 플랜에 대한 정보는 제공된 데이터에 포함되어 있지 않아 답변할 수 없어.
사용자의 질문에 대해서는 본 데이터를 참고해서 가장 적합한 답변을 제공하고 제공된 데이터에 없는 내용은 모른다고 솔직하게 답변해줘.

#### 각 보험사별 보험료 비교표 ####
{markdown_table}

#### 사용자 질문 ####
{query}
"""

            # if markdown_table is not None:
            #     with open("/app/prompt.txt", "w", encoding="utf-8") as f:
            #         f.write(system_prompt)

            # 4. LLM 스트리밍 응답 생성
            # 모델 선택: 파라미터가 있으면 사용, 없으면 기본값(self._human_data_llm) 사용
            selected_model = model if model else self._human_data_llm
            logger.info(f"선택된 LLM 모델: {selected_model}")

            async for chunk in self._stream_llm_response(system_prompt, selected_model):
                chunk["processing_time"] = time.time() - start_time
                yield chunk

        except Exception as e:
            error_msg = f"Chat Stream 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            yield {
                "status": "error",
                "message": f"처리 중 오류가 발생했습니다: {str(e)}",
                "progress": 0,
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
            }


# 전역 인스턴스 생성
rag_system = HybridRAGSystem()
