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
import json
from dotenv import load_dotenv

# LLM Tools import
from llm_tools import (
    OPENAI_TOOLS,
    GEMINI_TOOLS,
    TOOL_FUNCTIONS,
    execute_tool,
    search_by_company_name,
    search_by_coverage_name,
    compare_companies_by_coverage,
)

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
        self, prompt: str, model: str, llm_data: Dict[str, Any], query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        LLM을 통한 스트리밍 응답 생성 (Tool Calling 통합)

        Args:
            prompt: LLM에 전달할 프롬프트
            model: 사용할 모델 ("gemini" 또는 "openai")
            llm_data: 보험 데이터
            query: 사용자 질문

        Yields:
            스트리밍 응답 청크
        """
        try:
            yield {
                "status": "analyzing",
                "message": "질문 분석 중...",
                "progress": 20,
                "timestamp": time.time(),
            }

            response_text = ""
            tool_calls = []
            usage_metadata = None

            if model == "openai":
                # OpenAI Tool Calling
                if openai_client:
                    messages = [
                        {
                            "role": "system",
                            "content": "당신은 보험 비교 전문가입니다. 제공된 도구들을 사용하여 사용자의 질문에 답변하세요.",
                        },
                        {"role": "user", "content": prompt},
                    ]

                    # 첫 번째 호출: 도구 선택
                    response = await openai_client.chat.completions.create(
                        model="x-ai/grok-4.1-fast",
                        messages=messages,
                        tools=OPENAI_TOOLS,
                        tool_choice="auto",
                    )

                    message = response.choices[0].message

                    # 도구 호출이 있는 경우
                    if message.tool_calls:
                        logger.info(
                            f"OpenAI 도구 호출 감지: {len(message.tool_calls)}개"
                        )

                        # 프론트엔드에 도구 사용 상태 전송
                        yield {
                            "status": "searching",
                            "message": "보험 비교표외에 추가적인 세부 보장 정보 조회중...",
                            "progress": 65,
                            "timestamp": time.time(),
                        }

                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)

                            logger.info(
                                f"도구 실행: {function_name}, 파라미터: {function_args}"
                            )

                            # 도구 실행
                            result = execute_tool(
                                llm_data, function_name, function_args
                            )
                            tool_calls.append(
                                {
                                    "tool": function_name,
                                    "args": function_args,
                                    "result": result,
                                }
                            )

                        # 도구 결과를 포함하여 최종 응답 생성
                        tool_results_text = "\n\n[도구 검색 결과]\n"
                        for tc in tool_calls:
                            tool_results_text += f"\n{tc['tool']}({tc['args']}):\n{json.dumps(tc['result'], ensure_ascii=False, indent=2)}\n"

                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tc.get("id", "call_1"),
                                        "type": "function",
                                        "function": {
                                            "name": tc["tool"],
                                            "arguments": json.dumps(tc["args"]),
                                        },
                                    }
                                    for tc in tool_calls
                                ],
                            }
                        )

                        for tc in tool_calls:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", "call_1"),
                                    "content": json.dumps(
                                        tc["result"], ensure_ascii=False
                                    ),
                                }
                            )

                        # 최종 응답 스트리밍
                        yield {
                            "status": "generating",
                            "message": "답변 생성 중...",
                            "progress": 80,
                            "timestamp": time.time(),
                        }

                        final_response = await openai_client.chat.completions.create(
                            model="x-ai/grok-4.1-fast",
                            messages=messages,
                            stream=True,
                        )

                        async for chunk in final_response:
                            if chunk.choices[0].delta.content:
                                response_text += chunk.choices[0].delta.content
                                yield {
                                    "status": "streaming",
                                    "message": "답변 생성 중...",
                                    "progress": 90,
                                    "chunk": chunk.choices[0].delta.content,
                                    "partial_response": response_text,
                                    "timestamp": time.time(),
                                }

                            # 사용량 정보 추출
                            if hasattr(chunk, "usage") and chunk.usage:
                                usage_metadata = chunk.usage
                    else:
                        # 도구 호출 없음 - 바로 응답
                        response_text = message.content or ""
                        yield {
                            "status": "streaming",
                            "message": "답변 생성 중...",
                            "progress": 90,
                            "chunk": response_text,
                            "partial_response": response_text,
                            "timestamp": time.time(),
                        }

                else:
                    yield {
                        "status": "error",
                        "message": "OpenAI client가 초기화되지 않았습니다.",
                        "progress": 0,
                        "timestamp": time.time(),
                    }
                    return

            elif model == "gemini":
                # Gemini Tool Calling (수동 방식)
                if client:
                    tool_selection_prompt = f"""
사용자 질문: {query}

사용 가능한 도구들:
1. search_by_company_name - 회사명으로 검색 (예: "DB", "삼성")
2. search_by_coverage_name - 보장 항목으로 검색 (예: "암진단", "상해")
3. compare_companies_by_coverage - 특정 보장으로 회사 비교 (예: "통합암진단비")
4. get_cheapest_company - 가장 저렴한 회사 찾기
5. get_company_summary - 회사 전체 정보

이 질문에 어떤 도구를 사용해야 하나요? JSON 형식으로 답변하세요:
{{"tool": "도구명", "parameters": {{"파라미터명": "값"}}}}

또는 도구가 필요 없으면: {{"tool": null}}
"""

                    selection_response = client.models.generate_content(
                        model="gemini-3-flash-preview", contents=[tool_selection_prompt]
                    )

                    try:
                        selection_text = selection_response.text.strip()
                        if "```json" in selection_text:
                            selection_text = (
                                selection_text.split("```json")[1]
                                .split("```")[0]
                                .strip()
                            )
                        elif "```" in selection_text:
                            selection_text = (
                                selection_text.split("```")[1].split("```")[0].strip()
                            )

                        selection = json.loads(selection_text)
                        selected_tool = selection.get("tool")
                        parameters = selection.get("parameters", {})

                        if selected_tool and selected_tool in TOOL_FUNCTIONS:
                            logger.info(
                                f"Gemini 도구 선택: {selected_tool}, 파라미터: {parameters}"
                            )

                            yield {
                                "status": "searching",
                                "message": "보험 비교표외에 추가적인 세부 보장 정보 조회중...",
                                "progress": 65,
                                "timestamp": time.time(),
                            }

                            result = execute_tool(llm_data, selected_tool, parameters)
                            tool_calls.append(
                                {
                                    "tool": selected_tool,
                                    "args": parameters,
                                    "result": result,
                                }
                            )

                            tool_result_prompt = f"""
{prompt}

[도구 검색 결과]
도구: {selected_tool}
파라미터: {json.dumps(parameters, ensure_ascii=False)}
결과: {json.dumps(result, ensure_ascii=False, indent=2)}

위 결과를 바탕으로 사용자 질문에 답변해주세요.
"""
                            prompt = tool_result_prompt
                    except Exception as e:
                        logger.warning(f"Gemini 도구 선택 파싱 실패: {e}")

                    yield {
                        "status": "generating",
                        "message": "답변 생성 중...",
                        "progress": 80,
                        "timestamp": time.time(),
                    }

                    response = client.models.generate_content_stream(
                        model="gemini-3-flash-preview", contents=[prompt]
                    )

                    for chunk in response:
                        if chunk.text:
                            response_text += chunk.text
                            yield {
                                "status": "streaming",
                                "message": "답변 생성 중...",
                                "progress": 90,
                                "chunk": chunk.text,
                                "partial_response": response_text,
                                "timestamp": time.time(),
                            }

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

사용 가능한 도구들:
- search_by_company_name: 회사명으로 검색
- search_by_coverage_name: 보장 항목으로 검색  
- compare_companies_by_coverage: 특정 보장 항목으로 회사들 비교
- get_cheapest_company: 특정 보장에서 가장 저렴한 회사 찾기

#### 각 보험사별 보험료 비교표 ####
{markdown_table}

#### 사용자 질문 ####
{query}
"""

            # if markdown_table is not None:
            #     with open("/app/prompt.txt", "w", encoding="utf-8") as f:
            #         f.write(system_prompt)

            # 4. LLM 스트리밍 응답 생성 (Tool Calling 통합)
            # 모델 선택: 파라미터가 있으면 사용, 없으면 기본값(self._human_data_llm) 사용
            selected_model = model if model else self._human_data_llm
            logger.info(f"선택된 LLM 모델: {selected_model}")

            # llm_data를 포함하여 Tool Calling으로 실행
            logger.info(
                f"Tool Calling 모드로 실행 - llm_data 키 수: {len(llm_data) if llm_data else 0}"
            )
            async for chunk in self._stream_llm_response(
                system_prompt, selected_model, llm_data, query
            ):
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
