import chromadb
import google.genai as genai
from google.genai import types
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
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
    LANGCHAIN_AVAILABLE = False

    # Fallback dummy classes
    class DummyChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain packages not available")

    class DummyChatOpenAI:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain packages not available")

    def dummy_create_pandas_dataframe_agent(*args, **kwargs):
        raise ImportError("LangChain packages not available")

    ChatGoogleGenerativeAI = DummyChatGoogleGenerativeAI
    ChatOpenAI = DummyChatOpenAI
    create_pandas_dataframe_agent = dummy_create_pandas_dataframe_agent


class HybridRAGSystem:
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider.lower()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embedding_model = self.client.models.embed_content
        self.llm = self.client.models.generate_content
        self.vector_store = None
        self.qa_chain = None
        self._pandas_llm = None
        # 환경변수 또는 기본값으로 판다스 분석 단계 설정
        self.pandas_analysis_stages = int(os.getenv("PANDAS_ANALYSIS_STAGES", "2"))
        logger.debug(
            f"HybridRAGSystem initialized with llm_provider={self.llm_provider}, pandas_analysis_stages={os.getenv('PANDAS_ANALYSIS_STAGES')}"
        )

    def _get_pandas_llm(self):
        """LangChain pandas agent를 위한 LLM 초기화 (lazy loading)"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain을 사용할 수 없음 - pandas agent 생성 불가")
            raise ImportError("LangChain 패키지가 설치되지 않았습니다")

        if self._pandas_llm is None:
            try:
                if self.llm_provider == "gemini":
                    if ChatGoogleGenerativeAI is None:
                        raise ImportError("ChatGoogleGenerativeAI not available")

                    model = "gemini-3-flash-preview"

                    self._pandas_llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=0,
                        google_api_key=os.getenv("GOOGLE_API_KEY"),
                        # convert_system_message_to_human=True,
                        generate_content_config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                                thinking_level="LOW",
                            )
                        ),
                    )
                    logger.info(
                        f"LangChain ChatGoogleGenerativeAI 초기화 성공 - 사용 model: {model}"
                    )
                elif self.llm_provider == "openai":
                    model = "qwen/qwen3-235b-a22b-2507"
                    self._pandas_llm = ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        # base_url="https://api.groq.com/openai/v1",
                        model=model,
                        temperature=0,
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        default_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "MyManger Frontdesk",
                        },
                    )
                    logger.info(
                        f"LangChain ChatOpenAI 초기화 성공 - 사용 model: {model}"
                    )
                else:
                    raise ValueError(f"지원되지 않는 LLM 제공업체: {self.llm_provider}")
            except Exception as e:
                logger.error(f"LangChain {self.llm_provider} LLM 초기화 실패: {e}")
                raise
        return self._pandas_llm

    def _create_pandas_agent(self, df: pd.DataFrame):
        """LangChain pandas agent 생성"""
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.error("LangChain을 사용할 수 없음 - fallback 모드 사용")
                return None

            if create_pandas_dataframe_agent is None:
                logger.error("create_pandas_dataframe_agent not available")
                return None

            llm = self._get_pandas_llm()

            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                # agent_type=ZERO_SHOT_REACT_DESCRIPTION,
                agent_type=TOOL_CALLING_DESCRIPTION,
                handle_parsing_errors=True,
                max_iterations=20,
                max_execution_time=120,
                return_intermediate_steps=True,
                allow_dangerous_code=True,
            )

            logger.info(
                f"Pandas DataFrame Agent 생성 성공 - DataFrame shape: {df.shape}"
            )
            return agent

        except Exception as e:
            logger.error(f"Pandas Agent 생성 실패: {e}")
            return None

    def _extract_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """DataFrame의 기본 정보 추출"""
        try:
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "sample_data": df.head(3).to_dict() if len(df) > 0 else {},
            }
        except Exception as e:
            logger.error(f"데이터 정보 추출 실패: {e}")
            return {}

    def _generate_insurance_prompt(self, query: str, df_info: Dict) -> str:
        """보험 데이터 분석 특화 프롬프트 생성"""

        base_prompt = f"""
당신은 보험 전문가입니다. 다음 보험료 데이터를 분석하여 질문에 답변해주세요.

분석 목표: {query}
데이터 정보:
- 형태: {df_info.get('shape', 'Unknown')}
- 컬럼: {df_info.get('columns', [])}
- 데이터 타입: {df_info.get('dtypes', {})}

보험 분석 가이드:
1. 보험료 비교: 가장 저렴한 보험사 순위 제시
2. 보장 항목 분석: 암진단비, 상해보장 등 주요 보장 비교  
3. 특징 분석: 각 보험사의 장단점 및 차이점
4. 합리적인 추천: 비용-효과성 기준 추천

분석 지침:
- 전체 데이터 기반 통계적 분석 수행
- 특이값(outlier) 확인 및 분석
- 보험사별 보장 내용 상세 비교
- 한국어 보험 용어 사용
- 구체적인 수치 데이터 제공
"""

        # 질문 유형별 추가 프롬프트
        query_lower = query.lower()
        if "저렴" in query_lower or "싼" in query_lower or "가격" in query_lower:
            return (
                base_prompt
                + "\n\n특히 보험료 합계가 낮은 순으로 정렬하고, 가성비를 분석해주세요."
            )
        elif (
            "보장" in query_lower or "담보" in query_lower or "보장내용" in query_lower
        ):
            return (
                base_prompt
                + "\n\n각 보장 항목별 상세 비교와 보장 내용의 차이점을 분석해주세요."
            )
        elif "추천" in query_lower or "어떤" in query_lower:
            return (
                base_prompt
                + "\n\n고객의 입장에서 가장 합리적인 선택을 추천하고 그 이유를 설명해주세요."
            )
        elif "비교" in query_lower or "차이" in query_lower:
            return base_prompt + "\n\n보험사별 차이점을 명확하게 비교 분석해주세요."

        return base_prompt

    def initialize_vector_store(self, insurance_data: List[Dict[str, Any]]):
        """
        product_insur_premiums 데이터로 ChromaDB 벡터 저장소 초기화
        """
        try:
            if not insurance_data:
                logger.warning(
                    "No insurance data provided for vector store initialization"
                )
                return False

            # ChromaDB 클라이언트 생성
            self.vector_store = chromadb.Client()

            # 기존 컬렉션이 있으면 삭제
            try:
                self.vector_store.delete_collection("insurance_coverage")
            except:
                pass

            self.collection = self.vector_store.create_collection(
                name="insurance_coverage", metadata={"hnsw:space": "cosine"}
            )

            # 문서 준비
            documents = []
            metadatas = []
            ids = []

            for i, item in enumerate(insurance_data):
                # insur_bojang(보장설명) 텍스트 추출
                bojang_text = item.get("insur_bojang", "")
                if bojang_text:
                    documents.append(bojang_text)
                    metadatas.append(
                        {
                            "plan_id": item.get("plan_id", ""),
                            "insur_name": item.get("insur_name", ""),
                            "insur_code": item.get("insur_code", ""),
                            "premium_amount": str(item.get("premium_amount", 0)),
                        }
                    )
                    ids.append(f"doc_{i}")

            if not documents:
                logger.warning("No valid documents created from insurance data")
                return False

            # 임베딩 생성 및 저장
            logger.info(f"임베딩 생성 시작 - 문서 수: {len(documents)}")
            start_time = time.time()

            result = self.embedding_model(
                model="gemini-embedding-001", contents=documents
            )

            # 임베딩 결과 확인
            if not result or not hasattr(result, "embeddings") or not result.embeddings:
                logger.error("임베딩 생성 실패: result가 비어있음")
                return False

            embeddings = []
            for i, emb in enumerate(result.embeddings):
                if emb and hasattr(emb, "values"):
                    embeddings.append(emb.values)
                else:
                    logger.warning(f"임베딩 {i}가 비어있음")

            if len(embeddings) != len(documents):
                logger.error(
                    f"임베딩 수({len(embeddings)})와 문서 수({len(documents)})가 일치하지 않음"
                )
                return False

            logger.info(
                f"임베딩 생성 완료 - 임베딩 수: {len(embeddings)}, 소요 시간: {time.time() - start_time:.2f}초"
            )

            # ChromaDB에 저장
            logger.info("ChromaDB에 문서 저장 시작")
            try:
                # 임베딩 변환 시도
                self.collection.add(
                    embeddings=embeddings,  # 원본 임베딩 사용 (변환 시도 안 함)
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info("ChromaDB 저장 완료")
            except Exception as e:
                logger.error(f"ChromaDB 저장 실패 (첫 시도): {e}")

                # fallback: 간단한 리스트로 변환 시도
                try:
                    simple_embeddings = []
                    for emb in embeddings:
                        if emb is not None:
                            if hasattr(emb, "tolist"):
                                simple_embeddings.append(emb.tolist())
                            else:
                                simple_embeddings.append(list(emb))
                        else:
                            simple_embeddings.append([])

                    self.collection.add(
                        embeddings=simple_embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    logger.info("ChromaDB 저장 완료 (fallback)")
                except Exception as e2:
                    logger.error(f"ChromaDB 저장 실패 (fallback): {e2}")
                    return False

            logger.info(
                f"Vector store initialized successfully with {len(documents)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False

    def search_relevant_docs(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 문서 검색
        """
        start_time = time.time()
        logger.info(f"문서 검색 시작 - 쿼리: '{query}', 검색 수: {k}")

        try:
            if not hasattr(self, "collection"):
                logger.warning("Vector store not initialized")
                return []

            # 쿼리 임베딩 생성
            logger.info("쿼리 임베딩 생성 시작")
            embedding_start = time.time()

            query_embedding_result = self.embedding_model(
                model="gemini-embedding-001", contents=[query]
            )

            if (
                not query_embedding_result
                or not hasattr(query_embedding_result, "embeddings")
                or not query_embedding_result.embeddings
                or len(query_embedding_result.embeddings) == 0
                or not hasattr(query_embedding_result.embeddings[0], "values")
            ):
                logger.error("쿼리 임베딩 생성 실패")
                return []

            query_embedding = query_embedding_result.embeddings[0].values
            logger.info(
                f"쿼리 임베딩 생성 완료 - 소요 시간: {time.time() - embedding_start:.2f}초"
            )

            # 검색
            logger.info("벡터 검색 시작")
            search_start = time.time()

            # 쿼리 임베딩 처리
            if query_embedding is None:
                logger.error("쿼리 임베딩이 None")
                return []

            # ChromaDB 쿼리 - 여러 형식 시도
            try:
                # 첫 시도: 원본 임베딩 사용
                results = self.collection.query(
                    query_embeddings=[query_embedding], n_results=k
                )
            except Exception as e1:
                logger.warning(f"첫 쿼리 시도 실패: {e1}")
                try:
                    # 두 번째 시도: 리스트로 변환
                    import numpy as np

                    query_array = np.array(query_embedding, dtype=np.float32)
                    results = self.collection.query(
                        query_embeddings=[query_array], n_results=k
                    )
                except Exception as e2:
                    logger.error(f"쿼리 실패: {e2}")
                    return []

            logger.info(
                f"벡터 검색 완료 - 소요 시간: {time.time() - search_start:.2f}초"
            )

            # 결과 포맷팅
            docs = []
            if (
                results
                and isinstance(results, dict)
                and "documents" in results
                and results["documents"]
                and len(results["documents"]) > 0
                and results["documents"][0]
            ):

                documents_list = results["documents"][0]  # 첫 번째 결과 세트
                metadatas_list = []
                if (
                    results.get("metadatas")
                    and isinstance(results["metadatas"], list)
                    and len(results["metadatas"]) > 0
                    and results["metadatas"][0]
                ):
                    metadatas_list = results["metadatas"][0]

                for i in range(len(documents_list)):
                    doc_data = {
                        "page_content": (
                            documents_list[i] if i < len(documents_list) else ""
                        ),
                        "metadata": (
                            metadatas_list[i] if i < len(metadatas_list) else {}
                        ),
                    }
                    docs.append(doc_data)
                    content_length = len(str(doc_data["page_content"]))
                    logger.debug(
                        f"문서 {i+1}: {doc_data['metadata'].get('insur_name', 'Unknown')} - {content_length}자"
                    )

            total_time = time.time() - start_time
            logger.info(
                f"문서 검색 완료 - 찾은 문서 수: {len(docs)}, 총 소요 시간: {total_time:.2f}초"
            )
            return docs

        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {e}")
            logger.error(f"오류 상세: {type(e).__name__}: {str(e)}")
            return []

    def _execute_fallback_analysis(
        self, df: pd.DataFrame, query: str
    ) -> Dict[str, Any]:
        """LangChain을 사용할 수 없을 때의 fallback 분석"""
        start_time = time.time()

        try:
            logger.info("Fallback 분석 모드 실행 - 통계적 분석 수행")

            # 기본 통계 정보 계산
            df_info = self._extract_data_info(df)
            analysis_prompt = self._generate_insurance_prompt(query, df_info)

            # 데이터 통계 분석
            stats = {}
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats = df.describe().to_dict()
                    logger.info("기술 통계 계산 완료")
            except Exception as e:
                logger.warning(f"기술 통계 계산 실패: {e}")

            # 보험사별 요약 (가능한 경우)
            company_summary = {}
            for col in df.columns:
                if any(
                    keyword in col.lower()
                    for keyword in ["보험사", "company", "insurer"]
                ):
                    try:
                        company_summary[col] = df[col].value_counts().to_dict()
                    except:
                        pass

            # 보험료 관련 분석 (가능한 경우)
            premium_analysis = {}
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if any(
                        keyword in col.lower()
                        for keyword in ["보험료", "premium", "금액", "amount"]
                    ):
                        try:
                            premium_analysis[col] = {
                                "mean": float(df[col].mean()),
                                "min": float(df[col].min()),
                                "max": float(df[col].max()),
                                "std": float(df[col].std()),
                            }
                        except:
                            pass
            except:
                pass

            # 분석 결과 구성
            analysis_result = f"""
데이터 통계 분석 결과:

## 기본 정보
- 데이터 형태: {df.shape}
- 컬럼 수: {len(df.columns)}

## 수치형 데이터 요약
{df.describe().to_string() if df.select_dtypes(include=[np.number]).shape[1] > 0 else '수치형 데이터 없음'}

## 보험사별 현황
{chr(10).join([f'- {k}: {v}' for k, v in company_summary.items()]) if company_summary else '보험사 정보 없음'}

## 보험료 관련 통계
{chr(10).join([f'- {k}: 평균 {v["mean"]:,.0f}, 최소 {v["min"]:,.0f}, 최대 {v["max"]:,.0f}' for k, v in premium_analysis.items()]) if premium_analysis else '보험료 정보 없음'}

## 분석 제안
고객 질문에 따라 다음과 같은 추가 분석이 가능합니다:
1. 특정 보험사별 상세 비교
2. 보험료 수준별 순위 분석
3. 보장 항목별 차이점 분석
"""

            duration = time.time() - start_time
            logger.info(f"Fallback 분석 완료 - 소요 시간: {duration:.2f}초")

            return {
                "status": "success",
                "analysis": analysis_result,
                "steps": [("fallback_analysis", "통계적 분석 수행")],
                "duration": duration,
                "mode": "fallback",
            }

        except Exception as e:
            logger.error(f"Fallback 분석 오류: {type(e).__name__}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _validate_agent_result(self, result: Any) -> Dict[str, Any]:
        """Agent 실행 결과 유효성 검사 및 표준화"""
        try:
            # 결과 형식 검사
            if not isinstance(result, dict):
                logger.warning(f"Agent 결과가 dict 형식이 아님: {type(result)}")
                return {"status": "invalid_format", "message": "결과 형식 오류"}

            # output 필드 검사
            if "output" not in result:
                logger.warning("Agent 결과에 output 필드 없음")
                logger.debug(f"Agent 결과 키: {list(result.keys())}")
                return {"status": "missing_output", "message": "Output 필드 누락"}

            agent_output = result["output"]

            # output 내용 검사
            if not agent_output or not str(agent_output).strip():
                logger.warning("Agent output이 비어있음")
                return {"status": "empty_output", "message": "분석 결과 없음"}

            # Agent output이 리스트인 경우 문자열로 변환
            output_str = ""
            if isinstance(agent_output, list):
                # 리스트의 각 항목에서 text 필드 추출
                for item in agent_output:
                    if isinstance(item, dict) and "text" in item:
                        output_str += item["text"]
                    elif isinstance(item, str):
                        output_str += item
            else:
                output_str = str(agent_output)
            if len(output_str.strip()) < 5:
                logger.warning(f"Agent output이 너무 짧음: {len(output_str)}자")
                return {"status": "too_short", "message": "분석 결과가 너무 짧음"}

            # 한국어 내용 검사
            if not any(ord(char) > 127 for char in output_str):
                logger.warning("Agent output에 한국어 내용 없음")
                # 영어만 있어도 성공으로 처리 (fallback 방지)
                logger.info("영어 응답이지만 성공으로 처리")

            # 최종 유효성 검사 통과
            steps = result.get("intermediate_steps", [])

            return {
                "status": "success",
                "analysis": output_str,
                "steps": steps,
                "intermediate_steps_count": len(steps),
                "output_length": len(output_str),
                "result_keys": list(result.keys()),
                "validation_time": time.time(),
            }

        except Exception as e:
            logger.error(f"Agent 결과 검증 중 오류: {type(e).__name__}: {str(e)}")
            return {
                "status": "validation_error",
                "message": f"결과 검증 실패: {str(e)}",
            }

    def _execute_agent_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """LangChain agent를 통한 데이터 분석 실행 - 개선된 버전"""
        start_time = time.time()

        # LangChain 사용 가능 여부 체크
        if not LANGCHAIN_AVAILABLE:
            logger.info("LangChain을 사용할 수 없어 Fallback 모드로 전환")
            return self._execute_fallback_analysis(df, query)

        try:
            # Agent 생성
            logger.info("LangChain Pandas Agent 생성 시작")
            agent = self._create_pandas_agent(df)

            if agent is None:
                logger.warning("Agent 생성 실패 - Fallback 모드로 전환")
                return self._execute_fallback_analysis(df, query)

            # 보험 특화 프롬프트
            df_info = self._extract_data_info(df)
            analysis_prompt = self._generate_insurance_prompt(query, df_info)

            logger.info(f"Agent 분석 실행 시작 - 프롬프트 길이: {len(analysis_prompt)}")

            # Agent 실행 타임아웃 설정
            agent_start_time = time.time()
            result = agent.invoke({"input": analysis_prompt})
            agent_execution_time = time.time() - agent_start_time

            logger.info(f"Agent 실행 완료 - 소요 시간: {agent_execution_time:.2f}초")

            # 결과 유효성 검사 및 표준화
            validated_result = self._validate_agent_result(result)

            if validated_result["status"] == "success":
                # 성공 로깅
                logger.info(
                    f"Agent 분석 성공 - 결과 길이: {validated_result['output_length']}자, "
                    f"단계 수: {validated_result['intermediate_steps_count']}, "
                    f"실행 시간: {agent_execution_time:.2f}초"
                )

                # 상세 분석 단계 로깅 (선택적)
                if validated_result["intermediate_steps_count"] > 0:
                    logger.debug("=== Agent 분석 단계 상세 ===")
                    for i, step in enumerate(
                        validated_result["steps"][:3]
                    ):  # 처음 3단계만 로깅
                        logger.debug(f"단계 {i+1}: {str(step)[:100]}...")

                # 표준화된 결과 반환
                return {
                    "status": "success",
                    "analysis": validated_result["analysis"],
                    "steps": validated_result["steps"],
                    "duration": time.time() - start_time,
                    "mode": "langchain_agent",
                    "execution_time": agent_execution_time,
                    "validation_info": {
                        "output_length": validated_result["output_length"],
                        "steps_count": validated_result["intermediate_steps_count"],
                    },
                }
            else:
                # 유효성 검사 실패 - 상세 로그와 함께 fallback
                logger.error(
                    f"Agent 결과 유효성 검사 실패: {validated_result['status']} - {validated_result['message']}"
                )
                logger.debug(f"Agent 원본 결과: {str(result)[:500]}...")

                # 특정 실패 유형별 처리
                if validated_result["status"] in ["too_short", "empty_output"]:
                    logger.warning(
                        "Agent가 불충분한 응답을 제공 - Fallback 모드로 전환"
                    )
                elif validated_result["status"] in ["invalid_format", "missing_output"]:
                    logger.error("Agent 결과 형식 오류 - Fallback 모드로 전환")
                else:
                    logger.warning(
                        f"기타 유효성 검사 실패: {validated_result['message']}"
                    )

                return self._execute_fallback_analysis(df, query)

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"Agent 실행 치명적 오류: {type(e).__name__}: {str(e)} - "
                f"소요 시간: {error_time:.2f}초 - Fallback 모드로 전환"
            )

            # 에러 타입별 상세 처리
            error_str = str(e).lower()
            if "timeout" in error_str or "시간 초과" in error_str:
                logger.error("Agent 실행 시간 초과")
            elif "parsing" in error_str or "파싱" in error_str:
                logger.error("Agent 출력 파싱 오류")
            elif "memory" in error_str or "메모리" in error_str:
                logger.error("Agent 실행 메모리 오류")

            return self._execute_fallback_analysis(df, query)

    def _generate_final_analysis(
        self, agent_result: Dict, query: str, df: pd.DataFrame
    ) -> str:
        """최종 LLM을 통한 종합 분석"""
        start_time = time.time()

        try:
            if agent_result["status"] != "success":
                return f"데이터 분석 중 오류 발생: {agent_result['message']}"

            # 최종 종합 프롬프트
            final_prompt = f"""
다음은 LangChain pandas agent의 보험 데이터 분석 결과입니다:

{agent_result['analysis']}

고객 질문: {query}

위 분석 결과를 바탕으로, 보험 전문가로서 다음 내용을 포함하여 종합적으로 답변해주세요:

1. **핵심 분석 내용 요약**: 가장 중요한 분석 결과를 간결하게 요약
2. **보험사별 특징 비교**: 각 보험사의 장점, 단점, 차이점 명확히 비교  
3. **수치 기반 추천**: 구체적인 금액과 데이터를 근거로 추천
4. **실질적인 조언**: 고객의 입장에서 실질적으로 도움이 될 정보 제공

답변 형식:
- 명확하고 이해하기 쉬운 한국어 사용
- 구체적인 수치 데이터 포함  
- 불렛 포인트나 번호로 구조화
- 전문가적이면서 친절한 톤
"""

            logger.info(f"최종 분석 생성 시작 - LLM 호출")

            # 최종 LLM 응답 생성
            response = self.llm(model="gemini-3-flash-preview", contents=[final_prompt])

            result_text = "분석 결과 생성 실패"
            if response and hasattr(response, "text") and response.text:
                result_text = response.text

            return result_text

        except Exception as e:
            logger.error(f"최종 분석 생성 오류: {type(e).__name__}: {str(e)}")
            return f"최종 분석 생성 중 오류 발생: {str(e)}"

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 메모리 최적화"""
        try:
            original_memory = df.memory_usage(deep=True).sum()

            # 수치형 데이터 최적화
            for col in df.select_dtypes(include=["int64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")

            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

            # 문자열 데이터 최적화
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].nunique() / len(df) < 0.5:  # 카디널리티가 낮은 경우
                    df[col] = df[col].astype("category")

            optimized_memory = df.memory_usage(deep=True).sum()
            memory_reduction = (
                (original_memory - optimized_memory) / original_memory * 100
            )

            logger.info(
                f"메모리 최적화 완료 - {original_memory/1024/1024:.2f}MB → {optimized_memory/1024/1024:.2f}MB ({memory_reduction:.1f}% 감소)"
            )

            return df

        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {e}")
            return df

    def pandas_analysis(
        self,
        df: pd.DataFrame,
        query: str,
        comparison_table: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        개선된 보험료 데이터 분석 - LangChain pandas agent 통합
        가변 단계 구조: 1) 데이터 준비 → 2) Pandas Agent 분석 → 3) 최종 LLM 종합 (선택적)
        """
        start_time = time.time()
        logger.info(f"=== {self.pandas_analysis_stages}단계 Pandas 분석 시작 ===")
        logger.info(
            f"쿼리: '{query}', DataFrame 형태: {df.shape if df is not None else 'None'}"
        )

        try:
            # 입력 데이터 검증
            if df is None or df.empty:
                logger.warning("분석할 데이터가 없습니다")
                return "분석할 데이터가 없습니다."

            # 1단계: 데이터 준비
            prep_start = time.time()
            logger.info("1단계: 데이터 준비 시작")

            # 판다스 에이전트 분석을 위해 aggregated_df 사용 (보장금액 정보 보존)
            if comparison_table is not None and not comparison_table.empty:
                # comparison_table이 있더라도 aggregated_df를 우선 사용
                logger.info(
                    "comparison_table이 있지만 aggregated_df로 분석 진행 (보장금액 정보 보존)"
                )
                try:
                    from data_manager import data_manager

                    if (
                        data_manager.coverage_premiums_df is not None
                        and not data_manager.coverage_premiums_df.empty
                    ):
                        # 동적으로 aggregated_df 생성
                        logger.info("보험료 데이터 정규화 시작")
                        normalized_df = data_manager.normalize_coverage_amounts(
                            data_manager.coverage_premiums_df
                        )

                        logger.info("보험사별 데이터 집계 시작")
                        analysis_df = data_manager.aggregate_coverage_by_code(
                            normalized_df
                        )
                        data_type = "집계된 데이터프레임 (aggregated_df)"
                        logger.info(
                            f"집계된 데이터프레임 생성 완료 - 형태: {analysis_df.shape}"
                        )
                    else:
                        analysis_df = df
                        data_type = "원본 데이터"
                        logger.info("원본 데이터로 분석 진행")
                except Exception as e:
                    logger.error(f"집계된 데이터프레임 생성 실패: {e}")
                    analysis_df = df
                    data_type = "원본 데이터(집계 실패)"
            else:
                # 비교 표가 없는 경우 aggregated_df 생성 시도
                logger.info("동적 집계 데이터프레임 생성 시도")
                try:
                    from data_manager import data_manager

                    if (
                        data_manager.coverage_premiums_df is not None
                        and not data_manager.coverage_premiums_df.empty
                    ):
                        # 동적으로 aggregated_df 생성
                        logger.info("보험료 데이터 정규화 시작")
                        normalized_df = data_manager.normalize_coverage_amounts(
                            data_manager.coverage_premiums_df
                        )

                        logger.info("보험사별 데이터 집계 시작")
                        analysis_df = data_manager.aggregate_coverage_by_code(
                            normalized_df
                        )
                        data_type = "동적 생성 집계 데이터프레임"
                        logger.info(
                            f"동적 집계 데이터프레임 생성 완료 - 형태: {analysis_df.shape}"
                        )
                    else:
                        analysis_df = df
                        data_type = "원본 데이터"
                        logger.info("원본 데이터로 분석 진행")
                except Exception as e:
                    logger.error(f"동적 집계 데이터프레임 생성 실패: {e}")
                    analysis_df = df
                    data_type = "원본 데이터(집계 실패)"

            # 메모리 최적화
            analysis_df = self._optimize_dataframe_memory(analysis_df)

            prep_time = time.time() - prep_start
            logger.info(
                f"1단계 완료: 데이터 준비 - 유형: {data_type}, 소요 시간: {prep_time:.2f}초"
            )

            # 2단계: LangChain Pandas Agent 분석
            logger.info("2단계: Pandas Agent 분석 시작")
            agent_start = time.time()

            agent_result = self._execute_agent_analysis(analysis_df, query)

            agent_time = time.time() - agent_start
            logger.info(
                f"2단계 완료: Agent 분석 - 상태: {agent_result['status']}, 소요 시간: {agent_time:.2f}초"
            )

            # 3단계: 최종 LLM 종합 분석 (단계 수 설정에 따라 실행)

            # Agent 결과 안전 추출 - 새로운 유효성 검사 결과 구조 반영
            if agent_result.get("status") == "success":
                final_result = agent_result.get("analysis", "")
                # 성공 시 추가 정보 로깅
                validation_info = agent_result.get("validation_info", {})
                if validation_info:
                    logger.info(
                        f"Agent 분석 상세 - 길이: {validation_info.get('output_length', 0)}자, "
                        f"단계: {validation_info.get('steps_count', 0)}개"
                    )
            else:
                # Agent 실패 시 fallback 결과 사용
                logger.warning(
                    f"Agent 분석 실패: {agent_result.get('status', 'unknown')}"
                )
                final_result = agent_result.get(
                    "analysis", "분석 결과를 가져올 수 없습니다."
                )

            if self.pandas_analysis_stages >= 3:
                logger.info("3단계: 최종 LLM 종합 분석 시작")
                final_start = time.time()

                final_result = self._generate_final_analysis(
                    agent_result, query, analysis_df
                )

                final_time = time.time() - final_start
                logger.info(f"3단계 완료: 최종 분석 - 소요 시간: {final_time:.2f}초")
            else:
                logger.info(
                    f"3단계 건너뜀 - 설정된 단계 수: {self.pandas_analysis_stages}"
                )

            total_time = time.time() - start_time
            logger.info(
                f"=== 전체 분석 완료 ({self.pandas_analysis_stages}단계) === 총 소요 시간: {total_time:.2f}초, 결과 길이: {len(final_result)}자"
            )

            return final_result

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"pandas_analysis 치명적 오류: {type(e).__name__}: {str(e)}")
            logger.error(f"오류 발생 시점: {error_time:.2f}초")
            logger.error(f"데이터 정보: shape={df.shape if df is not None else 'None'}")

            # 사용자 친화적 에러 메시지
            error_message = self._generate_user_friendly_error(e)
            return error_message

    def _generate_user_friendly_error(self, error: Exception) -> str:
        """사용자 친화적 에러 메시지 생성"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "timeout" in error_str or "시간" in error_str:
            return "⏰ 분석 시간이 초과되었습니다. 데이터가 너무 많거나 복잡할 수 있습니다. 다시 시도해주세요."
        elif "memory" in error_str or "메모리" in error_str:
            return "💾 데이터가 너무 많아 메모리가 부족합니다. 일부 데이터만 다시 분석해주세요."
        elif "api" in error_str or "연결" in error_str:
            return "🔌 외부 API 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        elif "parse" in error_str or "형식" in error_str:
            return (
                "📋 데이터 형식에 문제가 있습니다. 데이터를 확인하고 다시 시도해주세요."
            )
        else:
            return f"❌ 분석 중 오류가 발생했습니다: {str(error)} (오류 타입: {error_type})"

    def hybrid_chat(
        self, query: str, df: pd.DataFrame, insurance_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Hybrid RAG 시스템을 사용한 종합적인 질의응답 - 비교 표 활용
        """
        start_time = time.time()
        logger.info(f"Hybrid RAG 챗 시작 - 쿼리: '{query}'")
        logger.info(
            f"입력 데이터 - DataFrame 형태: {df.shape if df is not None else 'None'}, 보험 데이터 수: {len(insurance_data) if insurance_data else 0}"
        )

        try:
            # 1. 벡터 검색을 통한 관련 문서 검색
            logger.info("1단계: 벡터 검색 시작")
            search_start = time.time()
            relevant_docs = self.search_relevant_docs(query)
            search_time = time.time() - search_start
            logger.info(
                f"1단계 완료: 벡터 검색 - 찾은 문서 수: {len(relevant_docs)}, 소요 시간: {search_time:.2f}초"
            )

            # 2. 비교 표 생성 및 Pandas 데이터 분석
            pandas_result = ""
            comparison_table = None

            if df is not None and not df.empty:
                # 집계 데이터프레임 생성 시도
                from data_manager import data_manager

                try:
                    if (
                        data_manager.coverage_premiums_df is not None
                        and not data_manager.coverage_premiums_df.empty
                    ):
                        # 동적으로 집계 데이터프레임 생성 (보장금액 정보 보존)
                        logger.info("하이브리드 챗에서 집계 데이터프레임 생성 시작")
                        normalized_df = data_manager.normalize_coverage_amounts(
                            data_manager.coverage_premiums_df
                        )
                        aggregated_df = data_manager.aggregate_coverage_by_code(
                            normalized_df
                        )
                        logger.info(
                            "집계 데이터프레임 생성 완료 - pandas_analysis 호출"
                        )

                        # 집계 데이터프레임을 사용한 분석
                        pandas_result = self.pandas_analysis(df, query, aggregated_df)
                    else:
                        pandas_result = self.pandas_analysis(df, query)
                except Exception as e:
                    logger.warning(
                        f"Failed to create aggregated dataframe for analysis: {e}"
                    )
                    pandas_result = self.pandas_analysis(df, query)

            # 3. 종합 응답 생성
            if relevant_docs and pandas_result:
                # 두 가지 결과 모두 있는 경우
                # Simple document-based QA using LLM directly
                context = "\n".join([doc["page_content"] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-flash-preview", contents=[prompt]
                ).text
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}\n\n📋 **보장내용 검색 결과:**\n{qa_result}"""

            elif relevant_docs:
                # 문서 검색 결과만 있는 경우
                context = "\n".join([doc["page_content"] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-flash-preview", contents=[prompt]
                ).text
                combined_response = f"""📋 **보장내용 검색 결과:**\n{qa_result}"""

            elif pandas_result:
                # Pandas 분석 결과만 있는 경우
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}"""

            else:
                combined_response = "죄송합니다. 관련 정보를 찾을 수 없습니다."

            return {
                "response": combined_response,
                "sources_found": len(relevant_docs) > 0,
                "data_analysis_available": df is not None and not df.empty,
                "source_count": len(relevant_docs),
            }

        except Exception as e:
            logger.error(f"Error in hybrid chat: {e}")
            return {
                "response": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                "sources_found": False,
                "data_analysis_available": False,
                "source_count": 0,
            }

    async def hybrid_chat_stream(
        self, query: str, df: pd.DataFrame, insurance_data: List[Dict[str, Any]]
    ):
        """
        스트리밍 방식의 Hybrid RAG 시스템 - 단계별 진행 상태 전송
        """
        start_time = time.time()
        logger.info(f"[STREAM START] Streaming Hybrid RAG 챗 시작 - 쿼리: '{query}'")

        # 진행률 가중치 정의 (preparing을 searching에 통합)
        PROGRESS_WEIGHTS = {
            "searching": 0.25,  # 벡터 검색 + 데이터 준비 25%
            "analyzing": 0.60,  # Pandas 분석 60%
            "finalizing": 0.15,  # 최종 정리 15%
        }

        try:
            logger.info(f"[STREAM] 스트리밍 시작 - 쿼리: '{query}'")

            # 1. 벡터 검색 단계
            logger.info("[STREAM] 1단계: 벡터 검색 시작")
            chunk1 = {
                "status": "searching",
                "message": "🔍 벡터 검색 중...",
                "progress": 0.0,
                "timestamp": time.time(),
            }
            logger.info(f"[STREAM YIELD] 첫 번째 청크 전송: {chunk1}")
            yield chunk1

            # 약간의 지연을 주어 청크가 전송되도록 함
            await asyncio.sleep(0.1)

            search_start = time.time()
            relevant_docs = self.search_relevant_docs(query)
            search_time = time.time() - search_start

            logger.info(
                f"[STREAM] 벡터 검색 완료 - 찾은 문서 수: {len(relevant_docs)}, 소요 시간: {search_time:.2f}초"
            )

            # 벡터 검색 완료 상태 전송
            chunk2 = {
                "status": "searching",
                "message": f"🔍 벡터 검색 완료 ({len(relevant_docs)}개 문서 발견)",
                "progress": PROGRESS_WEIGHTS["searching"] * 100,
                "timestamp": time.time(),
                "doc_count": len(relevant_docs),
            }
            logger.info(f"[STREAM YIELD] 벡터 검색 완료 청크 전송: {chunk2}")
            yield chunk2
            await asyncio.sleep(0.1)

            # 2. 데이터 준비 단계 (searching 상태로 통합)
            logger.info("[STREAM] 2단계: 데이터 준비 시작")
            chunk3 = {
                "status": "searching",
                "message": "📋 데이터 준비 중...",
                "progress": PROGRESS_WEIGHTS["searching"] * 100,
                "timestamp": time.time(),
            }
            logger.info(f"[STREAM YIELD] 데이터 준비 시작 청크 전송: {chunk3}")
            yield chunk3
            await asyncio.sleep(0.1)

            pandas_result = ""
            aggregated_df = df  # 기본값으로 df 설정

            if df is not None and not df.empty:
                # 데이터프레임 준비
                from data_manager import data_manager

                try:
                    if (
                        data_manager.coverage_premiums_df is not None
                        and not data_manager.coverage_premiums_df.empty
                    ):
                        logger.info("[STREAM] 집계 데이터프레임 생성 시작")
                        normalized_df = data_manager.normalize_coverage_amounts(
                            data_manager.coverage_premiums_df
                        )
                        aggregated_df = data_manager.aggregate_coverage_by_code(
                            normalized_df
                        )
                        logger.info("[STREAM] 집계 데이터프레임 생성 완료")
                except Exception as e:
                    logger.warning(f"[STREAM] 집계 데이터프레임 생성 실패: {e}")
                    aggregated_df = df  # 실패 시 원본 df 사용

            # 데이터 준비 완료 상태 전송
            chunk4 = {
                "status": "searching",
                "message": "📋 데이터 준비 완료",
                "progress": PROGRESS_WEIGHTS["searching"] * 100,
                "timestamp": time.time(),
                "data_shape": (
                    aggregated_df.shape if aggregated_df is not None else None
                ),
            }
            logger.info(f"[STREAM YIELD] 데이터 준비 완료 청크 전송: {chunk4}")
            yield chunk4
            await asyncio.sleep(0.1)

            # 3. Pandas 분석 단계 (단순화)
            if aggregated_df is not None and not aggregated_df.empty:
                logger.info("[STREAM] 3단계: Pandas 분석 시작")
                # 분석 시작 상태 전송
                chunk5 = {
                    "status": "analyzing",
                    "message": "📊 Pandas 분석 중...",
                    "progress": PROGRESS_WEIGHTS["searching"] * 100,
                    "timestamp": time.time(),
                }
                logger.info(f"[STREAM YIELD] Pandas 분석 시작 청크 전송: {chunk5}")
                yield chunk5
                await asyncio.sleep(0.1)

                # 실제 Pandas 분석 실행
                try:
                    logger.info("[STREAM] 실제 pandas_analysis 호출 시작")
                    pandas_result = self.pandas_analysis(df, query, aggregated_df)
                    logger.info(
                        f"[STREAM] Pandas 분석 완료 - 결과 길이: {len(pandas_result) if pandas_result else 0}"
                    )
                except Exception as e:
                    logger.error(f"[STREAM] Pandas 분석 실패: {e}")
                    pandas_result = f"데이터 분석 중 오류 발생: {str(e)}"

                # 분석 완료 상태 전송
                chunk6 = {
                    "status": "analyzing",
                    "message": "📊 Pandas 분석 완료",
                    "progress": (
                        PROGRESS_WEIGHTS["searching"] + PROGRESS_WEIGHTS["analyzing"]
                    )
                    * 100,
                    "timestamp": time.time(),
                    "result_length": len(pandas_result) if pandas_result else 0,
                }
                logger.info(f"[STREAM YIELD] Pandas 분석 완료 청크 전송: {chunk6}")
                yield chunk6
                await asyncio.sleep(0.1)

            # 4. 최종 응답 생성
            logger.info("[STREAM] 4단계: 최종 응답 생성 시작")
            chunk7 = {
                "status": "finalizing",
                "message": "🤖 최종 응답 생성 중...",
                "progress": (
                    PROGRESS_WEIGHTS["searching"]
                    + PROGRESS_WEIGHTS["analyzing"]
                    + PROGRESS_WEIGHTS["finalizing"] * 0.5
                )
                * 100,
                "timestamp": time.time(),
            }
            logger.info(f"[STREAM YIELD] 최종 응답 생성 시작 청크 전송: {chunk7}")
            yield chunk7
            await asyncio.sleep(0.1)

            # 최종 응답 생성
            logger.info("[STREAM] 최종 LLM 응답 생성 시작")
            if relevant_docs and pandas_result:
                context = "\n".join([doc["page_content"] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-pro-preview", contents=[prompt]
                ).text
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}\n\n📋 **보장내용 검색 결과:**\n{qa_result}"""
            elif relevant_docs:
                context = "\n".join([doc["page_content"] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-pro-preview", contents=[prompt]
                ).text
                combined_response = f"""📋 **보장내용 검색 결과:**\n{qa_result}"""
            elif pandas_result:
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}"""
            else:
                combined_response = "죄송합니다. 관련 정보를 찾을 수 없습니다."

            total_time = time.time() - start_time
            logger.info(
                f"[STREAM COMPLETE] Streaming Hybrid RAG 완료 - 총 소요 시간: {total_time:.2f}초, 응답 길이: {len(combined_response)}"
            )

            # 최종 완료 상태 전송
            chunk8 = {
                "status": "complete",
                "message": "✅ 분석 완료!",
                "progress": 100.0,
                "response": combined_response,
                "sources_found": len(relevant_docs) > 0,
                "data_analysis_available": df is not None and not df.empty,
                "source_count": len(relevant_docs),
                "total_time": total_time,
                "timestamp": time.time(),
            }
            logger.info(f"[STREAM YIELD] 최종 완료 청크 전송: {chunk8}")
            yield chunk8

        except Exception as e:
            logger.error(
                f"[STREAM ERROR] Streaming Hybrid RAG 오류: {type(e).__name__}: {e}"
            )
            error_chunk = {
                "status": "error",
                "message": f"❌ 처리 중 오류 발생: {str(e)}",
                "progress": 100.0,
                "timestamp": time.time(),
            }
            logger.info(f"[STREAM YIELD] 에러 청크 전송: {error_chunk}")
            yield error_chunk


# 전역 Hybrid RAG 시스템 인스턴스
rag_system = HybridRAGSystem()
