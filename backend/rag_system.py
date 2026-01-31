import chromadb
import google.genai as genai
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class HybridRAGSystem:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embedding_model = self.client.models.embed_content
        self.llm = self.client.models.generate_content
        self.vector_store = None
        self.qa_chain = None

        
    def initialize_vector_store(self, insurance_data: List[Dict[str, Any]]):
        """
        product_insur_premiums 데이터로 ChromaDB 벡터 저장소 초기화
        """
        try:
            if not insurance_data:
                logger.warning("No insurance data provided for vector store initialization")
                return False
            
            # ChromaDB 클라이언트 생성
            self.vector_store = chromadb.Client()
            
            # 기존 컬렉션이 있으면 삭제
            try:
                self.vector_store.delete_collection("insurance_coverage")
            except:
                pass
                
            self.collection = self.vector_store.create_collection(
                name="insurance_coverage",
                metadata={"hnsw:space": "cosine"}
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
                    metadatas.append({
                        "plan_id": item.get("plan_id", ""),
                        "insur_name": item.get("insur_name", ""),
                        "insur_code": item.get("insur_code", ""),
                        "premium_amount": str(item.get("premium_amount", 0))
                    })
                    ids.append(f"doc_{i}")
            
            if not documents:
                logger.warning("No valid documents created from insurance data")
                return False
            
            # 임베딩 생성 및 저장
            result = self.embedding_model(
                model="gemini-embedding-001",
                contents=documents
            )
            embeddings = [emb.values for emb in result.embeddings]
            
            # ChromaDB에 저장
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Vector store initialized successfully with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def search_relevant_docs(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 문서 검색
        """
        try:
            if not hasattr(self, 'collection'):
                logger.warning("Vector store not initialized")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model(
                model="gemini-embedding-001",
                contents=[query]
            )
            
            # 검색
            results = self.collection.query(
                query_embeddings=[query_embedding.embeddings[0].values],
                n_results=k
            )
            
            # 결과 포맷팅
            docs = []
            for i in range(len(results['documents'][0])):
                docs.append({
                    'page_content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
            
            logger.info(f"Found {len(docs)} relevant documents for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def pandas_analysis(self, df: pd.DataFrame, query: str, comparison_table: Optional[pd.DataFrame] = None) -> str:
        """
        보험료 데이터 분석 - 비교 표 우선 활용
        """
        try:
            if df is None or df.empty:
                return "데이터가 없습니다."
            
            # 비교 표가 있는 경우 비교 표를 우선적으로 분석
            if comparison_table is not None and not comparison_table.empty:
                analysis_df = comparison_table
                data_type = "보험사별 비교 표"
            else:
                # 비교 표가 없는 경우 원본 데이터로 비교 표 생성
                from data_manager import data_manager
                
                if data_manager.coverage_premiums_df is not None and not data_manager.coverage_premiums_df.empty:
                    # 동적으로 비교 표 생성
                    normalized_df = data_manager.normalize_coverage_amounts(data_manager.coverage_premiums_df)
                    aggregated_df = data_manager.aggregate_coverage_by_code(normalized_df)
                    analysis_df = data_manager.create_comparison_table(aggregated_df)
                    data_type = "생성된 비교 표"
                else:
                    analysis_df = df
                    data_type = "원본 데이터"
            
            # 데이터 분석 프롬프트
            prompt = f"""
            다음 {data_type}를 분석하여 질문에 답변해주세요:
            
            질문: {query}
            
            데이터 구조:
            - Shape: {analysis_df.shape}
            - Columns: {list(analysis_df.columns)}
            - Index: {list(analysis_df.index)}
            
            데이터 샘플:
            {analysis_df.head(10).to_string()}
            
            주요 통계:
            보험사별 평균 보험료:
            {str(analysis_df.mean()) if not analysis_df.empty else '데이터 없음'}
            
            분석 가이드:
            1. 보험료 비교 시 가장 저렴한 보험사를 추천해주세요
            2. 특정 보장 항목(암진단비, 상해보장 등)에 대해 비교 분석해주세요
            3. 보험료 합계를 기준으로 순위를 매겨주세요
            4. 각 보험사의 특징과 장단점을 분석해주세요
            """
            
            response = self.llm(
                model="gemini-3-pro-preview",
                contents=[prompt]
            )
            result = response.text if response and hasattr(response, 'text') else "분석 결과를 생성할 수 없습니다."
            
            return result or "분석 결과를 생성할 수 없습니다."
            
        except Exception as e:
            logger.error(f"Error in pandas analysis: {e}")
            return f"Error in data analysis: {str(e)}"
    
    def hybrid_chat(self, query: str, df: pd.DataFrame, insurance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hybrid RAG 시스템을 사용한 종합적인 질의응답 - 비교 표 활용
        """
        try:
            # 1. 벡터 검색을 통한 관련 문서 검색
            relevant_docs = self.search_relevant_docs(query)
            
            # 2. 비교 표 생성 및 Pandas 데이터 분석
            pandas_result = ""
            comparison_table = None
            
            if df is not None and not df.empty:
                # 비교 표 생성 시도
                from data_manager import data_manager
                
                try:
                    if data_manager.coverage_premiums_df is not None and not data_manager.coverage_premiums_df.empty:
                        # 동적으로 비교 표 생성
                        normalized_df = data_manager.normalize_coverage_amounts(data_manager.coverage_premiums_df)
                        aggregated_df = data_manager.aggregate_coverage_by_code(normalized_df)
                        comparison_table = data_manager.create_comparison_table(aggregated_df)
                        
                        # 비교 표를 사용한 분석
                        pandas_result = self.pandas_analysis(df, query, comparison_table)
                    else:
                        pandas_result = self.pandas_analysis(df, query)
                except Exception as e:
                    logger.warning(f"Failed to create comparison table for analysis: {e}")
                    pandas_result = self.pandas_analysis(df, query)
            
            # 3. 종합 응답 생성
            if relevant_docs and pandas_result:
                # 두 가지 결과 모두 있는 경우
                # Simple document-based QA using LLM directly
                context = "\n".join([doc['page_content'] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-pro-preview",
                    contents=[prompt]
                ).text
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}\n\n📋 **보장내용 검색 결과:**\n{qa_result}"""
                
            elif relevant_docs:
                # 문서 검색 결과만 있는 경우
                context = "\n".join([doc['page_content'] for doc in relevant_docs])
                prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
                qa_result = self.llm(
                    model="gemini-3-pro-preview",
                    contents=[prompt]
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
                "source_count": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid chat: {e}")
            return {
                "response": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                "sources_found": False,
                "data_analysis_available": False,
                "source_count": 0
            }

# 전역 Hybrid RAG 시스템 인스턴스
rag_system = HybridRAGSystem()