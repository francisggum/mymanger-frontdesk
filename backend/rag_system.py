import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from typing import List, Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class HybridRAGSystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        self.vector_store = None
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")
        
    def initialize_vector_store(self, insurance_data: List[Dict[str, Any]]):
        """
        product_insur_premiums 데이터로 ChromaDB 벡터 저장소 초기화
        """
        try:
            if not insurance_data:
                logger.warning("No insurance data provided for vector store initialization")
                return False
            
            # 보장내용 텍스트 추출 및 Document 객체 생성
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            for item in insurance_data:
                # insur_bojang(보장설명) 텍스트 추출
                bojang_text = item.get("insur_bojang", "")
                if bojang_text:
                    # 메타데이터 포함 Document 생성
                    doc = Document(
                        page_content=bojang_text,
                        metadata={
                            "plan_id": item.get("plan_id", ""),
                            "insur_name": item.get("insur_name", ""),
                            "insur_code": item.get("insur_code", ""),
                            "premium_amount": item.get("premium_amount", 0)
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                logger.warning("No valid documents created from insurance data")
                return False
            
            # 텍스트 분할
            texts = text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks from {len(documents)} documents")
            
            # ChromaDB 벡터 저장소 생성
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name="insurance_coverage"
            )
            
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def search_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """
        쿼리와 관련된 문서 검색
        """
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} relevant documents for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def pandas_analysis(self, df: pd.DataFrame, query: str) -> str:
        """
        Pandas DataFrame을 사용한 데이터 분석
        """
        try:
            if df is None or df.empty:
                return "No coverage data available for analysis."
            
            # Pandas Agent 생성
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                agent_type="zero-shot-react-description"
            )
            
            # 쿼리 실행
            result = agent.run(query)
            logger.info(f"Pandas analysis completed for query: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Error in pandas analysis: {e}")
            return f"Error in data analysis: {str(e)}"
    
    def hybrid_chat(self, query: str, df: pd.DataFrame, insurance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hybrid RAG 시스템을 사용한 종합적인 질의응답
        """
        try:
            # 1. 벡터 검색을 통한 관련 문서 검색
            relevant_docs = self.search_relevant_docs(query)
            
            # 2. Pandas 데이터 분석
            pandas_result = ""
            if df is not None and not df.empty:
                pandas_result = self.pandas_analysis(df, query)
            
            # 3. 종합 응답 생성
            if relevant_docs and pandas_result:
                # 두 가지 결과 모두 있는 경우
                qa_result = self.qa_chain.run(
                    input_documents=relevant_docs,
                    question=query
                )
                
                combined_response = f"""📊 **데이터 분석 결과:**\n{pandas_result}\n\n📋 **보장내용 검색 결과:**\n{qa_result}"""
                
            elif relevant_docs:
                # 문서 검색 결과만 있는 경우
                qa_result = self.qa_chain.run(
                    input_documents=relevant_docs,
                    question=query
                )
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