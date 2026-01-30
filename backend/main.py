from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from api_client import api_client
from data_manager import data_manager
from rag_system import rag_system

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

@app.get("/")
async def root():
    return {"message": "Insurance Comparison AI API is running"}

@app.post("/fetch-plans", response_model=List[PlanInfo])
async def fetch_plans(request: JWTRequest):
    try:
        plans = await api_client.fetch_plans(request.jwt_token)
        return [PlanInfo(**plan) for plan in plans]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-data")
async def load_data(request: LoadDataRequest):
    try:
        # 외부 API에서 데이터 가져오기
        premium_data = await api_client.fetch_product_premiums(
            request.jwt_token,
            request.plan_id,
            request.age,
            request.gender
        )
        
        # 데이터 처리 및 저장
        result = data_manager.process_premium_data(
            premium_data,
            request.plan_id,
            request.age,
            request.gender
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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 현재 세션 정보 확인
        session_info = data_manager.get_current_session_info()
        
        if not session_info["has_coverage_data"] and not session_info["has_insurance_data"]:
            raise HTTPException(
                status_code=400, 
                detail="No data loaded. Please load data first using /load-data endpoint."
            )
        
        # 데이터 가져오기
        coverage_df = data_manager.get_coverage_dataframe()
        insurance_data = data_manager.get_insurance_data()
        
        # Hybrid RAG 시스템으로 질의응답
        result = rag_system.hybrid_chat(request.query, coverage_df, insurance_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)