import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class ExternalAPIClient:
    def __init__(self):
        self.base_url = "https://mmlfcp.ohmymanager.com/api"
        self.timeout = 30.0
    
    async def fetch_plans(self, jwt_token: str) -> List[Dict[str, str]]:
        """
        API 1: 플랜 목록 조회
        JWT 토큰 유효성 검증 및 플랜 목록 반환
        """
        url = f"{self.base_url}/Auth"
        params = {
            "token": jwt_token,
            "access_path": "MMLFCP_WEB"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # 응답 형식에 따라 파싱 (예상: [{"plan_id": "...", "plan_name": "..."}, ...])
                if isinstance(data, list):
                    return [
                        {
                            "plan_id": str(item.get("plan_id", "")),
                            "plan_name": str(item.get("plan_name", ""))
                        }
                        for item in data
                        if item.get("plan_id") and item.get("plan_name")
                    ]
                else:
                    logger.error(f"Unexpected response format: {data}")
                    return []
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching plans: {e}")
            raise Exception(f"Failed to fetch plans: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching plans: {e}")
            raise Exception(f"Failed to fetch plans: {str(e)}")
    
    async def fetch_product_premiums(
        self, 
        jwt_token: str, 
        plan_id: str, 
        age: int, 
        gender: str
    ) -> Dict[str, Any]:
        """
        API 2: 플랜별 보험료 상세 조회
        RAG 분석을 위한 핵심 데이터 조회
        """
        url = f"{self.base_url}/ProductPremiums"
        params = {
            "plan_id": plan_id,
            "age": age,
            "gender": gender
        }
        headers = {
            "Authorization": f"Bearer {jwt_token}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Successfully fetched premiums for plan {plan_id}")
                return data
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching premiums: {e}")
            raise Exception(f"Failed to fetch premiums: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching premiums: {e}")
            raise Exception(f"Failed to fetch premiums: {str(e)}")

# 전역 클라이언트 인스턴스
api_client = ExternalAPIClient()