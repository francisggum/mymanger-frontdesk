import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ExternalAPIClient:
    def __init__(self):
        self.base_url = "https://mmlfcp.ohmymanager.com/api"
        self.timeout = 60.0

    async def fetch_plans(self, jwt_token: str) -> List[Dict[str, str]]:
        """
        API 1: 플랜 목록 조회

        외부 보험 API(/Auth 엔드포인트)를 호출하여 사용 가능한 플랜 목록을 조회합니다.
        JWT 토큰의 유효성을 검증하며, 유효한 토큰인 경우 플랜 목록을 반환합니다.

        Args:
            jwt_token: 사용자 인증을 위한 JWT 토큰

        Returns:
            List[Dict[str, str]]: 플랜 정보 목록 (plan_id, plan_name, plan_type 등)
            - plan_id: 플랜 고유 ID
            - plan_name: 플랜 이름
            - plan_type: 플랜 유형 코드
            - plan_type_name: 플랜 유형 이름
            - plan_payterm_type_name: 납입 기간 유형 이름
            - plan_min_m_age: 남자 최소 가입 연령
            - plan_max_m_age: 남자 최대 가입 연령
            - plan_min_f_age: 여자 최소 가입 연령
            - plan_max_f_age: 여자 최대 가입 연령

        Raises:
            Exception: API 호출 실패 시 (HTTP 에러, 네트워크 에러 등)
        """
        url = f"{self.base_url}/Auth"
        params = {"token": jwt_token, "access_path": "MMLFCP_WEB"}

        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=120.0,  # 50→120초로 증가
                write=10.0,
                pool=120.0,  # 60→120초로 증가
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # 응답 형식에 따라 파싱 (실제: {"plans": [{"plan_id": "...", "plan_name": "..."}, ...]})
                if isinstance(data, dict) and "plans" in data:
                    plans_data = data["plans"]
                    if isinstance(plans_data, list):
                        return [
                            {
                                "plan_id": str(item.get("plan_id", "")),
                                "plan_name": str(item.get("plan_name", "")),
                                "plan_type": item.get("plan_type", ""),
                                "plan_type_name": item.get("plan_type_name", ""),
                                "plan_payterm_type_name": item.get(
                                    "plan_payterm_type_name", ""
                                ),
                                "plan_min_m_age": item.get("plan_min_m_age", 0),
                                "plan_max_m_age": item.get("plan_max_m_age", 0),
                                "plan_min_f_age": item.get("plan_min_f_age", 0),
                                "plan_max_f_age": item.get("plan_max_f_age", 0),
                            }
                            for item in plans_data
                            if item.get("plan_id") and item.get("plan_name")
                        ]
                    else:
                        logger.error(f"Plans is not a list: {plans_data}")
                        return []
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
        self, jwt_token: str, plan_id: str, age: int, gender: str
    ) -> Dict[str, Any]:
        """
        API 2: 플랜별 보험료 상세 조회
        RAG 분석을 위한 핵심 데이터 조회
        """
        url = f"{self.base_url}/ProductPremiums"
        params = {"plan_id": plan_id, "age": age, "gender": gender}
        headers = {"Authorization": f"Bearer {jwt_token}"}

        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=120.0,  # 50→120초로 증가
                write=10.0,
                pool=120.0,  # 60→120초로 증가
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
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
