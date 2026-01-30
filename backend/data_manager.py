import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.coverage_premiums_df: pd.DataFrame = None
        self.product_insur_data: List[Dict[str, Any]] = []
        self.current_plan_id: str = None
        self.current_age: int = None
        self.current_gender: str = None
    
    def process_premium_data(self, data: Dict[str, Any], plan_id: str, age: int, gender: str) -> Dict[str, Any]:
        """
        API 2 응답 데이터를 처리하여 DataFrame과 VectorDB 데이터로 분리
        """
        try:
            # 데이터 상태 저장
            self.current_plan_id = plan_id
            self.current_age = age
            self.current_gender = gender
            
            # coverage_premiums -> DataFrame
            if "coverage_premiums" in data:
                coverage_data = data["coverage_premiums"]
                if isinstance(coverage_data, list) and len(coverage_data) > 0:
                    self.coverage_premiums_df = pd.DataFrame(coverage_data)
                    logger.info(f"Loaded {len(coverage_data)} coverage premium records")
                else:
                    logger.warning("No coverage_premiums data found")
                    self.coverage_premiums_df = pd.DataFrame()
            else:
                logger.warning("coverage_premiums key not found in response")
                self.coverage_premiums_df = pd.DataFrame()
            
            # product_insur_premiums -> VectorDB 데이터
            if "product_insur_premiums" in data:
                self.product_insur_data = data["product_insur_premiums"]
                logger.info(f"Loaded {len(self.product_insur_data)} product insurance records")
            else:
                logger.warning("product_insur_premiums key not found in response")
                self.product_insur_data = []
            
            return {
                "status": "success",
                "coverage_count": len(self.coverage_premiums_df) if self.coverage_premiums_df is not None else 0,
                "insurance_count": len(self.product_insur_data),
                "plan_info": {
                    "plan_id": plan_id,
                    "age": age,
                    "gender": gender
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing premium data: {e}")
            raise Exception(f"Failed to process premium data: {str(e)}")
    
    def get_coverage_dataframe(self) -> pd.DataFrame:
        """현재 로드된 coverage premiums DataFrame 반환"""
        return self.coverage_premiums_df if self.coverage_premiums_df is not None else pd.DataFrame()
    
    def get_insurance_data(self) -> List[Dict[str, Any]]:
        """현재 로드된 product insur premiums 데이터 반환"""
        return self.product_insur_data
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """현재 세션 정보 반환"""
        return {
            "plan_id": self.current_plan_id,
            "age": self.current_age,
            "gender": self.current_gender,
            "has_coverage_data": self.coverage_premiums_df is not None and not self.coverage_premiums_df.empty,
            "has_insurance_data": len(self.product_insur_data) > 0
        }
    
    def clear_data(self):
        """모든 데이터 초기화"""
        self.coverage_premiums_df = None
        self.product_insur_data = []
        self.current_plan_id = None
        self.current_age = None
        self.current_gender = None

# 전역 데이터 매니저 인스턴스
data_manager = DataManager()