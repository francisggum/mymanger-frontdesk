import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.coverage_premiums_df: Optional[pd.DataFrame] = None
        self.product_insur_data: List[Dict[str, Any]] = []
        self.current_plan_id: Optional[str] = None
        self.current_age: Optional[int] = None
        self.current_gender: Optional[str] = None
    
    def process_premium_data(self, data: Dict[str, Any], plan_id: str, age: int, gender: str) -> Dict[str, Any]:
        """
        API 2 응답 데이터를 처리하여 DataFrame과 VectorDB 데이터로 분리
        """
        try:
            # 데이터 상태 저장
            self.current_plan_id = plan_id
            self.current_age = age
            self.current_gender = gender
            
            # coverage_premiums와 required_premiums 결합
            combined_coverage_data = []
            
            # 1. coverage_premiums 처리 (보장코드 있는 데이터) - is_selected_coverage='Y'만 필터링
            if "coverage_premiums" in data:
                coverage_data = data["coverage_premiums"]
                if isinstance(coverage_data, list) and len(coverage_data) > 0:
                    # is_selected_coverage='Y'인 데이터만 필터링
                    filtered_coverage_data = [
                        item for item in coverage_data 
                        if item.get('is_selected_coverage') == 'Y'
                    ]
                    combined_coverage_data.extend(filtered_coverage_data)
                    logger.info(f"Loaded {len(filtered_coverage_data)} selected coverage premium records (filtered from {len(coverage_data)} total)")
                else:
                    logger.warning("No coverage_premiums data found")
            else:
                logger.warning("coverage_premiums key not found in response")
            
            # 2. required_premiums 처리 (보장코드 없는 데이터)
            if "required_premiums" in data:
                required_data = data["required_premiums"]
                if isinstance(required_data, list) and len(required_data) > 0:
                    # required_premiums에 coverage_cd를 "최저기본계약조건"으로 설정
                    for item in required_data:
                        item_copy = item.copy()
                        item_copy['coverage_cd'] = 'REQ_BASE'
                        item_copy['coverage_name'] = '최저기본계약조건'
                        item_copy['guide_coverage_amount'] = item_copy.get('min_insur_amount', 0)
                        item_copy['guide_coverage_premium'] = item_copy.get('min_premium', 0)
                        combined_coverage_data.append(item_copy)
                    logger.info(f"Loaded {len(required_data)} required premium records")
                else:
                    logger.warning("No required_premiums data found")
            else:
                logger.warning("required_premiums key not found in response")
            
            # DataFrame 생성
            if combined_coverage_data:
                self.coverage_premiums_df = pd.DataFrame(combined_coverage_data)
                logger.info(f"Total combined coverage records: {len(self.coverage_premiums_df)}")
            else:
                self.coverage_premiums_df = pd.DataFrame()
                logger.warning("No coverage data available")
            
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
    
    def normalize_coverage_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        가이드 보장금액 기준으로 보험료 비율 조정
        
        Args:
            df: coverage_premiums DataFrame
            
        Returns:
            adjusted_premium 컬럼이 추가된 DataFrame
        """
        try:
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # 각 (회사, 보장코드) 그룹별로 처리
            grouped = df.groupby(['company_code', 'coverage_cd'])
            for key, group in grouped:
                if isinstance(key, tuple):
                    company_code, coverage_cd = key
                else:
                    continue
                    
                # 최저기본계약조건은 그대로 사용 (보장금액 정규화 불필요)
                if coverage_cd == 'REQ_BASE':
                    result_df.loc[group.index, 'adjusted_premium'] = group['premium'].iloc[0]
                    continue
                    
                if len(group) == 1:
                    # 특약이 1개만 있는 경우 그대로 사용
                    result_df.loc[group.index, 'adjusted_premium'] = group['premium'].iloc[0]
                else:
                    # 여러 특약이 있는 경우 가이드 보장금액 정규화
                    base_amount = group.iloc[0]['coverage_amount']
                    
                    for idx, row in group.iterrows():
                        if row['coverage_amount'] != base_amount:
                            # 가이드 보장금액이 다른 경우 비율 조정
                            ratio = base_amount / row['coverage_amount']
                            adjusted_premium = row['premium'] * ratio
                            result_df.loc[idx, 'adjusted_premium'] = round(adjusted_premium, 0)
                        else:
                            result_df.loc[idx, 'adjusted_premium'] = row['premium']
            
            logger.info(f"Coverage amount normalization completed for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"Error normalizing coverage amounts: {e}")
            return df
    
    def aggregate_coverage_by_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        보장코드별로 보험료 합산
        
        Args:
            df: 정규화된 coverage_premiums DataFrame
            
        Returns:
            보장코드별로 합산된 DataFrame
        """
        try:
            if df.empty:
                return df
            
            # 보장코드별로 그룹화하여 합산
            aggregated = df.groupby(['company_code', 'company_name', 'coverage_cd', 'coverage_name']).agg({
                'adjusted_premium': 'sum',
                'coverage_amount': 'first',  # 첫 번째 특약의 보장금액 사용
                'guide_coverage_amount': 'first'
            }).reset_index()
            
            logger.info(f"Aggregated coverage by code: {len(aggregated)} groups created")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating coverage by code: {e}")
            return df
    
    def create_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        최종 비교 표 형태로 데이터 변환 (피벗 테이블)
        
        Args:
            df: 보장코드별로 합산된 DataFrame
            
        Returns:
            보장항목 vs 보험사 형태의 피벗 테이블
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            # 피벗 테이블 생성
            pivot_df = df.pivot_table(
                index='coverage_name',
                columns='company_name',
                values='adjusted_premium',
                aggfunc='sum',
                fill_value=0
            )
            
            # 보험료 합계 행 추가
            premium_sums = df.groupby('company_name')['adjusted_premium'].sum()
            pivot_df.loc['보험료 합계'] = premium_sums
            
            # 보장갯수 카운팅 (보장금액 > 0인 항목)
            coverage_counts = {}
            for company in df['company_name'].unique():
                company_data = df[df['company_name'] == company]
                # coverage_amount > 0이고 최저기본계약조건이 아닌 항목 카운트
                valid_coverages = company_data[
                    (company_data['coverage_amount'] > 0) & 
                    (company_data['coverage_cd'] != 'REQ_BASE')
                ]['coverage_cd'].nunique()
                coverage_counts[company] = valid_coverages
            
            pivot_df.loc['보장갯수'] = coverage_counts
            
            # 정렬: 보험료 합계 → 보장갯수 → 최저기본계약조건 순
            desired_order = ['보험료 합계', '보장갯수']
            if '최저기본계약조건' in pivot_df.index:
                desired_order.append('최저기본계약조건')
            
            other_coverages = [idx for idx in pivot_df.index if idx not in desired_order]
            final_order = desired_order + other_coverages
            
            pivot_df = pivot_df.reindex(final_order)
            
            logger.info(f"Comparison table created with shape: {pivot_df.shape}")
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error creating comparison table: {e}")
            return pd.DataFrame()
    
    def process_coverage_data_for_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        API 데이터를 받아 최종 비교 표 형태로 처리하는 통합 함수
        
        Args:
            data: API 응답 데이터
            
        Returns:
            처리된 비교 표 데이터
        """
        try:
            # 기존 프로세스: 기본 DataFrame 생성
            basic_result = self.process_premium_data(data, "", 0, "")
            
            if basic_result["status"] != "success":
                return basic_result
            
            # 1. 가이드 보장금액 정규화
            if self.coverage_premiums_df is None:
                raise Exception("No coverage premiums data available")
            normalized_df = self.normalize_coverage_amounts(self.coverage_premiums_df)
            
            # 2. 보장코드별 합산
            aggregated_df = self.aggregate_coverage_by_code(normalized_df)
            
            # 3. 최종 비교 표 생성
            comparison_table = self.create_comparison_table(aggregated_df)
            
            return {
                "status": "success",
                "coverage_count": len(aggregated_df),
                "insurance_count": len(self.product_insur_data),
                "comparison_table": self._clean_for_json_serialization(comparison_table.to_dict()),
                "raw_data": self._clean_for_json_serialization(aggregated_df.to_dict())
            }
            
        except Exception as e:
            logger.error(f"Error processing coverage data for comparison: {e}")
            raise Exception(f"Failed to process coverage data: {str(e)}")

    def _clean_for_json_serialization(self, data: Any) -> Any:
        """
        JSON 직렬화를 위해 무한대/NaN 값을 None으로 변환
        
        Args:
            data: 변환할 데이터
            
        Returns:
            JSON 직렬화 가능한 데이터
        """
        try:
            if isinstance(data, dict):
                return {key: self._clean_for_json_serialization(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [self._clean_for_json_serialization(item) for item in data]
            elif isinstance(data, (float, np.floating)):
                if math.isnan(data) or math.isinf(data):
                    return None
                return float(data)
            elif isinstance(data, (int, np.integer)):
                return int(data)
            else:
                return data
        except Exception:
            return None
    
    def clear_data(self):
        """모든 데이터 초기화"""
        self.coverage_premiums_df = None
        self.product_insur_data = []
        self.current_plan_id = None
        self.current_age = None
        self.current_gender = None

# 전역 데이터 매니저 인스턴스
data_manager = DataManager()