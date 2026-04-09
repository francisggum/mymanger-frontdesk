"""
MSSQL 데이터베이스 연결 모듈
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


PLAN_CATEGORY_MAPPING = {
    "생손보 건강(무해지)": "건강",
    "생손보 간편3.3.5(무해지)": "간편",
    "생손보 간편3.5.5(무해지)": "간편",
    "생손보 간편3.10.10(5)(무해지)": "간편",
    "손보 종합(표준환급)": "건강",
    "손보 종합(무해지)": "건강",
    "손보 5.10.10(무해지)": "건강",
    "손보 여성건강(무해지)": "건강",
    "생보 건강(무해지)": "건강",
    "생보 암(무해지)": "건강",
    "생보 간편3.3.5(무해지)": "간편",
    "생보 간편3.5.5(무해지)": "간편",
    "생보 간편3.10.5(무해지)": "간편",
    "손보 간편3.2.5(무해지)": "간편",
    "손보 간편3.3.5(무해지)": "간편",
    "손보 간편3.5.5(무해지)": "간편",
    "손보 간편3.10.10(5)(무해지)": "간편",
    "손보 어린이(표준환급)": "어린이",
    "손보 어린이(무해지)": "어린이",
    "손보 청소년(표준환급)": "건강",
    "손보 청소년(무해지)": "건강",
    "손보 청소년5.10.10(무해지)": "건강",
    "손보 실손": "실손",
    "손보 간편실손": "실손",
    "생보 치매(무해지)": "치매",
    "손보 치아": "치아",
    "생보 치아": "치아",
    "손보 운전자": "운전자",
}


def _get_plan_category(plan_name: str) -> str:
    """
    플랜명으로 카테고리 조회

    Args:
        plan_name: 플랜명

    Returns:
        카테고리명 (매핑되지 않으면 "기타")
    """
    return PLAN_CATEGORY_MAPPING.get(plan_name.strip(), "기타")


class DatabaseManager:
    """MSSQL 데이터베이스 관리자"""

    def __init__(self):
        self.server = os.getenv("DB_HOST", "localhost")
        self.database = os.getenv("DB_NAME")
        self.username = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

        if not self.username or not self.password:
            raise ValueError(
                "DB_USER and DB_PASSWORD environment variables are required"
            )

    def get_connection_string(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )

    def execute_query(
        self, query: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        SQL 쿼리 실행하고 결과 반환

        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 리스트 (선택적)

        Returns:
            쿼리 결과 리스트 (각 행이 딕셔너리 형태)
        """
        conn = None
        cursor = None
        try:
            conn = pyodbc.connect(self.get_connection_string())
            cursor = conn.cursor()

            logger.info(f"쿼리 실행: {query[:100]}...")
            if params:
                logger.info(f"쿼리 파라미터: {params}")
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # 컬럼명 가져오기
            columns = [column[0] for column in cursor.description]

            # 결과를 딕셔너리 리스트로 변환
            results = []
            for row in cursor:
                result_dict = {}
                for i, value in enumerate(row):
                    result_dict[columns[i]] = value
                results.append(result_dict)

            logger.info(f"쿼리 결과: {len(results)}개 행 반환")
            return results

        except pyodbc.Error as e:
            logger.error(f"데이터베이스 오류: {e}")
            raise Exception(f"데이터베이스 쿼리 실패: {e}")
        except Exception as e:
            logger.error(f"쿼리 실행 중 오류: {e}")
            raise Exception(f"쿼리 실행 실패: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def fetch_plans(self) -> List[Dict[str, Any]]:
        """
        플랜 목록 조회

        Returns:
            플랜 정보 리스트
        """
        query = """
        select
            plan_id as plan_id ,
            insu_compy_type as insu_compy_type,
            d.cd_nm as insu_compy_type_name,
            plan_type as plan_type,
            b.cd_nm as plan_type_name,
            a.refund_payment_type,
            a.simplified_underwriting_yn,
            a.renewal_yn,
            a.notice_type,
            plan_payterm_type as payment_due_type,
            c.cd_nm as payment_due_type_name,
            plan_min_m_age as min_m_age,
            plan_max_m_age as max_m_age,
            plan_min_f_age as min_f_age,
            plan_max_f_age as max_f_age
        from
            TB_MMLFCP_PLAN a
        join mmapi.dbo.TB_COMM_CD b
            on a.plan_type = b.CD_ID
            and b.UPP_CD_ID = 'MMLFCP_A'
        join mmapi.dbo.TB_COMM_CD c
            on a.plan_payterm_type = c.CD_ID
            and c.UPP_CD_ID = 'MMLFCP_B'
        join mmapi.dbo.TB_COMM_CD d
            on a.insu_compy_type = d.CD_ID
            and d.UPP_CD_ID = 'MMLFCP_C'
        where
            a.use_yn = 'Y'
        order by
            b.ORDER_SEQ
        """

        plans = self.execute_query(query)

        # 결과 데이터 정리
        for plan in plans:
            # plan_name을 plan_type_name으로 설정
            plan_name = plan.get("plan_type_name", "")
            plan["plan_name"] = plan_name

            # plan_category 매핑
            plan["plan_category"] = _get_plan_category(
                f'{plan["insu_compy_type_name"]} {plan_name}'
            )

            # 나이 필드가 None인 경우 0으로 설정
            for age_field in ["min_m_age", "max_m_age", "min_f_age", "max_f_age"]:
                if plan.get(age_field) is None:
                    plan[age_field] = 0

        return plans

    def fetch_products(self) -> List[Dict[str, Any]]:
        """
        상품 목록 조회

        TB_MMLFCP_PLAN_PRODUCT 테이블에서 사용 가능한 상품 목록을 조회합니다.
        보험사 코드는 TB_COMM_CD 테이블을 통해保险公司명을 조회합니다.

        Returns:
            상품 정보 리스트 (plan_id, company_code, product_code, company_nm 포함)
        """
        query = """
        SELECT 
            plan_id, 
            company_code, 
            product_code, 
            (SELECT TOP 1 cd_nm FROM mmapi.dbo.TB_COMM_CD WHERE cd_id = company_code AND UPP_CD_ID = 'COMPY') AS company_nm
        FROM mmlfcp.dbo.TB_MMLFCP_PLAN_PRODUCT
        WHERE use_yn = 'Y'
        """

        products = self.execute_query(query)

        logger.info(f"상품 목록 조회 완료: {len(products)}개 상품")

        return products

    def fetch_premium_data(
        self, plan_id: str, gender: str, age: int
    ) -> List[Dict[str, Any]]:
        """
        플랜별 보험료 데이터 조회

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 나이

        Returns:
            보험료 데이터 리스트
        """
        query = """
        SELECT
            a.compy_cd AS company_code,
            a.prdt_cd AS product_code,
            c.prdt_name AS product_name,
            e.coverage_cd AS coverage_cd,
            a.insur_cd AS insur_item_code,
            d.insur_nm AS insur_item_name,
            d.insur_bojang AS insur_item_coverage,
            d.pay_term AS payment_due,
            e.guide_contract_amount AS guide_contract_amount,
            CASE
                WHEN a.std_contract_amt <= 0 THEN 0
                ELSE CAST((e.guide_contract_amount * a.premium) / a.std_contract_amt AS INT)
            END AS guide_premium,
            (SELECT TOP 1 CD_NM FROM mmapi.dbo.TB_COMM_CD 
             WHERE CD_ID = a.compy_cd AND UPP_CD_ID = 'COMPY') AS company_name,
            ISNULL((SELECT TOP 1 coverage_name FROM TB_MMLFCP_COVERAGE 
            WHERE coverage_cd = e.coverage_cd), '최저기본계약조건') AS coverage_name
        FROM
            mmapi.dbo.TB_TIC_PRDT_PRICE a
        JOIN TB_MMLFCP_PLAN_PRODUCT b
            ON a.compy_cd = b.company_code
            AND a.prdt_cd = b.product_code
            AND b.plan_id = ?
        JOIN mmapi.dbo.TB_TIC_PRDT c
            ON a.compy_cd = c.compy_cd
            AND a.prdt_cd = c.prdt_cd
        JOIN mmapi.dbo.TB_TIC_PRDT_D d
            ON a.compy_cd = d.compy_cd
            AND a.prdt_cd = d.prdt_cd
            AND a.insur_cd = d.insur_cd
        JOIN (
            SELECT
                a.coverage_cd,
                b.insur_cd,
                b.guide_insur_amount AS guide_contract_amount
            FROM
                TB_MMLFCP_PLAN_COVERAGE a
            JOIN TB_MMLFCP_COVERAGE_INSUR_MAPPING b
                ON a.coverage_cd = b.coverage_cd
            WHERE
                a.plan_id = ?
                AND a.use_yn = 'Y'
        ) e ON a.insur_cd = e.insur_cd
        WHERE
            a.sex = ?
            AND a.age = ?
        ORDER BY
            a.compy_cd,
            e.coverage_cd,
            a.insur_cd
        """

        params = [plan_id, plan_id, gender, age]

        # 로깅 추가
        logger.info(
            f"보험료 데이터 조회 시작 - plan_id: {plan_id}, gender: {gender}, age: {age}"
        )

        results = self.execute_query(query, params)

        # 결과 로깅 - 수치 필드 상세 분석
        logger.info(f"보험료 데이터 조회 완료 - {len(results)}개 행 반환")
        if results:
            logger.info(f"첫 3개 결과 샘플: {results[:3]}")

            # 수치 필드 분석
            for i, row in enumerate(results[:3]):
                guide_premium = row.get("guide_premium")
                guide_contract_amount = row.get("guide_contract_amount")
                logger.info(
                    f"행 {i + 1} - guide_premium: {guide_premium} (타입: {type(guide_premium)}), guide_contract_amount: {guide_contract_amount} (타입: {type(guide_contract_amount)})"
                )

                # 비정상 값 확인
                if isinstance(guide_premium, (int, float)):
                    if isinstance(guide_premium, float) and (
                        math.isnan(guide_premium) or math.isinf(guide_premium)
                    ):
                        logger.warning(
                            f"행 {i + 1} - guide_premium에 비정상 값: {guide_premium}"
                        )

                if isinstance(guide_contract_amount, (int, float)):
                    if isinstance(guide_contract_amount, float) and (
                        math.isnan(guide_contract_amount)
                        or math.isinf(guide_contract_amount)
                    ):
                        logger.warning(
                            f"행 {i + 1} - guide_contract_amount에 비정상 값: {guide_contract_amount}"
                        )

        return results

    def process_premium_data_for_comparison(
        self, plan_id: str, gender: str, age: int
    ) -> Dict[str, Any]:
        """
        보험료 데이터를 비교표용으로 전처리

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 나이

        Returns:
            사람용 비교표와 LLM용 비교표 데이터
        """
        try:
            # 1. 기존 함수로 원본 데이터 조회
            raw_data = self.fetch_premium_data(plan_id, gender, age)

            if not raw_data:
                logger.warning("조회된 데이터가 없습니다.")
                return {
                    "status": "success",
                    "human_readable_table": {},
                    "llm_readable_data": {},
                    "summary": {
                        "total_companies": 0,
                        "total_coverages": 0,
                        "plan_id": plan_id,
                        "age": age,
                        "gender": gender,
                    },
                }

            # 2. pandas DataFrame으로 변환
            df = pd.DataFrame(raw_data)
            logger.info(f"DataFrame 생성 완료: {len(df)}행")

            # 3. 그룹화 기준에 따라 데이터 집계 - 다른 방식으로 접근
            unique_groups = df[
                [
                    "company_code",
                    "company_name",
                    "product_code",
                    "product_name",
                    "coverage_cd",
                    "coverage_name",
                ]
            ].drop_duplicates()

            processed_data = []
            for _, group_row in unique_groups.iterrows():
                company_code = group_row["company_code"]
                company_name = group_row["company_name"]
                product_code = group_row["product_code"]
                product_name = group_row["product_name"]
                coverage_cd = group_row["coverage_cd"]
                coverage_name = group_row["coverage_name"]

                # 해당 그룹의 데이터 필터링
                group = df[
                    (df["company_code"] == company_code)
                    & (df["company_name"] == company_name)
                    & (df["product_code"] == product_code)
                    & (df["product_name"] == product_name)
                    & (df["coverage_cd"] == coverage_cd)
                    & (df["coverage_name"] == coverage_name)
                ]
                # sum_premium 계산 - 비정상 값 처리
                premium_values = group["guide_premium"].dropna()
                # inf, -inf, NaN 값 제거
                premium_values = premium_values[
                    ~premium_values.apply(
                        lambda x: (
                            isinstance(x, float) and (math.isnan(x) or math.isinf(x))
                        )
                    )
                ]
                sum_premium = premium_values.sum() if len(premium_values) > 0 else 0
                if math.isnan(sum_premium) or math.isinf(sum_premium):
                    sum_premium = 0
                    logger.warning(
                        f"sum_premium이 비정상 값으로 0으로 설정됨: company_code={company_code}, coverage_cd={coverage_cd}"
                    )

                # insur_item_name_list 생성 (|로 조인)
                insur_item_name_list = "|".join(group["insur_item_name"].astype(str))

                # insur_item_name_coverage_list 생성 (|로 조인)
                insur_item_name_coverage_list = "|".join(
                    group["insur_item_coverage"].astype(str)
                )

                # payment_due_list 생성 (|로 조인)
                payment_due_list = "|".join(group["payment_due"].astype(str))

                # guide_premium_list 생성 (+로 조인) - 기존 함수 활용하여 정수 포맷팅
                guide_premium_list = "+".join(
                    self._safe_float_format(premium)
                    for premium in group["guide_premium"]
                )

                # guide_contract_amount_max 계산 - 비정상 값 처리
                contract_values = group["guide_contract_amount"].dropna()
                # inf, -inf, NaN 값 제거
                contract_values = contract_values[
                    ~contract_values.apply(
                        lambda x: (
                            isinstance(x, float) and (math.isnan(x) or math.isinf(x))
                        )
                    )
                ]
                guide_contract_amount_max = (
                    contract_values.max() if len(contract_values) > 0 else 0
                )
                if math.isnan(guide_contract_amount_max) or math.isinf(
                    guide_contract_amount_max
                ):
                    guide_contract_amount_max = 0
                    logger.warning(
                        f"guide_contract_amount_max이 비정상 값으로 0으로 설정됨: company_code={company_code}, coverage_cd={coverage_cd}"
                    )

                processed_data.append(
                    {
                        "company_code": company_code,
                        "company_name": company_name,
                        "product_code": product_code,
                        "product_name": product_name,
                        "coverage_cd": coverage_cd,
                        "coverage_name": coverage_name,
                        "sum_premium": sum_premium,
                        "insur_item_name_list": insur_item_name_list,
                        "insur_item_name_coverage_list": insur_item_name_coverage_list,
                        "payment_due_list": payment_due_list,
                        "guide_premium_list": guide_premium_list,
                        "guide_contract_amount_max": guide_contract_amount_max,
                    }
                )

            processed_df = pd.DataFrame(processed_data)
            logger.info(f"데이터 집계 완료: {len(processed_df)}개 그룹")

            # 4. 사람이 읽기 편한 비교표 생성 (피벗 테이블)
            human_readable_table = self._create_human_readable_table(processed_df)

            # 5. LLM이 읽기 편한 비교표 생성 (트리 구조)
            llm_readable_data = self._create_llm_readable_data(processed_df)

            # 6. 요약 정보 생성
            summary = {
                "total_companies": processed_df["company_code"].nunique(),
                "total_coverages": processed_df["coverage_cd"].nunique(),
                "plan_id": plan_id,
                "age": age,
                "gender": gender,
            }

            return {
                "status": "success",
                "human_readable_table": human_readable_table,
                "llm_readable_data": llm_readable_data,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}")
            raise Exception(f"Failed to process premium data for comparison: {str(e)}")

    def process_premium_data_for_coverages(
        self, plan_id: str, gender: str, age: int
    ) -> List[Dict[str, Any]]:
        """
        보험료 데이터를 coverage용으로 전처리

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 나이

        Returns:
            processed_df.to_dict(records) 형태의 리스트
        """
        try:
            # 1. rider 데이터 조회 (먼저 조회하여 위에 배치)
            rider_data = self.fetch_premium_data_require_rider(plan_id, gender, age)

            # 2. 기존 함수로 원본 데이터 조회
            raw_data = self.fetch_premium_data(plan_id, gender, age)

            # 3. rider 데이터를 먼저 배치하여 결합
            combined_data = []
            if rider_data:
                combined_data.extend(rider_data)
            if raw_data:
                combined_data.extend(raw_data)

            if not combined_data:
                logger.warning("조회된 데이터가 없습니다.")
                return []

            # 4. pandas DataFrame으로 변환
            df = pd.DataFrame(combined_data)
            logger.info(f"DataFrame 생성 완료: {len(df)}행")

            # 3. 그룹화 기준에 따라 데이터 집계
            unique_groups = df[
                [
                    "company_code",
                    "company_name",
                    "product_code",
                    "product_name",
                    "coverage_cd",
                    "coverage_name",
                ]
            ].drop_duplicates()

            processed_data = []
            for _, group_row in unique_groups.iterrows():
                company_code = group_row["company_code"]
                company_name = group_row["company_name"]
                product_code = group_row["product_code"]
                product_name = group_row["product_name"]
                coverage_cd = group_row["coverage_cd"]
                coverage_name = group_row["coverage_name"]

                # 해당 그룹의 데이터 필터링
                group = df[
                    (df["company_code"] == company_code)
                    & (df["company_name"] == company_name)
                    & (df["product_code"] == product_code)
                    & (df["product_name"] == product_name)
                    & (df["coverage_cd"] == coverage_cd)
                    & (df["coverage_name"] == coverage_name)
                ]

                # sum_premium 계산 - 비정상 값 처리
                premium_values = group["guide_premium"].dropna()
                premium_values = premium_values[
                    ~premium_values.apply(
                        lambda x: (
                            isinstance(x, float) and (math.isnan(x) or math.isinf(x))
                        )
                    )
                ]
                sum_premium = premium_values.sum() if len(premium_values) > 0 else 0
                if math.isnan(sum_premium) or math.isinf(sum_premium):
                    sum_premium = 0
                    logger.warning(
                        f"sum_premium이 비정상 값으로 0으로 설정됨: company_code={company_code}, coverage_cd={coverage_cd}"
                    )

                # insur_item_name_list 생성 (|로 조인)
                insur_item_name_list = "|".join(group["insur_item_name"].astype(str))

                # insur_item_name_coverage_list 생성 (|로 조인)
                insur_item_name_coverage_list = "|".join(
                    group["insur_item_coverage"].astype(str)
                )

                # payment_due_list 생성 (|로 조인)
                payment_due_list = "|".join(group["payment_due"].astype(str))

                # guide_premium_list 생성 (+로 조인)
                guide_premium_list = "+".join(
                    self._safe_float_format(premium)
                    for premium in group["guide_premium"]
                )

                # guide_contract_amount_max 계산 - 비정상 값 처리
                contract_values = group["guide_contract_amount"].dropna()
                contract_values = contract_values[
                    ~contract_values.apply(
                        lambda x: (
                            isinstance(x, float) and (math.isnan(x) or math.isinf(x))
                        )
                    )
                ]
                guide_contract_amount_max = (
                    contract_values.max() if len(contract_values) > 0 else 0
                )
                if math.isnan(guide_contract_amount_max) or math.isinf(
                    guide_contract_amount_max
                ):
                    guide_contract_amount_max = 0
                    logger.warning(
                        f"guide_contract_amount_max이 비정상 값으로 0으로 설정됨: company_code={company_code}, coverage_cd={coverage_cd}"
                    )

                processed_data.append(
                    {
                        "company_code": company_code,
                        "company_name": company_name,
                        "product_code": product_code,
                        "product_name": product_name,
                        "coverage_cd": coverage_cd,
                        "coverage_name": coverage_name,
                        "sum_premium": sum_premium,
                        "insur_item_name_list": insur_item_name_list,
                        "insur_item_name_coverage_list": insur_item_name_coverage_list,
                        "payment_due_list": payment_due_list,
                        "guide_premium_list": guide_premium_list,
                        "guide_contract_amount_max": guide_contract_amount_max,
                    }
                )

            processed_df = pd.DataFrame(processed_data)
            logger.info(f"데이터 집계 완료: {len(processed_df)}개 그룹")

            return processed_df.to_dict(orient="records")

        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}")
            raise Exception(f"Failed to process premium data for coverages: {str(e)}")

    def _safe_float_format(self, value, prefix="", suffix=""):
        """
        안전한 float 포맷팅 메서드

        Args:
            value: 포맷팅할 값
            prefix: 접두사
            suffix: 접미사

        Returns:
            포맷팅된 문자열
        """
        try:
            if value is None:
                return f"{prefix}0{suffix}"

            # NaN이나 inf 체크
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                logger.warning(f"비정상 float 값 감지: {value}, 0으로 대체")
                return f"{prefix}0{suffix}"

            # 정수로 변환 시도
            try:
                int_value = int(float(value))
                return f"{prefix}{int_value}{suffix}"
            except (ValueError, TypeError):
                return f"{prefix}0{suffix}"

        except Exception as e:
            logger.error(f"float 포맷팅 오류: {e}, value: {value}")
            return f"{prefix}0{suffix}"

    def _create_human_readable_table(self, df: pd.DataFrame) -> str:
        """
        사람이 읽기 편한 피벗 테이블 생성

        Args:
            df: 전처리된 DataFrame

        Returns:
            피벗 테이블 형식의 딕셔너리
        """
        try:
            # 인덱스 컬럼 생성: coverage_name(coverage_code:guide_contract_amount_max)
            df["coverage_name(coverage_code:guide_contract_amount_max[만원])"] = (
                df.apply(
                    lambda row: self._safe_float_format(
                        row["guide_contract_amount_max"],
                        f"{row['coverage_name']}({row['coverage_cd']}:",
                        ")",
                    ),
                    axis=1,
                )
            )

            # 컬럼명 생성: company_name(company_code)
            df["column_name"] = df.apply(
                lambda row: f"{row['company_name']}({row['company_code']})", axis=1
            )

            # 셀 값 생성: sum_premium(guide_premium_list)
            df["cell_value"] = df.apply(
                lambda row: (
                    f"{self._safe_float_format(row['sum_premium'], '', '')}({row['guide_premium_list']})"
                ),
                axis=1,
            )

            # 피벗 테이블 생성
            pivot_df = df.pivot_table(
                index="coverage_name(coverage_code:guide_contract_amount_max[만원])",
                columns="column_name",
                values="cell_value",
                aggfunc="first",
            ).fillna("")

            # 보험사별 총 보험료 합계 계산
            company_totals = df.groupby("column_name")["sum_premium"].sum()

            # 합계 행 생성 (맨 위에 추가할 행)
            total_row = {}
            for col in pivot_df.columns:
                if col in company_totals.index:
                    total_value = company_totals[col]
                    total_row[col] = f"{self._safe_float_format(total_value, '', '')}"
                else:
                    total_row[col] = ""

            # 합계 행을 DataFrame으로 변환하고 기존 피벗 테이블과 결합
            total_df = pd.DataFrame([total_row], index=["**보험사별 총 보험료 합계**"])

            # 합계 행을 맨 위에 추가
            pivot_df = pd.concat([total_df, pivot_df])

            # JSON 직렬화를 위해 변환 (pandas to_json 사용)
            json_result = pivot_df.fillna("").to_json(orient="table")
            result = (
                json_result
                if json_result is not None
                else '{"schema":{"fields":[],"primaryKey":[],"pandas_version":"1.4.0"},"data":[]}'
            )

            logger.info(
                f"사람용 비교표 생성 완료: {len(pivot_df)}행 x {len(pivot_df.columns)}열 (합계 행 포함)"
            )

            # 디버깅용으로 최종 데이터를 markdown 표로 저장
            try:
                # /app 디렉토리가 없는 경우 생성
                os.makedirs("/app", exist_ok=True)

                # pivot_df를 markdown 형식으로 변환하여 파일 저장
                # markdown_content = pivot_df.to_markdown()
                # if markdown_content is not None:
                #     with open("/app/human_df.md", "w", encoding="utf-8") as f:
                #         f.write(markdown_content)
                # logger.info("사람용 비교표 markdown 파일 저장 완료: /app/human_df.md")
            except Exception as e:
                logger.error(f"markdown 파일 저장 실패: {e}")
                raise Exception(f"디버깅용 markdown 파일 저장에 실패했습니다: {e}")

            return result

        except Exception as e:
            logger.error(f"사람용 비교표 생성 실패: {e}")
            return pd.DataFrame().to_json(orient="table")

    def _create_llm_readable_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        LLM이 읽기 편한 트리 구조 데이터 생성

        Args:
            df: 전처리된 DataFrame

        Returns:
            트리 구조의 딕셔너리
        """
        try:
            result = {}

            # 회사별로 그룹화
            unique_companies = df[
                ["company_code", "company_name", "product_name"]
            ].drop_duplicates()

            for _, company_row in unique_companies.iterrows():
                company_code = company_row["company_code"]
                company_name = company_row["company_name"]
                product_name = company_row["product_name"]
                group = df[
                    (df["company_code"] == company_code)
                    & (df["company_name"] == company_name)
                    & (df["product_name"] == product_name)
                ]
                company_key = f"{company_name}_{company_code}_{product_name}"

                # 보장 항목 리스트 생성
                coverages = []
                for _, row in group.iterrows():
                    # insur_item_list 생성 - 세 가지 리스트를 파싱하여 객체 배열로 변환
                    insur_item_list = []

                    # "|"로 구분된 리스트 파싱
                    names = (
                        row["insur_item_name_list"].split("|")
                        if row["insur_item_name_list"]
                        else []
                    )
                    coverages_list = (
                        row["insur_item_name_coverage_list"].split("|")
                        if row["insur_item_name_coverage_list"]
                        else []
                    )

                    # "+"로 구분된 프리미엄 리스트 파싱
                    premiums = (
                        row["guide_premium_list"].split("+")
                        if row["guide_premium_list"]
                        else []
                    )

                    # 세 리스트를 zip하여 객체 배열 생성 (길이가 다른 경우 가장 짧은 것에 맞춤)
                    for i in range(min(len(names), len(coverages_list), len(premiums))):
                        insur_item_list.append(
                            {
                                "name": names[i].strip(),
                                "coverage": coverages_list[i].strip(),
                                "premium": premiums[i].strip(),
                            }
                        )

                    coverage = {
                        "coverage_name": row["coverage_name"],
                        "coverage_code": row["coverage_cd"],
                        "guide_contract_amount_max": row["guide_contract_amount_max"],
                        "sum_premium": row["sum_premium"],
                        "insur_item_list": insur_item_list,
                    }
                    coverages.append(coverage)

                result[company_key] = coverages

            logger.info(f"LLM용 데이터 생성 완료: {len(result)}개 회사")

            # 디버그용: result를 JSON 파일로 저장
            # try:
            #     import json
            #     os.makedirs("/app", exist_ok=True)
            #     with open("/app/result.json", "w", encoding="utf-8") as f:
            #         json.dump(result, f, ensure_ascii=False, indent=2)
            #     logger.info("디버그용 결과 파일 저장 완료: /app/result.json")
            # except Exception as e:
            #     logger.error(f"디버그용 결과 파일 저장 실패: {e}")

            return result

        except Exception as e:
            logger.error(f"LLM용 데이터 생성 실패: {e}")
            return {}

    def fetch_plan_standard_coverages(self) -> List[Dict[str, Any]]:
        """
        플랜별 표준 보장 항목 조회

        Returns:
            표준 보장 항목 리스트
        """
        query = """
        SELECT plan_id, coverage_cd, guide_coverage_amount, 
               is_selected_coverage, coverage_seq
        FROM TB_MMLFCP_PLAN_COVERAGE
        WHERE use_yn = 'Y'
        ORDER BY plan_id, coverage_seq
        """
        return self.execute_query(query)

    def fetch_coverage_mapping(self) -> List[Dict[str, Any]]:
        """
        보장코드와 보장명 매핑 정보 조회

        Returns:
            보장코드, 보장명 리스트
        """
        query = """
                -- 1. 하드코딩된 가짜 레코드 (Dummy Data)
        SELECT 
            'Z000' AS coverage_cd, 
            '주계약' AS coverage_name
        UNION ALL
        SELECT coverage_cd, coverage_name
        FROM TB_MMLFCP_COVERAGE
        WHERE use_yn = 'Y'
        """
        return self.execute_query(query)

    def fetch_premium_by_age(
        self, plan_id: str, gender: str, age: int, company_cd: str
    ) -> List[Dict[str, Any]]:
        """
        연령별 보장 보험료 조회

        입력된 나이 이후부터 최대가입연령까지의 보장별 보험료를 조회합니다.

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 시작 나이
            company_cd: 보험사 코드

        Returns:
            연령별 보장 보험료 데이터 리스트
        """
        query = """
        SELECT
            a.age as age,
            e.coverage_cd as coverage_cd,
            MAX(e.guide_contract_amount) as guide_contract_amount,
            SUM(CASE
                WHEN a.std_contract_amt <= 0 THEN 0
                ELSE CAST((e.guide_contract_amount * a.premium) / a.std_contract_amt AS INT)
            END) as guide_premium
        FROM
            mmapi.dbo.TB_TIC_PRDT_PRICE a
        JOIN TB_MMLFCP_PLAN_PRODUCT b
            ON a.compy_cd = b.company_code
            AND a.prdt_cd = b.product_code
            AND b.plan_id = ?
        JOIN mmapi.dbo.TB_TIC_PRDT c
            ON a.compy_cd = c.compy_cd
            AND a.prdt_cd = c.prdt_cd
        JOIN mmapi.dbo.TB_TIC_PRDT_D d
            ON a.compy_cd = d.compy_cd
            AND a.prdt_cd = d.prdt_cd
            AND a.insur_cd = d.insur_cd
        JOIN (
            SELECT
                a.coverage_cd,
                b.insur_cd,
                b.guide_insur_amount as guide_contract_amount
            FROM
                TB_MMLFCP_PLAN_COVERAGE a
            JOIN TB_MMLFCP_COVERAGE_INSUR_MAPPING b
                ON a.coverage_cd = b.coverage_cd
            WHERE
                a.plan_id = ?
                AND a.use_yn = 'Y'
        ) e
            ON a.insur_cd = e.insur_cd
        WHERE
            a.sex = ?
            AND a.age >= ?
            AND a.compy_cd = ?
        GROUP BY
            a.age,
            e.coverage_cd
        ORDER BY
            a.age,
            e.coverage_cd
        """

        params = [plan_id, plan_id, gender, age, company_cd]

        logger.info(
            f"연령별 보장 보험료 조회 시작 - plan_id: {plan_id}, gender: {gender}, age: {age}, company_cd: {company_cd}"
        )

        results = self.execute_query(query, params)

        logger.info(f"연령별 보장 보험료 조회 완료 - {len(results)}개 행 반환")

        return results

    def fetch_premium_data_require_rider(
        self, plan_id: str, gender: str, age: int
    ) -> List[Dict[str, Any]]:
        """
        필수 담보 보험료 조회

        플랜에 해당하는 모든 보험사의 필수 담보(主계약) 보험료를 조회합니다.

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 나이

        Returns:
            필수 담보 보험료 데이터 리스트
            - plan_id: 플랜 ID
            - company_code: 보험사 코드
            - product_code: 상품 코드
            - product_name: 상품명
            - coverage_cd: 보장 코드 (항상 'Z000' - 주계약)
            - insur_item_code: 보험 항목 코드
            - insur_item_name: 보험 항목명
            - insur_item_coverage: 보험 항목 보장액
            - payment_due: 납입 기간
            - guide_contract_amount: 안내 계약 금액
            - guide_premium: 안내 보험료
            - company_name: 보험사명
            - coverage_name: 보장명 (항상 '주계약')
        """
        query = """
        SELECT
            b.plan_id AS plan_id,
            a.compy_cd AS company_code,
            a.prdt_cd AS product_code,
            c.prdt_name AS product_name,
            'Z000' AS coverage_cd,
            a.insur_cd AS insur_item_code,
            d.insur_nm AS insur_item_name,
            d.insur_bojang AS insur_item_coverage,
            d.pay_term AS payment_due,
            e.min_insur_amount AS guide_contract_amount,
            CASE
                WHEN a.std_contract_amt > 0 THEN
                    (e.min_insur_amount * a.premium) / a.std_contract_amt
                ELSE 0
            END AS guide_premium,
            (SELECT TOP 1 CD_NM FROM mmapi.dbo.TB_COMM_CD WHERE CD_ID = a.compy_cd AND UPP_CD_ID = 'COMPY') AS company_name,
            '주계약' AS coverage_name
        FROM
            mmapi.dbo.TB_TIC_PRDT_PRICE a
        JOIN TB_MMLFCP_PLAN_PRODUCT b
            ON a.compy_cd = b.company_code
            AND a.prdt_cd = b.product_code
            AND b.plan_id = ?
        JOIN mmapi.dbo.TB_TIC_PRDT c
            ON a.compy_cd = c.compy_cd
            AND a.prdt_cd = c.prdt_cd
        JOIN mmapi.dbo.TB_TIC_PRDT_D d
            ON a.compy_cd = d.compy_cd
            AND a.prdt_cd = d.prdt_cd
            AND a.insur_cd = d.insur_cd
        JOIN TB_MMLFCP_PRODUCT_REQUIRED_RULES e
            ON a.compy_cd = e.company_code
            AND a.prdt_cd = e.product_code
            AND a.insur_cd = e.insur_cd
        WHERE
            a.sex = ?
            AND a.age = ?
            AND a.use_yn = 'Y'
        ORDER BY
            a.compy_cd,
            a.prdt_cd,
            a.insur_cd
        """

        params = [plan_id, gender, age]

        logger.info(
            f"필수 담보 보험료 조회 시작 - plan_id: {plan_id}, gender: {gender}, age: {age}"
        )

        results = self.execute_query(query, params)

        logger.info(f"필수 담보 보험료 조회 완료 - {len(results)}개 행 반환")

        return results

    def get_coverage_insur_mapping(self) -> List[Dict[str, Any]]:
        """
        보장코드-담보코드 매핑 정보 조회

        Returns:
            보장코드, 보장명, 담보코드, 가이드보험금액, 사용여부 리스트
        """
        query = """
        SELECT TB_MMLFCP_COVERAGE_INSUR_MAPPING.[coverage_cd]
             , [coverage_name]
             , [insur_cd]
             , [guide_insur_amount]
             , [use_yn]
          FROM dbo.TB_MMLFCP_COVERAGE_INSUR_MAPPING
        INNER JOIN dbo.TB_MMLFCP_COVERAGE
            ON TB_MMLFCP_COVERAGE_INSUR_MAPPING.coverage_cd = TB_MMLFCP_COVERAGE.coverage_cd
        """
        return self.execute_query(query)

    def fetch_premium_data_coverage(
        self, plan_id: str, gender: str, age: int
    ) -> List[Dict[str, Any]]:
        """
        보장별 보험료 조회

        Args:
            plan_id: 플랜 ID
            gender: 성별 (M/F)
            age: 나이

        Returns:
            보장별 보험료 데이터 리스트
        """
        query = """
        SELECT
            a.company_code,
            e.CD_NM as company_name,
            a.product_code,
            d.prdt_name as product_name,
            d.attr1 as product_detail_name,
            d.mb_conditions as join_condition,
            a.coverage_cd,
            f.coverage_name,
            c.coverage_seq,
            c.is_selected_coverage,
            c.guide_coverage_amount,
            CASE
                WHEN a.coverage_amount > 0 THEN
                    CAST((c.guide_coverage_amount * a.premium) / a.coverage_amount AS INT)
                ELSE 0
            END as guide_coverage_premium,
            ISNULL((SELECT TOP 1 coverage_amount_ratio 
                    FROM TB_MMLFCP_AMOUNT_RATIO 
                    WHERE a.company_code = company_code 
                    AND a.product_code = product_code 
                    AND c.coverage_cd = coverage_cd), 1) as coverage_amount_ratio
        FROM
            TB_MMLFCP_COVERAGE_PRICE a
        JOIN TB_MMLFCP_PLAN_PRODUCT b
            ON a.company_code = b.company_code
            AND a.product_code = b.product_code
            AND b.plan_id = ?
        JOIN TB_MMLFCP_PLAN_COVERAGE c
            ON a.coverage_cd = c.coverage_cd
            AND c.plan_id = ?
            AND c.use_yn = 'Y'
        JOIN mmapi.dbo.TB_TIC_PRDT AS d
            ON a.company_code = d.compy_cd
            AND a.product_code = d.prdt_cd
        JOIN mmapi.dbo.TB_COMM_CD e
            ON a.company_code = e.CD_ID
            AND e.UPP_CD_ID = 'COMPY'
        JOIN TB_MMLFCP_COVERAGE f
            ON a.coverage_cd = f.coverage_cd
        WHERE
            a.gender = ?
            AND a.age = ?
        ORDER BY
            a.company_code,
            c.coverage_seq
        """
        params = [plan_id, plan_id, gender, age]

        logger.info(
            f"보장별 보험료 조회 시작 - plan_id: {plan_id}, gender: {gender}, age: {age}"
        )

        results = self.execute_query(query, params)

        logger.info(f"보장별 보험료 조회 완료 - {len(results)}개 행 반환")

        return results


# 전역 데이터베이스 관리자 인스턴스
db_manager = DatabaseManager()
