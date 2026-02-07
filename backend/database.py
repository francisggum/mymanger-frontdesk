"""
MSSQL 데이터베이스 연결 모듈
"""

import logging
import os
import pyodbc
import pandas as pd
import math
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """MSSQL 데이터베이스 관리자"""

    def __init__(self):
        self.server = os.getenv("DB_HOST", "localhost")
        self.database = os.getenv("DB_NAME", "mmapi")
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
            plan["plan_name"] = plan.get("plan_type_name", "")

            # 나이 필드가 None인 경우 0으로 설정
            for age_field in ["min_m_age", "max_m_age", "min_f_age", "max_f_age"]:
                if plan.get(age_field) is None:
                    plan[age_field] = 0

        return plans

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
        select
            a.compy_cd as company_code,
            a.prdt_cd as product_code,
            c.prdt_name as product_name,
            e.coverage_cd as coverage_cd,
            a.insur_cd as insur_item_code,
            d.insur_nm as insur_item_name,
            d.insur_bojang as insur_item_coverage,
            d.pay_term as payment_due,
            e.guide_contract_amount as guide_contract_amount,
            case
                when a.std_contract_amt <= 0 then 0
                else (e.guide_contract_amount * a.premium) / a.std_contract_amt
            end as guide_premium,
            (select top 1 CD_NM from mmapi.dbo.TB_COMM_CD 
             where CD_ID = a.compy_cd and UPP_CD_ID = 'COMPY') as company_name,
            (select top 1 coverage_name from TB_MMLFCP_COVERAGE 
             where coverage_cd = e.coverage_cd) as coverage_name
        from
            mmapi.dbo.TB_TIC_PRDT_PRICE a
        join TB_MMLFCP_PLAN_PRODUCT b
            on a.compy_cd = b.company_code
            and a.prdt_cd = b.product_code
            and b.plan_id = ?
        join mmapi.dbo.TB_TIC_PRDT c
            on a.compy_cd = c.compy_cd
            and a.prdt_cd = c.prdt_cd
        join mmapi.dbo.TB_TIC_PRDT_D d
            on a.compy_cd = d.compy_cd
            and a.prdt_cd = d.prdt_cd
            and a.insur_cd = d.insur_cd
        join (
            select
                a.coverage_cd,
                b.insur_cd,
                b.guide_insur_amount as guide_contract_amount
            from
                TB_MMLFCP_PLAN_COVERAGE a
            join TB_MMLFCP_COVERAGE_INSUR_MAPPING b
                on a.coverage_cd = b.coverage_cd
            where
                a.plan_id = ?
                and a.use_yn = 'Y'
        ) e on a.insur_cd = e.insur_cd
        where
            a.sex = ?
            and a.age = ?
        order by
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
                    f"행 {i+1} - guide_premium: {guide_premium} (타입: {type(guide_premium)}), guide_contract_amount: {guide_contract_amount} (타입: {type(guide_contract_amount)})"
                )

                # 비정상 값 확인
                if isinstance(guide_premium, (int, float)):
                    if isinstance(guide_premium, float) and (
                        math.isnan(guide_premium) or math.isinf(guide_premium)
                    ):
                        logger.warning(
                            f"행 {i+1} - guide_premium에 비정상 값: {guide_premium}"
                        )

                if isinstance(guide_contract_amount, (int, float)):
                    if isinstance(guide_contract_amount, float) and (
                        math.isnan(guide_contract_amount)
                        or math.isinf(guide_contract_amount)
                    ):
                        logger.warning(
                            f"행 {i+1} - guide_contract_amount에 비정상 값: {guide_contract_amount}"
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
                        lambda x: isinstance(x, float)
                        and (math.isnan(x) or math.isinf(x))
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
                        lambda x: isinstance(x, float)
                        and (math.isnan(x) or math.isinf(x))
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
                lambda row: f"{self._safe_float_format(row['sum_premium'], '', '')}({row['guide_premium_list']})",
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
                    coverage = {
                        "coverage_name": row["coverage_name"],
                        "coverage_code": row["coverage_cd"],
                        "guide_contract_amount_max": row["guide_contract_amount_max"],
                        "sum_premium": row["sum_premium"],
                        "guide_premium_list": row["guide_premium_list"],
                        # "insur_item_name_list": row["insur_item_name_list"],
                        # "insur_item_name_coverage_list": row[
                        #     "insur_item_name_coverage_list"
                        # ],
                    }
                    coverages.append(coverage)

                result[company_key] = coverages

            logger.info(f"LLM용 데이터 생성 완료: {len(result)}개 회사")
            return result

        except Exception as e:
            logger.error(f"LLM용 데이터 생성 실패: {e}")
            return {}


# 전역 데이터베이스 관리자 인스턴스
db_manager = DatabaseManager()
