 ## 1. 개요
 본 문서는 보험 플랜(`plan_id`)을 기준으로 보장별 보험료, 담보별 상세 보험료, 필수 담보 보험료를 계산하고 플랜의 상세 정보를 조회하는 로직을 정의합니다.
 
 **입력 파라미터 예시:**
 *   `@plan_id`: '921081111041' (플랜 ID)
 *   `@gender`: 'M' (성별)
 *   `@age`: 40 (보험 나이)
 
 ---
 
 ## 2. 테이블 레이아웃 (Schema)
 로직에 사용되는 주요 테이블과 컬럼 정보입니다.
 
 ### 2.1. 플랜 및 상품 구성 테이블
 | 테이블명 | 설명 | 주요 컬럼 |
 | :--- | :--- | :--- |
 | **TB_MMLFCP_PLAN** | 플랜 기본 정보 | `plan_id`, `plan_name`, `plan_type`, `use_yn` |
 | **TB_MMLFCP_PLAN_PRODUCT** | 플랜에 포함된 보험사/상품 매핑 | `plan_id`, `company_code`, `product_code` |
 | **TB_MMLFCP_PLAN_COVERAGE** | 플랜별 보장(Coverage) 설정 | `plan_id`, `coverage_cd`, `guide_coverage_amount` (가입금액 가이드) |
 | **TB_MMLFCP_COVERAGE** | 보장 마스터 정보 | `coverage_cd`, `coverage_name` |
 | **TB_MMLFCP_COVERAGE_INSUR_MAPPING** | 보장(Coverage)과 실제 담보(Insur) 매핑 | `coverage_cd`, `insur_cd`, `guide_insur_amount` |
 | **TB_MMLFCP_PRODUCT_REQUIRED_RULES** | 상품별 필수 가입 담보 규칙 | `company_code`, `product_code`, `insur_cd`, `min_insur_amount` |
 
 ### 2.2. 보험료 및 상품 상세 테이블
 | 테이블명 | 설명 | 주요 컬럼 |
 | :--- | :--- | :--- |
 | **TB_MMLFCP_COVERAGE_PRICE** | 보장 단위 보험료 테이블 | `company_code`, `product_code`, `coverage_cd`, `gender`, `age`, `premium`, `coverage_amount` |
 | **TB_TIC_PRDT** | 보험 상품 마스터 | `compy_cd`, `prdt_cd`, `prdt_name` |
 | **TB_TIC_PRDT_D** | 보험 담보(특약) 상세 | `compy_cd`, `prdt_cd`, `insur_cd`, `insur_nm` |
 | **TB_TIC_PRDT_PRICE** | 담보(특약) 단위 보험료 테이블 | `compy_cd`, `prdt_cd`, `insur_cd`, `sex`, `age`, `std_contract_amt`, `premium` |
 | **TB_MMLFCP_AMOUNT_RATIO** | 보장 금액 비율 조정 | `company_code`, `product_code`, `coverage_cd`, `coverage_amount_ratio` |
 
 ### 2.3. 공통 코드
 | 테이블명 | 설명 | 주요 컬럼 |
 | :--- | :--- | :--- |
 | **TB_COMM_CD** | 공통 코드 관리 | `CD_ID`, `CD_NM`, `UPP_CD_ID` (그룹코드) |
 
 ---
 
 ## 3. 비즈니스 로직 (Query Logic)
 
 ### 3.1. 보장 보험료 비교 (Coverage Premium)
 플랜에 설정된 보장(`coverage_cd`)별로 보험료를 계산합니다.
 *   **계산식:** `(가이드금액 * 기준보험료) / 기준가입금액`
     *   `guide_coverage_premium` = `(TB_MMLFCP_PLAN_COVERAGE.guide_coverage_amount * TB_MMLFCP_COVERAGE_PRICE.premium) / TB_MMLFCP_COVERAGE_PRICE.coverage_amount`
 *   **특이사항:** `TB_MMLFCP_AMOUNT_RATIO` 테이블에 비율 정보가 있으면 가져오고, 없으면 1로 처리합니다.
 
 ### 3.2. 담보별 보험료 (Item Premium)
 보장(`coverage_cd`) 하위에 매핑된 실제 담보(`insur_cd`) 단위의 보험료를 계산합니다.
 *   **매핑:** `TB_MMLFCP_COVERAGE_INSUR_MAPPING` 테이블을 통해 보장과 담보를 연결합니다.
 *   **계산식:** `(가이드담보금액 * 담보기준보험료) / 담보기준가입금액`
     *   `guide_premium` = `(Mapping.guide_insur_amount * TB_TIC_PRDT_PRICE.premium) / TB_TIC_PRDT_PRICE.std_contract_amt`
 
 ### 3.3. 필수 담보 보험료 (Required Rules)
 상품 가입 시 반드시 포함되어야 하는 필수 담보(`Required Rules`)에 대한 보험료를 계산합니다.
 *   **조건:** `TB_MMLFCP_PRODUCT_REQUIRED_RULES` 테이블에 정의된 담보만 조회합니다.
 *   **계산식:** `(최소가입금액 * 담보기준보험료) / 담보기준가입금액`
     *   `premium` = `(Rules.min_insur_amount * TB_TIC_PRDT_PRICE.premium) / TB_TIC_PRDT_PRICE.std_contract_amt`
 
 ### 3.4. 플랜 조회 (Plan Info)
 플랜의 기본 속성을 조회하며, 공통 코드(`TB_COMM_CD`)를 조인하여 사람이 읽을 수 있는 명칭(보험사 타입, 플랜 타입, 납입 주기 등)을 가져옵니다.
 
 ---
 
 ## 4. 데이터 샘플 (Sample Data)
 각 테이블별로 관계형 데이터베이스의 참조 무결성을 고려하여 생성된 샘플 데이터입니다. (CSV 형식)
 
 #### A. TB_MMLFCP_PLAN (플랜)
plan_id,plan_name,plan_type,plan_payterm_type,insu_compy_type,use_yn
921081111041,프리미엄 암보험 플랜,TYPE_A,TERM_20Y,COMP_L,Y
921081111042,실속 상해 플랜,TYPE_B,TERM_10Y,COMP_S,Y
 #### B. TB_MMLFCP_PLAN_PRODUCT (플랜-상품 매핑)
plan_id,company_code,product_code,use_yn
921081111041,INS01,PRD001,Y
921081111041,INS02,PRD002,Y
921081111042,INS01,PRD003,Y
 #### C. TB_MMLFCP_PLAN_COVERAGE (플랜-보장 설정)
plan_id,coverage_cd,guide_coverage_amount,is_selected_coverage,coverage_seq,use_yn
921081111041,COV_CANCER,50000000,Y,1,Y
921081111041,COV_DEATH,100000000,Y,2,Y
921081111041,COV_SURGERY,10000000,N,3,Y
 #### D. TB_MMLFCP_COVERAGE_PRICE (보장별 기준 보험료)
company_code,product_code,coverage_cd,gender,age,coverage_amount,premium
INS01,PRD001,COV_CANCER,M,40,10000000,15000
INS01,PRD001,COV_DEATH,M,40,10000000,5000
INS02,PRD002,COV_CANCER,M,40,10000000,14500
INS02,PRD002,COV_DEATH,M,40,10000000,4800
 #### E. TB_TIC_PRDT (상품 마스터)
compy_cd,prdt_cd,prdt_name,attr1,mb_conditions,use_yn
INS01,PRD001,든든한암보험,순수보장형,만기환급없음,Y
INS02,PRD002,행복한종신보험,만기환급형,80세만기,Y
INS01,PRD003,간편상해보험,갱신형,3년갱신,Y
 #### F. TB_TIC_PRDT_D (담보 상세)
compy_cd,prdt_cd,insur_cd,insur_nm,insur_bojang,pay_term
INS01,PRD001,TRT_001,일반암진단비,암진단시 지급,20년납
INS01,PRD001,TRT_002,유사암진단비,유사암진단시 지급,20년납
INS01,PRD001,TRT_MAIN,주계약(사망),사망시 지급,20년납
INS02,PRD002,TRT_001,일반암진단비,암진단시 지급,20년납
 #### G. TB_TIC_PRDT_PRICE (담보별 기준 보험료)
compy_cd,prdt_cd,insur_cd,sex,age,std_contract_amt,premium,use_yn
INS01,PRD001,TRT_001,M,40,10000000,12000,Y
INS01,PRD001,TRT_002,M,40,10000000,1000,Y
INS01,PRD001,TRT_MAIN,M,40,50000000,25000,Y
INS02,PRD002,TRT_001,M,40,10000000,11500,Y
 #### H. TB_MMLFCP_COVERAGE_INSUR_MAPPING (보장-담보 매핑)
coverage_cd,insur_cd,guide_insur_amount
COV_CANCER,TRT_001,30000000
COV_CANCER,TRT_002,10000000
COV_DEATH,TRT_MAIN,100000000
 #### I. TB_MMLFCP_PRODUCT_REQUIRED_RULES (필수 담보 규칙)
company_code,product_code,insur_cd,min_insur_amount
INS01,PRD001,TRT_MAIN,10000000
INS02,PRD002,TRT_MAIN,5000000
 #### J. TB_COMM_CD (공통 코드)
CD_ID,CD_NM,UPP_CD_ID,ORDER_SEQ
INS01,A생명,COMPY,1
INS02,B화재,COMPY,2
TYPE_A,종합형,MMLFCP_A,1
TERM_20Y,20년납,MMLFCP_B,1
COMP_L,생명보험,MMLFCP_C,1
 #### K. TB_MMLFCP_COVERAGE (보장 마스터)
coverage_cd,coverage_name,use_yn
COV_CANCER,암보장,Y
COV_DEATH,사망보장,Y
COV_SURGERY,수술비보장,Y
 #### L. TB_MMLFCP_AMOUNT_RATIO (금액 비율)
company_code,product_code,coverage_cd,coverage_amount_ratio
INS01,PRD001,COV_CANCER,1.0
INS01,PRD001,COV_DEATH,1.0