# 기본 기술 스택

웹서버 : IIS10.0(윈도우 서버 2016 또는 2019)

UI프레임웍 : 바닐라 자바스크립트

CSS프레임웍 : 없는듯

주소 : 하나의 도메인이 아닌 다수의 서브도메인으로 운영 중임

1. 메인 페이지 : https://gc.bojang114.com/index_real.html
2. 생손보플랜 보험료 : https://mmlfcp.ohmymanager.com/index.html?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjb25zdWx0YW50aWQiOiJzdGVzdCIsImNvbnN1bHRhbnRfbmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNvbXB5X2NkIjoiQTI0NSIsImNvbXB5X25hbWUiOiJHQeuztO2XmOy7qOyEp O2MhSIsImV4cCI6MTc2OTczMjc2NywiY29uc3VsdGFudF9pZCI6InN0ZXN0IiwibmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNsaWVudF9pZCI6IkEyNDUiLCJtZG4iOiJBMjQ1Iiwicm9sZSI6IiIsImNsaWVudF9pcCI6IjEuMjA5LjE3MC41MCJ9.g0-Zt0Bca5nrkBX8n-rOIFxw2LVoNU-gGM1HjR6_HZQ
3. 한장 보험료 : https://mmcp.ohmymanager.com/html/index.html?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjb25zdWx0YW50aWQiOiJzdGVzdCIsImNvbnN1bHRhbnRfbmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNvbXB5X2NkIjoiQTI0NSIsImNvbXB5X25hbWUiOiJHQeuztO2XmOy7qOyEpO2MhSIsImV4cCI6MTc2OTczMjgwOSwiY29uc3VsdGFudF9pZCI6InN0ZXN0IiwibmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNsaWVudF9pZCI6IkEyNDUiLCJtZG4iOiJBMjQ1Iiwicm9sZSI6IiIsImNsaWVudF9pcCI6IjEuMjA5LjE3MC41MCJ9.mhrCkbpG1iX9TrAl0GXSKTSSu4SRD8lKpNmVK6VAl3A

## 메인페이지에서 서브페이지로 넘어갈때 크로스 도메인 인증 하는 기술

1. 토큰(아마도 JWT)를 메인도메인에서 발급 받음
   POST 메서드로 https://gc.bojang114.com/Common/mmflyerjwt 호출 이때 body에 {consultantid : 쿠키의 consultant_id 변수값} 을 담아서 토큰받음
   curl -X POST -d "{\"consultantid\" : \"stest\"}" https://gc.bojang114.com/Common/mmflyerjwt&AspxAutoDetectCookieSupport=1

2. 이후 서브도메인/index.html?token=발급받은토큰 으로 인증처리가 되어 페이지로 이동

# "한장으로 보는 생손보플랜 보험료 비교" 소스코드 분석

[한장으로 보는 생손보플랜 보험료 비교](https://mmlfcp.ohmymanager.com/index.html?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjb25zdWx0YW50aWQiOiJzdGVzdCIsImNvbnN1bHRhbnRfbmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNvbXB5X2NkIjoiQTI0NSIsImNvbXB5X25hbWUiOiJHQeuztO2XmOy7qOyEpO2MhSIsImV4cCI6MTc2OTY2ODc3MSwiY29uc3VsdGFudF9pZCI6InN0ZXN0IiwibmFtZSI6IuqwnOuwnOyekO2FjOyKpO2KuOqzhOyglSIsImNsaWVudF9pZCI6IkEyNDUiLCJtZG4iOiJBMjQ1Iiwicm9sZSI6IiIsImNsaWVudF9pcCI6IjEuMjA5LjE3MC41MCJ9._r_C62HK2xxyXu2Pd1CWFq02v3E29HwtyJ2NSkzB9XM)

## 최초 바닐라 자바스크립트에서 Auth 인증 실시

```javascript
export const BASE_URL = "/";
export const MMLFCP_URL = "/html/index.html";

export const API_MMLFCP_URL = {
    API_AUTH: "api/Auth",
    API_PRODUCT_PREMIUMS: "api/ProductPremiums", //플랜기준 상품 보험료 조회
    API_PRODUCT_PREMIUMS_BY_AGES: "api/ProductPremiumsByAges", //플랜 연령별 보험료 조회
    API_PRINT_PRODUCTS: "api/PrintProducts", // 플랜별 기준보장, 상품별 담보별, 필수보험료 정보 한장출력
    API_ADD_USER_COVERAGES: "api/AddUserCoverages", //사용자 플랜 추가
    API_UPDATE_USER_COVERAGES: "api/UpdateUserCoverages" //사용자 플랜 수정

};

export const appConstants = {
    jwt: '',
    access_path: '',
};

```

## Auth에서 인증 후 아래 응답값을 클라이언트에게 전송

```json
{
    "is_success": true,
    "error_message": "",
    "ga_id": "A245",
    "consultant_id": "stest",
    "plans": [
        {
            "plan_id": "921081111041",
            "plan_name": "생손보건강무해지",
            "plan_type": "01",
            "plan_type_name": "생손보건강 손보종합무해지",
            "plan_payterm_type": "01",
            "plan_payterm_type_name": "20년/100세",
            "plan_min_m_age": 15,
            "plan_max_m_age": 70,
            "plan_min_f_age": 15,
            "plan_max_f_age": 70
        },
        {
            "plan_id": "761081221041",
            "plan_name": "간편 3.3.5 세만기 무해지",
            "plan_type": "02",
            "plan_type_name": "생보335 손보335",
            "plan_payterm_type": "01",
            "plan_payterm_type_name": "20년/100세",
            "plan_min_m_age": 30,
            "plan_max_m_age": 80,
            "plan_min_f_age": 30,
            "plan_max_f_age": 80
        },
        {
            "plan_id": "771081231041",
            "plan_name": "간편 3.5.5 세만기 무해지",
            "plan_type": "03",
            "plan_type_name": "생보355 손보355",
            "plan_payterm_type": "01",
            "plan_payterm_type_name": "20년/100세",
            "plan_min_m_age": 30,
            "plan_max_m_age": 80,
            "plan_min_f_age": 30,
            "plan_max_f_age": 80
        },
        {
            "plan_id": "751081471041",
            "plan_name": "간편 3.10.10(5) 세만기 무해지",
            "plan_type": "04",
            "plan_type_name": "생보3.10.5(N) 손보3.10.5(N)",
            "plan_payterm_type": "01",
            "plan_payterm_type_name": "20년/100세",
            "plan_min_m_age": 30,
            "plan_max_m_age": 80,
            "plan_min_f_age": 30,
            "plan_max_f_age": 80
        }
    ]
}
```

## 이후 플랜별 보험상품 정보 다운로드

> 모두 공히 기준나이, 성별이 존재함 플랜내에 남여별 최소/최대 나이 범위정보를 내려줌

### API_PRODUCT_PREMIUMS

주요 4조각의 데이터가 제공됨

#### plan_coverages

화면 제일 좌측의 보장항목명과 코드, 기본 가입금액이 제공

```json
"plan_coverages": [
    {
        "plan_id": "921081111041",
        "coverage_cd": "ZZ00",
        "coverage_name": "주계약",
        "guide_coverage_amount": 100,
        "is_selected_coverage": "N",
        "coverage_seq": 0
    },
    {
        "plan_id": "921081111041",
        "coverage_cd": "aa01",
        "coverage_name": "상해후유장해(3~100%)",
        "guide_coverage_amount": 1000,
        "is_selected_coverage": "N",
        "coverage_seq": 1
    }
]
```

#### coverage_premiums

상품의 정보와 특약정보를 제공

특약정보 목록 제공되며, 보장코드(제일 좌측의 보장항목과 매핑해야함), 기본 보장금액, 기본 보험료 정보가 수록됨, 통합암(bb01)보장의 경우 다수의 특약으로 소분된것이 묶여 있음, 이걸 다 합산해야 구현이 가능함

> coverage_amount/premium이 guide_ 가 붙는것 2종이 있는데, 어떤것을 사용해야 하는지?
> 
>  is_selected_coverage는 UI용 선택/해제 상태 유지용 변수인지?
> 
> coverage_cd의 전체 리스트는? 
> 
> coverage_amount_ratio는?

```json
"coverage_premiums": [
    {
        "company_code": "DB",
        "company_name": "DB손해보험",
        "product_code": "42601002",
        "product_name": "무)참좋은훼밀리더블플러스종합보험2601",
        "product_detail_name": "납입중0%/납입후50%_프리미어골드클래스_납면적용",
        "product_conditions": "기본계약-상해사망후유20~100% 100만",
        "coverage_cd": "aa01",
        "coverage_name": "상해후유장해(3~100%)",
        "is_selected_coverage": "N",
        "coverage_seq": 1,
        "gender": "M",
        "age": 46,
        "guide_coverage_amount": 1000,
        "guide_coverage_premium": 720,
        "coverage_amount": 1000,
        "premium": 720,
        "coverage_amount_ratio": 1
    },
    {
        "company_code": "DB",
        "company_name": "DB손해보험",
        "product_code": "42601002",
        "product_name": "무)참좋은훼밀리더블플러스종합보험2601",
        "product_detail_name": "납입중0%/납입후50%_프리미어골드클래스_납면적용",
        "product_conditions": "기본계약-상해사망후유20~100% 100만",
        "coverage_cd": "aa02",
        "coverage_name": "상해(사망)후유장해(20~100%)",
        "is_selected_coverage": "N",
        "coverage_seq": 2,
        "gender": "M",
        "age": 46,
        "guide_coverage_amount": 1000,
        "guide_coverage_premium": 1360,
        "coverage_amount": 1000,
        "premium": 1360,
        "coverage_amount_ratio": 1
    }
]
```

#### product_insur_premiums

상품의 보장문구를 제공, 화면에서 상품별/보장별 보험료를 클릭하면 모달창으로 세부 특약 목록과 보장내용을 담은 문장이 노출

> insur_cd 리스트는? contract_amount와 premium은 coverage_premiums.coverage_amount/premium과 중복되는 정보 아닌가?

```json
"product_insur_premiums": [
        {
            "company_code": "DB",
            "product_code": "42601002",
            "product_name": "무)참좋은훼밀리더블플러스종합보험2601",
            "product_detail_name": "납입중0%/납입후50%_프리미어골드클래스_납면적용",
            "product_conditions": "기본계약-상해사망후유20~100% 100만",
            "pay_term": "20년/100세",
            "coverage_cd": "aa01",
            "gender": "M",
            "age": 46,
            "insur_cd": "10101",
            "insur_nm": "상해후유장해(3-100%)",
            "insur_bojang": "피보험자가 보험기간 중 상해사고로 후유장해(3%~100%)가 발생한 경우 가입\n금액에 후유장해지급률을 곱한 금액을 지급",
            "contract_amount": 1000,
            "premium": 720
        },
        {
            "company_code": "DB",
            "product_code": "42601002",
            "product_name": "무)참좋은훼밀리더블플러스종합보험2601",
            "product_detail_name": "납입중0%/납입후50%_프리미어골드클래스_납면적용",
            "product_conditions": "기본계약-상해사망후유20~100% 100만",
            "pay_term": "20년/100세",
            "coverage_cd": "aa02",
            "gender": "M",
            "age": 46,
            "insur_cd": "10201",
            "insur_nm": "상해사망·후유장해(20-100%)",
            "insur_bojang": "피보험자가 보험기간 중 상해사고로 사망한 경우에는 보험가입금액 지급하고,\n상해사고로 후유장해(20%~100%)가 발생한 경우에는 가입금액에 후유장해지\n급률을 곱한 금액을 지급",
            "contract_amount": 1000,
            "premium": 1360
        }
]
```

#### required_premiums

최좌측 세로 방향 보장목록의 최상단인 "최저기본계약조건"용 데이터로 파악됨

> 3덩어리를 연결할 FK가 회사코드 하나면 되나?(회사별 상품 1건), 아니면 회사코드+상품명까지 가야 하나?

```json
"required_premiums": [
        {
            "company_code": "DB",
            "company_name": "DB손해보험",
            "product_code": "42601002",
            "product_name": "무)참좋은훼밀리더블플러스종합보험2601",
            "product_detail_name": "납입중0%/납입후50%_프리미어골드클래스_납면적용",
            "product_conditions": "기본계약-상해사망후유20~100% 100만",
            "pay_term": "20년/100세",
            "gender": "M",
            "age": 46,
            "insur_cd": "10201",
            "insur_nm": "상해사망·후유장해(20-100%)",
            "insur_bojang": "피보험자가 보험기간 중 상해사고로 사망한 경우에는 보험가입금액 지급하고,\n상해사고로 후유장해(20%~100%)가 발생한 경우에는 가입금액에 후유장해지\n급률을 곱한 금액을 지급",
            "min_insur_amount": 100,
            "min_premium": 136,
            "contract_amount": 1000,
            "premium": 1360
        },
        {
            "company_code": "HA",
            "company_name": "한화손해보험",
            "product_code": "42601002",
            "product_name": "무)더건강한 한아름종합보험2601",
            "product_detail_name": "3종(납입면제형,납입후50%해약환급금지급형)_1형(일반고지형)_올인원플랜",
            "product_conditions": "기본계약-상해사망 100만",
            "pay_term": "20년납100세만기",
            "gender": "M",
            "age": 46,
            "insur_cd": "11201",
            "insur_nm": "보통약관(상해사망)",
            "insur_bojang": "보험기간 중에 상해의 직접 결과로써 사망한 경우(질병으로 인한 사망은 제외) 보험가입금액 지급",
            "min_insur_amount": 100,
            "min_premium": 74,
            "contract_amount": 1000,
            "premium": 740
        }

] 

```

### API_PRODUCT_PREMIUMS_BY_AGES

크게 coverage_premiums_by_ages, coverage_required_premiums_by_ages 2종의 데이터셋이 존재함

상세내용보기 모달창에서 연령대별 보험료 비교용으로 사용되며, 연령 구간은 총 5개임

1. 현재나이

2. 현재나이+1

3. 현재나이+2

4. 현재나이+5

5. 현재나이+10

이 부분은 상품별로 가입가능 나이 범위가 다르기도 함


