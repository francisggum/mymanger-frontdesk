"""
보험 올림픽 페이지

보험 비교 AI - 보험상품 랭킹 및 통계 기능
"""

import streamlit as st
import logging
import json
import re
import pandas as pd
from pathlib import Path
from io import StringIO

from config import PAGE_CONFIG
from utils.session import (
    init_session_state,
    set_session_value,
    get_session_value,
    has_plans,
    get_selected_plan,
)
from utils.api import fetch_plans, get_comparison_tables
from config import MODEL_OPTIONS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 데이터 파일 경로
COVERAGE_TIERS_FILE = Path(__file__).parent.parent / "data" / "coverage_tiers.json"
PLAN_CATEGORY_FILE = (
    Path(__file__).parent.parent / "data" / "plan_category_mapping.json"
)


def load_plan_category_mapping():
    """종목별 플랜 매핑 데이터 로드"""
    try:
        with open(PLAN_CATEGORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"종목별 플랜 매핑 데이터 로드 실패: {e}")
        return pd.DataFrame()


def save_plan_category_mapping(df):
    """종목별 플랜 매핑 데이터 저장"""
    try:
        with open(PLAN_CATEGORY_FILE, "w", encoding="utf-8") as f:
            json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"종목별 플랜 매핑 데이터 저장 실패: {e}")
        return False


def get_plan_category(plan_type_name):
    """플랜 타입명으로 종목 반환"""
    mapping_df = load_plan_category_mapping()
    if mapping_df.empty:
        return "건강"  # 기본값

    # 세부플랜 컬럼에서 일치하는 행 찾기
    match = mapping_df[mapping_df["세부플랜"] == plan_type_name]
    if not match.empty:
        return match.iloc[0]["종목"]
    return "건강"  # 기본값


def calculate_wcei_ranking(llm_data, coverage_tiers_df, plan_type_name):
    """WCEI 랭킹 계산 (llm_readable_data 사용)"""
    try:
        # 1. 플랜 종목 확인
        category = get_plan_category(plan_type_name)

        # 2. 해당 종목의 보장항목 필터링
        category_col_map = {
            "건강": "건강보험",
            "어린이": "건강보험",
            "청소년": "건강보험",
            "실손": "실손보험",
            "치매": "치매보험",
            "치아": "치아보험",
            "운전자": "운전자보험",
        }
        category_col = category_col_map.get(category, "건강보험")

        # 해당 종목이 true인 보장항목만 필터링
        filtered_tiers = coverage_tiers_df[
            coverage_tiers_df[category_col] == True
        ].copy()

        if filtered_tiers.empty:
            logger.warning(f"해당 종목({category})의 보장항목이 없습니다.")
            return pd.DataFrame()

        # 3. 보장코드 → 티어 매핑 생성 (소문자로 정규화)
        tier_map = {
            k.lower(): v
            for k, v in zip(filtered_tiers["보장코드"], filtered_tiers["티어"])
        }

        # 4. 데이터 수집
        scores_data = []
        company_coverage_map = {}  # 보험사별 보장항목 목록

        for company_key, coverages in llm_data.items():
            # 보험사명 추출 (형식: "회사명(회사코드)")
            company_name = (
                company_key.split("(")[0] if "(" in company_key else company_key
            )

            if company_name not in company_coverage_map:
                company_coverage_map[company_name] = {}

            for coverage in coverages:
                coverage_code = coverage.get("coverage_code", "").lower()
                coverage_name = coverage.get("coverage_name", "")
                premium = coverage.get("sum_premium", 0)

                # 해당 종목의 보장항목인지 확인
                if coverage_code not in tier_map:
                    continue

                tier = tier_map[coverage_code]

                scores_data.append(
                    {
                        "company": company_name,
                        "coverage_code": coverage_code,
                        "coverage_name": coverage_name,
                        "tier": tier,
                        "premium": premium,
                    }
                )

                # 보험사별 보장항목 기록
                company_coverage_map[company_name][coverage_code] = {
                    "name": coverage_name,
                    "premium": premium,
                }

        if not scores_data:
            return pd.DataFrame()

        scores_df = pd.DataFrame(scores_data)

        # 5. 각 보장항목별 점수 계산
        # 보장코드별로 그룹화하여 최저가/최고가 계산
        coverage_stats = {}
        for coverage_code in scores_df["coverage_code"].unique():
            coverage_data = scores_df[scores_df["coverage_code"] == coverage_code]
            premiums = list(coverage_data["premium"])
            valid_premiums = [p for p in premiums if p > 0]

            if valid_premiums:
                coverage_stats[coverage_code] = {
                    "min": min(valid_premiums),
                    "max": max(valid_premiums),
                }

        # 점수 계산
        def calc_score(row):
            coverage_code = row["coverage_code"]
            premium = row["premium"]

            if premium == 0:
                return 0

            if coverage_code not in coverage_stats:
                return 0

            min_premium = coverage_stats[coverage_code]["min"]
            max_premium = coverage_stats[coverage_code]["max"]

            if min_premium == max_premium:
                return 1.0 if premium > 0 else 0

            return 0.1 + 0.9 * ((max_premium - premium) / (max_premium - min_premium))

        scores_df["score"] = scores_df.apply(calc_score, axis=1)

        # 6. 티어별 평균 점수 계산
        tier_scores = {}
        for tier in [1, 2, 3]:
            tier_data = scores_df[scores_df["tier"] == tier]
            if not tier_data.empty:
                avg_score = tier_data.groupby("company")["score"].mean().to_dict()
                tier_scores[tier] = avg_score
            else:
                tier_scores[tier] = {}

        # 7. 각 보험사별 누락된 보장항목 확인 (보험료가 0원인 경우)
        company_missing_map = {}
        all_coverage_codes_in_tier = set(tier_map.keys())

        for company_name, coverages in company_coverage_map.items():
            missing = []
            for coverage_code in all_coverage_codes_in_tier:
                if (
                    coverage_code not in coverages
                    or coverages[coverage_code]["premium"] == 0
                ):
                    # coverage_code로 보장명 찾기
                    coverage_name = filtered_tiers[
                        filtered_tiers["보장코드"].str.lower() == coverage_code
                    ]["보장내용"].values
                    if len(coverage_name) > 0:
                        missing.append(coverage_name[0])
            if missing:
                company_missing_map[company_name] = missing

        # 8. 최종 점수 계산 (100점 만점)
        companies = list(company_coverage_map.keys())
        final_scores = []

        for company in companies:
            tier1_score = tier_scores.get(1, {}).get(company, 0)
            tier2_score = tier_scores.get(2, {}).get(company, 0)
            tier3_score = tier_scores.get(3, {}).get(company, 0)

            total_score = (tier1_score * 50) + (tier2_score * 30) + (tier3_score * 20)

            # 누락된 보장항목 가져오기
            company_missing = company_missing_map.get(company, [])

            # 보험사별 총 보험료 계산
            company_coverages = company_coverage_map.get(company, {})
            total_premium = sum(
                coverage_info["premium"] for coverage_info in company_coverages.values()
            )

            final_scores.append(
                {
                    "보험사": company,
                    "보험사_표시": company,  # 원래 보험사명 저장
                    "총점": round(total_score, 1),
                    "보험료": f"{int(total_premium):,}",
                    "Tier1": round(tier1_score * 50, 1),
                    "Tier2": round(tier2_score * 30, 1),
                    "Tier3": round(tier3_score * 20, 1),
                    "누락건수": len(company_missing),
                    "누락보장": company_missing if company_missing else [],
                }
            )

        ranking_df = pd.DataFrame(final_scores)
        ranking_df = ranking_df.sort_values("총점", ascending=False).reset_index(
            drop=True
        )
        ranking_df.index = ranking_df.index + 1  # 순위는 1부터 시작
        ranking_df.index.name = "순위"

        # 1/2/3등에 메달 아이콘 추가 (표시용)
        def add_medal_icon(row):
            rank = row.name
            company = row["보험사"]
            if rank == 1:
                return f"🥇 {company}"
            elif rank == 2:
                return f"🥈 {company}"
            elif rank == 3:
                return f"🥉 {company}"
            return company

        ranking_df["보험사_표시"] = ranking_df.apply(add_medal_icon, axis=1)

        # 총 평가 건수 계산
        total_coverage_count = len(filtered_tiers)

        return ranking_df, total_coverage_count

    except Exception as e:
        logger.error(f"WCEI 랭킹 계산 실패: {e}")
        return pd.DataFrame(), 0


def load_coverage_tiers():
    """보장 티어 데이터 로드"""
    try:
        with open(COVERAGE_TIERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"보장 티어 데이터 로드 실패: {e}")
        return pd.DataFrame()


def save_coverage_tiers(df):
    """보장 티어 데이터 저장"""
    try:
        with open(COVERAGE_TIERS_FILE, "w", encoding="utf-8") as f:
            json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"보장 티어 데이터 저장 실패: {e}")
        return False


def render_model_selector():
    """AI 모델 선택 (사이드바)"""
    st.sidebar.subheader("🤖 AI 모델 선택")

    current_model = get_session_value("selected_model", "openai")

    ui_options = list(MODEL_OPTIONS.keys())
    current_ui_label = None
    for label, value in MODEL_OPTIONS.items():
        if value == current_model:
            current_ui_label = label
            break

    if not current_ui_label:
        current_ui_label = ui_options[0]

    selected_ui_label = st.sidebar.radio(
        "사용할 AI 모델",
        options=ui_options,
        index=ui_options.index(current_ui_label),
        help="질의응답에 사용할 AI 모델을 선택하세요",
        label_visibility="collapsed",
    )

    selected_model_value = MODEL_OPTIONS[selected_ui_label]
    set_session_value("selected_model", selected_model_value)

    st.sidebar.caption(f"선택된 모델: **{selected_ui_label}**")


def render_plan_loader():
    """플랜 목록 로드 (사이드바)"""
    plans = get_session_value("plans", [])

    if not plans:
        with st.sidebar.spinner("플랜 목록 로딩 중..."):
            response = fetch_plans()
            if response and len(response) > 0:
                set_session_value("plans", response)
                st.sidebar.success(f"✅ 총 {len(response)}개의 플랜 로드됨")
                st.rerun()
            else:
                st.sidebar.error("❌ 플랜 목록 로딩 실패")
    else:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.sidebar.success(f"✅ 총 {len(plans)}개 플랜 로드됨")
        with col2:
            if st.sidebar.button("🔄", help="플랜 목록 새로고침"):
                with st.sidebar.spinner("플랜 목록 새로고침 중..."):
                    response = fetch_plans()
                    if response:
                        set_session_value("plans", response)
                        st.sidebar.success(f"✅ {len(response)}개 플랜 새로고침 완료!")
                        st.rerun()


def render_plan_selector():
    """플랜 선택 (사이드바)"""
    plans = get_session_value("plans", [])

    plan_options = {
        f"{plan['plan_type_name']} ({plan['payment_due_type_name']})": plan
        for plan in plans
    }

    selected_key = get_session_value("selected_plan_key")
    if not selected_key or selected_key not in plan_options:
        # 기존 세션 키가 새 형식과 다륾면 초기화
        selected_key = list(plan_options.keys())[0]
        set_session_value("selected_plan_key", selected_key)
        set_session_value("data_loaded", False)  # 기존 분석 데이터도 초기화

    selected = st.sidebar.selectbox(
        "플랜 선택",
        list(plan_options.keys()),
        index=list(plan_options.keys()).index(selected_key),
    )

    set_session_value("selected_plan_key", selected)


def render_plan_info():
    """선택된 플랜 정보 (사이드바)"""
    plan = get_selected_plan()
    if not plan:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 플랜 정보")

    min_m_age = plan.get("min_m_age", 0)
    max_m_age = plan.get("max_m_age", 0)
    min_f_age = plan.get("min_f_age", 0)
    max_f_age = plan.get("max_f_age", 0)

    if min_m_age == 0 and max_m_age == 0:
        st.sidebar.info("👩 이 플랜은 **여성** 전용입니다")
        st.sidebar.write(f"👤 나이 조건: {min_f_age}세 ~ {max_f_age}세")
    elif min_f_age == 0 and max_f_age == 0:
        st.sidebar.info("👨 이 플랜은 **남성** 전용입니다")
        st.sidebar.write(f"👤 나이 조건: {min_m_age}세 ~ {max_m_age}세")
    else:
        st.sidebar.info("👫 이 플랜은 **남녀 공통**입니다")
        male_range = (
            f"남성: {min_m_age}~{max_m_age}세" if min_m_age > 0 else "남성: 불가"
        )
        female_range = (
            f"여성: {min_f_age}~{max_f_age}세" if min_f_age > 0 else "여성: 불가"
        )
        st.sidebar.write(f"   • {male_range}")
        st.sidebar.write(f"   • {female_range}")


def render_analysis_form():
    """데이터 분석 폼 (사이드바)"""
    plan = get_selected_plan()
    if not plan:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 가입 조건 입력")

    min_m_age = plan.get("min_m_age", 0)
    max_m_age = plan.get("max_m_age", 0)
    min_f_age = plan.get("min_f_age", 0)
    max_f_age = plan.get("max_f_age", 0)

    if min_m_age == 0 and max_m_age == 0:
        gender_options = ["여성"]
        default_gender = "여성"
        min_age, max_age = min_f_age, max_f_age
    elif min_f_age == 0 and max_f_age == 0:
        gender_options = ["남성"]
        default_gender = "남성"
        min_age, max_age = min_m_age, max_m_age
    else:
        gender_options = ["남성", "여성"]
        default_gender = "남성"
        min_age = (
            max(min_m_age, min_f_age)
            if min_m_age > 0 and min_f_age > 0
            else max(min_m_age, min_f_age)
        )
        max_age = (
            min(max_m_age, max_f_age)
            if max_m_age > 0 and max_f_age > 0
            else max(max_m_age, max_f_age)
        )

    gender = st.sidebar.radio(
        "성별", gender_options, index=gender_options.index(default_gender)
    )
    gender_code = "M" if gender == "남성" else "F"

    if min_age > 0 and max_age > 0:
        default_age = min((min_age + max_age) // 2, min_age + 1)
        default_age = max(min_age, min(default_age, max_age))
        age = st.sidebar.number_input(
            "나이",
            min_value=min_age,
            max_value=max_age,
            value=default_age,
            help=f"{min_age}세에서 {max_age}세까지 입력 가능합니다",
        )
    else:
        age = st.sidebar.number_input("나이", min_value=0, max_value=100, value=46)

    # int 타입으로 변환
    age = int(age)

    if st.sidebar.button("데이터 분석 시작", type="secondary"):
        if age < min_age or age > max_age:
            st.sidebar.error(
                f"❌ 나이를 {min_age}세에서 {max_age}세 사이로 입력해주세요"
            )
        else:
            with st.sidebar.spinner(f"데이터 분석 중..."):
                response = get_comparison_tables(plan["plan_id"], age, gender_code)

                if response:
                    set_session_value("data_loaded", True)
                    set_session_value(
                        "current_plan",
                        f"{plan['plan_type_name']} ({plan['payment_due_type_name']})",
                    )
                    set_session_value("current_gender", gender)
                    set_session_value("current_age", age)
                    set_session_value("plan_data", response)
                    set_session_value(
                        "human_readable_table", response.get("human_readable_table")
                    )
                    set_session_value(
                        "llm_readable_data", response.get("llm_readable_data")
                    )
                    set_session_value("comparison_summary", response.get("summary", {}))

                    summary = response.get("summary", {})
                    st.sidebar.success(
                        f"데이터 분석 완료!\n"
                        f"• 총 보험사 수: {summary.get('total_companies', 0)}개\n"
                        f"• 총 보장 항목: {summary.get('total_coverages', 0)}개"
                    )
                else:
                    st.sidebar.error("데이터 분석에 실패했습니다.")


@st.dialog("⚙️ 보장 티어 설정", width="large")
def show_tier_settings_dialog():
    """보장 티어 설정 다이얼로그"""
    # 데이터 로드
    df = load_coverage_tiers()

    if df.empty:
        st.error("❌ 보장 티어 데이터를 불러올 수 없습니다.")
        if st.button("닫기", use_container_width=True):
            st.rerun()
        return

    # 데이터 에디터 표시
    st.write(f"총 {len(df)}개의 보장 항목")

    edited_df = st.data_editor(
        df,
        column_config={
            "보장코드": st.column_config.TextColumn(
                "보장코드", disabled=True, width="small"
            ),
            "보장내용": st.column_config.TextColumn(
                "보장내용", disabled=True, width="large"
            ),
            "티어": st.column_config.SelectboxColumn(
                "티어", options=[1, 2, 3], required=True
            ),
            "건강보험": st.column_config.CheckboxColumn("건강보험"),
            "치아보험": st.column_config.CheckboxColumn("치아보험"),
            "치매보험": st.column_config.CheckboxColumn("치매보험"),
            "운전자보험": st.column_config.CheckboxColumn("운전자보험"),
            "실손보험": st.column_config.CheckboxColumn("실손보험"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",  # 행 추가/삭제 불가
        height=500,
    )

    st.markdown("---")

    # 버튼 영역
    col1, col2 = st.columns(2)

    with col1:
        # 저장 버튼
        if st.button("💾 저장", type="primary", use_container_width=True):
            if save_coverage_tiers(edited_df):
                st.success("✅ 설정이 저장되었습니다!")
                st.rerun()
            else:
                st.error("❌ 저장에 실패했습니다.")

    with col2:
        # 닫기 버튼
        if st.button("닫기", use_container_width=True):
            st.rerun()


@st.dialog("🗂️ 종목별 플랜 그룹 설정", width="large")
def show_plan_category_dialog():
    """종목별 플랜 그룹 설정 다이얼로그"""
    # 데이터 로드
    df = load_plan_category_mapping()

    if df.empty:
        st.error("❌ 종목별 플랜 매핑 데이터를 불러올 수 없습니다.")
        if st.button("닫기", use_container_width=True):
            st.rerun()
        return

    # 데이터 에디터 표시
    st.write(f"총 {len(df)}개의 플랜 매핑")

    edited_df = st.data_editor(
        df,
        column_config={
            "종목": st.column_config.SelectboxColumn(
                "종목",
                options=["건강", "어린이", "청소년", "실손", "치매", "치아", "운전자"],
                required=True,
            ),
            "세부플랜": st.column_config.TextColumn("세부 플랜", required=True),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",  # 행 추가/삭제 가능
        height=500,
    )

    st.markdown("---")

    # 버튼 영역
    col1, col2 = st.columns(2)

    with col1:
        # 저장 버튼
        if st.button("💾 저장", type="primary", use_container_width=True):
            if save_plan_category_mapping(edited_df):
                st.success("✅ 설정이 저장되었습니다!")
                st.rerun()
            else:
                st.error("❌ 저장에 실패했습니다.")

    with col2:
        # 닫기 버튼
        if st.button("닫기", use_container_width=True):
            st.rerun()


def show_company_detail_breakdown(
    selected_company,
    llm_data,
    coverage_tiers_df,
    plan_type_name,
    ranking_df,
    total_coverage_count,
):
    """선택된 보험사의 세부 득점표 표시"""

    # 선택된 보험사의 랭킹 정보 가져오기
    company_row = ranking_df[ranking_df["보험사"] == selected_company]
    if company_row.empty:
        st.error(f"보험사 '{selected_company}'의 랭킹 정보를 찾을 수 없습니다.")
        return

    company_info = company_row.iloc[0]
    rank = company_row.index[0]  # 인덱스가 순위
    total_score = company_info["총점"]
    premium = company_info["보험료"]
    tier1 = company_info["Tier1"]
    tier2 = company_info["Tier2"]
    tier3 = company_info["Tier3"]
    missing_count = company_info["누락건수"]

    # 플랜 종목 확인
    category = get_plan_category(plan_type_name)
    category_col_map = {
        "건강": "건강보험",
        "어린이": "건강보험",
        "청소년": "건강보험",
        "실손": "실손보험",
        "치매": "치매보험",
        "치아": "치아보험",
        "운전자": "운전자보험",
    }
    category_col = category_col_map.get(category, "건강보험")

    # 해당 종목의 보장항목 필터링
    filtered_tiers = coverage_tiers_df[coverage_tiers_df[category_col] == True].copy()

    if filtered_tiers.empty:
        st.warning("해당 종목의 보장항목이 없습니다.")
        return

    # 1. 모든 회사 목록 수집
    all_companies = set()
    for company_key in llm_data.keys():
        company_name = company_key.split("(")[0] if "(" in company_key else company_key
        all_companies.add(company_name)

    # 2. 모든 보험사 데이터 수집 (모든 회사를 포함하도록 초기화)
    all_coverage_data = {}  # {coverage_code: {company: premium}}

    for company_key, coverages in llm_data.items():
        company_name = company_key.split("(")[0] if "(" in company_key else company_key

        for coverage in coverages:
            coverage_code = coverage.get("coverage_code", "").lower()
            coverage_premium = coverage.get("sum_premium", 0)

            if coverage_code not in all_coverage_data:
                # 새 보장항목: 모든 회사를 0원으로 초기화
                all_coverage_data[coverage_code] = {comp: 0 for comp in all_companies}

            all_coverage_data[coverage_code][company_name] = coverage_premium

    # 세부 득점표 데이터 생성
    detail_rows = []

    for tier in [1, 2, 3]:
        # 해당 티어의 보장항목 (보장명 순으로 정렬)
        tier_coverages = filtered_tiers[filtered_tiers["티어"] == tier].sort_values(
            "보장내용"
        )

        tier_total_premium = 0
        tier_scores = []

        for _, coverage_row in tier_coverages.iterrows():
            coverage_code = coverage_row["보장코드"].lower()
            coverage_name = coverage_row["보장내용"]
            contract_amount = coverage_row.get(
                "보장금액", coverage_row.get("guide_contract_amount", "0")
            )

            # 해당 보장의 전체 보험사 데이터
            coverage_data = all_coverage_data.get(coverage_code, {})

            # coverage_data가 없으면 모든 보험사 0원으로 초기화 (아우터 조인)
            if not coverage_data:
                coverage_data = {comp: 0 for comp in all_companies}

            # 선택된 회사의 보험료
            company_premium = coverage_data.get(selected_company, 0)

            # 전체 랭킹 계산
            all_premiums = [
                (comp, prem) for comp, prem in coverage_data.items() if prem > 0
            ]
            all_premiums.sort(key=lambda x: x[1])  # 보험료 낮은 순

            if company_premium > 0:
                item_rank = next(
                    (
                        i
                        for i, (comp, _) in enumerate(all_premiums, 1)
                        if comp == selected_company
                    ),
                    len(all_premiums),
                )
                total = len(all_premiums)

                # 1등/꼴찌 정보
                first_company, first_premium = (
                    all_premiums[0] if all_premiums else ("-", 0)
                )
                last_company, last_premium = (
                    all_premiums[-1] if all_premiums else ("-", 0)
                )

                # 점수 계산
                min_prem = min(p for _, p in all_premiums) if all_premiums else 0
                max_prem = max(p for _, p in all_premiums) if all_premiums else 0

                if max_prem == min_prem:
                    score = 1.0
                else:
                    score = 0.1 + 0.9 * (
                        (max_prem - company_premium) / (max_prem - min_prem)
                    )

                tier_scores.append(score)
                tier_total_premium += company_premium
                is_missing = False
            else:
                item_rank = "-"
                total = len(all_premiums)
                first_company = all_premiums[0][0] if all_premiums else "-"
                first_premium = all_premiums[0][1] if all_premiums else 0
                last_company = all_premiums[-1][0] if all_premiums else "-"
                last_premium = all_premiums[-1][1] if all_premiums else 0
                score = 0
                is_missing = True

            # 누락 회사 목록 (보험료 0원인 회사)
            missing_companies = [
                comp for comp, prem in coverage_data.items() if prem == 0
            ]

            # 등수에 메달 아이콘 추가
            if item_rank == 1:
                rank_display = f"🥇 1/{total}"
            elif item_rank == 2:
                rank_display = f"🥈 2/{total}"
            elif item_rank == 3:
                rank_display = f"🥉 3/{total}"
            elif item_rank != "-":
                rank_display = f"{item_rank}/{total}"
            else:
                rank_display = "-"

            detail_rows.append(
                {
                    "티어": tier,
                    "보장항목": f"{coverage_name}({coverage_row['보장코드']})",
                    "스코어": round(score, 2),
                    "보험료": f"{company_premium:,.0f}",
                    "등수": rank_display,
                    "1등": (
                        f"{first_company}/{first_premium:,.0f}"
                        if first_premium > 0
                        else "-"
                    ),
                    "꼴찌": (
                        f"{last_company}/{last_premium:,.0f}"
                        if last_premium > 0
                        else "-"
                    ),
                    "누락회사수": len(missing_companies),
                    "누락회사": missing_companies if missing_companies else [],
                    "is_missing": is_missing,
                    "is_summary": False,
                }
            )

        # 티어 소계 추가
        if tier_scores:
            avg_score = sum(tier_scores) / len(tier_scores)
            detail_rows.append(
                {
                    "티어": tier,
                    "보장항목": f"📊 Tier {tier} 소계 (평균: {avg_score:.2f})",
                    "스코어": round(avg_score, 2),
                    "보험료": f"{tier_total_premium:,}",
                    "등수": "-",
                    "1등": "-",
                    "꼴찌": "-",
                    "누락회사수": 0,
                    "누락회사": [],
                    "is_missing": False,
                    "is_summary": True,
                }
            )

    # DataFrame 생성 및 표시
    if detail_rows:
        # 메달 현황 계산 (각 보장항목별 1/2/3등 카운트)
        gold_count = sum(
            1
            for row in detail_rows
            if row.get("is_summary") == False and row.get("등수", "").startswith("🥇")
        )
        silver_count = sum(
            1
            for row in detail_rows
            if row.get("is_summary") == False and row.get("등수", "").startswith("🥈")
        )
        bronze_count = sum(
            1
            for row in detail_rows
            if row.get("is_summary") == False and row.get("등수", "").startswith("🥉")
        )

        # 순위 아이콘 설정
        if rank == 1:
            rank_icon = "🥇"
        elif rank == 2:
            rank_icon = "🥈"
        elif rank == 3:
            rank_icon = "🥉"
        else:
            rank_icon = ""

        # 대시보드 계기판 표시
        st.subheader(f"📊 {selected_company} 성과 대시보드")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            try:
                rank_value = int(rank)
                st.metric(
                    label="🏆 랭킹 순위",
                    value=f"{rank_icon} {rank_value}위",
                    border=True,
                )
            except (ValueError, TypeError):
                st.metric(label="🏆 랭킹 순위", value=f"{rank_icon} -위", border=True)

        with col2:
            st.metric(label="💰 보험료", value=f"{premium}원", border=True)

        with col3:
            st.metric(
                label="⭐ 총점",
                value=f"{total_score:.1f}점({tier1:.1f} + {tier2:.1f} + {tier3:.1f})",
                border=True,
            )

        with col4:
            st.metric(
                label="🎖️ 메달 현황",
                value=f"🥇{gold_count} 🥈{silver_count} 🥉{bronze_count}",
                border=True,
            )

        with col5:
            st.metric(
                label="⚠️ 부족 보장",
                value=f"{missing_count}/{total_coverage_count}",
                border=True,
            )

        st.markdown("---")

        detail_df = pd.DataFrame(detail_rows)

        # 스타일링 적용 (is_missing, is_summary 컬럼 포함하여 스타일링 후 제거)
        def highlight_missing(row):
            # 소계 행은 어두운 초록색 (우선순위 가장 높음)
            if row.get("is_summary", False):
                return ["background-color: #1a472a; color: #ffffff"] * len(row)
            # 누락/0원 행은 어두운 붉은색
            elif row.get("is_missing", False):
                return ["background-color: #5c1a1a; color: #ffffff"] * len(row)
            return [""] * len(row)

        # 스타일링 적용
        styled_df = detail_df.style.apply(highlight_missing, axis=1)

        # 데이터 표시 (is_missing, is_summary 컬럼 제외)
        display_columns = [
            "보장항목",
            "스코어",
            "보험료",
            "등수",
            "1등",
            "꼴찌",
            "누락회사수",
            "누락회사",
        ]

        # 데이터 표시 (전체 폭, 높이 제한 없이 스크롤)
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=750,
            column_order=display_columns,
            column_config={
                "보장항목": st.column_config.TextColumn("보장항목"),
                "스코어": st.column_config.NumberColumn("스코어", format="%.2f"),
                "보험료": st.column_config.TextColumn("보험료"),
                "등수": st.column_config.TextColumn("등수"),
                "1등": st.column_config.TextColumn("1등"),
                "꼴찌": st.column_config.TextColumn("꼴찌"),
                "누락회사수": st.column_config.NumberColumn("누락회사수"),
                "누락회사": st.column_config.ListColumn("누락회사"),
            },
        )

        # WCEI 점수 산출 과정 설명
        st.markdown(
            """
        **💡 WCEI 점수 산출 과정:**
        - 각 보장항목별로 최저가(1.0점) ~ 최고가(0.1점) 사이 점수 부여
        - Tier 1 (핵심 보장): 평균 × 50점
        - Tier 2 (주요 보장): 평균 × 30점  
        - Tier 3 (일반 보장): 평균 × 20점
        - **총점 = Tier1 + Tier2 + Tier3 (100점 만점)**
        """
        )
    else:
        st.info("세부 득점표를 생성할 데이터가 없습니다.")


def main():
    """보험 올림픽 페이지"""
    # 페이지 설정
    st.set_page_config(**PAGE_CONFIG)

    # 선택된 플랜 정보 가져오기
    plan = get_selected_plan()
    plan_name = plan.get("plan_type_name", "") if plan else ""
    payment_due_type_name = plan.get("payment_due_type_name", "") if plan else ""

    # 타이틀 설정
    if plan_name:
        st.title(f"🏆 보험 올림픽 - 돌격!!! 최저가 대전!!! - ({plan_name})[{payment_due_type_name}]")
    else:
        st.title("🏆 보험 올림픽 - 돌격!!! 최저가 대전!!!")

    # 세션 상태 초기화
    init_session_state()

    # 설정 영역 (사이드바)
    st.sidebar.subheader("⚙️ 분석 설정")
    render_model_selector()
    st.sidebar.markdown("---")
    render_plan_loader()

    if has_plans():
        render_plan_selector()
        render_plan_info()
        render_analysis_form()

    # 보장 티어 설정 버튼 (사이드바 맨 아래)
    st.sidebar.markdown("---")
    if st.sidebar.button("⚙️ 보장 티어 설정", use_container_width=True, type="primary"):
        show_tier_settings_dialog()

    # 종목별 플랜 그룹 설정 버튼
    if st.sidebar.button(
        "🗂️ 종목별 플랜 그룹 설정", use_container_width=True, type="primary"
    ):
        show_plan_category_dialog()

    # 메인 컨텐츠
    st.markdown("---")

    # 데이터가 로드된 경우
    if get_session_value("data_loaded"):
        # 공통 데이터 로드
        coverage_tiers_df = load_coverage_tiers()
        llm_data = get_session_value("llm_readable_data", {})
        plan = get_selected_plan()
        plan_type_name = plan.get("plan_type_name", "") if plan else ""

        # 좌우 2분할 레이아웃 (비교표 + 랭킹/선택)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 보험사별 보장 항목 비교표")
            human_table = get_session_value("human_readable_table")
            if human_table:
                try:
                    df = pd.read_json(StringIO(human_table), orient="table")
                    st.dataframe(df, use_container_width=True, height=600)
                except Exception as e:
                    st.error(f"데이터 표시 오류: {e}")
            else:
                st.info("비교표 데이터가 없습니다.")

        # WCEI 랭킹 계산 (col2 밖에서 미리 계산)
        ranking_df = pd.DataFrame()
        total_coverage_count = 0
        if not coverage_tiers_df.empty and llm_data and plan:
            ranking_df, total_coverage_count = calculate_wcei_ranking(
                llm_data, coverage_tiers_df, plan_type_name
            )

        with col2:
            # WCEI 랭킹 표시
            st.subheader("🏆 WCEI 랭킹")

            if coverage_tiers_df.empty:
                st.error("보장 티어 데이터를 불러올 수 없습니다.")
            elif not llm_data:
                st.info(
                    "랭킹 계산을 위한 데이터가 없습니다. 데이터 분석을 다시 실행해주세요."
                )
            elif not plan:
                st.info("플랜을 선택해주세요.")
            elif not ranking_df.empty:
                # 표시용 DataFrame 생성 (보험사_표시 컬럼 사용)
                display_df = ranking_df.copy()
                display_df["보험사"] = display_df["보험사_표시"]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=600,
                    column_order=[
                        "보험사",
                        "총점",
                        "보험료",
                        "Tier1",
                        "Tier2",
                        "Tier3",
                        "누락건수",
                        "누락보장",
                    ],
                    column_config={
                        "보험사": st.column_config.TextColumn("보험사"),
                        "총점": st.column_config.NumberColumn("총점", format="%.1f"),
                        "보험료": st.column_config.TextColumn("보험료"),
                        "Tier1": st.column_config.NumberColumn("Tier1", format="%.1f"),
                        "Tier2": st.column_config.NumberColumn("Tier2", format="%.1f"),
                        "Tier3": st.column_config.NumberColumn("Tier3", format="%.1f"),
                        "누락건수": st.column_config.NumberColumn(
                            f"누락건수(총 {total_coverage_count}건)"
                        ),
                        "누락보장": st.column_config.ListColumn("누락보장"),
                    },
                )
            else:
                st.info("랭킹 계산을 위한 데이터가 부족합니다.")

        # --- 2분할 밖에서 전체 폭으로 보험사 선택 및 세부 득점표 표시 ---
        st.markdown("---")

        # 보험사 선택 selectbox (전체 폭)
        if not ranking_df.empty:
            selected_company = st.selectbox(
                "📊 세부 득점표 확인할 보험사 선택",
                options=ranking_df["보험사"].tolist(),
                index=None,
                placeholder="보험사를 선택하세요",
                key="selected_company",
            )

        if (
            st.session_state.get("selected_company")
            and llm_data
            and not coverage_tiers_df.empty
        ):
            show_company_detail_breakdown(
                st.session_state["selected_company"],
                llm_data,
                coverage_tiers_df,
                plan_type_name,
                ranking_df,
                total_coverage_count,
            )
    else:
        # 데이터가 없는 경우 안내 메시지
        st.info("👈 사이드바에서 플랜을 선택하고 '데이터 분석 시작' 버튼을 클릭하세요.")

    # 푸터
    st.markdown("---")
    st.caption("🤖 AI 기반 보험 비교 분석 시스템 | © 2026")


if __name__ == "__main__":
    main()
