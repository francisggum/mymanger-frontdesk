"""
사이드바 컴포넌트
"""

import streamlit as st
from utils.session import (
    set_session_value,
    get_session_value,
    has_plans,
    get_selected_plan,
)
from utils.api import fetch_plans, get_comparison_tables, check_backend_connection
from config import IS_DEVELOPMENT


def render_sidebar():
    """사이드바 전체 렌더링"""
    st.sidebar.title("보험 비교 AI 설정")
    st.sidebar.markdown("---")

    # 백엔드 연결 상태
    render_connection_status()

    st.sidebar.markdown("---")

    # 플랜 목록 로드
    render_plan_loader()

    # 플랜 선택 및 분석
    if has_plans():
        render_plan_selector()
        render_plan_info()
        render_analysis_form()


def render_connection_status():
    """백엔드 연결 상태 표시"""
    st.sidebar.subheader("🔗 연결 상태")

    if check_backend_connection():
        st.sidebar.success("✅ 백엔드 연결됨")
    else:
        st.sidebar.error("❌ 백엔드 연결 실패")


def render_plan_loader():
    """플랜 목록 자동 로드 및 새로고침"""
    plans = get_session_value("plans", [])

    if not plans:
        with st.spinner("플랜 목록 로딩 중..."):
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
                with st.spinner("플랜 목록 새로고침 중..."):
                    response = fetch_plans()
                    if response:
                        set_session_value("plans", response)
                        st.sidebar.success(f"✅ {len(response)}개 플랜 새로고침 완료!")
                        st.rerun()


def render_plan_selector():
    """플랜 선택 드롭다운"""
    plans = get_session_value("plans", [])

    plan_options = {
        f"{plan['plan_type_name']} ({plan['insu_compy_type_name']})": plan
        for plan in plans
    }

    selected_key = get_session_value("selected_plan_key")
    if not selected_key or selected_key not in plan_options:
        selected_key = list(plan_options.keys())[0]
        set_session_value("selected_plan_key", selected_key)

    selected = st.sidebar.selectbox(
        "플랜 선택",
        list(plan_options.keys()),
        index=list(plan_options.keys()).index(selected_key),
    )

    set_session_value("selected_plan_key", selected)


def render_plan_info():
    """선택된 플랜 정보 표시"""
    plan = get_selected_plan()
    if not plan:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 플랜 정보")

    # 성별 및 나이 조건 분석
    min_m_age = plan.get("plan_min_m_age", 0)
    max_m_age = plan.get("plan_max_m_age", 0)
    min_f_age = plan.get("plan_min_f_age", 0)
    max_f_age = plan.get("plan_max_f_age", 0)

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
    """데이터 분석 폼"""
    plan = get_selected_plan()
    if not plan:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 가입 조건 입력")

    # 성별 및 나이 범위 결정
    min_m_age = plan.get("plan_min_m_age", 0)
    max_m_age = plan.get("plan_max_m_age", 0)
    min_f_age = plan.get("plan_min_f_age", 0)
    max_f_age = plan.get("plan_max_f_age", 0)

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

    # 성별 선택
    gender = st.sidebar.radio(
        "성별", gender_options, index=gender_options.index(default_gender)
    )
    gender_code = "M" if gender == "남성" else "F"

    # 나이 선택
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

    # 분석 시작 버튼
    if st.sidebar.button("데이터 분석 시작", type="secondary"):
        if age < min_age or age > max_age:
            st.sidebar.error(
                f"❌ 나이를 {min_age}세에서 {max_age}세 사이로 입력해주세요"
            )
        else:
            with st.spinner(f"데이터 분석 중..."):
                response = get_comparison_tables(plan["plan_id"], age, gender_code)

                if response:
                    set_session_value("data_loaded", True)
                    set_session_value(
                        "current_plan",
                        f"{plan['plan_type_name']} ({plan['insu_compy_type_name']})",
                    )
                    set_session_value("plan_data", response)
                    set_session_value(
                        "human_readable_table", response.get("human_readable_table")
                    )
                    set_session_value(
                        "llm_readable_data", response.get("llm_readable_data")
                    )
                    set_session_value("comparison_summary", response.get("summary", {}))
                    set_session_value("messages", [])  # 이전 대화 내용 초기화

                    summary = response.get("summary", {})
                    st.sidebar.success(
                        f"데이터 분석 완료!\n"
                        f"• 총 보험사 수: {summary.get('total_companies', 0)}개\n"
                        f"• 총 보장 항목: {summary.get('total_coverages', 0)}개"
                    )
                else:
                    st.sidebar.error("데이터 분석에 실패했습니다.")
