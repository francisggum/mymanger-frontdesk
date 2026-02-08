"""
세션 상태 관리 유틸리티
"""
import streamlit as st
from config import SESSION_DEFAULTS


def init_session_state():
    """모든 세션 상태를 한 곳에서 초기화"""
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_session_value(key, default=None):
    """세션 상태 값 안전하게 가져오기"""
    return st.session_state.get(key, default)


def set_session_value(key, value):
    """세션 상태 값 설정"""
    st.session_state[key] = value


def reset_analysis_data():
    """분석 데이터 초기화 (새로고침 시 사용)"""
    keys_to_reset = [
        "data_loaded",
        "current_plan",
        "current_gender",
        "current_age",
        "plan_data",
        "human_readable_table",
        "llm_readable_data",
        "comparison_summary",
        "messages",
    ]
    for key in keys_to_reset:
        if key in SESSION_DEFAULTS:
            st.session_state[key] = SESSION_DEFAULTS[key]


def is_data_loaded():
    """데이터 로드 여부 확인"""
    return st.session_state.get("data_loaded", False)


def has_plans():
    """플랜 목록이 있는지 확인"""
    plans = st.session_state.get("plans", [])
    return len(plans) > 0


def get_selected_plan():
    """현재 선택된 플랜 정보 반환"""
    plan_key = st.session_state.get("selected_plan_key")
    plans = st.session_state.get("plans", [])
    
    if not plan_key or not plans:
        return None
    
    plan_options = {
        f"{plan['plan_type_name']} ({plan['insu_compy_type_name']})": plan
        for plan in plans
    }
    
    return plan_options.get(plan_key)
