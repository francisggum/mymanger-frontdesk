"""
사이드바 컴포넌트 - 공통 정보 표시
"""

import streamlit as st
from utils.api import check_backend_connection


def render_sidebar():
    """사이드바 공통 정보 렌더링"""
    st.sidebar.title("ℹ️ 정보")
    st.sidebar.markdown("---")

    # 백엔드 연결 상태
    render_connection_status()


def render_connection_status():
    """백엔드 연결 상태 표시"""
    st.sidebar.subheader("🔗 연결 상태")

    if check_backend_connection():
        st.sidebar.success("✅ 백엔드 연결됨")
    else:
        st.sidebar.error("❌ 백엔드 연결 실패")
