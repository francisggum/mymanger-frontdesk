"""
보험 비교 AI - 메인 애플리케이션

Refactored version with clean component structure
"""

import streamlit as st
import logging

from config import PAGE_CONFIG
from utils.session import init_session_state
from components.sidebar import render_sidebar
from components.chat import render_chat_interface
from components.modal import render_comparison_modal

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """메인 애플리케이션 진입점"""
    # 페이지 설정
    st.set_page_config(**PAGE_CONFIG)

    # 세션 상태 초기화
    init_session_state()

    # 사이드바 렌더링
    render_sidebar()

    # 메인 페이지 제목
    st.title("🏥 생손보플랜 보험료 분석")

    # 비교표 모달 (조걶적으로 표시)
    render_comparison_modal()

    # 채팅 인터페이스
    render_chat_interface()

    # 푸터
    st.markdown("---")
    st.caption("🤖 AI 기반 보험 비교 분석 시스템 | © 2026")


if __name__ == "__main__":
    main()
