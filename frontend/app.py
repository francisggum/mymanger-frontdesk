"""
보험 비교 AI - 메인 애플리케이션

다중 페이지 구조의 메인 진입점
"""

import streamlit as st

from config import PAGE_CONFIG


def main():
    """메인 애플리케이션 진입점"""
    # 페이지 설정
    st.set_page_config(
        page_title=PAGE_CONFIG["page_title"],
        page_icon=PAGE_CONFIG["page_icon"],
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 메인 랜딩 페이지
    st.title("🥇 보험 올림픽")
    st.markdown("---")

    st.markdown(
        """
    ### 👋 환영합니다!
    
    **보험 비교 AI**는 다양한 보험 상품을 비교 분석하여 
    최적의 선택을 도와드리는 인공지능 기반 서비스입니다.
    
    ### 📋 제공 기능
    
    - **🏥 보험료 분석**: AI 기반 보험 상품 비교 및 분석
    - **실시간 채팅**: 자연어로 보험 관련 질문에 답변
    - **데이터 시각화**: 보험사별 보험료 비교표 제공
    
    ### 🚀 시작하기
    
    왼쪽 사이드바에서 **보험료 분석** 메뉴를 선택하여 시작하세요.
    """
    )

    # 푸터
    st.markdown("---")
    st.caption("🤖 AI 기반 보험 비교 분석 시스템 | © 2026")


if __name__ == "__main__":
    main()
