"""
비교표 모달 컴포넌트
"""
import streamlit as st
import pandas as pd
import json
from io import StringIO

from utils.session import get_session_value, set_session_value


def render_comparison_modal():
    """비교표 모달 렌더링"""
    if not get_session_value("show_comparison_modal"):
        return
    
    # st.dialog 데코레이터 사용
    @st.dialog("📈 보험사별 보장 항목 비교 표", width="large")
    def modal_content():
        render_modal_content()
    
    # 모달 표시
    modal_content()
    
    # 모달이 닫히면 상태 초기화 (사용자가 X 버튼이나 ESC로 닫았을 때)
    set_session_value("show_comparison_modal", False)


def render_modal_content():
    """모달 내용 렌더링"""
    # 플랜 정보 확인
    plan = get_session_value("current_plan")
    if not plan:
        st.error("선택된 플랜이 없습니다.")
        return
    
    # 비교표 데이터 확인
    human_table = get_session_value("human_readable_table")
    if not human_table:
        st.error("먼저 '데이터 분석 시작' 버튼을 클릭하여 데이터를 로드해주세요.")
        return
    
    # DataFrame으로 변환 및 표시
    try:
        df = pd.read_json(StringIO(human_table), orient='table')
        st.dataframe(df, use_container_width=True, height=600)
    except Exception as e:
        st.error(f"데이터 표시 오류: {e}")
        return
    
    # 요약 정보
    render_summary()
    
    # 분석 가이드
    render_analysis_guide()
    
    # LLM용 데이터 (접을 수 있는 섹션)
    render_llm_data_preview()
    
    # 닫기 버튼
    if st.button("닫기", key="close_modal"):
        set_session_value("show_comparison_modal", False)
        st.rerun()


def render_summary():
    """요약 정보 표시"""
    summary = get_session_value("comparison_summary", {})
    age = summary.get("age", 30)
    gender = summary.get("gender", "M")
    
    st.markdown("### 📊 비교표 요약")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 보험사 수", summary.get("total_companies", 0))
    with col2:
        st.metric("총 보장 항목", summary.get("total_coverages", 0))
    with col3:
        st.metric("분석 대상", f"{age}세 {'남성' if gender == 'M' else '여성'}")


def render_analysis_guide():
    """분석 가이드 표시"""
    st.markdown("### 💡 분석 가이드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(
            """
            **📊 보험료 합계 비교**
            - 가장 낮은 보험사 추천
            - 연간/월간 보험료 절감
            """
        )
    
    with col2:
        st.info(
            """
            **🎯 핵심 보장 항목**
            - 암진단비 비교 분석
            - 상핵보장 검토
            """
        )
    
    with col3:
        st.warning(
            """
            **🔍 특화 보장 확인**
            - 각사별 특별 약관
            - 가입 조건 검토
            """
        )


def render_llm_data_preview():
    """LLM용 데이터 미리보기"""
    llm_data = get_session_value("llm_readable_data", {})
    
    with st.expander("🔍 LLM용 데이터 보기"):
        if llm_data:
            total_items = sum(len(coverages) for coverages in llm_data.values())
            
            if total_items > 2:
                # 최대 2개 항목만 표시
                limited_data = {}
                current_count = 0
                
                for company, coverages in llm_data.items():
                    if current_count >= 2:
                        break
                    
                    for coverage in coverages:
                        if current_count >= 2:
                            break
                        
                        insur_item_name_list = coverage.get("insur_item_name_list", "")
                        if "|" in insur_item_name_list:
                            if company not in limited_data:
                                limited_data[company] = []
                            limited_data[company].append(coverage)
                            current_count += 1
                        elif current_count < 1:
                            if company not in limited_data:
                                limited_data[company] = []
                            limited_data[company].append(coverage)
                            current_count += 1
                
                st.json(limited_data)
                st.info(f"⚠️ 전체 {total_items}개 항목 중 최대 2개만 표시됩니다.")
            else:
                st.json(llm_data)
        else:
            st.info("LLM용 데이터가 없습니다.")
