"""
채팅 컴포넌트
"""

import streamlit as st
import json
import time
import logging
from io import StringIO
import pandas as pd

from utils.session import get_session_value, set_session_value, is_data_loaded
from utils.api import stream_chat_response
from config import IS_DEVELOPMENT

logger = logging.getLogger(__name__)


def render_chat_interface():
    """채팅 인터페이스 전체 렌더링"""
    if not is_data_loaded():
        st.info("👈 사이드바에서 플랜을 조회한 후 데이터 분석을 시작해주세요.")
        return

    # 플랜 상태 표시
    render_plan_status()

    # 채팅 헤더
    st.subheader("💬 AI 보험 상담사")

    # 채팅 기록 표시
    render_chat_history()

    # 채팅 입력 처리
    handle_chat_input()


def render_plan_status():
    """현재 분석 중인 플랜 상태 표시"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        current_plan = get_session_value("current_plan", "Unknown")
        st.success(f"📊 현재 분석 중인 플랜: **{current_plan}**")

    with col2:
        if st.button("🔄 새로고침", help="데이터를 새로고칩니다"):
            reset_chat()
            st.rerun()

    with col3:
        if st.button("📈 비교표", help="보험사별 비교 표 보기"):
            set_session_value("show_comparison_modal", True)
            st.rerun()


def reset_chat():
    """채팅 및 분석 데이터 초기화"""
    # 1. 대화 내용 명시적 초기화
    st.session_state["messages"] = []

    # 2. 분석 데이터 초기화
    from utils.session import reset_analysis_data

    reset_analysis_data()


def render_chat_history():
    """채팅 기록 표시"""
    messages = get_session_value("messages", [])

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def set_prompt_callback(prompt):
    """프롬프트 설정 콜백 함수"""
    set_session_value("temp_prompt", prompt)


def handle_chat_input():
    """채팅 입력 처리 (버그 수정: chat_input 항상 표시)"""
    messages = get_session_value("messages", [])

    # 개발 모드: 첫 메시지에서 기본 질문 버튼 표시
    if IS_DEVELOPMENT:
        cols = st.columns([1, 3])
        with cols[0]:
            st.button(
                f"🚀 뇌출혈 진단이 없는 회사는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("뇌출혈 진단이 없는 회사는?",),
                width='stretch',
            )
            st.button(
                f"🚀 3대진단금이 가장 저렴한 회사는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("3대진단금이 가장 저렴한 회사는?",),
                width='stretch',
            )
            st.button(
                f"🚀 통합암 진단이 없는 회사는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("통합암 진단이 없는 회사는?",),
                width='stretch',
            )
            st.button(
                f"🚀 암진단금이 가장 저렴한 회사는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("암진단금이 가장 저렴한 회사는?",),
                width='stretch',
            )
            st.button(
                f"🚀 보험료가 가장 저렴한 회사는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("보험료가 가장 저렴한 회사는?",),
                width='stretch',
            )
            st.button(
                f"🚀 삼겹살 맛있게 굽는 법 알려줘.",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("삼겹살 맛있게 굽는 법 알려줘.",),
                width='stretch',
            )
            st.button(
                f"🚀 db손해의 통합암진단의 세부 보장은?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("db손해의 통합암진단의 세부 보장은?",),
                width='stretch',
            )
            st.button(
                f"🚀 현대해상의 통합암진단이 가장 비싼 이유는?",
                help="개발용 기본 질문 사용",
                on_click=set_prompt_callback,
                args=("현대해상의 통합암진단이 가장 비싼 이유는?",),
                width='stretch',
            )

        with cols[1]:
            st.caption("💡 개발 모드: 빠른 테스트용 버튼")

    # 항상 chat_input 표시 (여기가 핵심!)
    user_input = st.chat_input("보험료나 보장내용에 대해 질문해주세요")

    # 입력 처리 우선순위:
    # 1. 사용자가 chat_input에 직접 입력한 경우
    # 2. 버튼으로 설정된 임시 프롬프트
    prompt = None

    if user_input:
        prompt = user_input
    else:
        temp_prompt = get_session_value("temp_prompt")
        if temp_prompt:
            prompt = temp_prompt
            set_session_value("temp_prompt", None)  # 사용 후 초기화

    # 프롬프트가 있으면 처리
    if prompt:
        process_chat_message(prompt)


def process_chat_message(prompt: str):
    """채팅 메시지 처리 및 AI 응답 생성"""
    # 사용자 메시지 추가
    messages = get_session_value("messages", [])
    messages.append({"role": "user", "content": prompt})
    set_session_value("messages", messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("AI가 응답을 준비 중입니다..."):
            full_response, usage_info = stream_chat(prompt)

            if full_response:
                st.markdown(full_response)

                # 토큰 사용량 표시
                if usage_info:
                    total_tokens = usage_info.get("total_tokens", 0)
                    prompt_tokens = usage_info.get("prompt_tokens", 0)
                    completion_tokens = usage_info.get("completion_tokens", 0)
                    cost = usage_info.get("cost", 0)

                    # 비용 표시 형식 결정 (숫자 vs 문자열)
                    if isinstance(cost, (int, float)):
                        cost_display = f"${cost:.6f}"
                    else:
                        cost_display = str(cost)

                    # 작은 글씨로 사용량 표시
                    st.caption(
                        f"💰 토큰: {total_tokens:,}개 (입력: {prompt_tokens:,} / 출력: {completion_tokens:,}) | 비용: {cost_display}"
                    )

                messages.append({"role": "assistant", "content": full_response})
                set_session_value("messages", messages)
            else:
                error_msg = "응답을 받지 못했습니다. 다시 시도해주세요."
                st.error(error_msg)


def stream_chat(prompt: str) -> tuple:
    """채팅 스트리밍 응답 처리"""
    try:
        llm_data = get_session_value("llm_readable_data", {})
        human_data = get_session_value("human_readable_table", "")
        selected_model = get_session_value("selected_model", "openai")

        # 사용자 컨텍스트 정보 가져오기
        plan_name = get_session_value("current_plan", "")
        gender = get_session_value("current_gender", "")
        age = get_session_value("current_age", 0)

        response = stream_chat_response(
            prompt, llm_data, human_data, selected_model, plan_name, gender, age
        )

        full_response = ""
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        usage_info = None

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
                status = data.get("status", "processing")
                message = data.get("message", "")
                progress = data.get("progress", 0)

                # 상태 표시 업데이트
                if status in ["searching", "analyzing", "finalizing"]:
                    status_placeholder.info(f"⏳ {message}")
                    progress_bar.progress(progress / 100.0)
                elif status == "complete":
                    status_placeholder.success("✅ 분석 완료!")
                    progress_bar.progress(1.0)
                    full_response = data.get("response", "")
                    usage_info = data.get("usage")
                elif status == "error":
                    status_placeholder.error(f"❌ 오류: {message}")
                    return None, None

            except json.JSONDecodeError:
                continue

        return full_response, usage_info

    except Exception as e:
        logger.error(f"스트리밍 오류: {e}")
        return f"스트리밍 오류: {str(e)}", None
