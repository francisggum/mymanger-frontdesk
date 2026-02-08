"""
Frontend 설정값 중앙화
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 백엔드 API 설정
BACKEND_URL = os.getenv("BACKEND_URL") or "http://localhost:8000"

# 개발 모드 설정
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "development") == "development"

# 페이지 설정
PAGE_CONFIG = {
    "page_title": "보험 비교 AI",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# API 타임아웃 설정
API_TIMEOUT = 120  # seconds
CHAT_TIMEOUT = 180  # seconds

# AI 모델 설정
# UI 표시명: 실제 API에 전달할 값
MODEL_OPTIONS = {
    "Grok 4.1 Fast": "openai",  # UI에서는 Grok로 표시, 실제로는 openai로 전송
    "Gemini": "gemini",
}

# 세션 상태 기본값
SESSION_DEFAULTS = {
    "messages": [],
    "plans": [],
    "show_comparison_modal": False,
    "selected_plan_key": None,
    "data_loaded": False,
    "current_plan": None,
    "current_gender": None,
    "current_age": None,
    "plan_data": None,
    "human_readable_table": None,
    "llm_readable_data": None,
    "comparison_summary": None,
    "temp_prompt": None,  # 개발 모드 버튼용 임시 프롬프트
    "selected_model": "openai",  # 기본값: openai (UI에서는 Grok 4.1 Fast로 표시)
}
