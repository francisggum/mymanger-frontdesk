"""
API 호출 유틸리티
"""

import requests
import streamlit as st
from config import BACKEND_URL, API_TIMEOUT, CHAT_TIMEOUT


def check_backend_connection():
    """백엔드 연결 상태 확인"""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def call_api(endpoint: str, data: dict, method: str = "POST") -> dict | None:
    """백엔드 API 호출 헬퍼 함수"""
    try:
        url = f"{BACKEND_URL}{endpoint}"

        if method == "POST":
            response = requests.post(url, json=data, timeout=API_TIMEOUT)
        else:
            response = requests.get(url, params=data, timeout=API_TIMEOUT)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.error("요청 시간이 초과되었습니다. 다시 시도해주세요.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 오류: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"예상치 못한 오류: {str(e)}")
        return None


def fetch_plans():
    """플랜 목록 조회"""
    return call_api("/fetch-plans", {}, method="POST")


def get_comparison_tables(plan_id: str, age: int, gender: str):
    """비교 테이블 데이터 조회"""
    data = {
        "plan_id": plan_id,
        "age": age,
        "gender": gender,
    }
    return call_api("/get-comparison-tables", data, method="POST")


def stream_chat_response(query: str, llm_data: dict, human_data: str, model: str = "openai",
                         plan_name: str = None, gender: str = None, age: int = None):
    """채팅 스트리밍 응답 요청"""
    url = f"{BACKEND_URL}/chat-stream"

    response = requests.post(
        url,
        json={
            "query": query,
            "llm_data": llm_data,
            "human_data": human_data,
            "model": model,
            "plan_name": plan_name,
            "gender": gender,
            "age": age,
        },
        stream=True,
        timeout=CHAT_TIMEOUT,
        headers={
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        },
    )

    return response
