import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time
import json
import base64
import logging
import threading
from datetime import datetime, timedelta

load_dotenv()

# 로거 설정

logger = logging.getLogger(__name__)

if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# 로깅 레벨 설정 (더 상세한 로그를 위해 INFO로 설정)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 백엔드 API 기본 URL (Docker 환경 우선)
BACKEND_URL = os.getenv("BACKEND_URL") or "http://localhost:8000"


# JWT 토큰 관리 함수
def load_jwt_token():
    """secrets.toml에서 JWT 토큰 로드"""
    try:
        if hasattr(st.secrets, "JWT_TOKEN") and st.secrets.JWT_TOKEN:
            return st.secrets.JWT_TOKEN
        return ""
    except Exception as e:
        logger.error(f"JWT 토큰 로드 실패: {e}")
        return ""


def save_jwt_token(token: str):
    """secrets.toml에 JWT 토큰 저장 (개발 환경에서만)"""
    try:
        secrets_path = ".streamlit/secrets.toml"
        # 기존 파일 읽기
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = '# Streamlit Secrets Configuration\nJWT_TOKEN = ""\nRESET_JWT_TOKEN = false\n'

        # 토큰 값 업데이트
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("JWT_TOKEN ="):
                lines[i] = f'JWT_TOKEN = "{token}"'
                break

        # 파일 저장
        with open(secrets_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True
    except Exception as e:
        logger.error(f"JWT 토큰 저장 실패: {e}")
        return False


def clear_jwt_token():
    """secrets.toml에서 JWT 토큰 삭제"""
    try:
        return save_jwt_token("")
    except Exception as e:
        logger.error(f"JWT 토큰 삭제 실패: {e}")
        return False


def show_loading(message="처리 중..."):
    """로딩 스피너 표시"""
    return st.spinner(message)


def create_animated_message(message_base: str) -> str:
    """애니메이션 메시지 생성 (점이 1~3개까지 늘어나는 효과)"""
    dots = "." * ((int(time.time() * 2) % 3) + 1)  # 1, 2, 3개 점 순환
    return f"{message_base}{dots}"


def create_animated_loading_placeholder(container, message_base: str):
    """애니메이션 로딩 메시지를 표시하는 함수"""
    stop_animation = threading.Event()
    
    def update_animation():
        while not stop_animation.is_set():
            animated_message = create_animated_message(message_base)
            container.markdown(f"**{animated_message}**")
            time.sleep(0.5)  # 0.5초마다 애니메이션 업데이트
    
    # 애니메이션 스레드 시작
    animation_thread = threading.Thread(target=update_animation, daemon=True)
    animation_thread.start()
    
    return stop_animation


def call_api(endpoint: str, data: dict, method: str = "POST") -> dict | None:
    """백엔드 API 호출 헬퍼 함수"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        print(f"API 호출: {method} {url}")
        print(f"요청 데이터: {data}")

        if method == "POST":
            response = requests.post(url, json=data, timeout=120)
        else:
            response = requests.get(url, params=data, timeout=120)

        print(f"응답 상태: {response.status_code}")
        print(f"응답 내용: {response.text}")

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


# 페이지 설정
st.set_page_config(
    page_title="보험 비교 AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "plans" not in st.session_state:
    st.session_state.plans = []
if "show_comparison_modal" not in st.session_state:
    st.session_state.show_comparison_modal = False
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = load_jwt_token()

# 사이드바 설정
st.sidebar.title("보험 비교 AI 설정")
st.sidebar.markdown("---")

# JWT 토큰 관리
jwt_token = st.session_state.jwt_token

if jwt_token:
    st.sidebar.success("✅ JWT 토큰이 영구 저장되어 있습니다")
    st.sidebar.warning("⚠️ 개발 환경: .streamlit/secrets.toml에 저장됨")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 토큰 변경", help="새로운 JWT 토큰을 입력합니다"):
            clear_jwt_token()
            st.session_state.jwt_token = ""
            st.session_state.plans = []
            st.session_state.data_loaded = False
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("🗑️ 토큰 삭제", help="저장된 토큰을 영구 삭제합니다"):
            if clear_jwt_token():
                st.session_state.jwt_token = ""
                st.session_state.plans = []
                st.session_state.data_loaded = False
                st.session_state.messages = []
                st.rerun()

    # 저장된 토큰의 일부 표시
    masked_token = jwt_token[:8] + "..." + jwt_token[-8:]
    st.sidebar.caption(f"저장된 토큰: {masked_token}")
else:
    st.sidebar.info("🔑 JWT 토큰이 필요합니다")
    st.sidebar.caption("토큰을 입력하면 .streamlit/secrets.toml에 영구 저장됩니다")

    new_token = st.sidebar.text_input(
        "JWT 토큰",
        type="password",
        placeholder="JWT 토큰을 입력하세요",
        help="1일 유효기간의 JWT 토큰 (입력 시 영구 저장됨)",
    )

    if new_token:
        if save_jwt_token(new_token):
            st.session_state.jwt_token = new_token
            st.sidebar.success("✅ JWT 토큰이 저장되었습니다")
            st.rerun()
        else:
            st.sidebar.error("❌ 토큰 저장에 실패했습니다")

# 플랜 조회 버튼
if st.sidebar.button("플랜 조회", type="primary"):
    if st.session_state.jwt_token:
        with show_loading("플랜 목록 조회 중..."):
            try:
                # 백엔드 API 호출
                response = call_api(
                    "/fetch-plans", {"jwt_token": st.session_state.jwt_token}
                )

                if response:
                    st.session_state.plans = response
                    st.sidebar.success(f"{len(response)}개의 플랜 목록을 불러왔습니다!")
                else:
                    st.sidebar.error("플랜 목록을 불러오는데 실패했습니다.")

            except Exception as e:
                st.sidebar.error(f"플랜 조회 실패: {e}")
    else:
        st.sidebar.error("JWT 토큰을 입력해주세요.")

# 플랜 목록이 있을 경우 시뮬레이션 입력 표시
if "plans" in st.session_state and st.session_state.plans:
    st.sidebar.markdown("---")
    st.sidebar.subheader("시뮬레이션 조건")

    # 플랜 선택
    plan_options = {
        plan["plan_name"]: plan["plan_id"] for plan in st.session_state.plans
    }
    selected_plan_name = st.sidebar.selectbox("플랜 선택", list(plan_options.keys()))

    # 나이 입력
    age = st.sidebar.number_input("나이", min_value=0, max_value=100, value=46)

    # 성별 선택
    gender = st.sidebar.radio(
        "성별", ["남성", "여성"], format_func=lambda x: "M" if x == "남성" else "F"
    )
    gender_code = "M" if gender == "남성" else "F"

    # 데이터 분석 시작 버튼
    if st.sidebar.button("데이터 분석 시작", type="secondary"):
        plan_id = plan_options[selected_plan_name]
        try:
            with show_loading(f"{selected_plan_name} 데이터 분석 중..."):
                # 백엔드 API 호출
                data = {
                    "jwt_token": st.session_state.jwt_token,
                    "plan_id": plan_id,
                    "age": age,
                    "gender": gender_code,
                }

                response = call_api("/load-data", data)

                if response:
                    st.session_state.data_loaded = True
                    st.session_state.current_plan = selected_plan_name
                    st.session_state.plan_data = response  # 추가: 데이터 정보 저장

                    # 성공 메시지에 상세 정보 포함
                    coverage_count = response.get("coverage_count", 0)
                    insurance_count = response.get("insurance_count", 0)
                    st.sidebar.success(
                        f"{selected_plan_name} 데이터를 분석했습니다!\n"
                        f"• 보험료 데이터: {coverage_count}건\n"
                        f"• 보장내용 데이터: {insurance_count}건"
                    )
                else:
                    st.sidebar.error("데이터 분석에 실패했습니다.")

        except Exception as e:
            st.sidebar.error(f"데이터 분석 실패: {e}")

# 메인 페이지
st.title("🏥 생손보플랜 보험료 분석")

# 헤더에 상태 표시
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 연결 상태")

# 백엔드 연결 상태 확인
try:
    backend_check = requests.get(f"{BACKEND_URL}/", timeout=5)
    if backend_check.status_code == 200:
        st.sidebar.success("✅ 백엔드 연결됨")
        backend_available = True
    else:
        st.sidebar.error("❌ 백엔드 응답 오류")
        backend_available = False
except:
    st.sidebar.error("❌ 백엔드 연결 실패")
    backend_available = False

if not backend_available:
    st.error(
        "⚠️ 백엔드 서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요."
    )
    st.stop()


@st.dialog("📈 보험사별 보장 항목 비교 표", width="large")
def comparison_modal():
    """비교 표 모달창 표시"""
    # 비교 표 데이터 로드 및 표시
    with show_loading("비교 표 생성 중..."):
        result = call_api("/get-comparison-table", {})

    if result and result.get("status") == "success":
        comparison_data = result.get("comparison_table", {})

        if comparison_data:
            # DataFrame으로 변환
            df = pd.DataFrame(comparison_data)

            # 전체 너비 데이터프레임
            st.dataframe(
                df.style.format("{:,.0f}"), use_container_width=True, height=600
            )

            # 분석 정보
            st.markdown("### 💡 분석 가이드")
            tips_col1, tips_col2, tips_col3 = st.columns(3)

            with tips_col1:
                st.success(
                    """
                **📊 보험료 합계 비교**
                - 가장 낮은 보험사 추천
                - 연간/월간 보험료 절감
                """
                )

            with tips_col2:
                st.info(
                    """
                **🎯 핵심 보장 항목**
                - 암진단비 비교 분석
                - 상해보장 검토
                """
                )

            with tips_col3:
                st.warning(
                    """
                **🔍 특화 보장 확인**
                - 각사별 특별 약관
                - 가입 조건 검토
                """
                )
        else:
            st.warning("비교 표 데이터가 없습니다.")
    else:
        st.error("비교 표를 생성하는 데 실패했습니다.")
        if result:
            st.error(f"오류 상세: {result.get('detail', '알 수 없는 오류')}")


# 모달창 표시
if st.session_state.show_comparison_modal:
    # Dialog 실행
    comparison_modal()

    # 모달창 상태 초기화
    st.session_state.show_comparison_modal = False

if "data_loaded" not in st.session_state:
    st.info(
        "👈 사이드바에서 JWT 토큰을 입력하고 플랜을 조회한 후 데이터 분석을 시작해주세요."
    )
else:
    # 플랜 정보와 데이터 상태 표시
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.success(f"📊 현재 분석 중인 플랜: **{st.session_state.current_plan}**")
    with col2:
        if st.button("🔄 새로고침", help="데이터를 새로고칩니다"):
            st.session_state.data_loaded = False
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("📈 비교표", help="보험사별 비교 표 보기"):
            st.session_state.show_comparison_modal = True
            st.rerun()

    # 챗 인터페이스
    st.subheader("💬 AI 보험 상담사")

    # 챗 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 챗 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("보험료나 보장내용에 대해 질문해주세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 (백엔드 API 연동)
        with st.chat_message("assistant"):
            # 스트리밍 옵션 체크박스
            use_streaming = st.checkbox(
                "🚀 스트리밍 응답 사용 (더 빠름)",
                value=True,
                help="실시간 진행 상태를 확인하며 응답받습니다.",
            )

            if use_streaming:
                # 스트리밍 응답 처리
                with st.container():
                    # 상태 표시 컨테이너 (애니메이션용)
                    status_container = st.empty()
                    progress_container = st.empty()
                    # 애니메이션 컨트롤
                    animation_stop = None

                    try:
                        logger.info(f"[FRONTEND] 스트리밍 요청 시작 - 쿼리: '{prompt}'")
                        
                        response = requests.post(
                            f"{BACKEND_URL}/chat-stream",
                            json={"query": prompt},
                            stream=True,
                            timeout=180,
                            headers={
                                'Accept': 'text/event-stream',
                                'Cache-Control': 'no-cache',
                            }
                        )
                        
                        logger.info(f"[FRONTEND] 응답 상태 코드: {response.status_code}")

                        full_response = ""
                        current_status = ""
                        line_count = 0
                        chunk_count = 0

                        # 버퍼링을 방지하기 위해 iter_lines에 chunk_size 설정
                        for line in response.iter_lines(decode_unicode=True, chunk_size=512):
                            line_count += 1
                            
                            # 빈 라인도 로깅하여 스트리밍 흐름 확인
                            if not line:
                                logger.debug(f"[FRONTEND] 빈 라인 수신 (라인 {line_count})")
                                continue
                                
                            if line.startswith("data: "):
                                chunk_count += 1
                                try:
                                    json_text = line[6:]
                                    data = json.loads(json_text)
                                    
                                    status = data.get("status", "processing")
                                    message = data.get("message", "")
                                    progress = data.get("progress", 0)
                                    timestamp = data.get("timestamp", time.time())
                                    
                                    logger.info(f"[FRONTEND] 청크 {chunk_count} 수신: status={status}, message='{message}', progress={progress}%")

                                    # Windows 인코딩 문제 처리
                                    try:
                                        safe_message = message.encode(
                                            "utf-8", errors="ignore"
                                        ).decode("utf-8")
                                    except:
                                        safe_message = str(message)

                                    # 상태 메시지와 프로그레스 바 즉시 업데이트
                                    if status == "searching":
                                        logger.info(f"[FRONTEND] searching 상태 업데이트: {safe_message}")
                                        # 이전 애니메이션 중지
                                        if animation_stop:
                                            animation_stop.set()
                                        # 새로운 애니메이션 시작
                                        base_message = safe_message.replace("중...", "중")
                                        animation_stop = create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
                                        progress_container.progress(progress / 100.0)
                                    elif status == "analyzing":
                                        logger.info(f"[FRONTEND] analyzing 상태 업데이트: {safe_message}")
                                        # 이전 애니메이션 중지
                                        if animation_stop:
                                            animation_stop.set()
                                        # 새로운 애니메이션 시작
                                        base_message = safe_message.replace("중...", "중")
                                        animation_stop = create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
                                        progress_container.progress(progress / 100.0)
                                    elif status == "finalizing":
                                        logger.info(f"[FRONTEND] finalizing 상태 업데이트: {safe_message}")
                                        # 이전 애니메이션 중지
                                        if animation_stop:
                                            animation_stop.set()
                                        # 새로운 애니메이션 시작
                                        base_message = safe_message.replace("중...", "중")
                                        animation_stop = create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
                                        progress_container.progress(progress / 100.0)
                                    elif status == "complete":
                                        logger.info(f"[FRONTEND] complete 상태 수신")
                                        # 애니메이션 중지
                                        if animation_stop:
                                            animation_stop.set()
                                        status_container.success("✅ 분석 완료!")
                                        progress_container.progress(1.0)
                                        full_response = data.get("response", "")
                                        logger.info(f"[FRONTEND] 최종 응답 수신 - 길이: {len(full_response)}")
                                    elif status == "error":
                                        logger.error(f"[FRONTEND] error 상태 수신: {safe_message}")
                                        # 애니메이션 중지
                                        if animation_stop:
                                            animation_stop.set()
                                        status_container.error(f"❌ 오류: {safe_message}")
                                        progress_container.progress(1.0)
                                        full_response = f"처리 중 오류 발생: {safe_message}"

                                except json.JSONDecodeError as e:
                                    logger.error(f"[FRONTEND] JSON 파싱 오류: {e}, 라인: {line}")
                                    continue
                                except Exception as e:
                                    logger.error(f"[FRONTEND] 청크 처리 오류: {e}")
                                    continue

                        logger.info(f"[FRONTEND] 스트리밍 완료 - 총 라인 수: {line_count}, 총 청크 수: {chunk_count}, 응답 길이: {len(full_response)}")
                        
                        # 애니메이션 정리
                        if animation_stop:
                            animation_stop.set()
                        
                        # 최종 응답 표시
                        if full_response:
                            st.markdown(full_response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_response}
                            )
                        else:
                            error_msg = "응답을 받지 못했습니다. 다시 시도해주세요."
                            st.markdown(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

                    except Exception as e:
                        logger.error(f"[FRONTEND] 스트리밍 요청 오류: {type(e).__name__}: {e}")
                        # 애니메이션 정리
                        if animation_stop:
                            animation_stop.set()
                        error_msg = f"스트리밍 오류: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

            else:
                # 기존 방식 (일반 응답)
                with show_loading("AI 답변 생성 중..."):
                    try:
                        response = call_api("/chat", {"query": prompt})

                        if response:
                            ai_response = response.get(
                                "response", "죄송합니다. 응답을 생성할 수 없습니다."
                            )
                            st.markdown(ai_response)

                            # 추가 정보 표시
                            sources_found = response.get("sources_found", False)
                            data_analysis_available = response.get(
                                "data_analysis_available", False
                            )
                            source_count = response.get("source_count", 0)

                            if sources_found:
                                st.info(
                                    f"📋 {source_count}개의 관련 문서를 찾았습니다."
                                )
                            if data_analysis_available:
                                st.info("📊 데이터 분석 결과가 포함되어 있습니다.")

                            st.session_state.messages.append(
                                {"role": "assistant", "content": ai_response}
                            )
                        else:
                            error_msg = (
                                "AI 답변 생성에 실패했습니다. 다시 시도해주세요."
                            )
                            st.markdown(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    # 데이터 뷰 (접을 수 있는 섹션)
    with st.expander("📋 분석 데이터 보기"):
        if "plan_data" in st.session_state and st.session_state.plan_data:
            plan_data = st.session_state.plan_data

            # 기본 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("보험료 데이터 건수", plan_data.get("coverage_count", 0))
            with col2:
                st.metric("보장내용 데이터 건수", plan_data.get("insurance_count", 0))
            with col3:
                vector_status = (
                    "✅ 초기화됨"
                    if plan_data.get("vector_store_initialized", False)
                    else "❌ 초기화 안됨"
                )
                st.metric("벡터 저장소 상태", vector_status)

            # 플랜 정보
            if "plan_info" in plan_data:
                st.subheader("🎯 현재 분석 플랜 정보")
                plan_info = plan_data["plan_info"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**플랜 ID:** {plan_info.get('plan_id', 'N/A')}")
                with col2:
                    st.write(f"**나이:** {plan_info.get('age', 'N/A')}세")
                with col3:
                    gender = plan_info.get("gender", "N/A")
                    gender_text = (
                        "남성" if gender == "M" else "여성" if gender == "F" else gender
                    )
                    st.write(f"**성별:** {gender_text}")

            # 데이터 상태 표시
            st.subheader("📊 데이터 상태")
            if plan_data.get("status") == "success":
                st.success("✅ 데이터가 성공적으로 로드되었습니다.")
            else:
                st.warning("⚠️ 데이터 로드 중 일부 문제가 발생했습니다.")

        else:
            st.info(
                "👈 아직 분석된 데이터가 없습니다. 사이드바에서 데이터 분석을 시작해주세요."
            )
