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
        # print(f"응답 내용: {response.text}")

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


# 사이드바 설정
st.sidebar.title("보험 비교 AI 설정")
st.sidebar.markdown("---")

# 플랜 목록 자동 로드 (처음 접속 시)
if "plans" not in st.session_state or not st.session_state.plans:
    with show_loading("플랜 목록 자동 로딩 중..."):
        try:
            # 백엔드 API 호출 (DB에서 직접 조회)
            response = call_api("/fetch-plans", {})

            if response and len(response) > 0:
                st.session_state.plans = response
                st.sidebar.success(
                    f"✅ 총 {len(response)}개의 플랜 목록을 자동 로드했습니다!"
                )
                # UI 즉시 갱신을 위해 rerun 호출
                st.rerun()
            else:
                st.sidebar.error("❌ 플랜 목록을 불러오는데 실패했습니다.")
                st.session_state.plans = []

        except Exception as e:
            st.sidebar.error(f"❌ 플랜 로딩 실패: {e}")
            st.session_state.plans = []

# 디버그용 상태 정보 표시
logger.info(f"Session state - plans exists: {'plans' in st.session_state}")
if "plans" in st.session_state:
    logger.info(f"Plans count: {len(st.session_state.plans)}")
else:
    logger.info("No plans in session state")

# 플랜 목록이 있을 경우 상태 표시 및 새로고침 기능
if "plans" in st.session_state and st.session_state.plans:
    # 플랜 목록 상태 및 새로고침
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.sidebar.success(f"✅ 총 {len(st.session_state.plans)}개 플랜 로드됨")
    with col2:
        if st.sidebar.button("🔄", help="플랜 목록 새로고침"):
            with show_loading("플랜 목록 새로고침 중..."):
                try:
                    response = call_api("/fetch-plans", {})

                    if response:
                        st.session_state.plans = response
                        st.sidebar.success(f"✅ {len(response)}개 플랜 새로고침 완료!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ 플랜 목록 새로고침 실패!")

                except Exception as e:
                    st.sidebar.error(f"❌ 새로고침 실패: {e}")

# 플랜 목록이 있을 경우 플랜 선택 먼저 표시
if "plans" in st.session_state and st.session_state.plans:
    # 전체 플랜 목록으로 플랜 선택
    plan_options = {
        f"{plan['plan_type_name']} ({plan['insu_compy_type_name']})": plan
        for plan in st.session_state.plans
    }

    # 세션 상태 초기화
    if "selected_plan_key" not in st.session_state:
        st.session_state.selected_plan_key = list(plan_options.keys())[0]

    selected_plan_key = st.sidebar.selectbox(
        "플랜 선택",
        list(plan_options.keys()),
        index=(
            list(plan_options.keys()).index(st.session_state.selected_plan_key)
            if st.session_state.selected_plan_key in plan_options
            else 0
        ),
    )

    # 선택된 플랜 정보 저장
    st.session_state.selected_plan_key = selected_plan_key
    selected_plan = plan_options[selected_plan_key]

    # 플랜 정보 표시
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 플랜 정보")

    # 가입 조건 분석
    min_m_age = selected_plan.get("plan_min_m_age", 0)
    max_m_age = selected_plan.get("plan_max_m_age", 0)
    min_f_age = selected_plan.get("plan_min_f_age", 0)
    max_f_age = selected_plan.get("plan_max_f_age", 0)

    # 성별 제한 확인
    if min_m_age == 0 and max_m_age == 0:
        # 여성 전용
        gender_options = ["여성"]
        gender_code = "F"
        st.sidebar.info("👩 이 플랜은 **여성** 전용입니다")
        available_min_age = min_f_age
        available_max_age = max_f_age
        st.sidebar.write(f"👤 나이 조건: {available_min_age}세 ~ {available_max_age}세")
    elif min_f_age == 0 and max_f_age == 0:
        # 남성 전용
        gender_options = ["남성"]
        gender_code = "M"
        st.sidebar.info("👨 이 플랜은 **남성** 전용입니다")
        available_min_age = min_m_age
        available_max_age = max_m_age
        st.sidebar.write(f"👤 나이 조건: {available_min_age}세 ~ {available_max_age}세")
    else:
        # 남녀 공통
        gender_options = ["남성", "여성"]
        st.sidebar.info("👫 이 플랜은 **남녀 공통**입니다")

        # 남여 나이 범위 계산
        male_range = (
            f"남성: {min_m_age}~{max_m_age}세" if min_m_age > 0 else "남성: 불가"
        )
        female_range = (
            f"여성: {min_f_age}~{max_f_age}세" if min_f_age > 0 else "여성: 불가"
        )
        st.sidebar.write(f"👤 나이 조건:")
        st.sidebar.write(f"   • {male_range}")
        st.sidebar.write(f"   • {female_range}")

        # 공통 나이 범위 (교집합)
        common_min = (
            max(min_m_age, min_f_age)
            if min_m_age > 0 and min_f_age > 0
            else (min_m_age if min_f_age == 0 else min_f_age)
        )
        common_max = (
            min(max_m_age, max_f_age) if max_m_age > 0 and max_f_age > 0 else max_m_age
        )
        available_min_age = common_min
        available_max_age = common_max

        if common_min > 0 and common_max > 0:
            st.sidebar.write(f"📊 공통 가능 나이: {common_min}~{common_max}세")

        gender_code = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 가입 조건 입력")

    # 성별 선택 (단일 선택이 아닌 경우에만 표시)
    if len(gender_options) == 1:
        gender = gender_options[0]
        gender_code = "M" if gender == "남성" else "F"
        st.sidebar.write(f"🚻 성별: **{gender}** (자동 설정)")
    else:
        gender = st.sidebar.radio(
            "성별", gender_options, format_func=lambda x: "M" if x == "남성" else "F"
        )
        gender_code = "M" if gender == "남성" else "F"

    # 동적 나이 입력
    if gender_code == "M":
        min_age = min_m_age
        max_age = max_m_age
    else:
        min_age = min_f_age
        max_age = max_f_age

    # 유효한 나이 범위인지 확인
    if min_age > 0 and max_age > 0:
        # 기본값 설정: 범위 내의 중간값 또는 최소값+1
        default_age = min((min_age + max_age) // 2, min_age + 1)
        if default_age < min_age:
            default_age = min_age
        if default_age > max_age:
            default_age = max_age

        age = st.sidebar.number_input(
            "나이",
            min_value=min_age,
            max_value=max_age,
            value=default_age,
            help=f"{min_age}세에서 {max_age}세까지 입력 가능합니다",
        )
    else:
        # 나이 제한이 없는 경우
        age = st.sidebar.number_input("나이", min_value=0, max_value=100, value=46)
        st.sidebar.warning("⚠️ 이 성별은 해당 플랜에 가입할 수 없습니다")

    # 데이터 분석 시작 버튼
    if st.sidebar.button("데이터 분석 시작", type="secondary"):
        if min_age > 0 and max_age > 0 and (age < min_age or age > max_age):
            st.sidebar.error(
                f"❌ 나이를 {min_age}세에서 {max_age}세 사이로 입력해주세요"
            )
        else:
            try:
                with show_loading(f"{selected_plan_key} 데이터 분석 중..."):
                    # 백엔드 API 호출
                    data = {
                        "plan_id": selected_plan["plan_id"],
                        "age": age,
                        "gender": gender_code,
                    }

                    response = call_api("/get-comparison-tables", data)

                    if response:
                        st.session_state.data_loaded = True
                        st.session_state.current_plan = selected_plan_key
                        st.session_state.plan_data = response

                        # 비교표 데이터를 세션 상태에 저장
                        st.session_state.human_readable_table = response.get(
                            "human_readable_table", {}
                        )
                        st.session_state.llm_readable_data = response.get(
                            "llm_readable_data", {}
                        )

                        # JSON 데이터를 파일로 저장
                        with open("/app/dump.json", "w", encoding="utf-8") as f:
                            json.dump(
                                response.get("llm_readable_data", {}),
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                        st.session_state.comparison_summary = response.get(
                            "summary", {}
                        )

                        print(response.get("llm_readable_data", {}))

                        # 성공 메시지에 상세 정보 포함
                        summary = response.get("summary", {})
                        total_companies = summary.get("total_companies", 0)
                        total_coverages = summary.get("total_coverages", 0)
                        st.sidebar.success(
                            f"{selected_plan_key} 데이터를 분석했습니다!\n"
                            f"• 총 보험사 수: {total_companies}개\n"
                            f"• 총 보장 항목: {total_coverages}개"
                        )
                    else:
                        st.sidebar.error("데이터 분석에 실패했습니다.")

            except Exception as e:
                st.sidebar.error(f"데이터 분석 실패: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("시뮬레이션 조건")

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
    # 현재 선택된 플랜 정보 확인
    if "plans" not in st.session_state or not st.session_state.plans:
        st.error("먼저 플랜을 선택해주세요.")
        return

    # 선택된 플랜 정보 가져오기
    plan_options = {
        f"{plan['plan_type_name']} ({plan['insu_compy_type_name']})": plan
        for plan in st.session_state.plans
    }
    selected_plan = plan_options.get(st.session_state.get("selected_plan_key", ""))

    if not selected_plan:
        st.error("선택된 플랜이 없습니다.")
        return

    # 세션에 저장된 비교표 데이터 확인
    if not st.session_state.get("human_readable_table"):
        st.error("먼저 '데이터 분석 시작' 버튼을 클릭하여 데이터를 로드해주세요.")
        return

    # 세션에서 데이터 가져오기
    human_table = st.session_state.get("human_readable_table", {})
    summary = st.session_state.get("comparison_summary", {})

    # 현재 플랜 정보
    current_plan = st.session_state.get("current_plan", selected_plan_key)

    # 파라미터 추출 (summary에서 가져오기)
    plan_id = summary.get("plan_id", "")
    age = summary.get("age", 30)
    gender = summary.get("gender", "M")

    if human_table:
        # DataFrame으로 변환
        df = pd.DataFrame(human_table)

        # 전체 너비 데이터프레임
        st.dataframe(df, use_container_width=True, height=600)

        # 요약 정보 표시
        st.markdown("### 📊 비교표 요약")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("총 보험사 수", summary.get("total_companies", 0))
        with summary_col2:
            st.metric("총 보장 항목", summary.get("total_coverages", 0))
        with summary_col3:
            st.metric("분석 대상", f"{age}세 {'남성' if gender == 'M' else '여성'}")

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

        # LLM용 데이터도 표시 옵션 (최대 2개 항목으로 제한)
        llm_data = st.session_state.get("llm_readable_data", {})
        with st.expander("🔍 LLM용 데이터 보기"):
            if llm_data:
                # 전체 항목 수 계산
                total_items = sum(len(coverages) for coverages in llm_data.values())

                if total_items > 2:
                    # 최대 2개 항목만 표시
                    limited_data = {}
                    current_count = 0

                    for company, coverages in llm_data.items():
                        if current_count >= 2:
                            break

                        # insur_item_name_list에 "|" 구분자가 있는지 확인하여 복수 항목인지 체크
                        for coverage in coverages:
                            if current_count >= 2:
                                break

                            insur_item_name_list = coverage.get(
                                "insur_item_name_list", ""
                            )
                            if "|" in insur_item_name_list:
                                # 복수 항목인 경우 1건 추가로 표시
                                if company not in limited_data:
                                    limited_data[company] = []
                                limited_data[company].append(coverage)
                                current_count += 1
                            elif current_count < 1:
                                # 단일 항목인 경우 첫 1건만 표시
                                if company not in limited_data:
                                    limited_data[company] = []
                                limited_data[company].append(coverage)
                                current_count += 1

                    st.json(limited_data)
                    print(json.dumps(limited_data, ensure_ascii=False, indent=2))
                    st.info(f"⚠️ 전체 {total_items}개 항목 중 최대 2개만 표시됩니다.")
                else:
                    st.json(llm_data)
            else:
                st.info("LLM용 데이터가 없습니다.")
    else:
        st.warning("비교 표 데이터가 없습니다.")


# 모달창 표시
if st.session_state.show_comparison_modal:
    # Dialog 실행
    comparison_modal()

    # 모달창 상태 초기화
    st.session_state.show_comparison_modal = False

if "data_loaded" not in st.session_state:
    st.info("👈 사이드바에서 플랜을 조회한 후 데이터 분석을 시작해주세요.")
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
    default_prompt = (
        "뇌출혈 진단이 없는 회사는?"
        if os.getenv("ENVIRONMENT", "development") == "development"
        else ""
    )
    prompt = None

    # 개발 모드에서 기본값 버튼 제공 (첫 메시지가 없을 때만)
    if default_prompt and len(st.session_state.messages) == 0:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 개발 모드 질문", help="개발용 기본 질문 사용"):
                prompt = default_prompt
        with col2:
            st.caption("💡 개발 모드: 빠른 테스트용 기본 질문 버튼")

    # 항상 채팅 입력창 표시
    if prompt is None:  # 버튼으로 입력되지 않았을 때만
        prompt = st.chat_input("보험료나 보장내용에 대해 질문해주세요")

    if prompt:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 (백엔드 API 연동) - 스트리밍 방식
        with st.chat_message("assistant"):
            # 스트리밍 응답 처리
            with st.container():
                # 상태 표시 컨테이너 (애니메이션용)
                status_container = st.empty()
                progress_container = st.empty()
                # 애니메이션 컨트롤
                animation_stop = None

                try:
                    logger.info(f"[FRONTEND] 스트리밍 요청 시작 - 쿼리: '{prompt}'")

                    # 세션에서 LLM 데이터 가져오기
                    llm_data = st.session_state.get("llm_readable_data", {})

                    response = requests.post(
                        f"{BACKEND_URL}/chat-stream",
                        json={"query": prompt, "llm_data": llm_data},
                        stream=True,
                        timeout=180,
                        headers={
                            "Accept": "text/event-stream",
                            "Cache-Control": "no-cache",
                        },
                    )

                    logger.info(f"[FRONTEND] 응답 상태 코드: {response.status_code}")

                    full_response = ""
                    current_status = ""
                    line_count = 0
                    chunk_count = 0

                    # 버퍼링을 방지하기 위해 iter_lines에 chunk_size 설정
                    for line in response.iter_lines(
                        decode_unicode=True, chunk_size=512
                    ):
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

                                logger.info(
                                    f"[FRONTEND] 청크 {chunk_count} 수신: status={status}, message='{message}', progress={progress}%"
                                )

                                # Windows 인코딩 문제 처리
                                try:
                                    safe_message = message.encode(
                                        "utf-8", errors="ignore"
                                    ).decode("utf-8")
                                except:
                                    safe_message = str(message)

                                # 상태 메시지와 프로그레스 바 즉시 업데이트
                                if status == "searching":
                                    logger.info(
                                        f"[FRONTEND] searching 상태 업데이트: {safe_message}"
                                    )
                                    # 이전 애니메이션 중지
                                    if animation_stop:
                                        animation_stop.set()
                                    # 새로운 애니메이션 시작
                                    base_message = safe_message.replace("중...", "중")
                                    animation_stop = (
                                        create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
                                    )
                                    progress_container.progress(progress / 100.0)
                                elif status == "analyzing":
                                    logger.info(
                                        f"[FRONTEND] analyzing 상태 업데이트: {safe_message}"
                                    )
                                    # 이전 애니메이션 중지
                                    if animation_stop:
                                        animation_stop.set()
                                    # 새로운 애니메이션 시작
                                    base_message = safe_message.replace("중...", "중")
                                    animation_stop = (
                                        create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
                                    )
                                    progress_container.progress(progress / 100.0)
                                elif status == "finalizing":
                                    logger.info(
                                        f"[FRONTEND] finalizing 상태 업데이트: {safe_message}"
                                    )
                                    # 이전 애니메이션 중지
                                    if animation_stop:
                                        animation_stop.set()
                                    # 새로운 애니메이션 시작
                                    base_message = safe_message.replace("중...", "중")
                                    animation_stop = (
                                        create_animated_loading_placeholder(
                                            status_container, base_message
                                        )
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
                                    logger.info(
                                        f"[FRONTEND] 최종 응답 수신 - 길이: {len(full_response)}"
                                    )
                                elif status == "error":
                                    logger.error(
                                        f"[FRONTEND] error 상태 수신: {safe_message}"
                                    )
                                    # 애니메이션 중지
                                    if animation_stop:
                                        animation_stop.set()
                                    status_container.error(f"❌ 오류: {safe_message}")
                                    progress_container.progress(1.0)
                                    full_response = f"처리 중 오류 발생: {safe_message}"

                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"[FRONTEND] JSON 파싱 오류: {e}, 라인: {line}"
                                )
                                continue
                            except Exception as e:
                                logger.error(f"[FRONTEND] 청크 처리 오류: {e}")
                                continue

                    logger.info(
                        f"[FRONTEND] 스트리밍 완료 - 총 라인 수: {line_count}, 총 청크 수: {chunk_count}, 응답 길이: {len(full_response)}"
                    )

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
                    logger.error(
                        f"[FRONTEND] 스트리밍 요청 오류: {type(e).__name__}: {e}"
                    )
                    # 애니메이션 정리
                    if animation_stop:
                        animation_stop.set()
                    error_msg = f"스트리밍 오류: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # 데이터 뷰 (접을 수 있는 섹션)
    with st.expander("📋 분석 데이터 보기"):
        if "plan_data" in st.session_state and st.session_state.plan_data:
            plan_data = st.session_state.plan_data

            # 기본 정보 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("보험료 데이터 건수", plan_data.get("coverage_count", 0))
            with col2:
                st.metric("보장내용 데이터 건수", plan_data.get("insurance_count", 0))

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
