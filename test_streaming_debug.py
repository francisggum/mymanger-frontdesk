"""
스트리밍 디버깅 테스트 스크립트
"""
import asyncio
import json
import logging
import sys
import os

# 백엔드 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from rag_system import rag_system
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)

async def test_streaming_debug():
    """스트리밍 기능 테스트"""
    
    # 테스트 데이터프레임 생성
    test_df = pd.DataFrame({
        '보험사': ['삼성화재', '현대해상', 'KB손해보험'],
        '보험료': [100000, 95000, 110000],
        '보장내용': ['암진단비 5000만원', '암진단비 3000만원', '암진단비 4000만원']
    })
    
    test_insurance_data = [
        {'plan_id': 'test1', 'insur_name': '테스트보험', 'insur_bojang': '암진단비 5000만원 보장'},
        {'plan_id': 'test2', 'insur_name': '테스트보험2', 'insur_bojang': '상해보장 1000만원 보장'}
    ]
    
    query = "가장 저렴한 보험 추천해줘"
    
    print(f"=== 스트리밍 테스트 시작 ===")
    print(f"쿼리: {query}")
    print(f"테스트 데이터: {test_df.shape}")
    print()
    
    try:
        async for chunk in rag_system.hybrid_chat_stream(query, test_df, test_insurance_data):
            # 이모지 제거 및 출력
            chunk_text = json.dumps(chunk, ensure_ascii=False, indent=2)
            # 윈도우 콘솔에서 이모지가 깨지는 것을 방지
            safe_text = chunk_text.encode('cp949', 'ignore').decode('cp949')
            print(f"수신: {safe_text}")
            print("-" * 50)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming_debug())