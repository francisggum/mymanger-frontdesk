#!/usr/bin/env python3
"""
Plan B: 결과 형식 표준화 테스트
Agent 결과 유효성 검사 개선 확인
"""

import logging
import sys
import os
import pandas as pd
from rag_system import rag_system

logging.basicConfig(level=logging.INFO)

# 테스트 데이터 생성 (실제와 유사한 구조)
test_data = {
    'company_name': ['흥국생명', 'DB생명', '동양생명', 'ABL생명', 'KB라이프'],
    'total_premium': [188584, 207854, 215801, 233644, 235569],
    'coverage_count': [33, 30, 37, 28, 34],
    'basic_premium': [12200, 8500, 9200, 11500, 12200],
    'cancer_coverage': [10000000, 8000000, 12000000, 9000000, 11000000],
    'cancer_premium': [2500, 3200, 2100, 2800, 2400],
    'stroke_coverage': [5000000, 7000000, 6000000, 5500000, 6500000],
    'stroke_premium': [1800, 2100, 1900, 2000, 1700]
}

df = pd.DataFrame(test_data)
query = "보험료가 가장 저렴한 회사는?"

print("=== Plan B: 결과 형식 표준화 테스트 ===")
print(f"테스트 데이터: {df.shape}")
print(f"질문: {query}")
print()

# 2단계 설정
rag_system.pandas_analysis_stages = 2

try:
    result = rag_system.pandas_analysis(df, query)
    
    print("=== 분석 결과 ===")
    print(f"결과 길이: {len(result)}자")
    print(f"결과 내용:")
    print("-" * 50)
    print(result)
    print("-" * 50)
    
    # 결과 분석
    if len(result) > 100:
        print("[성공] 상세한 분석 결과가 반환됨")
        if "흥국생명" in result:
            print("[성공] 최저가 회사 정보 포함됨")
        if any(keyword in result for keyword in ["순위", "보험료", "추천"]):
            print("[성공] 관련 키워드 포함됨")
    else:
        print("[실패] 분석 결과가 너무 짧음")
        if "데이터 통계 분석 결과" in result:
            print("[실패] Fallback 모드로 전환됨")

except Exception as e:
    print(f"[오류] {type(e).__name__}: {str(e)}")

print("\n=== 테스트 완료 ===")