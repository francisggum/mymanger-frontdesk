#!/usr/bin/env python3
"""
판다스 분석 단계 제한 테스트
"""

import logging
import sys
import os
import pandas as pd
from rag_system import rag_system

logging.basicConfig(level=logging.INFO)

# 테스트 데이터 생성
test_data = {
    'insur_name': ['삼성화재', '메리츠화재', 'KB손해보험'],
    'premium_amount': [50000, 45000, 52000],
    'coverage_amount': [1000000, 900000, 1100000]
}

df = pd.DataFrame(test_data)
query = "가장 저렴한 보험 추천해줘"

print("=== 2단계 분석 테스트 ===")
rag_system.pandas_analysis_stages = 2
result_2stage = rag_system.pandas_analysis(df, query)
print(f"2단계 결과 길이: {len(result_2stage)}자")
print(f"2단계 결과: {result_2stage[:200]}...")

print("\n=== 3단계 분석 테스트 ===")
rag_system.pandas_analysis_stages = 3
result_3stage = rag_system.pandas_analysis(df, query)
print(f"3단계 결과 길이: {len(result_3stage)}자")
print(f"3단계 결과: {result_3stage[:200]}...")

print(f"\n결과 비교: 3단계가 2단계보다 {len(result_3stage) - len(result_2stage)}자 더 김")