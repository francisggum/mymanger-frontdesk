import pandas as pd
from rag_system import rag_system

# 테스트 데이터 생성
test_data = pd.DataFrame({
    '보험사': ['삼성생명', 'KB생명', '신한생명'],
    '보험료': [100000, 95000, 105000],
    '암진단비': [50000000, 45000000, 55000000]
})

print('테스트 데이터:')
print(test_data)
print()

# Pandas 분석 테스트
result = rag_system.pandas_analysis(test_data, '가장 저렴한 보험사 추천해줘')
print('분석 결과:')
print(result)