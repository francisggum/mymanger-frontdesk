import pandas as pd
import numpy as np
from rag_system import rag_system

# 실제 보험료 비교 테이블과 유사한 구조 생성
companies = [
    'ABL생명', 'DB생명', 'DB손해보험', '교보생명', 'KB라이프', '삼성생명', 
    '신한생명', '한화생명', '흥국생명', '메리츠화재', '미래에셋생명', '동양생명'
]

# 테스트 데이터 생성
coverage_data = [
    '뇌출혈진단비', '뇌경색증진단비', '뇌수술비', '급성심근경색증진단비', 
    '뇌졸중후유장해연금', '관상동맥우회로수술비', '최저기본계약조건'
]

# 비교 테이블 생성 (0.0 값 포함)
data = []
for coverage in coverage_data:
    row = {}
    for company in companies:
        # 뇌출혈진단비는 약 30% 보험사가 제공하지 않음 (0.0)
        if coverage == '뇌출혈진단비':
            if company in ['ABL생명', 'DB손해보험', '흥국생명', '메리츠화재']:  # 특정 보험사는 0.0
                row[company] = 0.0
            else:
                row[company] = np.random.randint(5000, 25000)
        else:
            # 다른 보장 항목은 20% 확률로 0.0
            if np.random.random() < 0.2:
                row[company] = 0.0
            else:
                row[company] = np.random.randint(3000, 20000)
    data.append(row)

# 보험료 합계와 보장갯수 행 추가
comparison_df = pd.DataFrame(data, index=coverage_data)

# 보험료 합계 계산
premium_sums = {}
for company in companies:
    premium_sums[company] = comparison_df[company].sum()
comparison_df.loc['보험료 합계'] = premium_sums

# 보장갯수 계산
coverage_counts = {}
for company in companies:
    coverage_counts[company] = (comparison_df[company] > 0).sum()
comparison_df.loc['보장갯수'] = coverage_counts

print('=== 생성된 비교 테이블 ===')
print(comparison_df)
print()

# 뇌출혈진단비 데이터 확인
brain_hemorrhage_data = comparison_df.loc['뇌출혈진단비']
zero_companies = brain_hemorrhage_data[brain_hemorrhage_data == 0.0].index.tolist()
nonzero_companies = brain_hemorrhage_data[brain_hemorrhage_data > 0].index.tolist()

print('뇌출혈진단비가 있는 보험사:', len(nonzero_companies), '개')
print('뇌출혈진단비가 없는 보험사:', len(zero_companies), '개')
for company in zero_companies:
    print(f'  - {company}')
print()

# Pandas 에이전트 테스트
print('=== Pandas 에이전트 테스트 ===')
query = '뇌출혈진단이 없는 보험사는?'
print(f'질문: {query}')

try:
    result = rag_system.pandas_analysis(comparison_df, query)
    print('분석 결과:')
    print(result)
except Exception as e:
    print(f'오류 발생: {e}')
    import traceback
    traceback.print_exc()

