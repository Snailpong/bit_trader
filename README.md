# Cryptocurrency trading algorithm

## 실험 진행 상황

- 4/19: 모델 구현 (0.5% 상승 여부 + CrossEntropy)
- 4/21: Loss 수정 (MSE, MAE 전체적으로 불안정해서 target 변경 고려)
- 4/25: target 수정 (max 일자 예측, MSE)
- 4/26: 가우시안 필터 전처리
- 5/2: 데이터셋 분리(train/validation)
- 5/6: 살지 말지에 대한 모델 제작
- 5/8: 5분봉, 15분봉 차트 제작
- 5/9: 2시간 출력 데이터의 평균 상승 여부 분리해 실험 
- 5/10: validation 케이스 투자 코드
- 5/11: cosine distance baseline
- 5/12: cosine 관련 5분봉 실험, 결과 비교