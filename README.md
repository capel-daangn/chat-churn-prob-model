# 채팅 이탈 확률 예측 모델 (Chat Churn Prediction Model)

DistilBERT + LightGBM 기반의 경량형 채팅 이탈 예측 시스템입니다. GPU 없이도 빠르게 실행 가능하며, 각 메시지 이후 대화가 끊길 확률을 예측합니다.

## 프로젝트 개요

### 특징
- **DistilBERT**: 문장 의미를 768차원 벡터로 표현 (한국어 지원)
- **LightGBM**: CPU 기반 빠른 이탈 확률 예측
- **경량화**: GPU 불필요, 맥/일반 서버 환경 모두 지원
- **높은 성능**: AUROC 0.8+ 수준 달성 가능
- **해석력**: Feature importance를 통한 모델 해석 가능

### 모델 구조
```
텍스트(content) → DistilBERT → 문장 임베딩(768차원)
                                         │
                                         ▼
              +──────────────────────────────────+
              │ 메타데이터 (시간차, 길이, 감정 등) │
              +──────────────────────────────────+
                                         │
                                         ▼
                          Feature Concatenation
                                         │
                                         ▼
                              LightGBM Classifier
                                         │
                                         ▼
                           이탈 확률(p_churn) 예측
```

## 프로젝트 구조

```
chat-churn-prob-model/
├── data/
│   ├── raw/                    # 생성된 원본 데이터
│   └── processed/              # 전처리 완료 데이터
├── models/                     # 학습된 모델 저장
├── src/
│   ├── data_generator.py       # 합성 채팅 데이터 생성
│   ├── feature_engineering.py  # 피처 추출 및 임베딩
│   ├── train.py                # 모델 학습
│   ├── evaluate.py             # 모델 평가
│   └── utils.py                # 공통 유틸리티
├── requirements.txt            # 패키지 의존성
├── config.yaml                 # 설정 파일
└── README.md
```

## 설치 방법

### 1. 저장소 클론 (또는 디렉토리 이동)
```bash
cd chat-churn-prob-model
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

**중요**: 모든 명령은 프로젝트 루트 디렉토리에서 실행해야 합니다 (`config.yaml`이 있는 위치)

### 빠른 시작: 전체 파이프라인 실행

가장 간단한 방법은 전체 파이프라인을 한 번에 실행하는 것입니다:

```bash
python run_pipeline.py
```

이 명령은 데이터 생성부터 모델 평가까지 모든 단계를 자동으로 실행합니다.

**옵션**:
```bash
# 데이터 생성 스킵 (기존 데이터 사용)
python run_pipeline.py --skip-data

# 피처 추출부터 시작
python run_pipeline.py --start-from features

# 학습부터 시작 (데이터와 피처는 이미 준비됨)
python run_pipeline.py --start-from train
```

---

### 단계별 실행 (고급)

각 단계를 개별적으로 실행하고 싶다면 아래 방법을 사용하세요.

#### Step 1: 합성 데이터 생성
```bash
python -m src.data_generator
```

출력:
- `data/raw/chat_data.csv`: 500개 대화, 약 5000개 메시지

### Step 2: 피처 추출 (DistilBERT 임베딩)
```bash
python -m src.feature_engineering
```

출력:
- `data/processed/features.npz`: 임베딩 + 메타피처

**참고**: 첫 실행 시 DistilBERT 모델을 다운로드하므로 시간이 걸릴 수 있습니다 (~500MB).

### Step 3: 모델 학습
```bash
python -m src.train
```

출력:
- `models/churn_model.txt`: 학습된 LightGBM 모델
- 학습 로그 및 성능 지표 출력

### Step 4: 모델 평가
```bash
python -m src.evaluate
```

출력:
- 성능 지표 (AUROC, PR-AUC, Precision/Recall)
- `models/evaluation_curves.png`: ROC & PR 곡선
- `models/feature_importance.png`: 피처 중요도
- 예측 샘플 출력

## SHAP 모델 해석 (NEW!)

이 프로젝트는 SHAP (SHapley Additive exPlanations)를 통합하여 모델의 예측을 설명합니다.

### SHAP 분석 실행

#### 방법 1: 개별 실행
```bash
# 기본 SHAP 분석
python -m src.shap_analysis

# 임베딩 차원 분석
python -m src.embedding_analysis

# HTML 리포트 생성
python -m src.report_generator
```

#### 방법 2: 파이프라인과 함께 실행
```bash
# SHAP 분석 포함
python run_pipeline.py --with-shap

# 임베딩 분석 포함
python run_pipeline.py --with-embedding-analysis

# HTML 리포트 포함
python run_pipeline.py --with-report

# 모든 분석 실행
python run_pipeline.py --full-analysis
```

### 인터랙티브 대시보드

Streamlit 기반 웹 인터페이스로 실시간 예측 및 SHAP 설명을 확인할 수 있습니다:

```bash
streamlit run app.py
```

**대시보드 기능**:
- 🔍 실시간 메시지 입력 및 이탈 확률 예측
- 📊 SHAP waterfall plot으로 예측 근거 시각화
- 📂 검증 데이터셋 탐색 및 필터링
- 📈 모델 성능 지표 및 피처 중요도 확인

### SHAP 분석 결과물

- `models/shap_summary.png`: 전역 피처 중요도 (beeswarm plot)
- `models/shap_importance_bar.png`: 피처 중요도 (bar plot)
- `models/shap_waterfall_*.png`: 개별 예측 설명
- `models/embedding_importance.png`: 임베딩 차원 중요도
- `models/churn_analysis_report.html`: 종합 분석 리포트

---

## 설정 파일 (config.yaml)

주요 파라미터를 `config.yaml`에서 조정할 수 있습니다:

```yaml
data:
  num_conversations: 500        # 생성할 대화 수
  churn_ratio: 0.3              # 이탈 비율

model:
  name: "distilbert-base-multilingual-cased"
  max_length: 64

lightgbm:
  learning_rate: 0.05
  num_leaves: 64
  num_boost_round: 500
```

## 예상 성능

| 지표       | 값              |
|----------|----------------|
| AUROC    | 0.80 ~ 0.83    |
| PR-AUC   | 0.50 ~ 0.55    |
| 추론 속도   | 10~30ms/대화    |
| 필요 리소스  | CPU만으로 가능    |

## 피처 설명

### 텍스트 임베딩
- **DistilBERT embedding**: 768차원 벡터 (문장 의미 표현)

### 메타데이터 피처
- `msg_len`: 메시지 길이 (문자 수)
- `dt_prev_sec`: 이전 메시지와의 시간 차이 (초)
- `sentiment_score`: 감정 점수 (-1: 부정 ~ +1: 긍정)
- `reaction`: 반응 여부 (0/1)
- `exit_intent_score`: 대화 종료 신호 강도 (0~1)

## 예측 결과 예시

| 메시지                  | 예측 이탈 확률 |
|-----------------------|---------|
| "가격 네고 가능할까요?"       | 0.11    |
| "죄송하지만 다시 생각해볼게요"   | 0.74    |
| "..."                 | 0.89    |

## 실제 데이터 사용

실제 채팅 데이터를 사용하려면:

1. CSV 파일을 다음 컬럼으로 준비:
   - `message_id`, `conversation_id`, `sender_role`, `content`, `created_at`, `label`

2. `config.yaml`에서 경로 수정:
   ```yaml
   paths:
     raw_data: "data/raw/your_real_data.csv"
   ```

3. Step 2부터 실행

## 문제 해결

### 메모리 부족
- `config.yaml`에서 `num_conversations` 줄이기
- 배치 처리로 임베딩 추출 수정

### 느린 실행 속도
- DistilBERT의 `max_length` 줄이기 (64 → 32)
- GPU 사용 (CUDA 설치 필요)

### 성능 저하
- 더 많은 데이터 생성
- LightGBM 하이퍼파라미터 튜닝
- 추가 피처 엔지니어링

## 라이선스

MIT License

## 참고 자료

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
