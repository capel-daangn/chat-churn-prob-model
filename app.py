"""
Interactive Streamlit Dashboard for Chat Churn Prediction
Real-time prediction and SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from transformers import AutoTokenizer, AutoModel

# Import from src
from src.train import ChurnModelTrainer
from src.utils import calculate_sentiment_score, calculate_exit_intent_score
from src.shap_analysis import ShapAnalyzer


# Page config
st.set_page_config(
    page_title="채팅 이탈 예측 대시보드",
    page_icon="💬",
    layout="wide"
)

# Cache heavy operations
@st.cache_resource
def load_model():
    """Load trained model"""
    trainer = ChurnModelTrainer()
    trainer.load_model()
    return trainer

@st.cache_resource
def load_bert_model():
    """Load DistilBERT model"""
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_shap_explainer(_trainer):
    """Load SHAP explainer"""
    analyzer = ShapAnalyzer()
    analyzer.trainer = _trainer
    X_train, _, _, _ = analyzer.load_model_and_data()
    analyzer.create_explainer(X_train)
    return analyzer

@st.cache_data
def load_validation_data():
    """Load validation dataset"""
    analyzer = ShapAnalyzer()
    X_train, X_valid, y_train, y_valid = analyzer.load_model_and_data()

    # Load original messages
    config = analyzer.config
    df = pd.read_csv(config['paths']['raw_data'])

    from sklearn.model_selection import train_test_split
    _, df_valid = train_test_split(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=df['label']
    )

    return X_valid, y_valid, df_valid.reset_index(drop=True)


def get_text_embedding(text, tokenizer, model):
    """Get DistilBERT embedding for text"""
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding='max_length'
        )
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def create_features_from_text(text, dt_prev_sec=60, reaction=0):
    """Create feature vector from text input"""
    # Load models
    tokenizer, bert_model = load_bert_model()

    # Get embedding
    embedding = get_text_embedding(text, tokenizer, bert_model)

    # Calculate meta features
    msg_len = len(text)
    sentiment_score = calculate_sentiment_score(text)
    exit_intent_score = calculate_exit_intent_score(text)

    # Combine features
    meta_features = np.array([
        msg_len,
        dt_prev_sec,
        sentiment_score,
        reaction,
        exit_intent_score
    ])

    features = np.concatenate([embedding, meta_features])

    return features, {
        'msg_len': msg_len,
        'dt_prev_sec': dt_prev_sec,
        'sentiment_score': sentiment_score,
        'reaction': reaction,
        'exit_intent_score': exit_intent_score
    }


def main():
    st.title("💬 채팅 이탈 예측 대시보드")
    st.markdown("DistilBERT + LightGBM 기반 실시간 이탈 확률 예측 및 SHAP 설명")

    # Sidebar
    st.sidebar.header("설정")
    mode = st.sidebar.radio(
        "모드 선택",
        ["실시간 예측", "검증 데이터 탐색", "모델 분석"]
    )

    # Load model
    with st.spinner("모델 로딩 중..."):
        trainer = load_model()

    if mode == "실시간 예측":
        show_real_time_prediction(trainer)
    elif mode == "검증 데이터 탐색":
        show_validation_explorer(trainer)
    else:
        show_model_analysis()


def show_real_time_prediction(trainer):
    """Real-time prediction mode"""
    st.header("📝 실시간 메시지 분석")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("메시지 입력")
        text = st.text_area(
            "분석할 메시지를 입력하세요",
            value="가격 네고 가능할까요?",
            height=100
        )

        col_a, col_b = st.columns(2)
        with col_a:
            dt_prev_sec = st.slider("이전 메시지와 시간 차이 (초)", 0, 3600, 60)
        with col_b:
            reaction = st.selectbox("반응 여부", [0, 1], format_func=lambda x: "있음" if x else "없음")

    with col2:
        st.subheader("빠른 테스트")
        if st.button("긍정 메시지"):
            text = "네 좋습니다! 구매할게요"
        if st.button("부정 메시지"):
            text = "죄송하지만 다시 생각해볼게요"
        if st.button("애매한 메시지"):
            text = "..."

    if st.button("🔍 분석 실행", type="primary"):
        with st.spinner("분석 중..."):
            # Create features
            features, meta_info = create_features_from_text(text, dt_prev_sec, reaction)

            # Predict
            prob = trainer.predict(features.reshape(1, -1))[0]

            # Display results
            st.markdown("---")
            st.subheader("📊 분석 결과")

            # Probability gauge
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("이탈 확률", f"{prob*100:.1f}%")
                if prob > 0.7:
                    st.error("⚠️ 높은 이탈 위험")
                elif prob > 0.4:
                    st.warning("⚡ 중간 이탈 위험")
                else:
                    st.success("✅ 낮은 이탈 위험")

            with col2:
                st.metric("감정 점수", f"{meta_info['sentiment_score']:.2f}")

            with col3:
                st.metric("종료 의도", f"{meta_info['exit_intent_score']:.2f}")

            # Meta features
            st.subheader("📋 메타 피처")
            meta_df = pd.DataFrame([meta_info])
            st.dataframe(meta_df, use_container_width=True)

            # SHAP explanation
            st.subheader("🔬 SHAP 설명 (모델이 왜 이렇게 예측했는지)")

            with st.spinner("SHAP 값 계산 중..."):
                analyzer = load_shap_explainer(trainer)
                shap_values = analyzer.explainer.shap_values(features.reshape(1, -1))

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                # Create explanation
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=analyzer.explainer.expected_value,
                    data=features,
                    feature_names=analyzer.feature_names
                )

                # Waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(explanation, max_display=15, show=False)
                st.pyplot(fig)
                plt.close()


def show_validation_explorer(trainer):
    """Validation data explorer"""
    st.header("📂 검증 데이터 탐색")

    # Load data
    with st.spinner("데이터 로딩 중..."):
        X_valid, y_valid, df_valid = load_validation_data()
        preds = trainer.predict(X_valid)

    st.success(f"✓ {len(df_valid)}개 메시지 로드 완료")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_label = st.selectbox("실제 라벨", ["전체", "이탈 (1)", "유지 (0)"])
    with col2:
        filter_sender = st.selectbox("발신자", ["전체", "buyer", "seller"])
    with col3:
        pred_threshold = st.slider("예측 확률 최소값", 0.0, 1.0, 0.0)

    # Apply filters
    mask = np.ones(len(df_valid), dtype=bool)

    if filter_label != "전체":
        label_val = 1 if "이탈" in filter_label else 0
        mask &= (y_valid == label_val)

    if filter_sender != "전체":
        mask &= (df_valid['sender_role'] == filter_sender)

    mask &= (preds >= pred_threshold)

    # Filtered data
    filtered_df = df_valid[mask].copy()
    filtered_df['churn_prob'] = preds[mask]
    filtered_df['actual_label'] = y_valid[mask]

    st.write(f"필터링된 메시지: {len(filtered_df)}개")

    # Sort by prediction
    sort_by = st.radio("정렬", ["이탈 확률 높은 순", "이탈 확률 낮은 순"], horizontal=True)
    ascending = "낮은" in sort_by
    filtered_df = filtered_df.sort_values('churn_prob', ascending=ascending)

    # Display messages
    st.subheader("메시지 목록")

    for idx, row in filtered_df.head(20).iterrows():
        with st.expander(
            f"[{row['churn_prob']*100:.1f}%] {row['content'][:50]}..."
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**메시지**: {row['content']}")
                st.write(f"**발신자**: {row['sender_role']}")
                st.write(f"**실제 라벨**: {'이탈' if row['actual_label'] == 1 else '유지'}")

            with col2:
                st.metric("예측 확률", f"{row['churn_prob']*100:.1f}%")
                st.metric("메시지 길이", int(row['msg_len']))


def show_model_analysis():
    """Model analysis mode"""
    st.header("🔬 모델 분석")

    tab1, tab2, tab3 = st.tabs(["피처 중요도", "SHAP 분석", "성능 지표"])

    with tab1:
        st.subheader("피처 중요도")
        try:
            from PIL import Image
            img = Image.open("models/feature_importance.png")
            st.image(img, use_container_width=True)
        except:
            st.warning("피처 중요도 이미지를 찾을 수 없습니다. 먼저 evaluate.py를 실행하세요.")

    with tab2:
        st.subheader("SHAP Summary Plot")
        try:
            from PIL import Image
            img = Image.open("models/shap_summary.png")
            st.image(img, use_container_width=True)
        except:
            st.warning("SHAP 분석을 먼저 실행하세요: `python -m src.shap_analysis`")

    with tab3:
        st.subheader("모델 성능 곡선")
        try:
            from PIL import Image
            img = Image.open("models/evaluation_curves.png")
            st.image(img, use_container_width=True)
        except:
            st.warning("평가 결과를 찾을 수 없습니다. 먼저 evaluate.py를 실행하세요.")


if __name__ == "__main__":
    main()
