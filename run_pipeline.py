#!/usr/bin/env python
"""
Complete Pipeline Runner
Executes the full chat churn prediction workflow
"""

import argparse
import time
import sys
from pathlib import Path


def print_step(step_num, step_name):
    """Print step header"""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 70)


def run_data_generation():
    """Step 1: Generate synthetic data"""
    print_step(1, "데이터 생성 (Synthetic Data Generation)")
    from src.data_generator import main as gen_main
    gen_main()


def run_feature_extraction():
    """Step 2: Extract features with DistilBERT"""
    print_step(2, "피처 추출 (Feature Extraction with DistilBERT)")
    from src.feature_engineering import main as feat_main
    feat_main()


def run_model_training():
    """Step 3: Train LightGBM model"""
    print_step(3, "모델 학습 (Model Training)")
    from src.train import main as train_main
    train_main()


def run_model_evaluation():
    """Step 4: Evaluate model"""
    print_step(4, "모델 평가 (Model Evaluation)")
    from src.evaluate import main as eval_main
    eval_main()


def run_shap_analysis():
    """Step 5: SHAP Analysis"""
    print_step(5, "SHAP 분석 (SHAP Analysis)")
    from src.shap_analysis import main as shap_main
    shap_main()


def run_embedding_analysis():
    """Step 6: Embedding Analysis"""
    print_step(6, "임베딩 분석 (Embedding Analysis)")
    from src.embedding_analysis import main as emb_main
    emb_main()


def run_report_generation():
    """Step 7: Generate Report"""
    print_step(7, "리포트 생성 (Report Generation)")
    from src.report_generator import main as report_main
    report_main()


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete chat churn prediction pipeline"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation (use existing data)"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction (use existing features)"
    )
    parser.add_argument(
        "--start-from",
        choices=["data", "features", "train", "evaluate"],
        help="Start from a specific step"
    )
    parser.add_argument(
        "--with-shap",
        action="store_true",
        help="Run SHAP analysis after evaluation"
    )
    parser.add_argument(
        "--with-embedding-analysis",
        action="store_true",
        help="Run embedding analysis"
    )
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="Generate HTML report"
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run all analyses (SHAP + embedding + report)"
    )

    args = parser.parse_args()

    # Determine which steps to run
    steps = {
        "data": not args.skip_data,
        "features": not args.skip_features,
        "train": True,
        "evaluate": True
    }

    if args.start_from:
        step_order = ["data", "features", "train", "evaluate"]
        start_idx = step_order.index(args.start_from)
        for i, step in enumerate(step_order):
            if i < start_idx:
                steps[step] = False

    print("\n" + "=" * 70)
    print("채팅 이탈 확률 예측 모델 - 전체 파이프라인 실행")
    print("Chat Churn Prediction Model - Full Pipeline")
    print("=" * 70)

    # Additional analysis steps
    run_shap = args.with_shap or args.full_analysis
    run_embedding = args.with_embedding_analysis or args.full_analysis
    run_report = args.with_report or args.full_analysis

    print("\n실행할 단계:")
    if steps["data"]:
        print("  ✓ Step 1: 데이터 생성")
    if steps["features"]:
        print("  ✓ Step 2: 피처 추출 (DistilBERT)")
    if steps["train"]:
        print("  ✓ Step 3: 모델 학습 (LightGBM)")
    if steps["evaluate"]:
        print("  ✓ Step 4: 모델 평가")
    if run_shap:
        print("  ✓ Step 5: SHAP 분석")
    if run_embedding:
        print("  ✓ Step 6: 임베딩 분석")
    if run_report:
        print("  ✓ Step 7: HTML 리포트 생성")

    input("\n계속하려면 Enter를 누르세요... (Ctrl+C로 취소)")

    start_time = time.time()

    try:
        # Step 1: Data Generation
        if steps["data"]:
            step_start = time.time()
            run_data_generation()
            print(f"\n✓ Step 1 완료 ({time.time() - step_start:.1f}초)")

        # Step 2: Feature Extraction
        if steps["features"]:
            step_start = time.time()
            run_feature_extraction()
            print(f"\n✓ Step 2 완료 ({time.time() - step_start:.1f}초)")

        # Step 3: Model Training
        if steps["train"]:
            step_start = time.time()
            run_model_training()
            print(f"\n✓ Step 3 완료 ({time.time() - step_start:.1f}초)")

        # Step 4: Model Evaluation
        if steps["evaluate"]:
            step_start = time.time()
            run_model_evaluation()
            print(f"\n✓ Step 4 완료 ({time.time() - step_start:.1f}초)")

        # Step 5: SHAP Analysis (optional)
        if run_shap:
            step_start = time.time()
            run_shap_analysis()
            print(f"\n✓ Step 5 완료 ({time.time() - step_start:.1f}초)")

        # Step 6: Embedding Analysis (optional)
        if run_embedding:
            step_start = time.time()
            run_embedding_analysis()
            print(f"\n✓ Step 6 완료 ({time.time() - step_start:.1f}초)")

        # Step 7: Report Generation (optional)
        if run_report:
            step_start = time.time()
            run_report_generation()
            print(f"\n✓ Step 7 완료 ({time.time() - step_start:.1f}초)")

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("전체 파이프라인 완료!")
        print(f"총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print("=" * 70)

        print("\n생성된 파일:")
        print("  - data/raw/chat_data.csv: 원본 데이터")
        print("  - data/processed/features.npz: 추출된 피처")
        print("  - models/churn_model.txt: 학습된 모델")
        print("  - models/evaluation_curves.png: 평가 곡선")
        print("  - models/feature_importance.png: 피처 중요도")

        if run_shap:
            print("  - models/shap_summary.png: SHAP summary plot")
            print("  - models/shap_importance_bar.png: SHAP 중요도")

        if run_embedding:
            print("  - models/embedding_importance.png: 임베딩 차원 중요도")
            print("  - models/embedding_shap_dist.png: SHAP 분포")

        if run_report:
            print("  - models/churn_analysis_report.html: HTML 리포트")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
