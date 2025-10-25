"""
Model Evaluation and Analysis
Provides detailed evaluation metrics and feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
import lightgbm as lgb
from .train import ChurnModelTrainer
from .utils import load_config


class ModelEvaluator:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.trainer = ChurnModelTrainer(config_path)

    def evaluate_model(self):
        """Comprehensive model evaluation"""

        # Load model
        self.trainer.load_model()

        # Load data
        X, y = self.trainer.load_features()

        # Split data (same as training)
        from sklearn.model_selection import train_test_split
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Predictions
        train_preds = self.trainer.predict(X_train)
        valid_preds = self.trainer.predict(X_valid)

        results = {
            'X_train': X_train,
            'X_valid': X_valid,
            'y_train': y_train,
            'y_valid': y_valid,
            'train_preds': train_preds,
            'valid_preds': valid_preds
        }

        # Print metrics
        self.print_metrics(results)

        # Plot ROC and PR curves
        self.plot_curves(results)

        # Feature importance
        self.plot_feature_importance()

        # Example predictions
        self.show_example_predictions(results)

        return results

    def print_metrics(self, results):
        """Print evaluation metrics"""
        y_train = results['y_train']
        y_valid = results['y_valid']
        train_preds = results['train_preds']
        valid_preds = results['valid_preds']

        print("\n=== Model Performance Metrics ===")

        # AUC scores
        train_auc = roc_auc_score(y_train, train_preds)
        valid_auc = roc_auc_score(y_valid, valid_preds)

        print(f"\nAUROC:")
        print(f"  Train: {train_auc:.4f}")
        print(f"  Valid: {valid_auc:.4f}")

        # PR-AUC scores
        train_ap = average_precision_score(y_train, train_preds)
        valid_ap = average_precision_score(y_valid, valid_preds)

        print(f"\nPR-AUC (Average Precision):")
        print(f"  Train: {train_ap:.4f}")
        print(f"  Valid: {valid_ap:.4f}")

        # Classification report at different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            print(f"\n=== Classification Report (threshold={threshold}) ===")
            valid_preds_binary = (valid_preds > threshold).astype(int)
            print(classification_report(
                y_valid,
                valid_preds_binary,
                target_names=['No Churn', 'Churn'],
                zero_division=0
            ))

            # Confusion matrix
            cm = confusion_matrix(y_valid, valid_preds_binary)
            print(f"Confusion Matrix:\n{cm}")

    def plot_curves(self, results):
        """Plot ROC and Precision-Recall curves"""
        y_valid = results['y_valid']
        valid_preds = results['valid_preds']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_valid, valid_preds)
        auc = roc_auc_score(y_valid, valid_preds)

        axes[0].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_valid, valid_preds)
        ap = average_precision_score(y_valid, valid_preds)

        axes[1].plot(recall, precision, label=f'AP = {ap:.3f}', linewidth=2)
        axes[1].axhline(y=y_valid.mean(), color='k', linestyle='--', label='Baseline')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/evaluation_curves.png', dpi=150, bbox_inches='tight')
        print("\nCurves saved to models/evaluation_curves.png")
        plt.close()

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        model = self.trainer.model

        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_names = [f'emb_{i}' for i in range(768)] + self.config['features']['meta_features']

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)

        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance (Gain)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved to models/feature_importance.png")
        plt.close()

        # Print top features
        print("\n=== Top 10 Most Important Features ===")
        print(importance_df.head(10).to_string(index=False))

    def show_example_predictions(self, results, n_samples=10):
        """Show example predictions"""
        X_valid = results['X_valid']
        y_valid = results['y_valid']
        valid_preds = results['valid_preds']

        # Load original data to get message content
        data_path = self.config['paths']['raw_data']
        df = pd.read_csv(data_path)

        # Get validation indices
        from sklearn.model_selection import train_test_split
        _, df_valid = train_test_split(
            df,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=df['label']
        )

        # Add predictions
        df_valid = df_valid.copy()
        df_valid['churn_prob'] = valid_preds

        print("\n=== Example Predictions ===")

        # Show high probability churns
        print("\nHigh Churn Probability (Top 5):")
        high_churn = df_valid.nlargest(5, 'churn_prob')
        for idx, row in high_churn.iterrows():
            print(f"\nMessage: {row['content']}")
            print(f"  Predicted Churn Prob: {row['churn_prob']:.3f}")
            print(f"  Actual Label: {row['label']}")
            print(f"  Sender: {row['sender_role']}")

        # Show low probability churns
        print("\n\nLow Churn Probability (Bottom 5):")
        low_churn = df_valid.nsmallest(5, 'churn_prob')
        for idx, row in low_churn.iterrows():
            print(f"\nMessage: {row['content']}")
            print(f"  Predicted Churn Prob: {row['churn_prob']:.3f}")
            print(f"  Actual Label: {row['label']}")
            print(f"  Sender: {row['sender_role']}")


def main():
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model()

    print("\n=== Evaluation Complete ===")

    # Run SHAP analysis
    print("\n" + "=" * 70)
    user_input = input("SHAP 분석을 실행하시겠습니까? (y/n): ")

    if user_input.lower() == 'y':
        from .shap_analysis import ShapAnalyzer
        print("\nRunning SHAP analysis...")
        shap_analyzer = ShapAnalyzer()
        shap_analyzer.full_analysis(max_samples=500)


if __name__ == "__main__":
    main()
