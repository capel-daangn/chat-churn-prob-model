"""
SHAP Analysis for Chat Churn Prediction Model
Provides model explainability using SHAP (SHapley Additive exPlanations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from .train import ChurnModelTrainer
from .utils import load_config


class ShapAnalyzer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.trainer = ChurnModelTrainer(config_path)
        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def load_model_and_data(self):
        """Load trained model and validation data"""
        print("Loading model and data...")

        # Load model
        self.trainer.load_model()

        # Load features
        X, y = self.trainer.load_features()

        # Split data (same as training)
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Create feature names
        embedding_names = [f'emb_{i}' for i in range(768)]
        meta_names = self.config['features']['meta_features']
        self.feature_names = embedding_names + meta_names

        print(f"Loaded {len(X_valid)} validation samples")

        return X_train, X_valid, y_train, y_valid

    def create_explainer(self, X_train):
        """Create SHAP TreeExplainer for LightGBM"""
        print("\nCreating SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.trainer.model)
        print("✓ Explainer created")

        return self.explainer

    def calculate_shap_values(self, X, max_samples=None):
        """Calculate SHAP values for given data"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")

        if max_samples and len(X) > max_samples:
            print(f"Sampling {max_samples} from {len(X)} samples for SHAP calculation...")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        else:
            print(f"Calculating SHAP values for {len(X)} samples...")

        self.shap_values = self.explainer.shap_values(X)

        # Handle multi-class output (LightGBM returns list for binary classification)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class

        print(f"✓ SHAP values calculated: {self.shap_values.shape}")

        return self.shap_values

    def plot_summary(self, X, y=None, save_path="models/shap_summary.png"):
        """Create SHAP summary plot (beeswarm)"""
        print("\nGenerating SHAP summary plot...")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Summary plot saved to {save_path}")
        plt.close()

    def plot_bar(self, save_path="models/shap_importance_bar.png"):
        """Create SHAP bar plot (mean absolute values)"""
        print("\nGenerating SHAP bar plot...")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Bar plot saved to {save_path}")
        plt.close()

    def plot_waterfall(self, X, index=0, save_path="models/shap_waterfall.png"):
        """Create waterfall plot for a single prediction"""
        print(f"\nGenerating waterfall plot for sample {index}...")

        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[index],
            base_values=self.explainer.expected_value,
            data=X[index],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Waterfall plot saved to {save_path}")
        plt.close()

    def analyze_top_predictions(self, X, y, n_samples=5):
        """Analyze top churn and no-churn predictions with SHAP"""
        print(f"\n=== Analyzing Top {n_samples} Predictions ===")

        # Get predictions
        preds = self.trainer.predict(X)

        # Top churns (highest probability)
        top_churn_idx = np.argsort(preds)[-n_samples:][::-1]

        # Top no-churns (lowest probability)
        top_no_churn_idx = np.argsort(preds)[:n_samples]

        print(f"\n--- Top {n_samples} Churn Predictions ---")
        for i, idx in enumerate(top_churn_idx):
            print(f"\n{i+1}. Sample {idx}")
            print(f"   Predicted Probability: {preds[idx]:.3f}")
            print(f"   Actual Label: {y[idx]}")

            # Save waterfall for first few
            if i < 3:
                save_path = f"models/shap_waterfall_churn_{i+1}.png"
                self.plot_waterfall(X, idx, save_path)

        print(f"\n--- Top {n_samples} No-Churn Predictions ---")
        for i, idx in enumerate(top_no_churn_idx):
            print(f"\n{i+1}. Sample {idx}")
            print(f"   Predicted Probability: {preds[idx]:.3f}")
            print(f"   Actual Label: {y[idx]}")

    def get_top_features(self, top_n=10):
        """Get top N most important features by mean absolute SHAP value"""
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # Get top indices
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': self.feature_names[idx],
                'mean_abs_shap': mean_abs_shap[idx],
                'feature_type': 'embedding' if idx < 768 else 'meta'
            })

        return pd.DataFrame(top_features)

    def full_analysis(self, max_samples=1000):
        """Run complete SHAP analysis"""
        print("\n" + "=" * 70)
        print("SHAP Analysis - Model Explainability")
        print("=" * 70)

        # Load data
        X_train, X_valid, y_train, y_valid = self.load_model_and_data()

        # Create explainer
        self.create_explainer(X_train)

        # Calculate SHAP values
        self.calculate_shap_values(X_valid, max_samples=max_samples)

        # Generate plots
        self.plot_summary(X_valid[:len(self.shap_values)], y_valid[:len(self.shap_values)])
        self.plot_bar()

        # Top features
        print("\n=== Top 10 Most Important Features ===")
        top_features_df = self.get_top_features(10)
        print(top_features_df.to_string(index=False))

        # Analyze top predictions
        self.analyze_top_predictions(
            X_valid[:len(self.shap_values)],
            y_valid[:len(self.shap_values)],
            n_samples=5
        )

        print("\n" + "=" * 70)
        print("SHAP Analysis Complete!")
        print("=" * 70)

        return {
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'top_features': top_features_df
        }


def main():
    analyzer = ShapAnalyzer()
    results = analyzer.full_analysis(max_samples=500)


if __name__ == "__main__":
    main()
