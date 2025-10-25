"""
DistilBERT Embedding Analysis with SHAP
Analyzes which embedding dimensions are most important for churn prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from .shap_analysis import ShapAnalyzer
from .utils import load_config


class EmbeddingAnalyzer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.shap_analyzer = ShapAnalyzer(config_path)
        self.embedding_dim = 768

    def analyze_embedding_importance(self, max_samples=1000):
        """Analyze which embedding dimensions are most important"""
        print("\n" + "=" * 70)
        print("DistilBERT Embedding Analysis")
        print("=" * 70)

        # Load data and create explainer
        X_train, X_valid, y_train, y_valid = self.shap_analyzer.load_model_and_data()
        self.shap_analyzer.create_explainer(X_train)

        # Calculate SHAP values
        print(f"\nCalculating SHAP values for {min(max_samples, len(X_valid))} samples...")
        shap_values = self.shap_analyzer.calculate_shap_values(X_valid, max_samples=max_samples)

        # Extract embedding SHAP values (first 768 dimensions)
        embedding_shap = shap_values[:, :self.embedding_dim]

        # Analyze
        results = self._analyze_dimensions(embedding_shap, X_valid[:len(shap_values)])

        # Visualizations
        self._plot_dimension_importance(results)
        self._plot_dimension_distribution(embedding_shap)
        self._plot_top_dimensions_detail(embedding_shap, X_valid[:len(shap_values)], results)

        return results

    def _analyze_dimensions(self, embedding_shap, X_valid):
        """Analyze embedding dimensions"""
        print("\nAnalyzing embedding dimensions...")

        # Calculate statistics for each dimension
        mean_abs_shap = np.abs(embedding_shap).mean(axis=0)
        std_shap = embedding_shap.std(axis=0)
        max_shap = np.abs(embedding_shap).max(axis=0)

        # Create results dataframe
        results = pd.DataFrame({
            'dimension': range(self.embedding_dim),
            'mean_abs_shap': mean_abs_shap,
            'std_shap': std_shap,
            'max_abs_shap': max_shap
        })

        results = results.sort_values('mean_abs_shap', ascending=False)

        print("\n=== Top 20 Most Important Embedding Dimensions ===")
        print(results.head(20).to_string(index=False))

        # Statistics
        print(f"\n=== Embedding Dimension Statistics ===")
        print(f"Total dimensions: {self.embedding_dim}")
        print(f"Mean importance: {mean_abs_shap.mean():.6f}")
        print(f"Std importance: {mean_abs_shap.std():.6f}")
        print(f"Max importance: {mean_abs_shap.max():.6f}")
        print(f"Min importance: {mean_abs_shap.min():.6f}")

        # Top 10%
        top_10_pct = int(self.embedding_dim * 0.1)
        top_dims = results.head(top_10_pct)
        print(f"\nTop 10% dimensions ({top_10_pct} dims) account for "
              f"{top_dims['mean_abs_shap'].sum() / mean_abs_shap.sum() * 100:.1f}% of total importance")

        return results

    def _plot_dimension_importance(self, results, save_path="models/embedding_importance.png"):
        """Plot embedding dimension importance"""
        print("\nGenerating dimension importance plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top 30 dimensions bar plot
        ax = axes[0, 0]
        top_30 = results.head(30)
        ax.barh(range(len(top_30)), top_30['mean_abs_shap'].values)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels([f"dim_{d}" for d in top_30['dimension'].values])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('Top 30 Embedding Dimensions')
        ax.grid(alpha=0.3)

        # Distribution of importance
        ax = axes[0, 1]
        ax.hist(results['mean_abs_shap'].values, bins=50, edgecolor='black')
        ax.set_xlabel('Mean |SHAP|')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Dimension Importance')
        ax.axvline(results['mean_abs_shap'].mean(), color='red', linestyle='--', label='Mean')
        ax.legend()
        ax.grid(alpha=0.3)

        # Cumulative importance
        ax = axes[1, 0]
        sorted_importance = np.sort(results['mean_abs_shap'].values)[::-1]
        cumsum = np.cumsum(sorted_importance) / sorted_importance.sum()
        ax.plot(range(len(cumsum)), cumsum * 100)
        ax.set_xlabel('Number of Dimensions')
        ax.set_ylabel('Cumulative Importance (%)')
        ax.set_title('Cumulative Importance')
        ax.grid(alpha=0.3)
        ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80%')
        ax.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90%')
        ax.legend()

        # Importance vs std
        ax = axes[1, 1]
        ax.scatter(results['mean_abs_shap'], results['std_shap'], alpha=0.5, s=20)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_ylabel('Std SHAP')
        ax.set_title('Importance vs Variability')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Dimension importance plot saved to {save_path}")
        plt.close()

    def _plot_dimension_distribution(self, embedding_shap, save_path="models/embedding_shap_dist.png"):
        """Plot distribution of SHAP values for top dimensions"""
        print("\nGenerating SHAP value distribution plot...")

        # Get top 12 dimensions
        mean_abs_shap = np.abs(embedding_shap).mean(axis=0)
        top_dims = np.argsort(mean_abs_shap)[-12:][::-1]

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.ravel()

        for i, dim in enumerate(top_dims):
            ax = axes[i]
            shap_vals = embedding_shap[:, dim]

            ax.hist(shap_vals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title(f'Dim {dim}')
            ax.set_xlabel('SHAP value')
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)

        plt.suptitle('SHAP Value Distribution for Top 12 Dimensions', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Distribution plot saved to {save_path}")
        plt.close()

    def _plot_top_dimensions_detail(self, embedding_shap, X_valid, results,
                                     save_path="models/embedding_top_dims.png"):
        """Detailed analysis of top dimensions"""
        print("\nGenerating detailed plot for top dimensions...")

        top_5_dims = results.head(5)['dimension'].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot: SHAP vs actual embedding value
        ax = axes[0]
        for dim in top_5_dims:
            ax.scatter(
                X_valid[:, dim],
                embedding_shap[:, dim],
                alpha=0.3,
                s=10,
                label=f'Dim {dim}'
            )

        ax.set_xlabel('Embedding Value')
        ax.set_ylabel('SHAP Value')
        ax.set_title('SHAP vs Embedding Value (Top 5 Dims)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Box plot
        ax = axes[1]
        data_for_box = [embedding_shap[:, dim] for dim in top_5_dims]
        bp = ax.boxplot(data_for_box, labels=[f'Dim {d}' for d in top_5_dims])
        ax.set_ylabel('SHAP Value')
        ax.set_title('SHAP Value Distribution (Top 5 Dims)')
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Top dimensions detail plot saved to {save_path}")
        plt.close()

    def compare_meta_vs_embedding(self):
        """Compare importance of meta features vs embeddings"""
        print("\n=== Comparing Meta Features vs Embedding ===")

        # Load SHAP values
        X_train, X_valid, y_train, y_valid = self.shap_analyzer.load_model_and_data()
        self.shap_analyzer.create_explainer(X_train)
        shap_values = self.shap_analyzer.calculate_shap_values(X_valid, max_samples=1000)

        # Split SHAP values
        embedding_shap = shap_values[:, :768]
        meta_shap = shap_values[:, 768:]

        # Calculate importance
        embedding_importance = np.abs(embedding_shap).mean()
        meta_importance_per_feature = np.abs(meta_shap).mean(axis=0)
        meta_importance_total = np.abs(meta_shap).mean()

        print(f"\nAverage importance per dimension:")
        print(f"  Embedding (768 dims): {embedding_importance:.6f}")
        print(f"  Meta features (5 dims): {meta_importance_total:.6f}")

        print(f"\nTotal importance:")
        print(f"  Embedding: {embedding_importance * 768:.4f} ({embedding_importance * 768 / (embedding_importance * 768 + meta_importance_total * 5) * 100:.1f}%)")
        print(f"  Meta features: {meta_importance_total * 5:.4f} ({meta_importance_total * 5 / (embedding_importance * 768 + meta_importance_total * 5) * 100:.1f}%)")

        print(f"\nMeta feature importance:")
        meta_names = self.config['features']['meta_features']
        for name, importance in zip(meta_names, meta_importance_per_feature):
            print(f"  {name}: {importance:.6f}")


def main():
    analyzer = EmbeddingAnalyzer()

    # Run analyses
    results = analyzer.analyze_embedding_importance(max_samples=500)
    analyzer.compare_meta_vs_embedding()

    print("\n" + "=" * 70)
    print("Embedding Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
