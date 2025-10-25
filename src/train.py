"""
Model Training with LightGBM
Trains a binary classifier to predict chat churn probability
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import yaml
from .utils import load_config
import os


class ChurnModelTrainer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.lgb_params = self.config['lightgbm']
        self.training_config = self.config['training']
        self.model = None

        # Print key parameters
        print("LightGBM Parameters:")
        print(f"  - is_unbalance: {self.lgb_params.get('is_unbalance', False)}")
        print(f"  - learning_rate: {self.lgb_params.get('learning_rate')}")
        print(f"  - num_leaves: {self.lgb_params.get('num_leaves')}")

    def load_features(self):
        """Load preprocessed features"""
        data_path = self.config['paths']['processed_data']

        print(f"Loading features from {data_path}")
        data = np.load(data_path)

        X = data['X']
        y = data['y']

        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y.astype(int))}")

        return X, y

    def train(self, X=None, y=None):
        """Train LightGBM model"""

        if X is None or y is None:
            X, y = self.load_features()

        # Split data
        test_size = self.training_config['test_size']
        random_state = self.training_config['random_state']

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Valid set: {len(X_valid)} samples")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        # Train model
        print("\nTraining LightGBM model...")
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=self.lgb_params['num_boost_round'],
            callbacks=[
                lgb.early_stopping(self.lgb_params['early_stopping_rounds']),
                lgb.log_evaluation(period=50)
            ]
        )

        # Evaluate
        print("\n=== Training Results ===")
        train_preds = self.model.predict(X_train)
        valid_preds = self.model.predict(X_valid)

        train_auc = roc_auc_score(y_train, train_preds)
        valid_auc = roc_auc_score(y_valid, valid_preds)

        train_ap = average_precision_score(y_train, train_preds)
        valid_ap = average_precision_score(y_valid, valid_preds)

        print(f"Train AUC: {train_auc:.4f} | Valid AUC: {valid_auc:.4f}")
        print(f"Train AP:  {train_ap:.4f} | Valid AP:  {valid_ap:.4f}")

        # Classification report (using 0.5 threshold)
        print("\n=== Classification Report (threshold=0.5) ===")
        valid_preds_binary = (valid_preds > 0.5).astype(int)
        print(classification_report(y_valid, valid_preds_binary, target_names=['No Churn', 'Churn']))

        # Save model
        self.save_model()

        return {
            'model': self.model,
            'train_auc': train_auc,
            'valid_auc': valid_auc,
            'train_ap': train_ap,
            'valid_ap': valid_ap,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'valid_preds': valid_preds
        }

    def save_model(self):
        """Save trained model"""
        model_path = self.training_config['model_output_path']

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save_model(model_path)
        print(f"\nModel saved to {model_path}")

    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = self.training_config['model_output_path']

        print(f"Loading model from {model_path}")
        self.model = lgb.Booster(model_file=model_path)

        return self.model

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        return self.model.predict(X)


def main():
    trainer = ChurnModelTrainer()

    # Train model
    results = trainer.train()

    print("\n=== Training Complete ===")
    print(f"Final Validation AUC: {results['valid_auc']:.4f}")
    print(f"Final Validation AP:  {results['valid_ap']:.4f}")


if __name__ == "__main__":
    main()
