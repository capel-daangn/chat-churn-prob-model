"""
Feature Engineering with DistilBERT Embeddings
Extracts text embeddings and combines with metadata features
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import yaml
from .utils import load_config, extract_meta_features


class FeatureExtractor:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.model_name = self.config['model']['name']
        self.max_length = self.config['model']['max_length']
        self.embedding_dim = self.config['model']['embedding_dim']

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def get_embedding(self, text):
        """
        Extract DistilBERT embedding for a single text
        Returns 768-dimensional vector
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embedding

    def extract_text_embeddings(self, df):
        """
        Extract embeddings for all messages in dataframe
        Returns numpy array of shape (n_samples, 768)
        """
        print(f"Extracting text embeddings for {len(df)} messages...")

        embeddings = []
        for text in tqdm(df['content'].values, desc="Processing"):
            emb = self.get_embedding(text)
            embeddings.append(emb)

        return np.vstack(embeddings)

    def extract_all_features(self, data_path=None):
        """
        Extract all features: text embeddings + metadata
        Returns feature matrix X and labels y
        """
        if data_path is None:
            data_path = self.config['paths']['raw_data']

        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        print(f"Total messages: {len(df)}")
        print(f"Churn ratio: {df['label'].mean()*100:.1f}%")

        # Extract metadata features
        df = extract_meta_features(df)

        # Extract text embeddings
        text_embeddings = self.extract_text_embeddings(df)

        # Extract metadata features
        meta_feature_names = self.config['features']['meta_features']
        meta_features = df[meta_feature_names].values

        # Combine features
        X = np.hstack([text_embeddings, meta_features])

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"  - Text embeddings: {text_embeddings.shape[1]} dims")
        print(f"  - Meta features: {meta_features.shape[1]} dims")

        # Labels
        y = df['label'].values

        # Save processed features
        output_path = self.config['paths']['processed_data']
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            feature_names=np.array(meta_feature_names),
            message_ids=df['message_id'].values
        )

        print(f"\nFeatures saved to {output_path}")

        return X, y, df


def main():
    extractor = FeatureExtractor()
    X, y, df = extractor.extract_all_features()

    # Print statistics
    print("\n=== Feature Statistics ===")
    print(f"Total samples: {len(X)}")
    print(f"Positive class (churn): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Negative class (no churn): {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

    # Print sample predictions
    print("\n=== Sample Messages ===")
    print(df[['content', 'sentiment_score', 'exit_intent_score', 'label']].head(10))


if __name__ == "__main__":
    main()
