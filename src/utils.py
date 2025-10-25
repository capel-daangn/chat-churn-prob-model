"""
Utility functions for the churn prediction model
"""

import yaml
import re


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_sentiment_score(text):
    """
    Simple rule-based sentiment analysis for Korean text
    Returns score between -1 (negative) and +1 (positive)
    """
    positive_words = [
        '좋', '감사', '괜찮', '네', '가능', '구매', '알겠습니다',
        '완벽', '만족', '빠른', '친절', '최고', '훌륭'
    ]

    negative_words = [
        '비싸', '아니', '어렵', '안', '죄송', '고민', '글쎄',
        '...', '음', '불가능', '거절', '실망', '별로'
    ]

    text_lower = text.lower()

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    score = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, score))


def calculate_exit_intent_score(text):
    """
    Calculate exit intent based on conversation-ending signals
    Returns score between 0 (no exit intent) and 1 (strong exit intent)
    """
    exit_keywords = [
        '다시 생각', '고민', '비교', '나중에', '연락드릴게',
        '그냥', '괜찮습니다', '됐습니다', '...', '음',
        '다른', '아 그렇군요', '알겠습니다만'
    ]

    text_lower = text.lower()

    # Check for exit keywords
    exit_count = sum(1 for keyword in exit_keywords if keyword in text_lower)

    # Very short messages can indicate disengagement
    if len(text) <= 3:
        exit_count += 1

    # Calculate score (cap at 1.0)
    score = min(1.0, exit_count * 0.3)

    return score


def extract_meta_features(df):
    """
    Extract metadata features from the dataframe
    Adds sentiment_score and exit_intent_score columns
    """
    print("Extracting sentiment scores...")
    df['sentiment_score'] = df['content'].apply(calculate_sentiment_score)

    print("Extracting exit intent scores...")
    df['exit_intent_score'] = df['content'].apply(calculate_exit_intent_score)

    return df
