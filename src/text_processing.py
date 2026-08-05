"""
Text Processing Module
=========================

Handles NLP preprocessing for the Student Doubt Triage pipeline (Pipeline 2):
  - Text cleaning (lowercase, punctuation, URLs, stopwords, lemmatization)
  - TF-IDF feature extraction
  - Text-based feature engineering
"""

import logging
import re
import string
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# NLTK resource downloads (done once)
_NLTK_INITIALIZED = False


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already available."""
    global _NLTK_INITIALIZED
    if _NLTK_INITIALIZED:
        return

    import nltk
    resources = ["punkt", "stopwords", "wordnet", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}" if resource in ["stopwords", "wordnet"]
                           else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    _NLTK_INITIALIZED = True


# =============================================================================
# Text Cleaning Pipeline
# =============================================================================


def clean_text(text: str) -> str:
    """
    Apply the full text cleaning pipeline to a single text string.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation
      4. Remove stopwords
      5. Lemmatization

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    _ensure_nltk_resources()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Step 3: Remove punctuation (keep spaces)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 4: Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Step 5: Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_text_column(
    df: pd.DataFrame,
    text_column: str,
    output_column: str = "cleaned_text",
) -> pd.DataFrame:
    """
    Apply text cleaning to an entire DataFrame column.

    Args:
        df: Input DataFrame.
        text_column: Name of the column containing raw text.
        output_column: Name for the cleaned text column.

    Returns:
        DataFrame with the new cleaned text column added.
    """
    logger.info(f"Cleaning text column '{text_column}' ({len(df)} rows)...")

    df_clean = df.copy()
    df_clean[output_column] = df_clean[text_column].astype(str).apply(clean_text)

    # Log stats
    avg_len_before = df_clean[text_column].str.len().mean()
    avg_len_after = df_clean[output_column].str.len().mean()
    empty_count = (df_clean[output_column].str.strip() == "").sum()

    logger.info(
        f"Text cleaning complete. "
        f"Avg length: {avg_len_before:.0f} → {avg_len_after:.0f} chars. "
        f"Empty after cleaning: {empty_count} rows."
    )

    return df_clean


# =============================================================================
# TF-IDF Feature Extraction
# =============================================================================


def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
) -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer.

    Args:
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Tuple of (min_n, max_n) for n-gram range.

    Returns:
        Configured (unfitted) TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b",
    )
    logger.info(
        f"TF-IDF vectorizer created: "
        f"max_features={max_features}, ngram_range={ngram_range}"
    )
    return vectorizer


def fit_transform_tfidf(
    train_texts: pd.Series,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on training texts and transform them.

    Args:
        train_texts: Series of cleaned text strings (training data).
        vectorizer: Pre-configured vectorizer. Creates default if None.

    Returns:
        Tuple of (TF-IDF matrix, fitted vectorizer).
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    X_tfidf = vectorizer.fit_transform(train_texts)

    logger.info(
        f"TF-IDF fitted on {len(train_texts)} documents. "
        f"Vocabulary size: {len(vectorizer.vocabulary_)}. "
        f"Matrix shape: {X_tfidf.shape}"
    )

    return X_tfidf, vectorizer


def transform_tfidf(
    texts: pd.Series,
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """
    Transform texts using a pre-fitted TF-IDF vectorizer.

    Args:
        texts: Series of cleaned text strings.
        vectorizer: Previously fitted TfidfVectorizer.

    Returns:
        TF-IDF feature matrix.
    """
    X_tfidf = vectorizer.transform(texts)
    logger.info(f"TF-IDF transformed {len(texts)} documents → shape {X_tfidf.shape}")
    return X_tfidf


def get_top_tfidf_features(
    vectorizer: TfidfVectorizer,
    class_label: str,
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    top_n: int = 15,
) -> List[Tuple[str, float]]:
    """
    Get the top TF-IDF features for a specific class.

    Useful for understanding what words characterize each question category.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        class_label: Target class label to analyze.
        tfidf_matrix: Full TF-IDF matrix.
        labels: Array of class labels corresponding to the matrix rows.
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, mean_tfidf_score) tuples, sorted descending.
    """
    feature_names = vectorizer.get_feature_names_out()
    mask = labels == class_label

    if mask.sum() == 0:
        logger.warning(f"No samples found for class '{class_label}'")
        return []

    # Mean TF-IDF score per feature for this class
    class_tfidf = tfidf_matrix[mask].mean(axis=0)
    if hasattr(class_tfidf, "A1"):
        class_tfidf = class_tfidf.A1  # Convert sparse matrix to array

    top_indices = np.argsort(class_tfidf)[-top_n:][::-1]
    top_features = [
        (feature_names[i], float(class_tfidf[i]))
        for i in top_indices
    ]

    return top_features
