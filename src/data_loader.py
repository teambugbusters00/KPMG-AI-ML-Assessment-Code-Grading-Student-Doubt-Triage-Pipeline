"""
Data Loader Module
====================

Handles downloading, loading, and initial parsing of datasets for both pipelines:
  - NASA KC1 Software Defect Dataset (via OpenML / sklearn)
  - CS1QA Educational Programming Questions Dataset (via GitHub)
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff

from src.config import (
    CS1QA_DATA_SUBDIR,
    CS1QA_REPO_URL,
    CS1QA_TARGET_COLUMN,
    NASA_DATASET_ID,
    NASA_DATASET_NAME,
    NASA_TARGET_COLUMN,
    RAW_DATA_DIR,
    URGENCY_HIGH_KEYWORDS,
    URGENCY_MEDIUM_KEYWORDS,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Pipeline 1: NASA KC1 Dataset
# =============================================================================


def download_nasa_dataset(save_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Download the NASA KC1 software defect dataset from OpenML.

    Uses sklearn.datasets.fetch_openml to retrieve the KC1 dataset (ID: 1067).
    Saves a CSV copy to the raw data directory for reproducibility.

    Args:
        save_dir: Directory to save the CSV. Defaults to RAW_DATA_DIR.

    Returns:
        DataFrame containing the KC1 dataset with features and target.

    Raises:
        RuntimeError: If the dataset cannot be fetched.
    """
    save_dir = save_dir or RAW_DATA_DIR
    csv_path = save_dir / f"{NASA_DATASET_NAME}.csv"

    # Check for cached copy
    if csv_path.exists():
        logger.info(f"Loading cached NASA KC1 dataset from {csv_path}")
        return pd.read_csv(csv_path)

    logger.info(f"Downloading NASA KC1 dataset (OpenML ID: {NASA_DATASET_ID})...")
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(
            data_id=NASA_DATASET_ID,
            as_frame=True,
            parser="auto",
        )
        df = data.frame
        if df is None:
            # Fallback: construct from data and target
            df = data.data.copy()
            df[NASA_TARGET_COLUMN] = data.target

        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(
            f"NASA KC1 dataset downloaded: {df.shape[0]} rows, "
            f"{df.shape[1]} columns. Saved to {csv_path}"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to download NASA KC1 dataset: {e}")
        raise RuntimeError(f"Failed to download NASA KC1 dataset: {e}") from e


def load_nasa_dataset(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load and prepare the NASA KC1 dataset for modeling.

    Converts target column to binary integer (0/1) if needed.

    Args:
        filepath: Path to the CSV file. If None, downloads first.

    Returns:
        Tuple of (DataFrame, target_column_name).

    Raises:
        FileNotFoundError: If filepath is specified but doesn't exist.
    """
    if filepath and Path(filepath).exists():
        logger.info(f"Loading NASA KC1 from {filepath}")
        df = pd.read_csv(filepath)
    else:
        df = download_nasa_dataset()

    # Ensure target is binary integer
    target_col = NASA_TARGET_COLUMN
    if target_col in df.columns:
        if df[target_col].dtype == object or df[target_col].dtype.name == 'category':
            # Map 'true'/'false' or 'Y'/'N' or string numbers to 1/0
            def _map_val(val):
                val_str = str(val).strip().lower()
                if val_str in ["true", "y", "1", "1.0", "t"]:
                    return 1
                if val_str in ["false", "n", "0", "0.0", "f"]:
                    return 0
                try:
                    return int(float(val_str))
                except (ValueError, TypeError):
                    return 0

            df[target_col] = df[target_col].apply(_map_val)
        df[target_col] = df[target_col].astype(int)
    else:
        # Try common alternative column names
        alt_names = ["Defective", "defective", "class", "bug", "label"]
        for alt in alt_names:
            if alt in df.columns:
                df.rename(columns={alt: target_col}, inplace=True)
                logger.info(f"Renamed '{alt}' to '{target_col}'")
                break

    logger.info(
        f"NASA KC1 loaded: {df.shape[0]} samples, "
        f"{df.shape[1]} features. "
        f"Target distribution: {df[target_col].value_counts().to_dict()}"
    )
    return df, target_col


# =============================================================================
# Pipeline 2: CS1QA Dataset
# =============================================================================


def download_cs1qa_dataset(save_dir: Optional[Path] = None) -> Path:
    """
    Clone or locate the CS1QA dataset repository.

    Args:
        save_dir: Directory to clone the repo into. Defaults to RAW_DATA_DIR.

    Returns:
        Path to the data subdirectory within the cloned repo.

    Raises:
        RuntimeError: If git clone fails and no cached copy exists.
    """
    save_dir = save_dir or RAW_DATA_DIR
    repo_dir = save_dir / "CS1QA"
    data_dir = repo_dir / CS1QA_DATA_SUBDIR

    if data_dir.exists():
        logger.info(f"CS1QA dataset already available at {data_dir}")
        return data_dir

    logger.info(f"Cloning CS1QA repository from {CS1QA_REPO_URL}...")
    try:
        subprocess.run(
            ["git", "clone", CS1QA_REPO_URL, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"CS1QA repository cloned to {repo_dir}")
        return data_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        raise RuntimeError(f"Failed to clone CS1QA: {e.stderr}") from e
    except FileNotFoundError:
        logger.error("git is not installed or not in PATH")
        raise RuntimeError("git is required to download CS1QA dataset")


def _parse_cs1qa_json_files(data_dir: Path) -> pd.DataFrame:
    """
    Parse CS1QA annotated JSON data files into a DataFrame.

    Walks through the data directory, finds JSON files, and extracts
    question text, question type, code context, and answers.

    Args:
        data_dir: Path to the CS1QA data directory.

    Returns:
        DataFrame with columns: question, question_type, code, answer.
    """
    records = []

    # Walk through all JSON files in the data directory
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            filepath = Path(root) / filename

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        record = _extract_cs1qa_record(item)
                        if record:
                            records.append(record)
                elif isinstance(data, dict):
                    # Might be a dict of lists or a single record
                    if "data" in data:
                        for item in data["data"]:
                            record = _extract_cs1qa_record(item)
                            if record:
                                records.append(record)
                    else:
                        record = _extract_cs1qa_record(data)
                        if record:
                            records.append(record)

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping {filepath}: {e}")

    if not records:
        logger.warning("No records parsed from CS1QA JSON files")

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} records from CS1QA data directory")
    return df


def _extract_cs1qa_record(item: dict) -> Optional[dict]:
    """
    Extract a single question-answer record from a CS1QA JSON item.

    Args:
        item: Dictionary from the CS1QA JSON data.

    Returns:
        Dictionary with question, question_type, code, answer fields,
        or None if essential fields are missing.
    """
    question = item.get("question", item.get("q", item.get("text", "")))
    q_type = item.get("question_type", item.get("type", item.get("category", "")))
    code = item.get("code", item.get("student_code", ""))
    answer = item.get("answer", item.get("a", ""))

    if not question or not q_type:
        return None

    return {
        "question": str(question).strip(),
        CS1QA_TARGET_COLUMN: str(q_type).strip(),
        "code": str(code).strip() if code else "",
        "answer": str(answer).strip() if answer else "",
    }


def derive_urgency(df: pd.DataFrame, text_column: str = "question") -> pd.Series:
    """
    Derive urgency labels heuristically from question text.

    Since CS1QA doesn't have explicit urgency labels, we infer them from:
      - Presence of error/crash keywords → HIGH
      - Presence of confusion keywords → MEDIUM
      - Otherwise → LOW

    This is a documented heuristic. See report limitations section.

    Args:
        df: DataFrame containing the text column.
        text_column: Name of the column with question text.

    Returns:
        Series with urgency labels ('HIGH', 'MEDIUM', 'LOW').
    """
    def _classify_urgency(text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in URGENCY_HIGH_KEYWORDS):
            return "HIGH"
        if any(kw in text_lower for kw in URGENCY_MEDIUM_KEYWORDS):
            return "MEDIUM"
        return "LOW"

    urgency = df[text_column].apply(_classify_urgency)
    logger.info(f"Derived urgency distribution: {urgency.value_counts().to_dict()}")
    return urgency


def load_cs1qa_dataset(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load, parse, and prepare the CS1QA dataset for the doubt triage pipeline.

    Downloads if not available, parses JSON files, derives urgency labels,
    and returns a clean DataFrame ready for text preprocessing.

    Args:
        data_dir: Path to the CS1QA data directory. Downloads if None.

    Returns:
        Tuple of (DataFrame, target_column_name).

    Raises:
        ValueError: If no valid records are found after parsing.
    """
    if data_dir is None:
        data_dir = download_cs1qa_dataset()
    elif not Path(data_dir).exists():
        data_dir = download_cs1qa_dataset()

    df = _parse_cs1qa_json_files(data_dir)

    if df.empty:
        # Fallback: create a synthetic educational programming Q&A dataset
        # This ensures the pipeline can still demonstrate functionality
        logger.warning(
            "CS1QA JSON parsing yielded no records. "
            "Creating synthetic educational programming dataset for demonstration."
        )
        df = _create_fallback_dataset()

    # Derive urgency
    df["urgency"] = derive_urgency(df)

    # Clean target labels
    target_col = CS1QA_TARGET_COLUMN
    df[target_col] = df[target_col].str.strip().str.lower()

    # Remove rows with missing target
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] != ""]
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"Dropped {n_before - n_after} rows with missing target")

    logger.info(
        f"CS1QA dataset ready: {len(df)} samples, "
        f"{df[target_col].nunique()} categories. "
        f"Target distribution: {df[target_col].value_counts().to_dict()}"
    )
    return df, target_col


def _create_fallback_dataset() -> pd.DataFrame:
    """
    Create a synthetic educational programming Q&A dataset.

    This is used ONLY when CS1QA JSON parsing fails (e.g., format changes).
    The synthetic data mirrors CS1QA's 9 question type categories.

    Returns:
        DataFrame with question, question_type, code, answer columns.
    """
    np.random.seed(42)

    categories = [
        "syntax", "logic", "runtime", "conceptual", "output",
        "debugging", "implementation", "design", "other"
    ]

    templates = {
        "syntax": [
            "Why am I getting a syntax error on line {n}?",
            "What is wrong with my indentation in this code?",
            "I get an unexpected token error, what does it mean?",
            "How do I fix this missing colon error?",
            "Why does Python say invalid syntax for my print statement?",
        ],
        "logic": [
            "My loop runs forever, how do I fix the logic?",
            "The output is wrong even though there's no error. Why?",
            "My if-else condition doesn't work as expected",
            "Why does my function return None instead of the result?",
            "The sorting algorithm gives incorrect output for this case",
        ],
        "runtime": [
            "I'm getting a ZeroDivisionError, how do I fix it?",
            "My program crashes with IndexError at line {n}",
            "TypeError: unsupported operand type, what does this mean?",
            "I get a RecursionError, is my base case wrong?",
            "FileNotFoundError when trying to read the data file",
        ],
        "conceptual": [
            "What is the difference between a list and a tuple?",
            "Can you explain how recursion works?",
            "What does object-oriented programming mean?",
            "How are dictionaries different from lists?",
            "What is the time complexity of this algorithm?",
        ],
        "output": [
            "What will this code print?",
            "I expected the output to be {n} but got something else",
            "Why is my output missing the last element?",
            "The print statement shows the wrong variable value",
            "How do I format the output to show 2 decimal places?",
        ],
        "debugging": [
            "Help me find the bug in this function",
            "Why does this code fail only for negative numbers?",
            "I can't figure out why this test case fails",
            "My code works for small inputs but crashes on large ones",
            "There's an error but I don't understand the traceback",
        ],
        "implementation": [
            "How do I implement a binary search in Python?",
            "What's the best way to read a CSV file?",
            "How can I sort a dictionary by its values?",
            "How do I create a class with inheritance?",
            "What's the correct way to handle exceptions?",
        ],
        "design": [
            "Should I use a class or a function for this?",
            "Is it better to use a list or a dictionary here?",
            "How should I structure my code for this assignment?",
            "What design pattern should I use for this problem?",
            "Should I break this into multiple functions?",
        ],
        "other": [
            "How do I submit my assignment?",
            "What IDE should I use for Python?",
            "Is there a deadline extension for this homework?",
            "Where can I find the lecture notes?",
            "Can I use external libraries for this project?",
        ],
    }

    records = []
    for category in categories:
        n_samples = np.random.randint(80, 150)
        for _ in range(n_samples):
            template = np.random.choice(templates[category])
            question = template.format(n=np.random.randint(1, 50))
            records.append({
                "question": question,
                CS1QA_TARGET_COLUMN: category,
                "code": f"# sample code for {category}",
                "answer": f"Answer related to {category}",
            })

    df = pd.DataFrame(records)
    logger.info(f"Created fallback dataset with {len(df)} samples")
    return df
