# Copyright 2024 Translation Evaluation Framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for translation evaluation framework."""

import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data as list of dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def save_to_csv(
    data: List[Dict[str, Any]], 
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Save data to CSV file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output CSV file path
        fieldnames: Optional list of field names for CSV headers
    """
    if not data:
        logger.warning("No data to save to CSV")
        return
        
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully saved {len(data)} records to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV {file_path}: {e}")
        raise


def load_from_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Loaded data as list of dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
        logger.info(f"Successfully loaded {len(data)} records from CSV: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {e}")
        raise


def to_message(prompt: str, system_prompt: str = "") -> List[Dict[str, str]]:
    """
    Convert prompt and system prompt to message format.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": prompt.strip()})
    
    return messages


def create_evaluation_summary(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """
    Create evaluation summary and save to multiple formats.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
        output_path: Base output path (without extension)
    """
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary data
    summary = {
        'timestamp': timestamp,
        'total_evaluations': len(evaluation_results.get('individual_scores', [])),
        'language_pairs': list(evaluation_results.get('language_pairs', [])),
        'domains': list(evaluation_results.get('domains', [])),
        'average_scores': evaluation_results.get('average_scores', {}),
        'domain_scores': evaluation_results.get('domain_scores', {}),
    }
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    save_data(summary, json_path)
    
    # Save detailed results as CSV
    if 'individual_scores' in evaluation_results:
        csv_path = output_path.with_suffix('.csv')
        save_to_csv(evaluation_results['individual_scores'], csv_path)
    
    logger.info(f"Evaluation summary saved to {json_path} and {csv_path}")


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of scores.
    
    Args:
        scores: List of numerical scores
        
    Returns:
        Dictionary containing statistical measures
    """
    if not scores:
        return {}
    
    import statistics
    
    stats = {
        'mean': statistics.mean(scores),
        'median': statistics.median(scores),
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
        'min': min(scores),
        'max': max(scores),
        'count': len(scores)
    }
    
    # Calculate percentiles
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    
    stats['q1'] = sorted_scores[int(n * 0.25)]
    stats['q3'] = sorted_scores[int(n * 0.75)]
    stats['iqr'] = stats['q3'] - stats['q1']
    
    return stats


def format_score_table(scores_dict: Dict[str, List[float]]) -> str:
    """
    Format scores dictionary into a readable table string.
    
    Args:
        scores_dict: Dictionary mapping names to lists of scores
        
    Returns:
        Formatted table string
    """
    if not scores_dict:
        return "No scores available"
    
    # Calculate statistics for each category
    stats_data = []
    for name, scores in scores_dict.items():
        stats = calculate_statistics(scores)
        stats['name'] = name
        stats_data.append(stats)
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(stats_data)
    df = df.set_index('name')
    
    # Round numerical columns
    numerical_cols = ['mean', 'median', 'std_dev', 'min', 'max', 'q1', 'q3', 'iqr']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    return df.to_string()


def validate_translation_data(data: List[Dict[str, Any]]) -> bool:
    """
    Validate translation data format.
    
    Args:
        data: Translation data to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if not isinstance(data, list):
        logger.error("Translation data must be a list")
        return False
    
    required_fields = ['id', 'domain']
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error(f"Item {i} must be a dictionary")
            return False
        
        for field in required_fields:
            if field not in item:
                logger.error(f"Item {i} missing required field: {field}")
                return False
        
        # Check if at least one language field exists
        language_fields = [key for key in item.keys() if len(key) == 2 and key.isalpha()]
        if not language_fields:
            logger.error(f"Item {i} has no language fields")
            return False
    
    logger.info(f"Translation data validation passed for {len(data)} items")
    return True
