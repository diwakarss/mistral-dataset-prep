#!/usr/bin/env python
"""
Script to analyze a dataset and generate statistics and visualizations.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.data_validator import DatasetAnalyzer


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
        
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a file.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        List of data examples
    """
    logger = logging.getLogger(__name__)
    
    # Determine file format from extension
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    logger.info(f"Loading data from {input_path}")
    
    if ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif ext == '.jsonl':
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif ext in ['.csv', '.tsv']:
        separator = ',' if ext == '.csv' else '\t'
        df = pd.read_csv(input_path, sep=separator)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    logger.info(f"Loaded {len(data)} examples")
    return data


# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def analyze_dataset(data: List[Dict[str, Any]], output_dir: str):
    """
    Analyze the dataset and generate statistics and visualizations.
    
    Args:
        data: List of data examples
        output_dir: Directory to save the output files
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Basic statistics
    stats = {
        "total_examples": len(data),
        "fields_present": {col: int(df[col].notna().sum()) for col in df.columns},
        "examples_with_input": int(df["input"].notna().sum()) if "input" in df.columns else 0
    }
    
    # Calculate length statistics
    if "instruction" in df.columns:
        df["instruction_length"] = df["instruction"].fillna("").apply(len)
        stats["instruction_length"] = {
            "min": int(df["instruction_length"].min()),
            "max": int(df["instruction_length"].max()),
            "mean": float(df["instruction_length"].mean()),
            "median": float(df["instruction_length"].median()),
            "std": float(df["instruction_length"].std())
        }
    
    if "output" in df.columns:
        df["output_length"] = df["output"].fillna("").apply(len)
        stats["output_length"] = {
            "min": int(df["output_length"].min()),
            "max": int(df["output_length"].max()),
            "mean": float(df["output_length"].mean()),
            "median": float(df["output_length"].median()),
            "std": float(df["output_length"].std())
        }
    
    if "input" in df.columns:
        df["input_length"] = df["input"].fillna("").apply(len)
        stats["input_length"] = {
            "min": int(df["input_length"].min()),
            "max": int(df["input_length"].max()),
            "mean": float(df["input_length"].mean()),
            "median": float(df["input_length"].median()),
            "std": float(df["input_length"].std())
        }
    
    # Save statistics to a JSON file
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved basic statistics to {stats_path}")
    
    # Generate visualizations
    if "instruction_length" in df.columns and "output_length" in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Instruction length histogram
        plt.subplot(2, 2, 1)
        sns.histplot(df["instruction_length"], kde=True)
        plt.title("Instruction Length Distribution")
        plt.xlabel("Length (characters)")
        
        # Output length histogram
        plt.subplot(2, 2, 2)
        sns.histplot(df["output_length"], kde=True)
        plt.title("Output Length Distribution")
        plt.xlabel("Length (characters)")
        
        # Input length histogram (if applicable)
        if "input_length" in df.columns:
            plt.subplot(2, 2, 3)
            # Filter out examples without input
            input_lengths = df[df["input_length"] > 0]["input_length"]
            if len(input_lengths) > 0:
                sns.histplot(input_lengths, kde=True)
                plt.title("Input Length Distribution (when present)")
                plt.xlabel("Length (characters)")
        
        # Instruction vs Output length scatter plot
        plt.subplot(2, 2, 4)
        sns.scatterplot(x="instruction_length", y="output_length", data=df, alpha=0.5)
        plt.title("Instruction vs Output Length")
        plt.xlabel("Instruction Length (characters)")
        plt.ylabel("Output Length (characters)")
        
        plt.tight_layout()
        
        # Save the figure
        viz_path = os.path.join(output_dir, "length_distributions.png")
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved visualizations to {viz_path}")
    
    # Use the DatasetAnalyzer for more detailed analysis
    analyzer = DatasetAnalyzer()
    detailed_analysis = analyzer.analyze_dataset(data)
    
    # Save detailed analysis
    detailed_path = os.path.join(output_dir, "detailed_analysis.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_analysis = convert_to_serializable(detailed_analysis)
        json.dump(serializable_analysis, f, indent=2)
    
    logger.info(f"Saved detailed analysis to {detailed_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze a dataset and generate statistics")
    parser.add_argument("--input", required=True, help="Path to the input data file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Load data
    data = load_data(args.input)
    
    # Analyze dataset
    analyze_dataset(data, args.output_dir)


if __name__ == "__main__":
    main() 