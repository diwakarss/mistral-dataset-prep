#!/usr/bin/env python
"""
Main script for preparing datasets for Mistral-7B fine-tuning.
"""
import os
import sys
import json
import argparse
import logging
import yaml
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tqdm import tqdm

from src.data_processor import TextCleaner, DatasetFormatter, DatasetSplitter
from src.data_validator import DataValidator, DatasetAnalyzer


def setup_logging(log_level="INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
        
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(input_path, file_format):
    """
    Load data from file.
    
    Args:
        input_path: Path to the input file
        file_format: Format of the input file (json, jsonl, csv)
        
    Returns:
        List of data examples
    """
    if file_format == 'json':
        with open(input_path, 'r') as f:
            return json.load(f)
    elif file_format == 'jsonl':
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif file_format == 'csv':
        df = pd.read_csv(input_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def main():
    """
    Main function for dataset preparation.
    """
    parser = argparse.ArgumentParser(description="Prepare datasets for Mistral-7B fine-tuning")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--format", default="jsonl", choices=["json", "jsonl", "csv"],
                        help="Format of the input file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting dataset preparation with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    raw_data = load_data(args.input, args.format)
    logger.info(f"Loaded {len(raw_data)} examples")
    
    # Create processors
    text_cleaner = TextCleaner(config.get("text_cleaning", {}))
    dataset_formatter = DatasetFormatter(text_cleaner)
    data_validator = DataValidator(config.get("validation", {}))
    dataset_splitter = DatasetSplitter(
        val_size=config.get("val_size", 0.1),
        seed=config.get("seed", 42)
    )
    dataset_analyzer = DatasetAnalyzer()
    
    # Format data
    logger.info("Formatting dataset")
    formatted_data = dataset_formatter.format_dataset(
        data=raw_data,
        instruction_key=config["keys"]["instruction"],
        output_key=config["keys"]["output"],
        input_key=config["keys"].get("input")
    )
    logger.info(f"Formatted {len(formatted_data)} examples")
    
    # Validate data
    logger.info("Validating dataset")
    valid_data = data_validator.filter_dataset(formatted_data)
    logger.info(f"Validation complete: {len(valid_data)} valid examples")
    
    # Analyze dataset
    logger.info("Analyzing dataset")
    analysis = dataset_analyzer.analyze_dataset(valid_data)
    
    # Save analysis
    os.makedirs(args.output, exist_ok=True)
    analysis_path = os.path.join(args.output, "dataset_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Saved dataset analysis to {analysis_path}")
    
    # Split dataset
    logger.info("Splitting dataset into train/validation sets")
    splits = dataset_splitter.train_val_split(valid_data)
    
    # Save datasets
    for split_name, split_data in splits.items():
        output_path = os.path.join(args.output, f"{split_name}.jsonl")
        dataset_formatter.save_jsonl(split_data, output_path)
        logger.info(f"Saved {split_name} dataset with {len(split_data)} examples to {output_path}")
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main() 