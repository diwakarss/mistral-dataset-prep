#!/usr/bin/env python
"""
Converts datasets to the specific format required for Mistral-7B fine-tuning.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tqdm import tqdm

from src.data_processor import TextCleaner, DatasetFormatter


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
    Load data from various formats.
    
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


def convert_to_mistral_format(
    data: List[Dict[str, Any]], 
    instruction_key: str,
    output_key: str,
    input_key: Optional[str] = None,
    cleaner: Optional[TextCleaner] = None
) -> List[Dict[str, str]]:
    """
    Convert data to Mistral-7B fine-tuning format.
    
    Args:
        data: List of data examples
        instruction_key: Key for the instruction field
        output_key: Key for the output field
        input_key: Optional key for the input context field
        cleaner: Optional TextCleaner instance
        
    Returns:
        Data formatted for Mistral-7B fine-tuning
    """
    logger = logging.getLogger(__name__)
    formatter = DatasetFormatter(cleaner=cleaner)
    
    logger.info("Converting data to Mistral format")
    formatted_data = []
    
    for example in tqdm(data, desc="Converting data"):
        try:
            instruction = example.get(instruction_key, "")
            output = example.get(output_key, "")
            input_text = example.get(input_key, "") if input_key else None
            
            if not instruction or not output:
                continue
                
            formatted_example = formatter.format_example(
                instruction=instruction,
                output=output,
                input_text=input_text
            )
            
            formatted_data.append(formatted_example)
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            continue
    
    logger.info(f"Converted {len(formatted_data)} examples")
    return formatted_data


def save_output(data: List[Dict[str, str]], output_path: str):
    """
    Save formatted data to a file.
    
    Args:
        data: Formatted data
        output_path: Path to save the output file
    """
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()
    
    logger.info(f"Saving {len(data)} examples to {output_path}")
    
    if ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif ext == '.jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    else:
        logger.warning(f"Unsupported output format {ext}, defaulting to .jsonl")
        output_path = os.path.splitext(output_path)[0] + '.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Data saved successfully to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert datasets to Mistral-7B fine-tuning format")
    parser.add_argument("--input", required=True, help="Path to the input data file")
    parser.add_argument("--output", required=True, help="Path to save the formatted output file")
    parser.add_argument("--instruction-key", default="instruction", help="Key for instruction field")
    parser.add_argument("--output-key", default="output", help="Key for output field")
    parser.add_argument("--input-key", default=None, help="Key for input context field (optional)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Load data
    data = load_data(args.input)
    
    # Setup text cleaner
    cleaner = TextCleaner(config={
        "lowercase": False,
        "remove_urls": True
    })
    
    # Convert to Mistral format
    formatted_data = convert_to_mistral_format(
        data=data,
        instruction_key=args.instruction_key,
        output_key=args.output_key,
        input_key=args.input_key,
        cleaner=cleaner
    )
    
    # Save output
    save_output(formatted_data, args.output)


if __name__ == "__main__":
    main() 