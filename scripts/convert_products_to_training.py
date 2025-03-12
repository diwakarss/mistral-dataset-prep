#!/usr/bin/env python
"""
Script to convert product CSV data into Mistral-7B fine-tuning format with specific instruction types.
"""
import os
import sys
import json
import argparse
import logging
import csv
import random
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import TextCleaner


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


def load_csv_data(input_path: str) -> List[Dict[str, str]]:
    """
    Load product data from CSV file.
    
    Args:
        input_path: Path to the CSV file
        
    Returns:
        List of product dictionaries
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading product data from {input_path}")
    
    products = []
    with open(input_path, 'r', encoding='utf-8') as f:
        # Use csv.DictReader to handle quotes and commas within fields
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    
    logger.info(f"Loaded {len(products)} products")
    return products


def clean_html(text: str) -> str:
    """
    Basic HTML tag removal. For more sophisticated cleaning, use a dedicated HTML parser.
    
    Args:
        text: HTML text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    import re
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Replace &nbsp; with space
    text = text.replace("&nbsp;", " ")
    
    return text.strip()


def create_training_examples(products: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Create three types of training examples for each product:
    1. Given Variant SKU and Supplier, generate Title
    2. Given Title, Variant SKU, and Supplier, generate Features
    3. Given Title, Variant SKU, and Supplier, generate Categories
    
    Args:
        products: List of product dictionaries
        
    Returns:
        List of training examples in Mistral format
    """
    logger = logging.getLogger(__name__)
    
    examples = []
    skipped_count = 0
    
    for product in products:
        # Skip products with missing essential fields
        if not all(field in product and product[field] for field in ['Title', 'Variant SKU', 'Supplier']):
            skipped_count += 1
            continue
            
        # Clean fields
        title = product['Title'].strip()
        features = clean_html(product.get('Features', ''))
        supplier = product.get('Supplier', '').strip()
        categories = product.get('Categories', '').strip()
        variant_sku = product.get('Variant SKU', '').strip()
        
        # Skip if after cleaning, we don't have valid data
        if not title or not variant_sku or not supplier:
            skipped_count += 1
            continue
        
        # Example 1: Generate Title from Variant SKU and Supplier
        if title:
            examples.append({
                "instruction": "Given the variant SKU and supplier information, generate an appropriate product title.",
                "input": f"Variant SKU: {variant_sku}\nSupplier: {supplier}",
                "output": title
            })
        
        # Example 2: Generate Features from Title, Variant SKU, and Supplier
        if features:
            examples.append({
                "instruction": "Create descriptive product features based on the product title, variant SKU, and supplier.",
                "input": f"Title: {title}\nVariant SKU: {variant_sku}\nSupplier: {supplier}",
                "output": features
            })
        
        # Example 3: Generate Categories from Title, Variant SKU, and Supplier
        if categories:
            examples.append({
                "instruction": "Determine appropriate product categories based on the product title, variant SKU, and supplier.",
                "input": f"Title: {title}\nVariant SKU: {variant_sku}\nSupplier: {supplier}",
                "output": categories
            })
    
    logger.info(f"Created {len(examples)} training examples")
    logger.info(f"Skipped {skipped_count} products due to missing required fields")
    
    return examples


def save_examples(examples: List[Dict[str, str]], output_path: str):
    """
    Save training examples to JSONL file.
    
    Args:
        examples: List of training examples
        output_path: Path to save the output file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSONL (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(examples)} examples to {output_path}")


def create_train_val_split(examples: List[Dict[str, str]], val_ratio: float = 0.1, seed: int = 42):
    """
    Split the examples into training and validation sets.
    
    Args:
        examples: List of training examples
        val_ratio: Ratio of validation examples (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train' and 'val' splits
    """
    random.seed(seed)
    random.shuffle(examples)
    
    val_size = int(len(examples) * val_ratio)
    
    return {
        'train': examples[val_size:],
        'val': examples[:val_size]
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert product CSV to Mistral training format")
    parser.add_argument("--input", required=True, help="Path to the product CSV file")
    parser.add_argument("--output", required=True, help="Path to save the formatted examples")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load product data
    products = load_csv_data(args.input)
    
    # Create training examples
    examples = create_training_examples(products)
    
    # Create train/val split
    splits = create_train_val_split(examples, args.val_ratio, args.seed)
    
    # Save splits
    output_base, ext = os.path.splitext(args.output)
    if not ext:
        ext = '.jsonl'
        
    for split_name, split_data in splits.items():
        split_path = f"{output_base}_{split_name}{ext}"
        save_examples(split_data, split_path)
        
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main() 