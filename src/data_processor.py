"""
Data processing utilities for Mistral-7B fine-tuning dataset preparation.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
from tqdm import tqdm


class TextCleaner:
    """Cleans and normalizes text data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config: Configuration options for text cleaning
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Additional cleaning based on config
        if self.config.get("lowercase", False):
            text = text.lower()
            
        if self.config.get("remove_urls", True):
            # Simple URL removal - can be improved with regex
            text = text.replace("http://", " ")
            text = text.replace("https://", " ")
            text = " ".join(text.split())
            
        if self.config.get("max_length"):
            max_length = self.config["max_length"]
            if len(text) > max_length:
                text = text[:max_length]
                
        return text


class DatasetFormatter:
    """Formats data into the structure required for Mistral-7B fine-tuning."""
    
    def __init__(self, cleaner: Optional[TextCleaner] = None):
        """
        Initialize the dataset formatter.
        
        Args:
            cleaner: Text cleaner for preprocessing
        """
        self.cleaner = cleaner or TextCleaner()
        self.logger = logging.getLogger(__name__)
        
    def format_example(self, 
                      instruction: str, 
                      output: str, 
                      input_text: Optional[str] = None) -> Dict[str, str]:
        """
        Format a single example into the required structure.
        
        Args:
            instruction: The instruction for the model
            output: The expected output from the model
            input_text: Optional context/input for the instruction
            
        Returns:
            Formatted example as a dictionary
        """
        # Clean texts
        instruction = self.cleaner.clean_text(instruction)
        output = self.cleaner.clean_text(output)
        
        example = {
            "instruction": instruction,
            "output": output
        }
        
        if input_text:
            example["input"] = self.cleaner.clean_text(input_text)
            
        return example
    
    def format_dataset(self, 
                      data: List[Dict[str, str]], 
                      instruction_key: str,
                      output_key: str,
                      input_key: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format a list of examples into the required structure.
        
        Args:
            data: List of data examples as dictionaries
            instruction_key: Key for the instruction field in input data
            output_key: Key for the output field in input data
            input_key: Optional key for the input field in input data
            
        Returns:
            List of formatted examples
        """
        formatted_data = []
        
        for item in tqdm(data, desc="Formatting dataset"):
            try:
                instruction = item.get(instruction_key, "")
                output = item.get(output_key, "")
                
                if not instruction or not output:
                    self.logger.warning(f"Missing required fields: {item}")
                    continue
                    
                input_text = item.get(input_key, "") if input_key else None
                
                formatted_example = self.format_example(
                    instruction=instruction,
                    output=output,
                    input_text=input_text
                )
                
                formatted_data.append(formatted_example)
                
            except Exception as e:
                self.logger.error(f"Error formatting example: {e}")
                continue
                
        return formatted_data
    
    def save_jsonl(self, data: List[Dict[str, str]], output_path: str) -> None:
        """
        Save formatted data as JSONL file.
        
        Args:
            data: Formatted dataset
            output_path: Path to save the JSONL file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
        self.logger.info(f"Saved {len(data)} examples to {output_path}")


class DatasetSplitter:
    """Splits datasets into train/validation sets."""
    
    def __init__(self, val_size: float = 0.1, seed: int = 42):
        """
        Initialize the dataset splitter.
        
        Args:
            val_size: Proportion of data to use for validation
            seed: Random seed for reproducibility
        """
        self.val_size = val_size
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
    def train_val_split(self, data: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Split data into training and validation sets.
        
        Args:
            data: Dataset to split
            
        Returns:
            Dictionary with 'train' and 'val' keys containing the split datasets
        """
        data_size = len(data)
        indices = np.random.permutation(data_size)
        
        val_samples = max(1, int(data_size * self.val_size))
        val_idx, train_idx = indices[:val_samples], indices[val_samples:]
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        self.logger.info(f"Split dataset: {len(train_data)} training examples, {len(val_data)} validation examples")
        
        return {
            "train": train_data,
            "val": val_data
        } 