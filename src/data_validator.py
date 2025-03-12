"""
Data validation utilities for Mistral-7B fine-tuning dataset preparation.
"""
import re
import logging
from typing import Dict, List, Any, Optional, Callable

import numpy as np
from tqdm import tqdm


class DataValidator:
    """Validates and filters dataset examples based on various criteria."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration options for validation
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default validation config
        self.min_instruction_length = self.config.get("min_instruction_length", 3)
        self.max_instruction_length = self.config.get("max_instruction_length", 2048)
        self.min_output_length = self.config.get("min_output_length", 1)
        self.max_output_length = self.config.get("max_output_length", 4096)
        
    def is_valid_example(self, example: Dict[str, str]) -> bool:
        """
        Check if an example meets all validation criteria.
        
        Args:
            example: Dataset example to validate
            
        Returns:
            True if the example is valid, False otherwise
        """
        # Check required fields
        if "instruction" not in example or "output" not in example:
            return False
            
        instruction = example["instruction"]
        output = example["output"]
        
        # Length checks
        if not instruction or len(instruction) < self.min_instruction_length:
            return False
            
        if not output or len(output) < self.min_output_length:
            return False
            
        if len(instruction) > self.max_instruction_length:
            return False
            
        if len(output) > self.max_output_length:
            return False
            
        # Check for minimum content quality (customize as needed)
        if self.config.get("check_instruction_quality", True):
            if not self._has_meaningful_content(instruction):
                return False
                
        # Add custom validation rules here
        if self.config.get("custom_validators"):
            for validator in self.config["custom_validators"]:
                if not validator(example):
                    return False
                    
        return True
    
    def _has_meaningful_content(self, text: str) -> bool:
        """
        Check if text has meaningful content.
        
        Args:
            text: Text to check
            
        Returns:
            True if the text has meaningful content, False otherwise
        """
        # Simple heuristic - can be improved
        words = re.findall(r'\w+', text.lower())
        
        # Check if it has a minimum number of words
        if len(words) < 2:
            return False
            
        # Check if it's not just repeated characters
        if len(set(text)) < 3:
            return False
            
        return True
    
    def filter_dataset(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter a dataset to include only valid examples.
        
        Args:
            data: Dataset to filter
            
        Returns:
            Filtered dataset
        """
        valid_data = []
        invalid_count = 0
        
        for example in tqdm(data, desc="Validating examples"):
            if self.is_valid_example(example):
                valid_data.append(example)
            else:
                invalid_count += 1
                
        self.logger.info(f"Filtered dataset: {len(valid_data)} valid examples, {invalid_count} invalid examples")
        
        return valid_data


class DatasetAnalyzer:
    """Analyzes dataset characteristics and provides statistics."""
    
    def __init__(self):
        """Initialize the dataset analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def get_length_stats(self, data: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Get length statistics for each field in the dataset.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            Dictionary with statistics for each field
        """
        stats = {}
        
        if not data:
            return stats
            
        # Determine fields to analyze
        fields = set()
        for example in data:
            fields.update(example.keys())
            
        # Collect lengths for each field
        field_lengths = {field: [] for field in fields}
        
        for example in data:
            for field in fields:
                if field in example and isinstance(example[field], str):
                    field_lengths[field].append(len(example[field]))
        
        # Calculate statistics
        for field, lengths in field_lengths.items():
            if lengths:
                stats[field] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": np.mean(lengths),
                    "median": np.median(lengths),
                    "std": np.std(lengths),
                    "count": len(lengths)
                }
            
        return stats
    
    def get_vocabulary_stats(self, data: List[Dict[str, str]], field: str = "instruction") -> Dict[str, Any]:
        """
        Get vocabulary statistics for a specific field in the dataset.
        
        Args:
            data: Dataset to analyze
            field: Field to analyze
            
        Returns:
            Dictionary with vocabulary statistics
        """
        if not data:
            return {}
            
        # Extract text from the specified field
        texts = []
        for example in data:
            if field in example and isinstance(example[field], str):
                texts.append(example[field])
                
        if not texts:
            return {}
            
        # Tokenize and analyze
        all_words = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.extend(words)
            
        # Calculate statistics
        vocab = set(all_words)
        word_freq = {}
        
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        return {
            "vocabulary_size": len(vocab),
            "total_words": len(all_words),
            "unique_ratio": len(vocab) / max(1, len(all_words)),
            "top_words": sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        
    def analyze_dataset(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the dataset.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "dataset_size": len(data),
            "length_stats": self.get_length_stats(data),
        }
        
        # Analyze vocabulary for different fields
        for field in ["instruction", "output"]:
            results[f"{field}_vocab"] = self.get_vocabulary_stats(data, field)
            
        return results 