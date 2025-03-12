# Mistral-7B Dataset Preparation Workflow Guide

This guide provides step-by-step instructions for preparing datasets for Mistral-7B fine-tuning using the tools in this repository.

## Overview of the Process

The dataset preparation workflow consists of the following steps:

1. **Data Collection**: Gather raw data from various sources
2. **Data Preprocessing**: Clean and normalize the text data
3. **Format Conversion**: Convert data to Mistral-7B format
4. **Data Validation**: Filter out low-quality examples
5. **Dataset Splitting**: Create train/validation splits
6. **Export**: Export the final dataset for fine-tuning

## 1. Data Collection

Place your raw data in the `data/raw/` directory. The data can be in various formats:

- JSON: A list of objects with fields for instruction, input (optional), and output
- JSONL: One JSON object per line
- CSV/TSV: With columns for instruction, input (optional), and output

Example of expected data structure:

```json
[
  {
    "instruction": "Explain the concept of deep learning in simple terms.",
    "output": "Deep learning is a type of artificial intelligence..."
  },
  {
    "instruction": "Write a function in Python to check if a string is a palindrome.",
    "input": "Use any approach you like.",
    "output": "```python\ndef is_palindrome(s):\n    ..."
  }
]
```

## 2. Data Preprocessing and Format Conversion

Use the `convert_to_mistral_format.py` script to preprocess and convert your data:

```bash
python scripts/convert_to_mistral_format.py \
  --input data/raw/your_data.json \
  --output data/interim/formatted_data.jsonl \
  --instruction-key instruction \
  --output-key output \
  --input-key input
```

Parameters:
- `--input`: Path to your input data file
- `--output`: Path to save the formatted output
- `--instruction-key`: Field name for instructions in your data
- `--output-key`: Field name for outputs in your data
- `--input-key`: Field name for inputs in your data (optional)

## 3. Data Validation and Filtering

Use the `prepare_dataset.py` script to validate and filter your formatted data:

```bash
python scripts/prepare_dataset.py \
  --config config/default_config.yaml \
  --input data/interim/formatted_data.jsonl \
  --output data/processed/filtered_data.jsonl
```

Parameters:
- `--config`: Path to the configuration file
- `--input`: Path to your formatted data
- `--output`: Path to save the filtered data

## 4. Customizing the Process

### Configuration Options

You can customize the dataset preparation process by modifying the config file:

```yaml
# Example config.yaml
keys:
  instruction: "instruction"
  output: "output"
  input: "input"

text_cleaning:
  lowercase: false
  remove_urls: true
  max_length: null

validation:
  min_instruction_length: 3
  max_instruction_length: 2048
  min_output_length: 1
  max_output_length: 4096
  check_instruction_quality: true

val_size: 0.1
seed: 42
```

### Creating a Custom Preprocessing Pipeline

For advanced customization, you can extend the base classes in `src/data_processor.py` and `src/data_validator.py`:

```python
from src.data_processor import TextCleaner

class MyCustomCleaner(TextCleaner):
    def clean_text(self, text):
        # Call the parent method first
        text = super().clean_text(text)
        
        # Add your custom cleaning logic
        text = text.replace("foo", "bar")
        
        return text
```

## 5. Final Data Format for Mistral-7B Fine-tuning

The final dataset should be in the following format:

```json
[
  {
    "instruction": "Your instruction here",
    "input": "Optional context or input",
    "output": "Desired model output"
  }
]
```

Key requirements:
- The `instruction` field is required and should be a clear, well-formed instruction/question
- The `input` field is optional and provides additional context
- The `output` field is required and represents the desired model response

## 6. Best Practices

1. **Data Quality**: Focus on high-quality examples rather than quantity
2. **Instructions**: Make instructions clear, specific, and diverse
3. **Outputs**: Ensure outputs are well-formatted and helpful
4. **Validation**: Always validate your dataset before fine-tuning
5. **Balance**: Try to maintain a balance of different instruction types
6. **Length**: Include examples with varying lengths to handle diverse uses

## 7. Common Issues and Solutions

### Missing or Incomplete Outputs

Problem: Some examples have missing or incomplete outputs.
Solution: Filter them out using the validation script.

### Poor Instruction Quality

Problem: Instructions are vague or poorly worded.
Solution: Improve them manually or filter out low-quality instructions.

### Imbalanced Dataset

Problem: The dataset is dominated by one type of instruction.
Solution: Stratify your examples and balance the dataset by adding more diverse examples.

## Next Steps

After preparing your dataset, you can proceed with fine-tuning Mistral-7B using the formatted data. Follow the Mistral AI documentation for fine-tuning instructions. 