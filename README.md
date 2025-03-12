# Mistral-7B Fine-Tuning Dataset Preparation

This repository contains tools and scripts for preparing custom datasets to fine-tune the Mistral-7B language model. 

## Overview

The Mistral-7B model is a powerful open-source large language model that can be fine-tuned for specific tasks. This project provides utilities to:

1. Clean and preprocess raw data
2. Convert data into the correct format for fine-tuning
3. Validate and filter examples
4. Create train/validation splits
5. Export to compatible formats for training
6. Analyze and visualize dataset statistics

## Project Structure

- `src/` - Source code for data processing and validation
- `scripts/` - Utility scripts for dataset preparation and analysis
- `data/` - Directory for raw and processed data (not included in repo)
- `config/` - Configuration files for different dataset types
- `docs/` - Detailed documentation and guides
- `notebooks/` - Jupyter notebooks for data exploration and analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/yourusername/mistral-dataset-prep.git
cd mistral-dataset-prep
pip install -r requirements.txt
```

### Workflow

The typical workflow for dataset preparation consists of these steps:

1. **Prepare your data** in a format with instruction, input (optional), and output fields
2. **Convert to Mistral format** using the `convert_to_mistral_format.py` script
3. **Validate and filter** your data using the `prepare_dataset.py` script
4. **Analyze** your dataset with the `analyze_dataset.py` script
5. Use the final data for fine-tuning Mistral-7B

For detailed instructions, see the [Workflow Guide](docs/workflow_guide.md).

## Scripts

### Data Conversion

```bash
python scripts/convert_to_mistral_format.py \
  --input data/raw/your_data.json \
  --output data/interim/formatted_data.jsonl \
  --instruction-key instruction \
  --output-key output \
  --input-key input
```

### Data Validation and Filtering

```bash
python scripts/prepare_dataset.py \
  --config config/default_config.yaml \
  --input data/interim/formatted_data.jsonl \
  --output data/processed/filtered_data.jsonl
```

### Dataset Analysis

```bash
python scripts/analyze_dataset.py \
  --input data/processed/filtered_data.jsonl \
  --output-dir data/analysis
```

## Data Format

The final dataset should be in the following format for Mistral-7B fine-tuning:

```json
[
  {
    "instruction": "Your instruction here",
    "input": "Optional input context here",
    "output": "Desired output from the model"
  },
  ...
]
```

## Configuration

You can customize the dataset preparation process by modifying the config files in the `config/` directory. The default configuration is provided in `config/default_config.yaml`.

Key configuration options include:
- Text cleaning settings
- Validation criteria
- Dataset splitting parameters

## Sample Data

A sample dataset is provided in `data/raw/sample_data.json` to demonstrate the expected format.

## Best Practices

For best results when fine-tuning Mistral-7B:

1. **Data Quality**: Focus on high-quality examples rather than quantity
2. **Instructions**: Make instructions clear, specific, and diverse
3. **Outputs**: Ensure outputs are well-formatted and helpful
4. **Validation**: Always validate your dataset before fine-tuning
5. **Balance**: Try to maintain a balance of different instruction types
6. **Length**: Include examples with varying lengths to handle diverse uses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 