# CodeBuddy: Natural Language to Code Explanation Generator

An AI-Powered Tool for Code Understanding and Documentation

## Project Overview

CodeBuddy is an intelligent system that explains Python code in plain English and answers questions about it using advanced NLP techniques and RAG (Retrieval-Augmented Generation).

## Architecture

```
Code Input → CodeT5 Encoder → Explanation Generator
                    ↓
            Embeddings/Context
                    ↓
            RAG Pipeline ← User Questions
                    ↓
            Interactive Answers
```

## Key Components

- **CodeT5**: For code understanding and summarization
- **Vector Database**: For code snippet retrieval
- **RAG Pipeline**: For context-aware Q&A
- **Fine-tuning**: On domain-specific explanations

## Features

- **Code Explanation**: Generate human-readable explanations of Python code
- **Interactive Q&A**: Ask questions about code and get contextual answers
- **Code Search**: Find similar code snippets and their explanations
- **Streamlit Frontend**: User-friendly web interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CodeBuddy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets (optional for training):
```bash
python scripts/download_datasets.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. **Code Explanation**: Paste Python code and get explanations
2. **Code Q&A**: Ask questions about specific code snippets
3. **Code Search**: Find similar code examples

## Model Training

For training on Google Colab Pro with A100 GPU:

1. Upload the training scripts to Colab
2. Install requirements
3. Run training with GPU acceleration
4. Download the trained model

## Evaluation Metrics

- **BLEU Score**: Measure explanation quality against human references
- **BERTScore**: Semantic similarity of generated explanations
- **Exact Match (EM)**: For Q&A accuracy
- **Response Time**: System efficiency

## Datasets

- CodeSearchNet (Python subset)
- DocString Dataset (CodeXGLUE)
- HumanEval
- MBPP (Mostly Basic Python Problems)

## Project Structure

```
CodeBuddy/
├── app.py                 # Streamlit frontend
├── requirements.txt       # Dependencies
├── config.py             # Configuration
├── models/               # Model implementations
├── data/                 # Data processing
├── training/             # Training scripts
├── evaluation/           # Evaluation metrics
├── utils/                # Utility functions
└── scripts/              # Helper scripts
```

## License

MIT License
