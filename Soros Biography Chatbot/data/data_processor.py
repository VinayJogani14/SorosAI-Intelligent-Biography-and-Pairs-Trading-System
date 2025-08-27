"""
Data Processor: Handles dataset loading and preprocessing for training
"""

import datasets
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from pathlib import Path
import json
from config import DATASET_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles loading and preprocessing of datasets for CodeBuddy training
    """
    
    def __init__(self):
        """Initialize the DataProcessor"""
        self.datasets = {}
        self.processed_data = {}
    
    def load_code_search_net(self, subset: str = 'python', split: str = 'train', 
                            max_samples: Optional[int] = None) -> Dataset:
        """
        Load CodeSearchNet dataset
        
        Args:
            subset: Dataset subset (e.g., 'python')
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            logger.info(f"Loading CodeSearchNet {subset} dataset...")
            
            dataset = datasets.load_dataset(
                DATASET_CONFIG["code_search_net"]["name"],
                subset,
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Filter for Python code
            if subset == 'python':
                dataset = dataset.filter(lambda x: x.get('language', '').lower() == 'python')
            
            logger.info(f"Loaded {len(dataset)} samples from CodeSearchNet {subset}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading CodeSearchNet: {str(e)}")
            raise
    
    def load_docstring_dataset(self, split: str = 'train', 
                              max_samples: Optional[int] = None) -> Dataset:
        """
        Load DocString dataset from CodeXGLUE
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            logger.info("Loading DocString dataset...")
            
            dataset = datasets.load_dataset(
                DATASET_CONFIG["docstring"]["name"],
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            logger.info(f"Loaded {len(dataset)} samples from DocString dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading DocString dataset: {str(e)}")
            raise
    
    def load_human_eval(self, split: str = 'test', 
                        max_samples: Optional[int] = None) -> Dataset:
        """
        Load HumanEval dataset
        
        Args:
            split: Dataset split ('test')
            max_samples: Maximum number of samples to load
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            logger.info("Loading HumanEval dataset...")
            
            dataset = datasets.load_dataset(
                DATASET_CONFIG["human_eval"]["name"],
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            logger.info(f"Loaded {len(dataset)} samples from HumanEval dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading HumanEval: {str(e)}")
            raise
    
    def load_mbpp(self, split: str = 'train', 
                  max_samples: Optional[int] = None) -> Dataset:
        """
        Load MBPP (Mostly Basic Python Problems) dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            logger.info("Loading MBPP dataset...")
            
            dataset = datasets.load_dataset(
                DATASET_CONFIG["mbpp"]["name"],
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            logger.info(f"Loaded {len(dataset)} samples from MBPP dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading MBPP: {str(e)}")
            raise
    
    def preprocess_for_training(self, dataset: Dataset, 
                               code_column: str = 'code',
                               explanation_column: str = 'docstring') -> Dataset:
        """
        Preprocess dataset for training
        
        Args:
            dataset: Input dataset
            code_column: Column name containing code
            explanation_column: Column name containing explanations
            
        Returns:
            Preprocessed dataset
        """
        try:
            logger.info("Preprocessing dataset for training...")
            
            def clean_and_filter(example):
                """Clean and filter individual examples"""
                # Clean code
                if code_column in example and example[code_column]:
                    code = str(example[code_column]).strip()
                    if len(code) < 10:  # Filter out very short code
                        return False
                    example[code_column] = code
                else:
                    return False
                
                # Clean explanation
                if explanation_column in example and example[explanation_column]:
                    explanation = str(example[explanation_column]).strip()
                    if len(explanation) < 5:  # Filter out very short explanations
                        return False
                    example[explanation_column] = explanation
                else:
                    return False
                
                return True
            
            # Apply filtering
            filtered_dataset = dataset.filter(clean_and_filter)
            
            # Rename columns for consistency (only if they're different)
            if code_column != 'code':
                filtered_dataset = filtered_dataset.rename_column(code_column, 'code')
            if explanation_column != 'explanation':
                filtered_dataset = filtered_dataset.rename_column(explanation_column, 'explanation')
            
            # Add length information
            def add_lengths(example):
                example['code_length'] = len(example['code'])
                example['explanation_length'] = len(example['explanation'])
                return example
            
            processed_dataset = filtered_dataset.map(add_lengths)
            
            logger.info(f"Preprocessing complete. Final dataset size: {len(processed_dataset)}")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise
    
    def create_training_pairs(self, dataset: Dataset) -> Dataset:
        """
        Create training pairs for code-to-explanation generation
        
        Args:
            dataset: Input dataset with code and explanation columns
            
        Returns:
            Dataset with training pairs
        """
        try:
            logger.info("Creating training pairs...")
            
            def create_prompt(example):
                """Create training prompt"""
                example['input_text'] = f"Explain this Python code: {example['code']}"
                example['target_text'] = example['explanation']
                return example
            
            training_dataset = dataset.map(create_prompt)
            
            logger.info("Training pairs created successfully")
            return training_dataset
            
        except Exception as e:
            logger.error(f"Error creating training pairs: {str(e)}")
            raise
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> DatasetDict:
        """
        Split dataset into train/validation/test sets
        
        Args:
            dataset: Input dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        try:
            logger.info("Splitting dataset...")
            
            # Ensure ratios sum to 1
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError("Ratios must sum to 1.0")
            
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            
            # Split dataset
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, train_size + val_size))
            test_dataset = dataset.select(range(train_size + val_size, total_size))
            
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })
            
            logger.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            return dataset_dict
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise
    
    def save_processed_dataset(self, dataset: Dataset, 
                              output_path: str,
                              format: str = 'parquet'):
        """
        Save processed dataset to disk
        
        Args:
            dataset: Dataset to save
            output_path: Output file path
            format: Output format ('parquet', 'json', 'csv')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'parquet':
                dataset.to_parquet(output_path)
            elif format == 'json':
                dataset.to_json(output_path)
            elif format == 'csv':
                dataset.to_csv(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Dataset saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise
    
    def load_processed_dataset(self, file_path: str, format: str = 'parquet') -> Dataset:
        """
        Load processed dataset from disk
        
        Args:
            file_path: Path to dataset file
            format: Dataset format ('parquet', 'json', 'csv')
            
        Returns:
            Loaded dataset
        """
        try:
            file_path = Path(file_path)
            
            if format == 'parquet':
                dataset = datasets.load_dataset('parquet', data_files=str(file_path))['train']
            elif format == 'json':
                dataset = datasets.load_dataset('json', data_files=str(file_path))['train']
            elif format == 'csv':
                dataset = datasets.load_dataset('csv', data_files=str(file_path))['train']
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Dataset loaded from {file_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about a dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            info = {
                'size': len(dataset),
                'columns': list(dataset.column_names),
                'features': dataset.features,
                'split': getattr(dataset, 'split', 'unknown')
            }
            
            # Add sample statistics if available
            if 'code_length' in dataset.column_names:
                code_lengths = dataset['code_length']
                info['code_length_stats'] = {
                    'min': min(code_lengths),
                    'max': max(code_lengths),
                    'mean': sum(code_lengths) / len(code_lengths)
                }
            
            if 'explanation_length' in dataset.column_names:
                explanation_lengths = dataset['explanation_length']
                info['explanation_length_stats'] = {
                    'min': min(explanation_lengths),
                    'max': max(explanation_lengths),
                    'mean': sum(explanation_lengths) / len(explanation_lengths)
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {}
