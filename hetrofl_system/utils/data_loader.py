import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import dask
import dask.dataframe as dd
import os
import gc
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of the main dataset."""
    
    def __init__(self, dataset_path: str, target_column: str, columns_to_drop: list = None):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
        # Configure Dask to avoid ThreadPoolExecutor issues
        dask.config.set(scheduler='synchronous')
        
    def load_data(self, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load dataset with optional sampling for large files."""
        try:
            if not os.path.exists(self.dataset_path):
                logger.error(f"Dataset not found at {self.dataset_path}")
                return None
                
            file_size_mb = os.path.getsize(self.dataset_path) / (1024 * 1024)
            logger.info(f"Loading dataset: {file_size_mb:.2f} MB")
            
            # Use Dask for large files with proper configuration
            if file_size_mb > 500:
                logger.info("Using Dask for large dataset")
                
                # Use synchronous scheduler to avoid ThreadPoolExecutor issues
                with dask.config.set(scheduler='synchronous'):
                    ddf = dd.read_csv(self.dataset_path, blocksize='64MB')
                    
                    if sample_size:
                        total_rows = ddf.shape[0].compute()
                        sample_frac = min(sample_size / total_rows, 1.0)
                        ddf = ddf.sample(frac=sample_frac, random_state=42)
                        
                    df = ddf.compute()
            else:
                df = pd.read_csv(self.dataset_path)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
            
            # Optimize dtypes
            df = self._optimize_dtypes(df)
            logger.info(f"Dataset loaded with shape: {df.shape}")
            
            # Log class distribution
            if self.target_column in df.columns:
                class_dist = df[self.target_column].value_counts()
                logger.info(f"Class distribution:\n{class_dist}")
            
            gc.collect()
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame, fit_transformers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the dataset."""
        try:
            # Drop specified columns
            for col in self.columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    logger.info(f"Dropped column: {col}")
            
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            
            # Separate features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Store original feature names for consistency
            feature_names = [f'feature_{i}' for i in range(len(X.columns))]
            
            # Handle missing values
            if fit_transformers:
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)
            
            # Convert to DataFrame with consistent feature names
            X = pd.DataFrame(X_imputed, columns=feature_names, index=X.index)
            
            # Scale features
            if fit_transformers:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Convert back to DataFrame with consistent feature names
            X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
            
            # Encode target
            if fit_transformers:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            
            logger.info(f"Preprocessing completed. Features: {X.shape[1]}, Classes: {len(self.label_encoder.classes_)}")
            logger.info(f"Using consistent feature names: {feature_names[:5]}..." if len(feature_names) > 5 else f"Feature names: {feature_names}")
            
            return X.values, y_encoded
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def get_train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        return df
    
    def get_feature_names(self) -> list:
        """Get feature names after preprocessing."""
        return self.scaler.feature_names_in_.tolist() if hasattr(self.scaler, 'feature_names_in_') else []
    
    def get_class_names(self) -> list:
        """Get class names."""
        return self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else []
    
    def get_balanced_test_data(self, sample_size: int = 10000, random_state: int = 42) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get balanced test data for fair model evaluation."""
        try:
            logger.info(f"Loading balanced test data with sample size: {sample_size}")
            
            # Load data
            df = self.load_data(sample_size=sample_size * 2)  # Load more to ensure balance
            if df is None:
                return None, None
            
            # Get class distribution
            class_counts = df[self.target_column].value_counts()
            logger.info(f"Original class distribution: {class_counts.to_dict()}")
            
            # Calculate samples per class for balanced dataset
            min_class_count = class_counts.min()
            samples_per_class = min(min_class_count, sample_size // len(class_counts))
            
            logger.info(f"Creating balanced dataset with {samples_per_class} samples per class")
            
            # Create balanced dataset
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[self.target_column] == class_label].sample(
                    n=samples_per_class, 
                    random_state=random_state
                )
                balanced_dfs.append(class_df)
            
            # Combine balanced data
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
            # Shuffle the balanced dataset
            balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Verify balance
            final_counts = balanced_df[self.target_column].value_counts()
            logger.info(f"Balanced class distribution: {final_counts.to_dict()}")
            
            # Preprocess the balanced data
            X, y = self.preprocess_data(balanced_df, fit_transformers=False)
            
            logger.info(f"Created balanced test set: {X.shape[0]} samples, {len(np.unique(y))} classes")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating balanced test data: {e}")
            return None, None
    
    def get_imbalanced_test_data(self, sample_size: int = 10000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get imbalanced test data (original distribution) for comparison."""
        try:
            logger.info(f"Loading imbalanced test data with sample size: {sample_size}")
            
            # Load data with original distribution
            df = self.load_data(sample_size=sample_size)
            if df is None:
                return None, None
            
            # Get class distribution
            class_counts = df[self.target_column].value_counts()
            logger.info(f"Imbalanced class distribution: {class_counts.to_dict()}")
            
            # Preprocess the data
            X, y = self.preprocess_data(df, fit_transformers=False)
            
            logger.info(f"Created imbalanced test set: {X.shape[0]} samples, {len(np.unique(y))} classes")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating imbalanced test data: {e}")
            return None, None 