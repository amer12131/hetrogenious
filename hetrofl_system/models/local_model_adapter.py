import pickle
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import os
import joblib
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

class LocalModelAdapter(ABC):
    """Abstract base class for local model adapters."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the trained model and preprocessors."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def get_model_weights(self) -> Dict[str, Any]:
        """Extract model weights/parameters for aggregation."""
        pass
    
    @abstractmethod
    def update_model_weights(self, weights: Dict[str, Any]) -> bool:
        """Update model with new weights from aggregation."""
        pass
    
    @abstractmethod
    def fine_tune(self, X: np.ndarray, y: np.ndarray, global_knowledge: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, float]:
        """Fine-tune the model with new data and global knowledge."""
        pass
    
    def load_preprocessors(self) -> bool:
        """Load scaler and label encoder."""
        try:
            scaler_path = self.model_config['path'] / self.model_config['scaler_file']
            encoder_path = self.model_config['path'] / self.model_config['encoder_file']
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"Loaded label encoder from {encoder_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {e}")
            return False
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess input data using loaded scaler."""
        try:
            if self.scaler is not None:
                # Handle feature names properly to avoid sklearn warnings
                if isinstance(X, np.ndarray):
                    # Check if scaler was fitted with feature names
                    if hasattr(self.scaler, 'feature_names_in_'):
                        feature_names = self.scaler.feature_names_in_
                        if len(feature_names) == X.shape[1]:
                            X_df = pd.DataFrame(X, columns=feature_names)
                            # Transform and return as DataFrame to preserve feature names
                            X_transformed = self.scaler.transform(X_df)
                            return pd.DataFrame(X_transformed, columns=feature_names)
                        else:
                            # Feature count mismatch, log warning and use available features
                            logger.warning(f"Feature count mismatch: expected {len(feature_names)}, got {X.shape[1]}")
                            # Use the first N features or pad with zeros if needed
                            if X.shape[1] < len(feature_names):
                                # Pad with zeros for missing features
                                X_padded = np.zeros((X.shape[0], len(feature_names)))
                                X_padded[:, :X.shape[1]] = X
                                X_df = pd.DataFrame(X_padded, columns=feature_names)
                            else:
                                # Use only the first N features
                                X_df = pd.DataFrame(X[:, :len(feature_names)], columns=feature_names)
                            
                            X_transformed = self.scaler.transform(X_df)
                            return pd.DataFrame(X_transformed, columns=feature_names)
                    else:
                        # Scaler was fitted without feature names, safe to use numpy array
                        return self.scaler.transform(X)
                elif hasattr(X, 'columns'):
                    # X is already a DataFrame
                    if hasattr(self.scaler, 'feature_names_in_'):
                        # Ensure DataFrame has the correct feature names
                        expected_features = self.scaler.feature_names_in_
                        if not all(col in X.columns for col in expected_features):
                            logger.warning("DataFrame missing some expected features, reordering/filling")
                            # Create new DataFrame with expected features
                            X_reordered = pd.DataFrame(index=X.index)
                            for feature in expected_features:
                                if feature in X.columns:
                                    X_reordered[feature] = X[feature]
                                else:
                                    X_reordered[feature] = 0.0  # Fill missing features with 0
                            X = X_reordered
                    
                    X_transformed = self.scaler.transform(X)
                    if hasattr(self.scaler, 'feature_names_in_'):
                        return pd.DataFrame(X_transformed, columns=self.scaler.feature_names_in_)
                    else:
                        return X_transformed
                else:
                    # Fallback to direct transformation
                    return self.scaler.transform(X)
            return X
        except Exception as e:
            logger.warning(f"Error in preprocessing: {e}, returning original data")
            return X
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        info = {
            'type': self.model_config['type'],
            'path': str(self.model_config['path']),
            'is_loaded': self.is_loaded,
            'has_scaler': self.scaler is not None,
            'has_encoder': self.label_encoder is not None,
            'model_type': 'XGBoost',  # Always set to XGBoost for clarity
        }
        
        # Add feature count from scaler (this is more reliable)
        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
            info['n_features'] = int(self.scaler.n_features_in_)
        
        # Add class count from label encoder (this is more reliable)
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            info['n_classes'] = int(len(self.label_encoder.classes_))
        
        # Try to add more information about the model if available
        if self.is_loaded and self.model is not None:
            try:
                # Check if this is an sklearn-wrapped XGBoost model
                is_sklearn_wrapper = hasattr(self.model, 'get_booster')
                
                if is_sklearn_wrapper:
                    # For sklearn-wrapped XGBoost
                    if hasattr(self.model, 'n_features_in_'):
                        info['n_features'] = int(self.model.n_features_in_)
                    
                    if hasattr(self.model, 'n_classes_'):
                        info['n_classes'] = int(self.model.n_classes_)
                    elif hasattr(self.model, 'get_params') and 'num_class' in self.model.get_params():
                        info['n_classes'] = int(self.model.get_params()['num_class'])
                    
                    # Get feature importance if available
                    if hasattr(self.model, 'feature_importances_'):
                        info['has_feature_importance'] = True
                        info['feature_importances'] = self.model.feature_importances_.tolist()
                else:
                    # For native XGBoost
                    if hasattr(self.model, 'num_features'):
                        info['n_features'] = int(self.model.num_features())
                    
                    if hasattr(self.model, 'num_boosted_rounds'):
                        info['num_boosted_rounds'] = int(self.model.num_boosted_rounds())
                
                # Check for dump capability
                if hasattr(self.model, 'get_dump'):
                    try:
                        dumps = self.model.get_dump()
                        info['num_trees'] = len(dumps)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Error getting extended model info: {e}")
        
        # Set default values if we couldn't get the information
        if 'n_features' not in info or info['n_features'] is None:
            info['n_features'] = 35  # Default from typical dataset
        
        if 'n_classes' not in info or info['n_classes'] is None:
            info['n_classes'] = 10  # Default from typical dataset
        
        # Always provide these for the GUI
        if 'has_feature_importance' not in info:
            info['has_feature_importance'] = False
                
        return info

class XGBoostAdapter(LocalModelAdapter):
    """Adapter for XGBoost models."""
    
    def load_model(self) -> bool:
        """Load XGBoost model."""
        try:
            model_path = self.model_config['path'] / self.model_config['model_file']
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.load_preprocessors()
                self.is_loaded = True
                logger.info(f"Loaded XGBoost model from {model_path}")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # First, determine what type of XGBoost model we have
            is_sklearn_wrapper = False
            if hasattr(self.model, 'get_booster'):
                is_sklearn_wrapper = True
                logger.info("Using sklearn-wrapped XGBoost model for prediction")
            else:
                logger.info("Using direct XGBoost booster for prediction")
            
            if is_sklearn_wrapper:
                # Using sklearn wrapper's predict method
                if hasattr(X_processed, 'values'):
                    X_proc = X_processed.values
                else:
                    X_proc = X_processed
                
                # Check if we need to get class predictions (not probabilities)
                try:
                    # This is more reliable for multi-class predictions
                    predictions = self.model.predict(X_proc)
                    return predictions.astype(int)
                except Exception as e:
                    logger.warning(f"Error in sklearn prediction: {e}, trying alternative approach")
                    # Try predict_proba and convert to class predictions
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(X_proc)
                        return np.argmax(proba, axis=1).astype(int)
                    raise e
            else:
                # Using XGBoost Booster directly - MUST use DMatrix
                try:
                    # Create DMatrix with feature names if available
                    if hasattr(X_processed, 'columns'):
                        # X_processed is a DataFrame with feature names
                        feature_names = list(X_processed.columns)
                        X_values = X_processed.values
                        dtest = xgb.DMatrix(X_values, feature_names=feature_names)
                    else:
                        # X_processed is a numpy array, create DMatrix without feature names
                        dtest = xgb.DMatrix(X_processed)
                    
                    # Get raw predictions
                    raw_predictions = self.model.predict(dtest)
                    
                    # Convert probabilities to class predictions based on shape
                    if len(raw_predictions.shape) > 1:
                        # Multi-class
                        return np.argmax(raw_predictions, axis=1).astype(int)
                    else:
                        # Binary classification
                        return (raw_predictions > 0.5).astype(int)
                except Exception as e:
                    logger.error(f"Error using XGBoost booster predict: {e}")
                    
                    # Last resort - try to use a different approach
                    try:
                        logger.warning("Attempting alternative prediction approach")
                        # Try to use xgboost's predict method directly without DMatrix
                        # This is a fallback that shouldn't be necessary, but just in case
                        if hasattr(self.model, 'predict') and callable(self.model.predict):
                            preds = self.model.predict(X_processed.values if hasattr(X_processed, 'values') else X_processed)
                            if len(preds.shape) > 1:
                                return np.argmax(preds, axis=1).astype(int)
                            else:
                                return (preds > 0.5).astype(int)
                    except Exception as fallback_e:
                        logger.error(f"Fallback prediction also failed: {fallback_e}")
                        raise e  # Re-raise the original error
                
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            # Return default predictions if error occurs
            # But let's make this more informative in logs
            if hasattr(X, 'shape'):
                logger.error(f"Returning zeros with shape {X.shape[0]}")
            
            # Create a fixed-size array of zeros
            return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # First, determine what type of XGBoost model we have
            is_sklearn_wrapper = False
            if hasattr(self.model, 'get_booster'):
                is_sklearn_wrapper = True
            
            # Determine the number of classes from label encoder
            n_classes = 2  # Default binary classification
            if hasattr(self.label_encoder, 'classes_'):
                n_classes = len(self.label_encoder.classes_)
                
            if is_sklearn_wrapper:
                # Using sklearn wrapper's predict_proba method
                if hasattr(X_processed, 'values'):
                    X_proc = X_processed.values
                else:
                    X_proc = X_processed
                
                try:
                    proba = self.model.predict_proba(X_proc)
                    # Verify correct shape for number of classes
                    if proba.shape[1] != n_classes:
                        logger.warning(f"Probability shape mismatch: got {proba.shape[1]} columns, expected {n_classes}")
                        # Resize to match expected number of classes
                        if proba.shape[1] < n_classes:
                            # Add missing columns with zero probability
                            padding = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                            proba = np.hstack([proba, padding])
                        else:
                            # Truncate extra columns
                            proba = proba[:, :n_classes]
                    return proba
                except Exception as e:
                    logger.warning(f"Error in sklearn predict_proba: {e}, trying fallback")
                    # If predict_proba fails, try regular predict with output_margin=True
                    if hasattr(self.model, 'predict'):
                        try:
                            # For binary classification
                            raw_preds = self.model.predict(X_proc, output_margin=True)
                            # Convert logits to probabilities using sigmoid
                            probs = 1.0 / (1.0 + np.exp(-raw_preds))
                            if n_classes == 2:
                                return np.column_stack([1 - probs, probs])
                            else:
                                # For multi-class, create uniform distribution with higher prob for predicted class
                                y_pred = self.predict(X)
                                proba = np.ones((len(y_pred), n_classes)) * 0.1 / (n_classes - 1)
                                for i, cls in enumerate(y_pred):
                                    if cls < n_classes:  # Ensure predicted class is valid
                                        proba[i, cls] = 0.9
                                return proba
                        except:
                            # Can't use output_margin, return default
                            logger.error("Failed to get probabilities, returning uniform distribution")
                            pass
            else:
                # Using XGBoost Booster directly - MUST use DMatrix
                try:
                    # Create DMatrix with feature names if available
                    if hasattr(X_processed, 'columns'):
                        feature_names = list(X_processed.columns)
                        X_values = X_processed.values
                        dtest = xgb.DMatrix(X_values, feature_names=feature_names)
                    else:
                        # X_processed is a numpy array
                        dtest = xgb.DMatrix(X_processed)
                    
                    # Get raw probabilities from model
                    probabilities = self.model.predict(dtest)
                    
                    # Check if we're in a multi-class scenario
                    if len(probabilities.shape) > 1:
                        # Already multi-class format, verify correct shape
                        if probabilities.shape[1] != n_classes:
                            logger.warning(f"Probability shape mismatch: got {probabilities.shape[1]} columns, expected {n_classes}")
                            # Resize to match expected number of classes
                            if probabilities.shape[1] < n_classes:
                                # Add missing columns with zero probability
                                padding = np.zeros((probabilities.shape[0], n_classes - probabilities.shape[1]))
                                probabilities = np.hstack([probabilities, padding])
                            else:
                                # Truncate extra columns
                                probabilities = probabilities[:, :n_classes]
                        return probabilities
                    else:
                        # Binary classification - convert to 2-column format
                        if n_classes == 2:
                            return np.column_stack([1 - probabilities, probabilities])
                        else:
                            # We need to convert binary to multi-class format
                            logger.warning(f"Converting binary probabilities to {n_classes} classes")
                            # Create probabilities for predicted classes
                            y_pred = (probabilities > 0.5).astype(int)
                            multi_proba = np.zeros((len(y_pred), n_classes))
                            # Give majority probability to predicted class (0 or 1)
                            for i, p in enumerate(probabilities):
                                if y_pred[i] == 1:
                                    multi_proba[i, 1] = p
                                    # Distribute remaining probability uniformly
                                    remaining = (1 - p) / (n_classes - 1)
                                    for j in range(n_classes):
                                        if j != 1:
                                            multi_proba[i, j] = remaining
                                else:
                                    multi_proba[i, 0] = 1 - p
                                    # Distribute remaining probability uniformly
                                    remaining = p / (n_classes - 1)
                                    for j in range(n_classes):
                                        if j != 0:
                                            multi_proba[i, j] = remaining
                            return multi_proba
                        
                except Exception as e:
                    logger.error(f"Error using XGBoost booster predict_proba: {e}")
                    # Last resort fallback - use uniform distribution
                    n_samples = X.shape[0]
                    return np.ones((n_samples, n_classes)) / n_classes
            
            # If we get here, we couldn't properly get probabilities
            n_samples = X.shape[0]
            return np.ones((n_samples, n_classes)) / n_classes
                
        except Exception as e:
            logger.error(f"Error in XGBoost predict_proba: {e}")
            # Return default probabilities with correct number of classes
            n_samples = X.shape[0]
            n_classes = 2  # Default
            if hasattr(self.label_encoder, 'classes_'):
                n_classes = len(self.label_encoder.classes_)
            return np.ones((n_samples, n_classes)) / n_classes
    
    def get_model_weights(self) -> Dict[str, Any]:
        """Extract XGBoost model parameters."""
        if not self.is_loaded:
            return {}
        
        # For XGBoost, we can extract tree information
        # This is a simplified version - in practice, you might want more sophisticated extraction
        try:
            # Fix: XGBoost models don't have get_booster() method directly
            # They are already booster objects when loaded from pickle
            if hasattr(self.model, 'get_dump'):
                trees = self.model.get_dump()
            else:
                trees = []
            
            weights = {
                'num_trees': int(len(trees)),
                'trees_info': int(len(trees))  # Simplified representation
            }
            
            # Add feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                weights['feature_importance'] = [float(x) for x in self.model.feature_importances_]
            
            # Add model parameters (convert numpy types to Python types)
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                json_safe_params = {}
                for key, value in params.items():
                    if isinstance(value, (np.integer, np.int32, np.int64)):
                        json_safe_params[key] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        json_safe_params[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        json_safe_params[key] = value.tolist()
                    else:
                        json_safe_params[key] = value
                weights['model_params'] = json_safe_params
            
            return weights
            
        except Exception as e:
            logger.error(f"Error extracting XGBoost weights: {e}")
            return {}
    
    def update_model_weights(self, weights: Dict[str, Any]) -> bool:
        """Update XGBoost model (limited capability)."""
        # XGBoost doesn't support direct weight updates easily
        # This would require retraining or model ensemble approaches
        logger.warning("XGBoost direct weight updates not implemented - using ensemble approach")
        return False
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, global_knowledge: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, float]:
        """Fine-tune XGBoost model with new data and global knowledge."""
        try:
            X_processed = self.preprocess_data(X)
            
            # Make sure y contains valid class indices
            if hasattr(self.label_encoder, 'classes_'):
                # Ensure y values are valid class indices (0 to n_classes-1)
                n_classes = len(self.label_encoder.classes_)
                if np.any(y >= n_classes) or np.any(y < 0):
                    logger.warning(f"Found invalid class indices in y, clipping to valid range 0-{n_classes-1}")
                    y = np.clip(y, 0, n_classes - 1)
            
            # Determine if using sklearn wrapper or direct XGBoost booster
            is_sklearn_wrapper = hasattr(self.model, 'get_booster')
            if is_sklearn_wrapper:
                logger.info("Using sklearn-wrapped XGBoost model for fine-tuning")
                xgb_booster = self.model.get_booster()
            else:
                logger.info("Using direct XGBoost booster for fine-tuning")
                xgb_booster = self.model
            
            # Handle feature names for XGBoost DMatrix
            if hasattr(X_processed, 'columns'):
                feature_names = list(X_processed.columns)
                X_values = X_processed.values if hasattr(X_processed, 'values') else X_processed
                dtrain = xgb.DMatrix(X_values, label=y, feature_names=feature_names)
            else:
                X_values = X_processed
                dtrain = xgb.DMatrix(X_values, label=y)
            
            # Get baseline performance using DMatrix for prediction
            try:
                # Safer prediction approach using DMatrix directly
                dtest = xgb.DMatrix(X_values)
                if is_sklearn_wrapper:
                    y_pred_raw = xgb_booster.predict(dtest)
                else:
                    y_pred_raw = self.model.predict(dtest)
                
                # Convert to class predictions
                if len(y_pred_raw.shape) > 1:
                    y_pred_before = np.argmax(y_pred_raw, axis=1)
                else:
                    y_pred_before = (y_pred_raw > 0.5).astype(int)
                
                accuracy_before = accuracy_score(y, y_pred_before)
                logger.info(f"Baseline accuracy before fine-tuning: {accuracy_before:.4f}")
            except Exception as e:
                logger.warning(f"Error getting baseline performance: {e}")
                # Fallback to a default value
                accuracy_before = 0.0
                y_pred_before = np.zeros(len(y), dtype=int)
            
            # Strategy 1: Incremental training with new data
            logger.info(f"XGBoost incremental training with {len(X)} samples")
            
            # Continue training the existing model with new data
            num_boost_rounds = kwargs.get('num_boost_rounds', 15)  # Increased rounds
            
            try:                
                # If we have global knowledge, use it to guide training
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    global_prob = global_knowledge.get('global_probabilities', None)
                    
                    # Create mixed training data: combine original labels with global predictions
                    # This implements knowledge distillation from global model
                    mixed_labels = y.copy()
                    
                    # Use global predictions for a portion of the data to transfer knowledge
                    distill_ratio = kwargs.get('distill_ratio', 0.3)  # 30% of data uses global knowledge
                    n_distill = int(len(y) * distill_ratio)
                    
                    if n_distill > 0 and len(global_pred) >= len(y):
                        # Select samples for knowledge distillation
                        distill_indices = np.random.choice(len(y), n_distill, replace=False)
                        
                        # For knowledge distillation, we can either:
                        # 1. Use global predictions directly
                        # 2. Use soft targets if probabilities are available
                        if global_prob is not None and len(global_prob) >= len(y):
                            # Use soft targets - blend original labels with global probabilities
                            alpha = 0.7  # Weight for original labels
                            for idx in distill_indices:
                                # If global model is confident, use its prediction
                                global_confidence = np.max(global_prob[idx])
                                if global_confidence > 0.8:  # High confidence threshold
                                    mixed_labels[idx] = global_pred[idx]
                        else:
                            # Use hard targets from global predictions
                            mixed_labels[distill_indices] = global_pred[distill_indices]
                        
                        logger.info(f"Applied global knowledge distillation to {n_distill} samples")
                    
                    # Create DMatrix with mixed labels for knowledge transfer
                    if hasattr(X_processed, 'columns'):
                        dtrain_mixed = xgb.DMatrix(X_values, label=mixed_labels, feature_names=feature_names)
                    else:
                        dtrain_mixed = xgb.DMatrix(X_values, label=mixed_labels)
                    
                    # Train with mixed data for knowledge transfer
                    if is_sklearn_wrapper:
                        # For sklearn-style XGBoost, we need to retrain
                        logger.info("Retraining sklearn-wrapped XGBoost with global knowledge")
                        
                        # Extract parameters from model for proper re-fitting
                        params = self.model.get_params()
                        fit_params = {k: v for k, v in params.items() 
                                     if k not in ['n_estimators', 'early_stopping_rounds', 'callbacks']}
                        
                        # Set num_class explicitly for multi-class problems
                        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 2:
                            fit_params['num_class'] = len(self.label_encoder.classes_)
                        
                        # Combine both original and knowledge-enhanced data
                        X_expanded = np.vstack([X_values, X_values])
                        y_expanded = np.hstack([y, mixed_labels])
                        
                        # Fit with a slightly higher learning rate to improve convergence
                        if 'learning_rate' in fit_params:
                            original_lr = fit_params['learning_rate']
                            fit_params['learning_rate'] = min(0.1, original_lr * 1.5)
                        
                        # Retrain with expanded data
                        try:
                            self.model.fit(X_expanded, y_expanded)
                            logger.info(f"Successfully retrained model with {len(X_expanded)} samples")
                        except Exception as e:
                            logger.error(f"Error retraining with expanded data: {e}")
                            # Fallback to just retraining with original data
                            try:
                                self.model.fit(X_values, y)
                                logger.info("Fallback: retrained with original data only")
                            except Exception as e2:
                                logger.error(f"Also failed retraining with original data: {e2}")
                    else:
                        # For native XGBoost, we can update the existing model
                        try:
                            # First update with mixed data (knowledge transfer)
                            base_iteration = 0
                            if hasattr(xgb_booster, 'num_boosted_rounds'):
                                try:
                                    base_iteration = xgb_booster.num_boosted_rounds()
                                except:
                                    pass
                                    
                            # First update with mixed data (knowledge transfer)
                            half_rounds = num_boost_rounds // 2
                            for i in range(half_rounds):
                                try:
                                    iteration = base_iteration + i if base_iteration > 0 else i
                                    xgb_booster.update(dtrain_mixed, iteration)
                                except Exception as e:
                                    logger.warning(f"Error in update at iteration {i}: {e}")
                            
                            # Then update with original data (fine-tuning)
                            for i in range(half_rounds):
                                try:
                                    iteration = base_iteration + half_rounds + i if base_iteration > 0 else half_rounds + i
                                    xgb_booster.update(dtrain, iteration)
                                except Exception as e:
                                    logger.warning(f"Error in update at iteration {half_rounds + i}: {e}")
                                
                            logger.info(f"Updated native XGBoost with {num_boost_rounds} rounds")
                        except Exception as e:
                            logger.error(f"Error updating native XGBoost: {e}")
                            # Try simpler approach
                            for i in range(num_boost_rounds):
                                try:
                                    xgb_booster.update(dtrain, i)
                                except Exception as update_e:
                                    logger.error(f"Update failed at iteration {i}: {update_e}")
                                    break
                            logger.info("Fallback: updated with simplified approach")
                else:
                    # Standard incremental training without global knowledge
                    if is_sklearn_wrapper:
                        try:
                            # For sklearn models, we can use warm_start to continue training
                            if hasattr(self.model, 'n_estimators'):
                                current_n_estimators = self.model.n_estimators
                                self.model.n_estimators = current_n_estimators + num_boost_rounds
                                self.model.fit(X_values, y)
                                logger.info(f"Updated sklearn XGBoost from {current_n_estimators} to {self.model.n_estimators} trees")
                            else:
                                # Fallback: just refit with current data
                                self.model.fit(X_values, y)
                                logger.info(f"Refitted sklearn XGBoost with {len(X)} samples")
                        except Exception as e:
                            logger.warning(f"Error updating sklearn XGBoost: {e}")
                            # Last resort: train a new model with default params
                            try:
                                params = {'objective': 'multi:softprob', 'n_estimators': 100}
                                if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 2:
                                    params['num_class'] = len(self.label_encoder.classes_)
                                self.model = xgb.XGBClassifier(**params)
                                self.model.fit(X_values, y)
                                logger.info("Trained new XGBClassifier as fallback")
                            except Exception as e2:
                                logger.error(f"Complete failure in XGBoost training: {e2}")
                    else:
                        # For native XGBoost, update the model directly
                        base_iteration = 0
                        if hasattr(xgb_booster, 'num_boosted_rounds'):
                            try:
                                base_iteration = xgb_booster.num_boosted_rounds()
                            except:
                                pass
                                
                        try:
                            for i in range(num_boost_rounds):
                                try:
                                    iteration = base_iteration + i if base_iteration > 0 else i
                                    xgb_booster.update(dtrain, iteration)
                                except Exception as update_e:
                                    logger.error(f"Update failed at iteration {i}: {update_e}")
                                    break
                            logger.info(f"Updated native XGBoost with {num_boost_rounds} boosting rounds")
                        except Exception as e:
                            logger.warning(f"Error in native XGBoost update: {e}")
                            # Try simplest possible update
                            try:
                                for i in range(num_boost_rounds):
                                    try:
                                        xgb_booster.update(dtrain, i)
                                    except:
                                        break
                                logger.info(f"Applied simplified update with {num_boost_rounds} rounds")
                            except Exception as e2:
                                logger.error(f"Complete failure in XGBoost update: {e2}")
            except Exception as e:
                logger.error(f"XGBoost incremental training failed: {e}")
            
            # Evaluate final performance using DMatrix for prediction
            try:
                # Safer prediction approach using DMatrix directly
                dtest = xgb.DMatrix(X_values)
                if is_sklearn_wrapper and hasattr(xgb_booster, 'predict'):
                    y_pred_raw = xgb_booster.predict(dtest)
                else:
                    y_pred_raw = self.model.predict(dtest)
                
                # Convert to class predictions
                if len(y_pred_raw.shape) > 1:
                    y_pred = np.argmax(y_pred_raw, axis=1)
                else:
                    y_pred = (y_pred_raw > 0.5).astype(int)
                
                accuracy_after = accuracy_score(y, y_pred)
            except Exception as e:
                logger.warning(f"Error getting performance after fine-tuning: {e}")
                try:
                    # Try our regular predict method as fallback
                    y_pred = self.predict(X)
                    accuracy_after = accuracy_score(y, y_pred)
                except:
                    # Last resort fallback
                    logger.error("Both prediction methods failed, using zeros")
                    y_pred = np.zeros_like(y)
                    accuracy_after = 0.0
            
            # Calculate actual improvement
            actual_improvement = accuracy_after - accuracy_before
            
            # Calculate comprehensive metrics
            try:
                f1 = f1_score(y, y_pred, average='weighted')
            except:
                f1 = 0.0
                
            metrics = {
                'accuracy': float(accuracy_after),
                'f1_score': float(f1),
                'accuracy_improvement': float(actual_improvement),
                'samples_processed': int(len(X)),
                'boost_rounds_applied': num_boost_rounds,
                'training_time': 0.0,
                'global_knowledge_used': global_knowledge is not None
            }
            
            # Add precision and recall
            try:
                from sklearn.metrics import precision_score, recall_score
                metrics['precision'] = float(precision_score(y, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y, y_pred, average='weighted', zero_division=0))
            except:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
            
            logger.info(f"XGBoost fine-tuning completed: accuracy {accuracy_before:.3f} -> {accuracy_after:.3f} (improvement: {actual_improvement:.3f})")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fine-tuning XGBoost model: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'accuracy_improvement': 0.0,
                'samples_processed': 0,
                'error': str(e)
            }

class SklearnAdapter(LocalModelAdapter):
    """Adapter for scikit-learn models (Random Forest, etc.)."""
    
    def load_model(self) -> bool:
        """Load sklearn model."""
        try:
            model_path = self.model_config['path'] / self.model_config['model_file']
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.load_preprocessors()
                self.is_loaded = True
                logger.info(f"Loaded sklearn model from {model_path}")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading sklearn model: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with sklearn model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # Keep DataFrame if it has feature names that match the model
            if hasattr(X_processed, 'columns') and hasattr(self.model, 'feature_names_in_'):
                # Check if the DataFrame has the expected feature names
                expected_features = self.model.feature_names_in_
                if all(col in X_processed.columns for col in expected_features):
                    # Use DataFrame directly to preserve feature names
                    return self.model.predict(X_processed[expected_features])
            
            # Convert to numpy array if feature names don't match or aren't available
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
            
            return self.model.predict(X_processed)
        except Exception as e:
            logger.error(f"Error in sklearn prediction: {e}")
            # Return default predictions if error occurs
            return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # Keep DataFrame if it has feature names that match the model
            if hasattr(X_processed, 'columns') and hasattr(self.model, 'feature_names_in_'):
                # Check if the DataFrame has the expected feature names
                expected_features = self.model.feature_names_in_
                if all(col in X_processed.columns for col in expected_features):
                    # Use DataFrame directly to preserve feature names
                    return self.model.predict_proba(X_processed[expected_features])
            
            # Convert to numpy array if feature names don't match or aren't available
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
            
            return self.model.predict_proba(X_processed)
        except Exception as e:
            logger.error(f"Error in sklearn predict_proba: {e}")
            # Return default probabilities if error occurs
            n_samples = X.shape[0]
            return np.ones((n_samples, 2)) * 0.5
    
    def get_model_weights(self) -> Dict[str, Any]:
        """Extract sklearn model parameters."""
        if not self.is_loaded:
            return {}
        
        try:
            weights = {
                'n_features': int(self.model.n_features_in_) if hasattr(self.model, 'n_features_in_') else 0,
                'n_classes': int(len(self.model.classes_)) if hasattr(self.model, 'classes_') else 0
            }
            
            # Extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                weights['feature_importance'] = [float(x) for x in self.model.feature_importances_]
            
            # For Random Forest, extract tree information
            if hasattr(self.model, 'estimators_'):
                weights['n_estimators'] = int(len(self.model.estimators_))
            
            # Add model parameters (convert numpy types to Python types)
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                json_safe_params = {}
                for key, value in params.items():
                    if isinstance(value, (np.integer, np.int32, np.int64)):
                        json_safe_params[key] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        json_safe_params[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        json_safe_params[key] = value.tolist()
                    else:
                        json_safe_params[key] = value
                weights['model_params'] = json_safe_params
            
            return weights
            
        except Exception as e:
            logger.error(f"Error extracting sklearn weights: {e}")
            return {}
    
    def update_model_weights(self, weights: Dict[str, Any]) -> bool:
        """Update sklearn model (limited capability)."""
        # Most sklearn models don't support direct weight updates
        logger.warning("Sklearn direct weight updates not implemented")
        return False
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, global_knowledge: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, float]:
        """Fine-tune sklearn model with new data and global knowledge."""
        try:
            X_processed = self.preprocess_data(X)
            
            # Convert to numpy array if it's a DataFrame
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
            
            # Get baseline performance
            y_pred_before = self.predict(X)
            accuracy_before = accuracy_score(y, y_pred_before)
            
            # Strategy 1: For models that support partial_fit (incremental learning)
            if hasattr(self.model, 'partial_fit'):
                logger.info(f"Using partial_fit for incremental learning")
                
                # If we have global knowledge, use it for knowledge distillation
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    global_prob = global_knowledge.get('global_probabilities', None)
                    
                    # Create mixed training data for knowledge transfer
                    mixed_labels = y.copy()
                    distill_ratio = kwargs.get('distill_ratio', 0.4)  # 40% for sklearn models
                    n_distill = int(len(y) * distill_ratio)
                    
                    if n_distill > 0 and len(global_pred) >= len(y):
                        distill_indices = np.random.choice(len(y), n_distill, replace=False)
                        
                        # Use global predictions for knowledge transfer
                        if global_prob is not None and len(global_prob) >= len(y):
                            # Use confidence-based selection
                            for idx in distill_indices:
                                global_confidence = np.max(global_prob[idx])
                                if global_confidence > 0.75:  # Lower threshold for sklearn
                                    mixed_labels[idx] = global_pred[idx]
                        else:
                            mixed_labels[distill_indices] = global_pred[distill_indices]
                        
                        logger.info(f"Applied global knowledge distillation to {n_distill} samples")
                    
                    # Apply partial fit with mixed data first (knowledge transfer)
                    unique_classes = np.unique(np.concatenate([y, mixed_labels]))
                    if hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                        self.model.partial_fit(X_processed, mixed_labels)
                    else:
                        self.model.partial_fit(X_processed, mixed_labels, classes=unique_classes)
                    
                    # Then apply partial fit with original data
                    self.model.partial_fit(X_processed, y)
                    
                    logger.info(f"Applied global knowledge transfer via partial_fit")
                
                else:
                    # Standard partial fit without global knowledge
                    unique_classes = np.unique(y)
                    if hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                        self.model.partial_fit(X_processed, y)
                    else:
                        all_classes = np.arange(10)  # Assuming 10 classes for network traffic
                        self.model.partial_fit(X_processed, y, classes=all_classes)
                
                logger.info(f"Applied partial_fit with {len(X)} samples")
                
            # Strategy 2: For tree-based models, implement ensemble-based improvement
            elif hasattr(self.model, 'estimators_'):
                logger.info(f"Applying ensemble-based learning for tree model")
                
                # For ensemble models like Random Forest, we can implement knowledge transfer
                # by creating additional training data based on global knowledge
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    
                    # Create augmented training set
                    if len(global_pred) >= len(y):
                        # Use global predictions to create pseudo-labeled data
                        # This simulates learning from global model knowledge
                        X_augmented = np.vstack([X_processed, X_processed])
                        y_augmented = np.hstack([y, global_pred[:len(y)]])
                        
                        # Retrain with augmented data (knowledge transfer)
                        try:
                            # For Random Forest, we can't easily update, but we can retrain
                            # with the augmented dataset to incorporate global knowledge
                            self.model.fit(X_augmented, y_augmented)
                            logger.info(f"Retrained ensemble model with global knowledge")
                        except Exception as e:
                            logger.warning(f"Failed to retrain with global knowledge: {e}")
                
                # Evaluate improvement potential based on data characteristics
                data_quality_score = min(1.0, len(X) / 1000)  # More data = better quality
                improvement_factor = 0.02 * data_quality_score  # Realistic improvement
                
                logger.info(f"Applied ensemble learning with improvement factor: {improvement_factor:.3f}")
                
            # Strategy 3: For other models, evaluate and apply knowledge if available
            else:
                logger.info(f"Model doesn't support incremental learning - applying knowledge transfer")
                
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    
                    # For models that don't support incremental learning,
                    # we can still benefit from global knowledge by retraining
                    # with a combination of original and global-guided data
                    if len(global_pred) >= len(y):
                        try:
                            # Create mixed dataset for retraining
                            X_mixed = np.vstack([X_processed, X_processed])
                            y_mixed = np.hstack([y, global_pred[:len(y)]])
                            
                            # Retrain the model with mixed data
                            self.model.fit(X_mixed, y_mixed)
                            logger.info(f"Retrained model with global knowledge")
                        except Exception as e:
                            logger.warning(f"Failed to retrain with global knowledge: {e}")
            
            # Evaluate final performance
            y_pred = self.predict(X)
            accuracy_after = accuracy_score(y, y_pred)
            
            # Calculate actual improvement
            actual_improvement = accuracy_after - accuracy_before
            
            # Calculate comprehensive metrics
            metrics = {
                'accuracy': float(accuracy_after),
                'f1_score': float(f1_score(y, y_pred, average='weighted')),
                'accuracy_improvement': float(actual_improvement),
                'samples_processed': int(len(X)),
                'learning_method': 'partial_fit' if hasattr(self.model, 'partial_fit') else 'retrain',
                'global_knowledge_used': global_knowledge is not None
            }
            
            # Add precision and recall if possible
            try:
                from sklearn.metrics import precision_score, recall_score
                metrics['precision'] = float(precision_score(y, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y, y_pred, average='weighted', zero_division=0))
            except:
                pass
            
            logger.info(f"Sklearn fine-tuning completed: accuracy {accuracy_before:.3f} -> {accuracy_after:.3f} (improvement: {actual_improvement:.3f})")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fine-tuning sklearn model: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'accuracy_improvement': 0.0,
                'samples_processed': 0,
                'error': str(e)
            }

class CatBoostAdapter(LocalModelAdapter):
    """Adapter for CatBoost models."""
    
    def load_model(self) -> bool:
        """Load CatBoost model."""
        try:
            model_path = self.model_config['path'] / self.model_config['model_file']
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.load_preprocessors()
                self.is_loaded = True
                logger.info(f"Loaded CatBoost model from {model_path}")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading CatBoost model: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with CatBoost model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # Handle ensemble models (VotingClassifier) and single CatBoost models
            if hasattr(self.model, 'estimators_'):
                # This is an ensemble model, check if estimators have feature names
                if hasattr(X_processed, 'columns'):
                    # Try to use DataFrame directly for ensemble
                    return self.model.predict(X_processed)
                else:
                    # Convert to numpy array
                    if hasattr(X_processed, 'values'):
                        X_processed = X_processed.values
                    return self.model.predict(X_processed)
            else:
                # Single CatBoost model
                if hasattr(X_processed, 'columns') and hasattr(self.model, 'feature_names_'):
                    # Check if the DataFrame has compatible feature names
                    return self.model.predict(X_processed)
                else:
                    # Convert to numpy array if feature names don't match
                    if hasattr(X_processed, 'values'):
                        X_processed = X_processed.values
                    return self.model.predict(X_processed)
        except Exception as e:
            logger.error(f"Error in CatBoost prediction: {e}")
            # Return default predictions if error occurs
            return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            X_processed = self.preprocess_data(X)
            
            # Handle ensemble models (VotingClassifier) and single CatBoost models
            if hasattr(self.model, 'estimators_'):
                # This is an ensemble model, check if estimators have feature names
                if hasattr(X_processed, 'columns'):
                    # Try to use DataFrame directly for ensemble
                    return self.model.predict_proba(X_processed)
                else:
                    # Convert to numpy array
                    if hasattr(X_processed, 'values'):
                        X_processed = X_processed.values
                    return self.model.predict_proba(X_processed)
            else:
                # Single CatBoost model
                if hasattr(X_processed, 'columns') and hasattr(self.model, 'feature_names_'):
                    # Check if the DataFrame has compatible feature names
                    return self.model.predict_proba(X_processed)
                else:
                    # Convert to numpy array if feature names don't match
                    if hasattr(X_processed, 'values'):
                        X_processed = X_processed.values
                    return self.model.predict_proba(X_processed)
        except Exception as e:
            logger.error(f"Error in CatBoost predict_proba: {e}")
            # Return default probabilities if error occurs
            n_samples = X.shape[0]
            return np.ones((n_samples, 2)) * 0.5
    
    def get_model_weights(self) -> Dict[str, Any]:
        """Extract CatBoost model parameters."""
        if not self.is_loaded:
            return {}
        
        try:
            weights = {}
            
            # Handle ensemble models (VotingClassifier)
            if hasattr(self.model, 'estimators_'):
                weights['model_type'] = 'ensemble'
                weights['n_estimators'] = int(len(self.model.estimators_))
                
                # Get info from individual estimators
                estimator_info = []
                for i, estimator in enumerate(self.model.estimators_):
                    if hasattr(estimator, 'tree_count_'):
                        estimator_info.append({
                            'estimator_id': i,
                            'tree_count': int(estimator.tree_count_)
                        })
                weights['estimators_info'] = estimator_info
                
            # Handle single CatBoost models
            elif hasattr(self.model, 'tree_count_'):
                weights['model_type'] = 'single'
                weights['tree_count'] = int(self.model.tree_count_)
                
                # Add feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    weights['feature_importance'] = [float(x) for x in self.model.feature_importances_]
            
            else:
                weights['model_type'] = 'unknown'
                weights['tree_count'] = 0
            
            # Add basic model parameters (avoid serializing the model object)
            if hasattr(self.model, 'get_params'):
                try:
                    params = self.model.get_params()
                    json_safe_params = {}
                    for key, value in params.items():
                        # Skip non-serializable objects
                        if key in ['estimators', 'estimators_']:
                            continue
                        if isinstance(value, (np.integer, np.int32, np.int64)):
                            json_safe_params[key] = int(value)
                        elif isinstance(value, (np.floating, np.float32, np.float64)):
                            json_safe_params[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            json_safe_params[key] = value.tolist()
                        elif isinstance(value, (str, int, float, bool, type(None))):
                            json_safe_params[key] = value
                        # Skip complex objects that can't be serialized
                    weights['model_params'] = json_safe_params
                except Exception as e:
                    logger.warning(f"Could not extract CatBoost parameters: {e}")
                    weights['model_params'] = {}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error extracting CatBoost weights: {e}")
            return {
                'model_type': 'error',
                'tree_count': 0,
                'error': str(e)
            }
    
    def update_model_weights(self, weights: Dict[str, Any]) -> bool:
        """Update CatBoost model (limited capability)."""
        logger.warning("CatBoost direct weight updates not implemented")
        return False
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, global_knowledge: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, float]:
        """Fine-tune CatBoost model with new data and global knowledge."""
        try:
            X_processed = self.preprocess_data(X)
            
            # Convert to numpy array if it's a DataFrame
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
            
            # Get baseline performance
            y_pred_before = self.predict(X)
            accuracy_before = accuracy_score(y, y_pred_before)
            
            # Check if this is an ensemble model (VotingClassifier)
            if hasattr(self.model, 'estimators_'):
                logger.info(f"CatBoost ensemble model: applying learning to {len(self.model.estimators_)} estimators")
                
                # For ensemble models, we can try to update individual estimators
                improvement_count = 0
                total_estimators = len(self.model.estimators_)
                
                # If we have global knowledge, use it for knowledge transfer
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    global_prob = global_knowledge.get('global_probabilities', None)
                    
                    # Create mixed training data for knowledge transfer
                    mixed_labels = y.copy()
                    distill_ratio = kwargs.get('distill_ratio', 0.35)  # 35% for CatBoost
                    n_distill = int(len(y) * distill_ratio)
                    
                    if n_distill > 0 and len(global_pred) >= len(y):
                        distill_indices = np.random.choice(len(y), n_distill, replace=False)
                        
                        # Use global predictions for knowledge transfer
                        if global_prob is not None and len(global_prob) >= len(y):
                            # Use confidence-based selection
                            for idx in distill_indices:
                                global_confidence = np.max(global_prob[idx])
                                if global_confidence > 0.8:  # High threshold for CatBoost
                                    mixed_labels[idx] = global_pred[idx]
                        else:
                            mixed_labels[distill_indices] = global_pred[distill_indices]
                        
                        logger.info(f"Applied global knowledge distillation to {n_distill} samples")
                    
                    # Try to retrain ensemble with global knowledge
                    try:
                        # Create augmented training set
                        X_augmented = np.vstack([X_processed, X_processed])
                        y_augmented = np.hstack([y, mixed_labels])
                        
                        # Retrain the entire ensemble with augmented data
                        self.model.fit(X_augmented, y_augmented)
                        improvement_count = total_estimators
                        logger.info(f"Retrained entire CatBoost ensemble with global knowledge")
                        
                    except Exception as e:
                        logger.warning(f"Failed to retrain ensemble with global knowledge: {e}")
                        
                        # Fallback: try to update individual estimators
                        for i, estimator in enumerate(self.model.estimators_):
                            try:
                                if hasattr(estimator, 'fit'):
                                    # For CatBoost models, retrain with mixed data
                                    if hasattr(estimator, 'tree_count_'):
                                        logger.info(f"Updating CatBoost estimator {i+1}/{total_estimators}")
                                        estimator.fit(X_augmented, y_augmented)
                                        improvement_count += 1
                                        
                            except Exception as e:
                                logger.warning(f"Failed to update estimator {i}: {e}")
                
                else:
                    # Standard ensemble training without global knowledge
                    try:
                        # Retrain with original data
                        self.model.fit(X_processed, y)
                        improvement_count = total_estimators
                        logger.info(f"Retrained CatBoost ensemble with new data")
                    except Exception as e:
                        logger.warning(f"Failed to retrain ensemble: {e}")
                
                logger.info(f"Updated {improvement_count}/{total_estimators} estimators")
                
            # Handle single CatBoost models
            elif hasattr(self.model, 'tree_count_'):
                logger.info(f"Single CatBoost model: applying incremental learning")
                
                try:
                    # For single CatBoost models, we can use incremental training
                    num_iterations = kwargs.get('iterations', 20)  # More iterations for better learning
                    
                    if global_knowledge and 'global_predictions' in global_knowledge:
                        global_pred = global_knowledge['global_predictions']
                        global_prob = global_knowledge.get('global_probabilities', None)
                        
                        # Create mixed training data for knowledge transfer
                        mixed_labels = y.copy()
                        distill_ratio = kwargs.get('distill_ratio', 0.35)
                        n_distill = int(len(y) * distill_ratio)
                        
                        if n_distill > 0 and len(global_pred) >= len(y):
                            distill_indices = np.random.choice(len(y), n_distill, replace=False)
                            
                            # Use global predictions for knowledge transfer
                            if global_prob is not None and len(global_prob) >= len(y):
                                for idx in distill_indices:
                                    global_confidence = np.max(global_prob[idx])
                                    if global_confidence > 0.8:
                                        mixed_labels[idx] = global_pred[idx]
                            else:
                                mixed_labels[distill_indices] = global_pred[distill_indices]
                            
                            logger.info(f"Applied global knowledge distillation to {n_distill} samples")
                        
                        # Retrain with mixed data for knowledge transfer
                        try:
                            X_augmented = np.vstack([X_processed, X_processed])
                            y_augmented = np.hstack([y, mixed_labels])
                            
                            # Use CatBoost's incremental training capabilities
                            self.model.fit(X_augmented, y_augmented, 
                                         init_model=self.model,  # Continue from current model
                                         verbose=False)
                            
                            logger.info(f"Applied global knowledge transfer with {num_iterations} iterations")
                            
                        except Exception as e:
                            logger.warning(f"Global knowledge transfer failed, using standard training: {e}")
                            # Fallback to standard training
                            self.model.fit(X_processed, y, verbose=False)
                    
                    else:
                        # Standard incremental training without global knowledge
                        try:
                            # Use CatBoost's incremental training
                            self.model.fit(X_processed, y, 
                                         init_model=self.model,  # Continue from current model
                                         verbose=False)
                            logger.info(f"Applied incremental training with {num_iterations} iterations")
                        except Exception as e:
                            logger.warning(f"Incremental training failed: {e}")
                            # Fallback to retraining
                            self.model.fit(X_processed, y, verbose=False)
                    
                except Exception as e:
                    logger.warning(f"CatBoost incremental training failed: {e}")
                    
            else:
                logger.info(f"Unknown CatBoost model type - using standard training")
                
                # For unknown model types, try standard retraining
                if global_knowledge and 'global_predictions' in global_knowledge:
                    global_pred = global_knowledge['global_predictions']
                    
                    if len(global_pred) >= len(y):
                        try:
                            # Create mixed dataset for retraining
                            X_mixed = np.vstack([X_processed, X_processed])
                            y_mixed = np.hstack([y, global_pred[:len(y)]])
                            
                            # Retrain the model with mixed data
                            self.model.fit(X_mixed, y_mixed)
                            logger.info(f"Retrained unknown CatBoost model with global knowledge")
                        except Exception as e:
                            logger.warning(f"Failed to retrain with global knowledge: {e}")
                            # Fallback to standard training
                            self.model.fit(X_processed, y)
                else:
                    # Standard training without global knowledge
                    self.model.fit(X_processed, y)
            
            # Evaluate final performance
            y_pred = self.predict(X)
            accuracy_after = accuracy_score(y, y_pred)
            
            # Calculate actual improvement
            actual_improvement = accuracy_after - accuracy_before
            
            # Calculate comprehensive metrics
            metrics = {
                'accuracy': float(accuracy_after),
                'f1_score': float(f1_score(y, y_pred, average='weighted')),
                'accuracy_improvement': float(actual_improvement),
                'samples_processed': int(len(X)),
                'model_type': 'ensemble' if hasattr(self.model, 'estimators_') else 'single',
                'training_time': 0.0,
                'global_knowledge_used': global_knowledge is not None
            }
            
            # Add precision and recall
            try:
                from sklearn.metrics import precision_score, recall_score
                metrics['precision'] = float(precision_score(y, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y, y_pred, average='weighted', zero_division=0))
            except:
                pass
            
            # Add ensemble-specific metrics
            if hasattr(self.model, 'estimators_'):
                metrics['n_estimators'] = len(self.model.estimators_)
                metrics['estimators_updated'] = improvement_count if 'improvement_count' in locals() else 0
            
            logger.info(f"CatBoost fine-tuning completed: accuracy {accuracy_before:.3f} -> {accuracy_after:.3f} (improvement: {actual_improvement:.3f})")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fine-tuning CatBoost model: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'accuracy_improvement': 0.0,
                'samples_processed': 0,
                'error': str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        info = {
            'type': self.model_config['type'],
            'path': str(self.model_config['path']),
            'is_loaded': self.is_loaded,
            'has_scaler': self.scaler is not None,
            'has_encoder': self.label_encoder is not None,
            'model_type': 'CatBoost',  # Always set to CatBoost for clarity
        }
        
        # Add feature count from scaler
        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
            info['n_features'] = int(self.scaler.n_features_in_)
        
        # Add class count from label encoder
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            info['n_classes'] = int(len(self.label_encoder.classes_))
        
        # Try to add more information about the model if available
        if self.is_loaded and self.model is not None:
            try:
                # Add model-specific info
                if hasattr(self.model, 'tree_count_'):
                    info['tree_count'] = int(self.model.tree_count_)
                
                # For ensemble models
                if hasattr(self.model, 'estimators_'):
                    info['n_estimators'] = int(len(self.model.estimators_))
                
                # Feature importance
                if hasattr(self.model, 'feature_importances_'):
                    info['has_feature_importance'] = True
                
                # Feature and class counts
                if hasattr(self.model, 'feature_names_'):
                    info['n_features'] = len(self.model.feature_names_)
                
                if hasattr(self.model, 'classes_'):
                    info['n_classes'] = int(len(self.model.classes_))
                
            except Exception as e:
                logger.warning(f"Error getting extended model info: {e}")
        
        # Default values if we couldn't get the information
        if 'n_features' not in info:
            info['n_features'] = 35  # Default from scaler
        
        if 'n_classes' not in info:
            info['n_classes'] = 10  # Default from label encoder
                
        return info

def create_model_adapter(model_config: Dict[str, Any]) -> LocalModelAdapter:
    """Factory function to create appropriate model adapter."""
    model_type = model_config.get('type', '').lower()
    
    if model_type == 'xgboost':
        return XGBoostAdapter(model_config)
    elif model_type == 'sklearn':
        return SklearnAdapter(model_config)
    elif model_type == 'catboost':
        return CatBoostAdapter(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 