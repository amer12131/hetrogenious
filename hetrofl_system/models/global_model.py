import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
import os
import pickle
from typing import Dict, Any, Tuple, Optional, List
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class GlobalMLPModel:
    """Global MLP model for federated learning aggregation."""
    
    def __init__(self, config: Dict[str, Any], num_features: int, num_classes: int):
        self.config = config
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.history = []
        self.is_trained = False
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def build_model(self) -> tf.keras.Model:
        """Build the MLP architecture."""
        try:
            model = tf.keras.Sequential([
                layers.Input(shape=(self.num_features,)),
                layers.BatchNormalization(),
            ])
            
            # Add hidden layers
            for i, units in enumerate(self.config['hidden_layers']):
                model.add(layers.Dense(
                    units, 
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name=f'hidden_{i+1}'
                ))
                model.add(layers.Dropout(self.config['dropout_rate']))
                model.add(layers.BatchNormalization())
            
            # Output layer
            if self.num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid', name='output'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            logger.info(f"Built MLP model with {model.count_params()} parameters")
            logger.info(f"Model architecture:\n{model.summary()}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the global model."""
        try:
            start_time = time.time()
            
            # Build model if not exists
            if self.model is None:
                self.model = self.build_model()
            
            # Prepare validation data
            if X_val is None or y_val is None:
                validation_split = self.config['validation_split']
                validation_data = None
            else:
                validation_split = 0.0
                validation_data = (X_val, y_val)
            
            # Callbacks
            callback_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data or validation_split > 0 else 'loss',
                    patience=self.config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data or validation_split > 0 else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Add model checkpoint if save path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                callback_list.append(
                    callbacks.ModelCheckpoint(
                        save_path,
                        monitor='val_loss' if validation_data or validation_split > 0 else 'loss',
                        save_best_only=True,
                        verbose=1
                    )
                )
            
            # Train model
            logger.info("Starting global model training...")
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=validation_split,
                validation_data=validation_data,
                callbacks=callback_list,
                verbose=1
            )
            
            training_time = time.time() - start_time
            self.history.append(history.history)
            self.is_trained = True
            
            # Calculate final metrics
            if validation_data:
                val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
                y_pred = self.predict(X_val)
                y_prob = self.predict_proba(X_val)
            else:
                val_loss, val_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
                y_pred = self.predict(X_train)
                y_prob = self.predict_proba(X_train)
                y_val = y_train
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred, average='weighted'),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'loss': val_loss,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss'])
            }
            
            logger.info(f"Global model training completed in {training_time:.2f}s")
            logger.info(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training global model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if self.num_classes == 2:
            return np.column_stack([1 - probabilities, probabilities])
        else:
            return probabilities
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights for federated aggregation."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> bool:
        """Set model weights from federated aggregation."""
        try:
            if self.model is None:
                self.model = self.build_model()
            
            self.model.set_weights(weights)
            logger.info("Updated global model weights")
            return True
            
        except Exception as e:
            logger.error(f"Error setting model weights: {e}")
            return False
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> Dict[str, float]:
        """Fine-tune the model with new data."""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            start_time = time.time()
            
            # Reduce learning rate for fine-tuning
            current_lr = self.model.optimizer.learning_rate.numpy()
            self.model.optimizer.learning_rate = current_lr * 0.1
            
            # Fine-tune
            history = self.model.fit(
                X, y,
                batch_size=self.config['batch_size'] // 2,
                epochs=epochs,
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = self.predict(X)
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'f1_score': f1_score(y, y_pred, average='weighted'),
                'training_time': training_time
            }
            
            logger.info(f"Global model fine-tuning completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fine-tuning global model: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Use the newer Keras format instead of HDF5 to avoid warnings
            if filepath.endswith('.h5'):
                # Change extension to .keras for the new format
                keras_filepath = filepath.replace('.h5', '.keras')
                self.model.save(keras_filepath, save_format='keras')
                logger.info(f"Model saved in Keras format: {keras_filepath}")
            else:
                # If no extension or different extension, add .keras
                if not filepath.endswith('.keras'):
                    keras_filepath = filepath + '.keras'
                else:
                    keras_filepath = filepath
                self.model.save(keras_filepath, save_format='keras')
                logger.info(f"Model saved in Keras format: {keras_filepath}")
            
            # Save additional metadata
            metadata = {
                'config': self.config,
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'is_trained': self.is_trained,
                'history': self.history,
                'saved_at': datetime.now().isoformat()
            }
            
            # Update metadata path to match the actual saved file
            if filepath.endswith('.h5'):
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
            else:
                metadata_path = filepath.replace('.keras', '_metadata.pkl') if filepath.endswith('.keras') else filepath + '_metadata.pkl'
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Global model saved to {keras_filepath if 'keras_filepath' in locals() else filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            # Check for both .h5 and .keras formats
            keras_filepath = None
            if filepath.endswith('.h5'):
                # Check if there's a corresponding .keras file
                keras_filepath = filepath.replace('.h5', '.keras')
                if os.path.exists(keras_filepath):
                    filepath = keras_filepath
                    logger.info(f"Found newer Keras format, loading: {keras_filepath}")
                elif not os.path.exists(filepath):
                    logger.error(f"Model file not found: {filepath}")
                    return False
            elif not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            self.model = tf.keras.models.load_model(filepath)
            
            # Load metadata if available
            if filepath.endswith('.keras'):
                metadata_path = filepath.replace('.keras', '_metadata.pkl')
            elif filepath.endswith('.h5'):
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
            else:
                metadata_path = filepath + '_metadata.pkl'
                
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.config = metadata.get('config', self.config)
                self.num_features = metadata.get('num_features', self.num_features)
                self.num_classes = metadata.get('num_classes', self.num_classes)
                self.is_trained = metadata.get('is_trained', True)
                self.history = metadata.get('history', [])
            
            logger.info(f"Global model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and statistics."""
        if self.model is None:
            return {'status': 'not_built'}
        
        try:
            # Get basic model info
            total_params = self.model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            
            summary = {
                'status': 'trained' if self.is_trained else 'built',
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'num_layers': int(len(self.model.layers)),
                'num_features': int(self.num_features),
                'num_classes': int(self.num_classes),
                'architecture': [str(layer.name) for layer in self.model.layers],
                'training_history_length': int(len(self.history))
            }
            
            # Add configuration info
            if self.config:
                config_copy = {}
                for key, value in self.config.items():
                    if isinstance(value, (np.integer, np.int32, np.int64)):
                        config_copy[key] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        config_copy[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        config_copy[key] = value.tolist()
                    elif isinstance(value, list):
                        config_copy[key] = [int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else 
                                          float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else x 
                                          for x in value]
                    else:
                        config_copy[key] = value
                summary['config'] = config_copy
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'num_features': int(self.num_features) if self.num_features else 0,
                'num_classes': int(self.num_classes) if self.num_classes else 0
            }
    
    def create_knowledge_distillation_targets(self, X: np.ndarray, temperature: float = 3.0) -> np.ndarray:
        """Create soft targets for knowledge distillation."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get logits (before softmax)
        logits_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output  # Before final activation
        )
        
        logits = logits_model.predict(X, verbose=0)
        
        # Apply temperature scaling
        soft_targets = tf.nn.softmax(logits / temperature).numpy()
        
        return soft_targets
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive evaluation on test set."""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Get predictions
            y_pred = self.predict(X_test)
            y_prob = self.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Add loss
            loss = self.model.evaluate(X_test, y_test, verbose=0)
            if isinstance(loss, list):
                metrics['loss'] = loss[0]
            else:
                metrics['loss'] = loss
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {} 