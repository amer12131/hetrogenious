import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import os
import pickle
from datetime import datetime
import threading
import queue

from .local_model_adapter import LocalModelAdapter, create_model_adapter
from .global_model import GlobalMLPModel
from ..utils.data_loader import DataLoader
from ..utils.metrics import MetricsTracker
from ..utils.visualization import PlotGenerator

logger = logging.getLogger(__name__)

class FederatedCoordinator:
    """Coordinates federated learning between local and global models."""
    
    def __init__(self, config: Dict[str, Any], local_models_config: Dict[str, Any],
                 global_model_config: Dict[str, Any], data_loader: DataLoader):
        self.config = config
        self.local_models_config = local_models_config
        self.global_model_config = global_model_config
        self.data_loader = data_loader
        
        # Initialize components
        self.local_models: Dict[str, LocalModelAdapter] = {}
        self.global_model: Optional[GlobalMLPModel] = None
        self.metrics_tracker: Optional[MetricsTracker] = None
        self.plot_generator: Optional[PlotGenerator] = None
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.training_thread = None
        self.stop_training = False
        
        # Results storage
        self.round_results = []
        self.aggregation_weights = {}
        
        # Initialize local models
        self._initialize_local_models()
        
    def _initialize_local_models(self):
        """Initialize and load local models."""
        try:
            for model_name, model_config in self.local_models_config.items():
                logger.info(f"Initializing local model: {model_name}")
                
                adapter = create_model_adapter(model_config)
                if adapter.load_model():
                    self.local_models[model_name] = adapter
                    logger.info(f"Successfully loaded {model_name}")
                else:
                    logger.warning(f"Failed to load {model_name}")
            
            logger.info(f"Initialized {len(self.local_models)} local models")
            
        except Exception as e:
            logger.error(f"Error initializing local models: {e}")
    
    def initialize_global_model(self, X_sample: np.ndarray, y_sample: np.ndarray) -> bool:
        """Initialize the global model based on data characteristics."""
        try:
            num_features = X_sample.shape[1]
            num_classes = len(np.unique(y_sample))
            
            logger.info(f"Initializing global model: {num_features} features, {num_classes} classes")
            
            self.global_model = GlobalMLPModel(
                config=self.global_model_config,
                num_features=num_features,
                num_classes=num_classes
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing global model: {e}")
            return False
    
    def train_global_model_initial(self, sample_size: int = 100000) -> Dict[str, Any]:
        """Train the global model initially on the main dataset."""
        try:
            logger.info("Loading main dataset for global model training...")
            
            # Load and preprocess data
            df = self.data_loader.load_data(sample_size=sample_size)
            if df is None:
                raise ValueError("Failed to load main dataset")
            
            X, y = self.data_loader.preprocess_data(df, fit_transformers=True)
            X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split(X, y)
            
            # Initialize global model
            if not self.initialize_global_model(X_train, y_train):
                raise ValueError("Failed to initialize global model")
            
            # Train global model
            logger.info("Training global model on main dataset...")
            save_path = os.path.join(self.config.get('save_dir', 'models'), 'global_model_initial.h5')
            
            metrics = self.global_model.train(
                X_train, y_train,
                X_test, y_test,
                save_path=save_path
            )
            
            # Evaluate on test set
            test_metrics = self.global_model.evaluate_on_test_set(X_test, y_test)
            metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
            
            logger.info(f"Global model initial training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training global model: {e}")
            return {}
    
    def evaluate_local_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all local models on test data."""
        results = {}
        
        for model_name, model_adapter in self.local_models.items():
            try:
                logger.info(f"Evaluating local model: {model_name}")
                
                # Make predictions
                y_pred = model_adapter.predict(X_test)
                y_prob = model_adapter.predict_proba(X_test)
                
                # Calculate metrics
                if self.metrics_tracker:
                    metrics = self.metrics_tracker.calculate_metrics(y_test, y_pred, y_prob)
                    results[model_name] = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {}
        
        return results
    
    def aggregate_model_knowledge(self) -> Dict[str, Any]:
        """Aggregate knowledge from local models using ensemble approach."""
        try:
            logger.info("Aggregating knowledge from local models...")
            
            # Get model weights/information
            local_weights = {}
            local_info = {}
            
            for model_name, model_adapter in self.local_models.items():
                weights = model_adapter.get_model_weights()
                info = model_adapter.get_model_info()
                
                local_weights[model_name] = weights
                local_info[model_name] = info
            
            # Calculate aggregation weights based on model performance
            # Use performance-based weighting instead of equal weights
            aggregation_weights = {}
            total_weight = 0
            
            # Get recent performance metrics for weighting
            if self.metrics_tracker:
                for model_name in self.local_models.keys():
                    try:
                        # Get latest metrics for this model
                        model_df = self.metrics_tracker.get_metrics_dataframe(model_name)
                        if not model_df.empty:
                            # Use latest accuracy as weight basis
                            latest_accuracy = model_df['accuracy'].iloc[-1]
                            # Add small base weight to prevent zero weights
                            weight = max(0.1, latest_accuracy)
                        else:
                            weight = 0.33  # Default equal weight
                    except:
                        weight = 0.33  # Fallback to equal weight
                    
                    aggregation_weights[model_name] = weight
                    total_weight += weight
            else:
                # Fallback to equal weights
                for model_name in self.local_models.keys():
                    weight = 1.0 / len(self.local_models)
                    aggregation_weights[model_name] = weight
                    total_weight += weight
            
            # Normalize weights
            for model_name in aggregation_weights:
                aggregation_weights[model_name] /= total_weight
            
            self.aggregation_weights = aggregation_weights
            
            aggregation_result = {
                'local_weights': local_weights,
                'local_info': local_info,
                'aggregation_weights': aggregation_weights,
                'num_models': len(self.local_models)
            }
            
            logger.info(f"Knowledge aggregation completed with performance-based weights: {aggregation_weights}")
            return aggregation_result
            
        except Exception as e:
            logger.error(f"Error aggregating model knowledge: {e}")
            return {}
    
    def create_ensemble_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble predictions from local models."""
        try:
            all_predictions = []
            all_probabilities = []
            
            for model_name, model_adapter in self.local_models.items():
                try:
                    pred = model_adapter.predict(X)
                    prob = model_adapter.predict_proba(X)
                    
                    all_predictions.append(pred)
                    all_probabilities.append(prob)
                    
                    logger.debug(f"Model {model_name}: pred shape {pred.shape}, prob shape {prob.shape}")
                    
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model_name}: {e}")
            
            if not all_predictions:
                raise ValueError("No valid predictions from local models")
            
            # Check and standardize probability shapes
            n_samples = X.shape[0]
            max_classes = max(prob.shape[1] for prob in all_probabilities)
            
            # Standardize all probability arrays to have the same number of classes
            standardized_probs = []
            for i, prob in enumerate(all_probabilities):
                if prob.shape[1] < max_classes:
                    # Pad with zeros for missing classes
                    padded_prob = np.zeros((n_samples, max_classes))
                    padded_prob[:, :prob.shape[1]] = prob
                    standardized_probs.append(padded_prob)
                    logger.debug(f"Padded probability array from shape {prob.shape} to {padded_prob.shape}")
                else:
                    standardized_probs.append(prob)
            
            # Weighted ensemble with standardized probabilities
            ensemble_prob = np.zeros((n_samples, max_classes))
            
            for i, (model_name, prob) in enumerate(zip(self.local_models.keys(), standardized_probs)):
                weight = self.aggregation_weights.get(model_name, 1.0 / len(standardized_probs))
                ensemble_prob += weight * prob
                logger.debug(f"Added {model_name} with weight {weight}, prob shape {prob.shape}")
            
            # Get final predictions
            ensemble_pred = np.argmax(ensemble_prob, axis=1)
            
            logger.info(f"Ensemble created: pred shape {ensemble_pred.shape}, prob shape {ensemble_prob.shape}")
            return ensemble_pred, ensemble_prob
            
        except Exception as e:
            logger.error(f"Error creating ensemble predictions: {e}")
            return np.array([]), np.array([])
    
    def knowledge_distillation_update(self, X_distill: np.ndarray, y_distill: np.ndarray) -> Dict[str, Any]:
        """Update global model using knowledge distillation from local models."""
        try:
            if self.global_model is None:
                raise ValueError("Global model not initialized")
            
            logger.info("Performing knowledge distillation update...")
            
            # Get ensemble predictions as soft targets
            ensemble_pred, ensemble_prob = self.create_ensemble_predictions(X_distill)
            
            if len(ensemble_pred) == 0:
                raise ValueError("Failed to create ensemble predictions")
            
            # Create soft targets using temperature scaling
            temperature = self.config.get('temperature', 3.0)
            alpha = self.config.get('knowledge_distillation_alpha', 0.7)
            
            # Use ensemble predictions as pseudo-labels for global model training
            # This creates a form of knowledge distillation
            logger.info(f"Training global model with {len(X_distill)} distillation samples")
            
            # Fine-tune global model with ensemble knowledge
            distill_metrics = self.global_model.fine_tune(
                X_distill, ensemble_pred,  # Use ensemble predictions as targets
                epochs=self.config.get('distillation_epochs', 3)
            )
            
            logger.info(f"Knowledge distillation completed: {distill_metrics}")
            return distill_metrics
            
        except Exception as e:
            logger.error(f"Error in knowledge distillation: {e}")
            return {}
    
    def update_local_models(self, X_update: np.ndarray, y_update: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Update local models with global knowledge and cross-model learning."""
        results = {}
        
        # Get global model predictions for knowledge transfer
        global_knowledge = None
        
        if self.global_model and self.global_model.is_trained:
            try:
                global_predictions = self.global_model.predict(X_update)
                global_probabilities = self.global_model.predict_proba(X_update)
                
                # Package global knowledge for transfer
                global_knowledge = {
                    'global_predictions': global_predictions,
                    'global_probabilities': global_probabilities,
                    'model_confidence': np.max(global_probabilities, axis=1).mean() if global_probabilities is not None else 0.0
                }
                
                logger.info(f"Prepared global knowledge for transfer: {len(global_predictions)} predictions, avg confidence: {global_knowledge['model_confidence']:.3f}")
                
            except Exception as e:
                logger.warning(f"Could not get global model predictions: {e}")
                global_knowledge = None
        
        # Get ensemble predictions from other models for cross-learning
        ensemble_pred, ensemble_prob = self.create_ensemble_predictions(X_update)
        
        for model_name, model_adapter in self.local_models.items():
            try:
                logger.info(f"Updating local model: {model_name}")
                
                # Strategy 1: Fine-tune with original data and global knowledge
                update_metrics = model_adapter.fine_tune(
                    X_update, y_update, 
                    global_knowledge=global_knowledge,
                    distill_ratio=0.3,  # Use 30% of data for knowledge distillation
                    num_boost_rounds=20  # More training rounds for better learning
                )
                
                # Strategy 2: Cross-model ensemble learning
                if len(ensemble_pred) > 0:
                    try:
                        # Create ensemble knowledge for cross-model learning
                        ensemble_knowledge = {
                            'global_predictions': ensemble_pred,
                            'global_probabilities': ensemble_prob,
                            'model_confidence': np.max(ensemble_prob, axis=1).mean() if ensemble_prob is not None else 0.0
                        }
                        
                        # Additional fine-tuning with ensemble knowledge
                        ensemble_metrics = model_adapter.fine_tune(
                            X_update, y_update,
                            global_knowledge=ensemble_knowledge,
                            distill_ratio=0.2,  # Lower ratio for ensemble learning
                            num_boost_rounds=10  # Fewer rounds for ensemble learning
                        )
                        
                        # Combine metrics from ensemble learning
                        for key, value in ensemble_metrics.items():
                            if key in update_metrics and key != 'accuracy':
                                update_metrics[f'ensemble_{key}'] = value
                            elif key == 'accuracy':
                                # Track ensemble accuracy separately
                                update_metrics['ensemble_accuracy'] = value
                                
                        logger.info(f"Applied ensemble learning to {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Ensemble learning failed for {model_name}: {e}")
                
                # Strategy 3: Additional global knowledge transfer if available
                if global_knowledge and 'global_predictions' in global_knowledge:
                    try:
                        # Apply additional knowledge transfer with different parameters
                        additional_metrics = model_adapter.fine_tune(
                            X_update, y_update,
                            global_knowledge=global_knowledge,
                            distill_ratio=0.4,  # Higher ratio for additional learning
                            num_boost_rounds=15
                        )
                        
                        # Track additional improvements
                        for key, value in additional_metrics.items():
                            if key == 'accuracy_improvement':
                                update_metrics['additional_improvement'] = value
                            elif key == 'accuracy':
                                update_metrics['final_accuracy'] = value
                                
                        logger.info(f"Applied additional global knowledge transfer to {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Additional global knowledge transfer failed for {model_name}: {e}")
                
                results[model_name] = update_metrics
                
            except Exception as e:
                logger.error(f"Error updating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def run_federated_round(self, round_num: int, X_test: np.ndarray, y_test: np.ndarray,
                           X_distill: Optional[np.ndarray] = None, y_distill: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run a single federated learning round."""
        try:
            logger.info(f"Starting federated learning round {round_num}")
            round_start_time = time.time()
            
            round_results = {
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'local_metrics': {},
                'global_metrics': {},
                'aggregation_info': {},
                'update_metrics': {}
            }
            
            # 1. Evaluate local models
            local_metrics = self.evaluate_local_models(X_test, y_test)
            round_results['local_metrics'] = local_metrics
            
            # Update metrics tracker
            if self.metrics_tracker:
                for model_name, metrics in local_metrics.items():
                    self.metrics_tracker.update_local_metrics(
                        model_name, metrics, round_num
                    )
            
            # 2. Aggregate knowledge from local models
            aggregation_info = self.aggregate_model_knowledge()
            round_results['aggregation_info'] = aggregation_info
            
            # 3. Evaluate global model
            if self.global_model and self.global_model.is_trained:
                global_metrics = self.global_model.evaluate_on_test_set(X_test, y_test)
                round_results['global_metrics'] = global_metrics
                
                if self.metrics_tracker:
                    self.metrics_tracker.update_global_metrics(
                        global_metrics, round_num
                    )
            
            # 4. Knowledge distillation (if data provided)
            if X_distill is not None and y_distill is not None:
                distill_metrics = self.knowledge_distillation_update(X_distill, y_distill)
                round_results['distillation_metrics'] = distill_metrics
            
            # 5. Update local models with global knowledge
            if X_distill is not None and y_distill is not None:
                update_metrics = self.update_local_models(X_distill, y_distill)
                round_results['update_metrics'] = update_metrics
            
            round_time = time.time() - round_start_time
            round_results['round_time'] = round_time
            
            logger.info(f"Federated round {round_num} completed in {round_time:.2f}s")
            
            return round_results
            
        except Exception as e:
            logger.error(f"Error in federated round {round_num}: {e}")
            return {'round': round_num, 'error': str(e)}
    
    def start_federated_training(self, X_test: np.ndarray, y_test: np.ndarray,
                                X_distill: Optional[np.ndarray] = None, y_distill: Optional[np.ndarray] = None,
                                metrics_tracker: Optional[MetricsTracker] = None,
                                plot_generator: Optional[PlotGenerator] = None) -> bool:
        """Start federated training in a separate thread."""
        try:
            if self.is_training:
                logger.warning("Federated training already in progress")
                return False
            
            self.metrics_tracker = metrics_tracker
            self.plot_generator = plot_generator
            self.stop_training = False
            self.is_training = True
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._federated_training_loop,
                args=(X_test, y_test, X_distill, y_distill)
            )
            self.training_thread.start()
            
            logger.info("Federated training started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting federated training: {e}")
            self.is_training = False
            return False
    
    def _federated_training_loop(self, X_test: np.ndarray, y_test: np.ndarray,
                                X_distill: Optional[np.ndarray], y_distill: Optional[np.ndarray]):
        """Main federated training loop."""
        try:
            num_rounds = self.config.get('num_rounds', 10)
            
            for round_num in range(1, num_rounds + 1):
                if self.stop_training:
                    logger.info("Training stopped by user")
                    break
                
                # Run federated round
                round_results = self.run_federated_round(
                    round_num, X_test, y_test, X_distill, y_distill
                )
                
                self.round_results.append(round_results)
                self.current_round = round_num
                
                # Save intermediate results
                self._save_round_results(round_num)
                
                # Generate plots
                if self.plot_generator and round_num % 5 == 0:  # Every 5 rounds
                    self._generate_round_plots(round_num)
                
                # Sleep between rounds
                time.sleep(1)
            
            logger.info("Federated training completed")
            
        except Exception as e:
            logger.error(f"Error in federated training loop: {e}")
        
        finally:
            self.is_training = False
    
    def stop_federated_training(self):
        """Stop federated training."""
        self.stop_training = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        self.is_training = False
        logger.info("Federated training stopped")
    
    def _save_round_results(self, round_num: int):
        """Save results for a specific round."""
        try:
            save_dir = self.config.get('save_dir', 'results')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save round results
            results_file = os.path.join(save_dir, f'round_{round_num}_results.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump(self.round_results[-1], f)
            
            # Save global model
            if self.global_model:
                model_file = os.path.join(save_dir, f'global_model_round_{round_num}.h5')
                self.global_model.save_model(model_file)
            
        except Exception as e:
            logger.error(f"Error saving round results: {e}")
    
    def _generate_round_plots(self, round_num: int):
        """Generate plots for the current round."""
        try:
            if not self.plot_generator or not self.metrics_tracker:
                return
            
            # Get metrics data
            global_df = self.metrics_tracker.get_metrics_dataframe()
            
            if not global_df.empty:
                # Create metrics plot
                metrics_data = {
                    'accuracy': global_df['accuracy'].tolist(),
                    'f1_score': global_df['f1_score'].tolist(),
                    'precision': global_df['precision'].tolist(),
                    'recall': global_df['recall'].tolist()
                }
                
                self.plot_generator.plot_metrics_over_rounds(
                    metrics_data,
                    title=f"Global Model Performance - Round {round_num}",
                    save_name=f"global_metrics_round_{round_num}"
                )
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'current_round': self.current_round,
            'total_rounds': self.config.get('num_rounds', 10),
            'num_local_models': len(self.local_models),
            'global_model_trained': self.global_model is not None and self.global_model.is_trained,
            'results_available': len(self.round_results)
        }
    
    def get_latest_results(self) -> Dict[str, Any]:
        """Get latest round results."""
        if self.round_results:
            return self.round_results[-1]
        return {}
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all round results."""
        return self.round_results.copy()
    
    def evaluate_models_on_balanced_data(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on balanced test data and generate comprehensive plots."""
        try:
            logger.info("Starting comprehensive model evaluation on balanced data...")
            
            # Get balanced and imbalanced test data
            X_balanced, y_balanced = self.data_loader.get_balanced_test_data(sample_size=5000)
            X_imbalanced, y_imbalanced = self.data_loader.get_imbalanced_test_data(sample_size=5000)
            
            if X_balanced is None or X_imbalanced is None:
                logger.error("Failed to load test data for evaluation")
                return {}
            
            logger.info(f"Loaded balanced test data: {X_balanced.shape[0]} samples")
            logger.info(f"Loaded imbalanced test data: {X_imbalanced.shape[0]} samples")
            
            all_results = {}
            
            # Evaluate each local model
            for model_name, model_adapter in self.local_models.items():
                try:
                    logger.info(f"Evaluating {model_name} on balanced and imbalanced data...")
                    
                    # Evaluate on balanced data
                    y_pred_balanced = model_adapter.predict(X_balanced)
                    y_prob_balanced = model_adapter.predict_proba(X_balanced)
                    balanced_metrics = self.metrics_tracker.calculate_metrics(
                        y_balanced, y_pred_balanced, y_prob_balanced
                    ) if self.metrics_tracker else {}
                    
                    # Evaluate on imbalanced data
                    y_pred_imbalanced = model_adapter.predict(X_imbalanced)
                    y_prob_imbalanced = model_adapter.predict_proba(X_imbalanced)
                    imbalanced_metrics = self.metrics_tracker.calculate_metrics(
                        y_imbalanced, y_pred_imbalanced, y_prob_imbalanced
                    ) if self.metrics_tracker else {}
                    
                    # Get training history from metrics tracker
                    training_history = []
                    if self.metrics_tracker:
                        df = self.metrics_tracker.get_metrics_dataframe(model_name)
                        if not df.empty:
                            training_history = df.to_dict('records')
                    
                    # Store results
                    all_results[model_name] = {
                        'balanced_metrics': balanced_metrics,
                        'imbalanced_metrics': imbalanced_metrics,
                        'training_history': training_history
                    }
                    
                    logger.info(f"{model_name} - Balanced accuracy: {balanced_metrics.get('accuracy', 0.0):.3f}")
                    logger.info(f"{model_name} - Imbalanced accuracy: {imbalanced_metrics.get('accuracy', 0.0):.3f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    all_results[model_name] = {
                        'balanced_metrics': {},
                        'imbalanced_metrics': {},
                        'training_history': []
                    }
            
            # Evaluate global model if available
            if self.global_model and self.global_model.is_trained:
                try:
                    logger.info("Evaluating global model on balanced and imbalanced data...")
                    
                    # Evaluate on balanced data
                    y_pred_balanced = self.global_model.predict(X_balanced)
                    y_prob_balanced = self.global_model.predict_proba(X_balanced)
                    balanced_metrics = self.metrics_tracker.calculate_metrics(
                        y_balanced, y_pred_balanced, y_prob_balanced
                    ) if self.metrics_tracker else {}
                    
                    # Evaluate on imbalanced data
                    y_pred_imbalanced = self.global_model.predict(X_imbalanced)
                    y_prob_imbalanced = self.global_model.predict_proba(X_imbalanced)
                    imbalanced_metrics = self.metrics_tracker.calculate_metrics(
                        y_imbalanced, y_pred_imbalanced, y_prob_imbalanced
                    ) if self.metrics_tracker else {}
                    
                    # Get training history from metrics tracker
                    training_history = []
                    if self.metrics_tracker:
                        df = self.metrics_tracker.get_metrics_dataframe()  # Global model
                        if not df.empty:
                            training_history = df.to_dict('records')
                    
                    # Store results
                    all_results['global_model'] = {
                        'balanced_metrics': balanced_metrics,
                        'imbalanced_metrics': imbalanced_metrics,
                        'training_history': training_history
                    }
                    
                    logger.info(f"Global model - Balanced accuracy: {balanced_metrics.get('accuracy', 0.0):.3f}")
                    logger.info(f"Global model - Imbalanced accuracy: {imbalanced_metrics.get('accuracy', 0.0):.3f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating global model: {e}")
                    all_results['global_model'] = {
                        'balanced_metrics': {},
                        'imbalanced_metrics': {},
                        'training_history': []
                    }
            
            # Generate comprehensive plots
            if self.plot_generator:
                self._generate_comprehensive_plots(all_results)
            
            logger.info(f"Completed evaluation of {len(all_results)} models")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive model evaluation: {e}")
            return {}
    
    def _generate_comprehensive_plots(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate comprehensive plots for all models."""
        try:
            logger.info("Generating comprehensive plots for all models...")
            
            # Generate plots for each model
            for model_name, results in all_results.items():
                try:
                    balanced_metrics = results.get('balanced_metrics', {})
                    imbalanced_metrics = results.get('imbalanced_metrics', {})
                    training_history = results.get('training_history', [])
                    
                    # Create model-specific plots
                    plots = self.plot_generator.create_model_evaluation_plots(
                        model_name=model_name,
                        balanced_metrics=balanced_metrics,
                        imbalanced_metrics=imbalanced_metrics,
                        training_history=training_history,
                        save_plots=True
                    )
                    
                    logger.info(f"Generated {len(plots)} plots for {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error generating plots for {model_name}: {e}")
            
            # Generate federated learning summary
            try:
                summary_fig = self.plot_generator.create_federated_learning_summary(
                    all_results, save_name="federated_learning_comprehensive_summary"
                )
                logger.info("Generated federated learning summary plot")
                
            except Exception as e:
                logger.error(f"Error generating federated learning summary: {e}")
            
            logger.info("Completed comprehensive plot generation")
            
        except Exception as e:
            logger.error(f"Error in comprehensive plot generation: {e}")
    
    def run_complete_evaluation_cycle(self) -> Dict[str, Any]:
        """Run a complete evaluation cycle including training and comprehensive evaluation."""
        try:
            logger.info("Starting complete federated learning evaluation cycle...")
            
            # Step 1: Train global model if not already trained
            if not self.global_model or not self.global_model.is_trained:
                logger.info("Training global model initially...")
                initial_metrics = self.train_global_model_initial(sample_size=50000)
                if not initial_metrics:
                    logger.error("Failed to train global model")
                    return {}
            
            # Step 2: Run federated training rounds
            logger.info("Running federated training rounds...")
            
            # Load training data
            df_train = self.data_loader.load_data(sample_size=20000)
            if df_train is None:
                logger.error("Failed to load training data")
                return {}
            
            X_train, y_train = self.data_loader.preprocess_data(df_train, fit_transformers=False)
            
            # Load test data for evaluation during training
            df_test = self.data_loader.load_data(sample_size=10000)
            if df_test is None:
                logger.error("Failed to load test data")
                return {}
            
            X_test, y_test = self.data_loader.preprocess_data(df_test, fit_transformers=False)
            
            # Run multiple federated rounds
            num_rounds = self.config.get('num_rounds', 10)
            round_results = []
            
            for round_num in range(1, num_rounds + 1):
                logger.info(f"Running federated round {round_num}/{num_rounds}")
                
                # Run federated round
                round_result = self.run_federated_round(
                    round_num, X_test, y_test, X_train, y_train
                )
                round_results.append(round_result)
                
                # Log progress
                if 'local_metrics' in round_result:
                    for model_name, metrics in round_result['local_metrics'].items():
                        accuracy = metrics.get('accuracy', 0.0)
                        logger.info(f"Round {round_num} - {model_name}: {accuracy:.3f} accuracy")
            
            # Step 3: Comprehensive evaluation on balanced data
            logger.info("Running comprehensive evaluation on balanced data...")
            evaluation_results = self.evaluate_models_on_balanced_data()
            
            # Step 4: Compile final results
            final_results = {
                'training_rounds': round_results,
                'evaluation_results': evaluation_results,
                'summary': {
                    'total_rounds': len(round_results),
                    'models_evaluated': len(evaluation_results),
                    'global_model_trained': self.global_model is not None and self.global_model.is_trained
                }
            }
            
            logger.info("Completed federated learning evaluation cycle")
            logger.info(f"Summary: {final_results['summary']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in complete evaluation cycle: {e}")
            return {} 