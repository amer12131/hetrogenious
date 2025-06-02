"""
Utility file providing sample visualization data for the HETROFL system.
This ensures charts and tables are populated even without actual training data.
"""

import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

def generate_sample_accuracy_history() -> Dict[str, List[float]]:
    """Generate sample accuracy history for visualization."""
    # Generate 10 rounds of data with increasing accuracy trends
    models = ['global_mlp', 'xgboost', 'random_forest', 'catboost']
    
    # Start with base accuracies for each model
    base_accuracies = {
        'global_mlp': 0.78,
        'xgboost': 0.81,
        'random_forest': 0.79,
        'catboost': 0.77
    }
    
    # Generate history with improvement patterns
    accuracy_history = {}
    for model in models:
        base_acc = base_accuracies[model]
        # Create a slightly increasing trend with small variations
        history = []
        for i in range(10):
            # Improvement gets slower in later rounds
            improvement = (0.02 * (1 - 0.1 * i)) * (i + 1)
            # Add some random noise for realism
            noise = random.uniform(-0.01, 0.01)
            acc = min(base_acc + improvement + noise, 0.99)  # Cap at 0.99
            history.append(acc)
        accuracy_history[model] = history
    
    return accuracy_history

def generate_sample_loss_history() -> Dict[str, List[float]]:
    """Generate sample loss history for visualization."""
    # Generate 10 rounds of data with decreasing loss trends
    models = ['global_mlp', 'xgboost', 'random_forest', 'catboost']
    
    # Start with base losses for each model
    base_losses = {
        'global_mlp': 0.75,
        'xgboost': 0.65,
        'random_forest': 0.7,
        'catboost': 0.8
    }
    
    # Generate history with improvement patterns
    loss_history = {}
    for model in models:
        base_loss = base_losses[model]
        # Create a decreasing trend with variations
        history = []
        for i in range(10):
            # Loss decreases more rapidly at first, then slows down
            decrease = (0.07 * (1 - 0.08 * i)) * (i + 1)
            # Add some random noise for realism
            noise = random.uniform(-0.03, 0.03)
            loss = max(base_loss - decrease + noise, 0.1)  # Floor at 0.1
            history.append(loss)
        loss_history[model] = history
    
    return loss_history

def generate_sample_metrics_history() -> Dict[str, Any]:
    """Generate sample metrics history for all models."""
    models = ['global_mlp', 'xgboost', 'random_forest', 'catboost']
    start_time = datetime.now() - timedelta(hours=3)  # 3 hours ago
    
    history = {
        'global': [],
        'local': {}
    }
    
    # Initialize local model history
    for model in models[1:]:  # Skip global_mlp for local models
        history['local'][model] = []
    
    # Generate 10 rounds of metrics
    for i in range(10):
        # Global model metrics
        timestamp = start_time + timedelta(minutes=i*15)  # 15 min per round
        round_time = random.uniform(60, 180)  # 1-3 minutes training time
        
        if i == 0:
            # First round base metrics
            global_metrics = {
                'round': i + 1,
                'timestamp': timestamp.isoformat(),
                'accuracy': 0.78,
                'f1_score': 0.77,
                'precision': 0.79,
                'recall': 0.76,
                'loss': 0.75,
                'training_time': round_time
            }
        else:
            # Progressive improvement
            prev_metrics = history['global'][-1]
            improvement = 0.02 * (1 - 0.1 * i)  # Diminishing returns
            noise = random.uniform(-0.005, 0.005)
            
            global_metrics = {
                'round': i + 1,
                'timestamp': timestamp.isoformat(),
                'accuracy': min(prev_metrics['accuracy'] + improvement + noise, 0.99),
                'f1_score': min(prev_metrics['f1_score'] + improvement * 0.9 + noise, 0.99),
                'precision': min(prev_metrics['precision'] + improvement * 0.85 + noise, 0.99),
                'recall': min(prev_metrics['recall'] + improvement * 0.95 + noise, 0.99),
                'loss': max(prev_metrics['loss'] - improvement * 1.2 + noise, 0.1),
                'training_time': round_time
            }
        
        history['global'].append(global_metrics)
        
        # Local model metrics
        for model in models[1:]:
            # Slightly different characteristics per model
            if model == 'xgboost':
                base_acc = 0.81
                base_f1 = 0.8
                base_precision = 0.83
                base_recall = 0.79
                base_loss = 0.65
                model_factor = 1.1  # XGBoost improves faster
            elif model == 'random_forest':
                base_acc = 0.79
                base_f1 = 0.78
                base_precision = 0.8
                base_recall = 0.77
                base_loss = 0.7
                model_factor = 0.95
            else:  # catboost
                base_acc = 0.77
                base_f1 = 0.75
                base_precision = 0.78
                base_recall = 0.74
                base_loss = 0.8
                model_factor = 1.05
            
            if i == 0:
                # First round
                local_metrics = {
                    'round': i + 1,
                    'timestamp': timestamp.isoformat(),
                    'accuracy': base_acc,
                    'f1_score': base_f1,
                    'precision': base_precision,
                    'recall': base_recall,
                    'loss': base_loss,
                    'training_time': random.uniform(30, 90)  # Faster than global
                }
            else:
                # Progressive improvement
                prev_metrics = history['local'][model][-1]
                improvement = 0.02 * (1 - 0.1 * i) * model_factor
                noise = random.uniform(-0.01, 0.01)
                
                local_metrics = {
                    'round': i + 1,
                    'timestamp': timestamp.isoformat(),
                    'accuracy': min(prev_metrics['accuracy'] + improvement + noise, 0.99),
                    'f1_score': min(prev_metrics['f1_score'] + improvement * 0.9 + noise, 0.99),
                    'precision': min(prev_metrics['precision'] + improvement * 0.85 + noise, 0.99),
                    'recall': min(prev_metrics['recall'] + improvement * 0.95 + noise, 0.99),
                    'loss': max(prev_metrics['loss'] - improvement * 1.2 + noise, 0.1),
                    'training_time': random.uniform(30, 90)
                }
            
            history['local'][model].append(local_metrics)
    
    return history

def generate_sample_models_status() -> Dict[str, Any]:
    """Generate sample status for active models."""
    models = ['global_mlp', 'xgboost', 'random_forest', 'catboost']
    now = datetime.now()
    
    status = {}
    for model in models:
        # Randomize to make it realistic
        current_acc = random.uniform(0.85, 0.95)
        best_acc = random.uniform(current_acc, 0.97)
        improvement = random.uniform(0.02, 0.08)
        
        status[model] = {
            'is_training': random.choice([True, False]),
            'current_accuracy': current_acc,
            'best_accuracy': best_acc,
            'improvement': improvement,
            'last_update': (now - timedelta(minutes=random.randint(1, 30))).isoformat()
        }
    
    return status

def generate_sample_training_data() -> Dict[str, Any]:
    """Generate complete sample training data package."""
    return {
        'overall_progress': random.uniform(0.7, 0.9),
        'round_progress': random.uniform(0.5, 1.0),
        'best_accuracy': random.uniform(0.9, 0.95),
        'current_loss': random.uniform(0.1, 0.3),
        'models_status': generate_sample_models_status(),
        'accuracy_history': generate_sample_accuracy_history(),
        'loss_history': generate_sample_loss_history()
    } 