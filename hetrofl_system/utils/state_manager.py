import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemStateManager:
    """Manages system state persistence across sessions."""
    
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "system_state.json"
        self.training_state_file = self.state_dir / "training_state.json"
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default state
        self.system_state = {
            'initialized': False,
            'training': False,
            'error': None,
            'last_update': None,
            'active_models': 0,
            'current_round': 0,
            'total_rounds': 0
        }
        
        self.training_state = {
            'is_training': False,
            'current_round': 0,
            'max_rounds': 50,
            'start_time': None,
            'last_round_time': None,
            'sample_size': 50000,
            'convergence_threshold': 0.001
        }
        
        # Load existing state
        self._load_system_state()
        self._load_training_state()
    
    def _load_system_state(self):
        """Load system state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.system_state.update(loaded_state)
                logger.info(f"Loaded system state from {self.state_file}")
            else:
                logger.info("No existing system state found, using defaults")
        except Exception as e:
            logger.warning(f"Could not load system state: {e}, using defaults")
    
    def _load_training_state(self):
        """Load training state from file."""
        try:
            if self.training_state_file.exists():
                with open(self.training_state_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.training_state.update(loaded_state)
                logger.info(f"Loaded training state from {self.training_state_file}")
            else:
                logger.info("No existing training state found, using defaults")
        except Exception as e:
            logger.warning(f"Could not load training state: {e}, using defaults")
    
    def _save_system_state(self):
        """Save system state to file."""
        try:
            self.system_state['last_update'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.system_state, f, indent=2, default=str)
            logger.debug(f"Saved system state to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def _save_training_state(self):
        """Save training state to file."""
        try:
            with open(self.training_state_file, 'w') as f:
                json.dump(self.training_state, f, indent=2, default=str)
            logger.debug(f"Saved training state to {self.training_state_file}")
        except Exception as e:
            logger.error(f"Error saving training state: {e}")
    
    def update_system_status(self, **kwargs):
        """Update system status and save to file."""
        try:
            for key, value in kwargs.items():
                if key in self.system_state:
                    self.system_state[key] = value
            self._save_system_state()
            logger.debug(f"Updated system status: {kwargs}")
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    def update_training_status(self, **kwargs):
        """Update training status and save to file."""
        try:
            for key, value in kwargs.items():
                if key in self.training_state:
                    self.training_state[key] = value
            self._save_training_state()
            logger.debug(f"Updated training status: {kwargs}")
        except Exception as e:
            logger.error(f"Error updating training status: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return self.system_state.copy()
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return self.training_state.copy()
    
    def start_training(self, max_rounds: int = 50, sample_size: int = 50000):
        """Mark training as started."""
        self.update_training_status(
            is_training=True,
            start_time=datetime.now().isoformat(),
            max_rounds=max_rounds,
            sample_size=sample_size,
            current_round=0
        )
        self.update_system_status(training=True)
    
    def stop_training(self):
        """Mark training as stopped."""
        self.update_training_status(
            is_training=False,
            start_time=None
        )
        self.update_system_status(training=False)
    
    def update_training_round(self, round_num: int):
        """Update current training round."""
        self.update_training_status(
            current_round=round_num,
            last_round_time=datetime.now().isoformat()
        )
        self.update_system_status(current_round=round_num)
    
    def is_training(self) -> bool:
        """Check if system is currently training."""
        return self.training_state.get('is_training', False)
    
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self.system_state.get('initialized', False)
    
    def set_initialized(self, initialized: bool = True):
        """Set system initialization status."""
        self.update_system_status(initialized=initialized)
    
    def set_error(self, error: Optional[str]):
        """Set system error status."""
        self.update_system_status(error=error)
    
    def clear_error(self):
        """Clear system error status."""
        self.update_system_status(error=None)
    
    def reset_state(self):
        """Reset all state to defaults."""
        try:
            # Remove state files
            if self.state_file.exists():
                self.state_file.unlink()
            if self.training_state_file.exists():
                self.training_state_file.unlink()
            
            # Reset in-memory state
            self.system_state = {
                'initialized': False,
                'training': False,
                'error': None,
                'last_update': None,
                'active_models': 0,
                'current_round': 0,
                'total_rounds': 0
            }
            
            self.training_state = {
                'is_training': False,
                'current_round': 0,
                'max_rounds': 50,
                'start_time': None,
                'last_round_time': None,
                'sample_size': 50000,
                'convergence_threshold': 0.001
            }
            
            logger.info("System state reset to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting state: {e}") 