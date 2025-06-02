#!/usr/bin/env python3
"""
HETROFL System - Heterogeneous Federated Learning System
Main application runner

This script initializes and runs the complete HETROFL system with:
- Local model adapters for XGBoost, Random Forest, and CatBoost
- Global MLP model trained on the main dataset
- Federated learning coordinator with knowledge distillation
- Real-time web GUI with live visualizations
- Performance tracking and improvement monitoring

Usage:
    python run_hetrofl.py [--host HOST] [--port PORT] [--debug]

Author: HETROFL Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hetrofl_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask', 'flask_socketio', 'pandas', 'numpy', 'scikit-learn',
        'xgboost', 'catboost', 'tensorflow', 'plotly', 'matplotlib',
        'seaborn', 'dask', 'tqdm', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Special case for scikit-learn which is imported as sklearn
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n💡 Or install all requirements with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_data_files():
    """Check if required data files exist."""
    from hetrofl_system.config import MAIN_DATASET_PATH, LOCAL_MODELS
    
    # Check main dataset
    if not os.path.exists(MAIN_DATASET_PATH):
        print(f"❌ Main dataset not found: {MAIN_DATASET_PATH}")
        return False
    
    print(f"✅ Main dataset found: {MAIN_DATASET_PATH}")
    
    # Check local model files
    missing_models = []
    for model_name, config in LOCAL_MODELS.items():
        model_path = config['path'] / config['model_file']
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("⚠️  Some local model files are missing:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n💡 The system will still work, but these models won't be available for federated learning.")
    else:
        print("✅ All local model files found")
    
    return True

def create_directories():
    """Create necessary directories."""
    from hetrofl_system.config import MODELS_DIR, RESULTS_DIR, PLOTS_DIR
    
    directories = [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██╗  ██╗███████╗████████╗██████╗  ██████╗ ███████╗██╗                    ║
║    ██║  ██║██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██╔════╝██║                    ║
║    ███████║█████╗     ██║   ██████╔╝██║   ██║█████╗  ██║                    ║
║    ██╔══██║██╔══╝     ██║   ██╔══██╗██║   ██║██╔══╝  ██║                    ║
║    ██║  ██║███████╗   ██║   ██║  ██║╚██████╔╝██║     ███████╗               ║
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝               ║
║                                                                              ║
║              Heterogeneous Federated Learning System                        ║
║                           Version 1.0.0                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 Features:
   • Federated Learning with XGBoost, Random Forest, and CatBoost
   • Global MLP Model with Knowledge Distillation
   • Real-time Web GUI with Live Visualizations
   • Performance Tracking and Improvement Monitoring
   • Bidirectional Learning Loop with Model Updates

📊 System Components:
   • Local Model Adapters
   • Global Neural Network Model
   • Federated Learning Coordinator
   • Metrics Tracking and Visualization
   • Web-based Dashboard

"""
    print(banner)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='HETROFL - Heterogeneous Federated Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hetrofl.py                    # Run with default settings
  python run_hetrofl.py --host 0.0.0.0     # Allow external connections
  python run_hetrofl.py --port 8080        # Use custom port
  python run_hetrofl.py --debug            # Enable debug mode
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🔍 Checking system requirements...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    print("\n🚀 Starting HETROFL System...")
    
    try:
        # Import and run the Flask application
        from hetrofl_system.gui.app import app, socketio, initialize_system
        
        # Initialize the system
        print("⚙️  Initializing system components...")
        initialize_system()
        
        # Update GUI config
        from hetrofl_system.config import GUI_CONFIG
        GUI_CONFIG['host'] = args.host
        GUI_CONFIG['port'] = args.port
        GUI_CONFIG['debug'] = args.debug
        
        print(f"\n✅ HETROFL System is ready!")
        print(f"🌐 Web GUI available at: http://{args.host}:{args.port}")
        print(f"📊 Dashboard: http://{args.host}:{args.port}/")
        print(f"🧠 Models: http://{args.host}:{args.port}/models")
        print(f"🎯 Training: http://{args.host}:{args.port}/training")
        print(f"📈 Metrics: http://{args.host}:{args.port}/metrics")
        
        if not args.no_browser and args.host in ['127.0.0.1', 'localhost']:
            try:
                import webbrowser
                webbrowser.open(f"http://{args.host}:{args.port}")
                print("🌐 Opening browser...")
            except Exception:
                pass
        
        print("\n" + "="*80)
        print("🎉 HETROFL System is now running!")
        print("   Press Ctrl+C to stop the system")
        print("="*80 + "\n")
        
        # Run the Flask application
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down HETROFL System...")
        logger.info("System shutdown requested by user")
        
    except Exception as e:
        print(f"\n❌ Error starting HETROFL System: {e}")
        logger.error(f"System startup error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        print("👋 HETROFL System stopped. Goodbye!")

if __name__ == '__main__':
    main() 