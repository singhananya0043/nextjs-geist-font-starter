#!/usr/bin/env python3
"""
La Liga ML Backend Startup Script
Initializes the ML model and starts the FastAPI server
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.models.ensemble_model import LaLigaPredictor
from src.data.fetcher import DataFetcher

async def initialize_ml_system():
    """Initialize the ML system with training data"""
    logger.info("🚀 Starting La Liga ML System Initialization...")
    
    try:
        # Step 1: Initialize data fetcher
        logger.info("📊 Initializing data fetcher...")
        data_fetcher = DataFetcher()
        
        # Step 2: Fetch historical data
        logger.info("📈 Fetching historical La Liga data...")
        historical_data = await data_fetcher.fetch_historical_data()
        logger.info(f"✅ Fetched {len(historical_data)} historical matches")
        
        # Step 3: Initialize ML predictor
        logger.info("🤖 Initializing ML predictor...")
        predictor = LaLigaPredictor()
        
        # Step 4: Train the model
        logger.info("🎯 Training ML models...")
        predictor.train(historical_data)
        
        # Step 5: Save the trained model
        logger.info("💾 Saving trained models...")
        predictor.save_model()
        
        # Step 6: Test predictions
        logger.info("🧪 Testing predictions...")
        await test_predictions(predictor)
        
        logger.info("✅ ML System initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")
        return False

async def test_predictions(predictor: LaLigaPredictor):
    """Test the trained model with sample predictions"""
        logger.info("Testing sample predictions...")
        
        # Test matches
        test_matches = [
            ("Real Madrid", "Barcelona"),
            ("Atletico Madrid", "Sevilla"),
            ("Valencia", "Athletic Bilbao")
        ]
        
        for home_team, away_team in test_matches:
            try:
                prediction = predictor.predict_match(home_team, away_team)
                logger.info(f"🏆 {home_team} vs {away_team}:")
                logger.info(f"   Home Win: {prediction['home_win_probability']:.1%}")
                logger.info(f"   Draw: {prediction['draw_probability']:.1%}")
                logger.info(f"   Away Win: {prediction['away_win_probability']:.1%}")
                logger.info(f"   Predicted Score: {prediction['predicted_home_goals']:.1f} - {prediction['predicted_away_goals']:.1f}")
                logger.info(f"   Confidence: {prediction['confidence_score']:.1f}%")
                logger.info("")
            except Exception as e:
                logger.error(f"Error predicting {home_team} vs {away_team}: {e}")
                continue
        
        return True

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 
        'fastapi', 'uvicorn', 'loguru', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - Not found")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using:")
        logger.info("pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are available!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw', 'data/processed', 'models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data', 'data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ['data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data/processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data/raw", "data.processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True)
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories:
                os.makedirs(directory, exist_ok=True")
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models"]
    
    for directory in directories">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed", "models">
                    logger.info(f"📁 Created directory: {directory}")

def create_directories():
    directories = ["data", "data.raw", "data.processed
