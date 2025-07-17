from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime, date
import os
from loguru import logger

from src.models.ensemble_model import LaLigaPredictor
from src.data.fetcher import DataFetcher
from src.utils.metrics import ModelMetrics

# Initialize FastAPI app
app = FastAPI(
    title="La Liga ML Prediction API",
    description="Machine Learning API for La Liga football match predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Pydantic models for API
class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    match_date: Optional[str] = None
    venue: Optional[str] = None

class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_home_goals: float
    predicted_away_goals: float
    confidence_score: float
    key_factors: List[str]

class TeamForm(BaseModel):
    team_name: str
    recent_form: List[str]  # W, D, L for last 5 matches
    goals_scored_avg: float
    goals_conceded_avg: float
    home_advantage: float

@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global predictor
    try:
        logger.info("Initializing La Liga ML Predictor...")
        predictor = LaLigaPredictor()
        
        # Check if model exists, if not train it
        if not predictor.is_model_trained():
            logger.info("No trained model found. Starting training process...")
            await train_initial_model()
        else:
            predictor.load_model()
            logger.info("Loaded existing trained model")
            
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        # Continue without predictor for now
        predictor = None

async def train_initial_model():
    """Train the initial model with historical data"""
    try:
        data_fetcher = DataFetcher()
        training_data = await data_fetcher.fetch_historical_data()
        predictor.train(training_data)
        predictor.save_model()
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "La Liga ML Prediction API",
        "status": "running",
        "model_loaded": predictor is not None and predictor.is_model_trained()
    }

@app.post("/api/predictions/match", response_model=MatchPrediction)
async def predict_match(request: MatchPredictionRequest):
    """Predict outcome for a specific match"""
    if not predictor or not predictor.is_model_trained():
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        prediction = predictor.predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            match_date=request.match_date,
            venue=request.venue
        )
        return prediction
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/round/{round_number}")
async def predict_round(round_number: int):
    """Predict all matches for a specific round"""
    if not predictor or not predictor.is_model_trained():
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        predictions = predictor.predict_round(round_number)
        return {"round": round_number, "predictions": predictions}
    except Exception as e:
        logger.error(f"Error predicting round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teams/form/{team_name}", response_model=TeamForm)
async def get_team_form(team_name: str):
    """Get current form and statistics for a team"""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        form_data = predictor.get_team_form(team_name)
        return form_data
    except Exception as e:
        logger.error(f"Error getting team form: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/standings/predicted")
async def get_predicted_standings():
    """Get predicted final league standings"""
    if not predictor or not predictor.is_model_trained():
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        standings = predictor.simulate_season()
        return {"predicted_standings": standings}
    except Exception as e:
        logger.error(f"Error simulating season: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/retrain")
async def retrain_model():
    """Retrain the model with latest data"""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        await train_initial_model()
        return {"message": "Model retrained successfully"}
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    if not predictor or not predictor.is_model_trained():
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        metrics = predictor.get_model_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
