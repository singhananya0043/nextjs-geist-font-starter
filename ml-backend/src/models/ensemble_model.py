import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple, Optional
from loguru import logger
import json
from datetime import datetime

from ..data.fetcher import DataFetcher
from ..data.preprocessor import LaLigaPreprocessor

class LaLigaPredictor:
    """Ensemble ML model for La Liga match predictions"""
    
    def __init__(self):
        self.outcome_model = None
        self.goals_model = None
        self.preprocessor = LaLigaPreprocessor()
        self.is_trained = False
        self.model_metrics = {}
        self.models_dir = "models"
        self.ensure_models_directory()
        
        # La Liga teams for validation
        self.laliga_teams = {
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Athletic Bilbao',
            'Real Sociedad', 'Real Betis', 'Villarreal', 'Valencia',
            'Getafe', 'Girona', 'Sevilla', 'Osasuna', 'Las Palmas',
            'Celta Vigo', 'Mallorca', 'Cadiz', 'Rayo Vallecano',
            'Alaves', 'Almeria', 'Granada'
        }
    
    def ensure_models_directory(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def train(self, training_data: pd.DataFrame):
        """Train the ensemble model"""
        logger.info("Starting model training...")
        
        try:
            # Preprocess the data
            features, outcome_target, goals_target = self.preprocessor.preprocess_training_data(training_data)
            
            # Split data for training and validation
            X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
                features, outcome_target, test_size=0.2, random_state=42, stratify=outcome_target
            )
            
            _, _, y_goals_train, y_goals_test = train_test_split(
                features, goals_target, test_size=0.2, random_state=42
            )
            
            # Train outcome prediction model (Win/Draw/Loss)
            logger.info("Training outcome prediction model...")
            self.outcome_model = self._train_outcome_model(X_train, y_outcome_train)
            
            # Train goals prediction model
            logger.info("Training goals prediction model...")
            self.goals_model = self._train_goals_model(X_train, y_goals_train)
            
            # Evaluate models
            self._evaluate_models(X_test, y_outcome_test, y_goals_test)
            
            self.is_trained = True
            logger.info("Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _train_outcome_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train ensemble model for match outcome prediction"""
        
        # Create ensemble of classifiers
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            
            logger.info(f"{name} CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Create ensemble with weighted voting based on CV scores
        ensemble_model = EnsembleClassifier(trained_models, model_scores)
        
        return ensemble_model
    
    def _train_goals_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Train model for goals prediction"""
        
        models = {}
        
        # Separate models for home goals, away goals, and total goals
        for target_col in ['home_goals', 'away_goals', 'total_goals']:
            logger.info(f"Training model for {target_col}...")
            
            if target_col == 'total_goals':
                # Use Poisson regression for total goals
                model = PoissonRegressor(alpha=1.0, max_iter=1000)
            else:
                # Use Random Forest for individual team goals
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42
                )
            
            model.fit(X_train, y_train[target_col])
            models[target_col] = model
        
        return models
    
    def _evaluate_models(self, X_test: pd.DataFrame, y_outcome_test: pd.Series, y_goals_test: pd.DataFrame):
        """Evaluate trained models"""
        logger.info("Evaluating models...")
        
        # Evaluate outcome model
        outcome_pred = self.outcome_model.predict(X_test)
        outcome_accuracy = accuracy_score(y_outcome_test, outcome_pred)
        
        # Evaluate goals models
        goals_metrics = {}
        for target_col in ['home_goals', 'away_goals', 'total_goals']:
            goals_pred = self.goals_model[target_col].predict(X_test)
            mse = mean_squared_error(y_goals_test[target_col], goals_pred)
            rmse = np.sqrt(mse)
            goals_metrics[target_col] = {'mse': mse, 'rmse': rmse}
        
        # Store metrics
        self.model_metrics = {
            'outcome_accuracy': outcome_accuracy,
            'goals_metrics': goals_metrics,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_test)
        }
        
        logger.info(f"Outcome prediction accuracy: {outcome_accuracy:.3f}")
        for target, metrics in goals_metrics.items():
            logger.info(f"{target} RMSE: {metrics['rmse']:.3f}")
    
    def predict_match(self, home_team: str, away_team: str, 
                     match_date: str = None, venue: str = None) -> Dict:
        """Predict outcome for a specific match"""
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Validate team names
        if home_team not in self.laliga_teams or away_team not in self.laliga_teams:
            logger.warning(f"Unknown team(s): {home_team}, {away_team}")
        
        try:
            # Preprocess match data
            match_features = self.preprocessor.preprocess_prediction_data(
                home_team, away_team, match_date
            )
            
            # Predict outcome probabilities
            outcome_probs = self.outcome_model.predict_proba(match_features)[0]
            
            # Predict goals
            home_goals_pred = max(0, self.goals_model['home_goals'].predict(match_features)[0])
            away_goals_pred = max(0, self.goals_model['away_goals'].predict(match_features)[0])
            total_goals_pred = max(0, self.goals_model['total_goals'].predict(match_features)[0])
            
            # Calculate confidence score based on prediction certainty
            confidence_score = max(outcome_probs) * 100
            
            # Generate key factors
            key_factors = self._generate_key_factors(home_team, away_team, match_features)
            
            prediction = {
                'home_team': home_team,
                'away_team': away_team,
                'away_win_probability': float(outcome_probs[0]),  # Class 0 = Away Win
                'draw_probability': float(outcome_probs[1]),      # Class 1 = Draw
                'home_win_probability': float(outcome_probs[2]),  # Class 2 = Home Win
                'predicted_home_goals': round(home_goals_pred, 1),
                'predicted_away_goals': round(away_goals_pred, 1),
                'predicted_total_goals': round(total_goals_pred, 1),
                'confidence_score': round(confidence_score, 1),
                'key_factors': key_factors,
                'prediction_date': datetime.now().isoformat()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            raise
    
    def _generate_key_factors(self, home_team: str, away_team: str, features: pd.DataFrame) -> List[str]:
        """Generate key factors influencing the prediction"""
        factors = []
        
        # Get feature values
        feature_row = features.iloc[0]
        
        # Home advantage
        factors.append("Home advantage for " + home_team)
        
        # Team form comparison
        home_recent_wins = feature_row.get('home_recent_wins', 0)
        away_recent_wins = feature_row.get('away_recent_wins', 0)
        
        if home_recent_wins > away_recent_wins:
            factors.append(f"{home_team} has better recent form")
        elif away_recent_wins > home_recent_wins:
            factors.append(f"{away_team} has better recent form")
        
        # Goal scoring ability
        home_goals_avg = feature_row.get('home_goals_per_match', 0)
        away_goals_avg = feature_row.get('away_goals_per_match', 0)
        
        if home_goals_avg > 1.5:
            factors.append(f"{home_team} strong attacking record at home")
        if away_goals_avg > 1.2:
            factors.append(f"{away_team} good away scoring record")
        
        # Head-to-head
        h2h_matches = feature_row.get('h2h_total_matches', 0)
        if h2h_matches > 0:
            factors.append("Historical head-to-head record considered")
        
        return factors[:4]  # Return top 4 factors
    
    def predict_round(self, round_number: int) -> List[Dict]:
        """Predict all matches for a specific round"""
        # This would fetch actual fixtures for the round
        # For demo, return sample predictions
        sample_matches = [
            ('Real Madrid', 'Barcelona'),
            ('Atletico Madrid', 'Sevilla'),
            ('Valencia', 'Athletic Bilbao')
        ]
        
        predictions = []
        for home_team, away_team in sample_matches:
            try:
                prediction = self.predict_match(home_team, away_team)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting {home_team} vs {away_team}: {e}")
                continue
        
        return predictions
    
    def get_team_form(self, team_name: str) -> Dict:
        """Get current form and statistics for a team"""
        if team_name not in self.laliga_teams:
            raise ValueError(f"Unknown team: {team_name}")
        
        # Get team stats from preprocessor
        team_stats = self.preprocessor.team_stats.get(team_name, {})
        
        # Generate recent form (mock data for demo)
        recent_form = ['W', 'D', 'W', 'L', 'W']  # Last 5 matches
        
        return {
            'team_name': team_name,
            'recent_form': recent_form,
            'goals_scored_avg': team_stats.get('goals_per_match', 1.2),
            'goals_conceded_avg': team_stats.get('goals_conceded_per_match', 1.1),
            'home_advantage': team_stats.get('home_win_rate', 0.5) - team_stats.get('away_win_rate', 0.3)
        }
    
    def simulate_season(self) -> List[Dict]:
        """Simulate remaining season and predict final standings"""
        # This would simulate all remaining matches
        # For demo, return mock standings
        mock_standings = [
            {'position': 1, 'team': 'Real Madrid', 'points': 85, 'predicted': True},
            {'position': 2, 'team': 'Barcelona', 'points': 82, 'predicted': True},
            {'position': 3, 'team': 'Atletico Madrid', 'points': 75, 'predicted': True},
            {'position': 4, 'team': 'Athletic Bilbao', 'points': 68, 'predicted': True},
        ]
        
        return mock_standings
    
    def save_model(self):
        """Save trained models to disk"""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        try:
            # Save outcome model
            joblib.dump(self.outcome_model, f"{self.models_dir}/outcome_model.pkl")
            
            # Save goals models
            joblib.dump(self.goals_model, f"{self.models_dir}/goals_model.pkl")
            
            # Save preprocessor
            joblib.dump(self.preprocessor, f"{self.models_dir}/preprocessor.pkl")
            
            # Save metrics
            with open(f"{self.models_dir}/metrics.json", 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_model(self):
        """Load trained models from disk"""
        try:
            # Check if model files exist
            outcome_path = f"{self.models_dir}/outcome_model.pkl"
            goals_path = f"{self.models_dir}/goals_model.pkl"
            preprocessor_path = f"{self.models_dir}/preprocessor.pkl"
            
            if not all(os.path.exists(path) for path in [outcome_path, goals_path, preprocessor_path]):
                logger.warning("Model files not found")
                return False
            
            # Load models
            self.outcome_model = joblib.load(outcome_path)
            self.goals_model = joblib.load(goals_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Load metrics if available
            metrics_path = f"{self.models_dir}/metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            
            self.is_trained = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def is_model_trained(self) -> bool:
        """Check if model is trained"""
        return self.is_trained
    
    def get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        return self.model_metrics


class EnsembleClassifier:
    """Ensemble classifier with weighted voting"""
    
    def __init__(self, models: Dict, scores: Dict):
        self.models = models
        self.weights = self._calculate_weights(scores)
    
    def _calculate_weights(self, scores: Dict) -> Dict:
        """Calculate weights based on model performance"""
        total_score = sum(scores.values())
        return {name: score / total_score for name, score in scores.items()}
    
    def predict(self, X):
        """Make predictions using weighted voting"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights[name]
            predictions.append(pred * weight)
        
        # Sum weighted predictions and take argmax
        ensemble_pred = np.sum(predictions, axis=0)
        return np.argmax(ensemble_pred, axis=1) if len(ensemble_pred.shape) > 1 else ensemble_pred
    
    def predict_proba(self, X):
        """Make probability predictions using weighted voting"""
        probabilities = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                weight = self.weights[name]
                probabilities.append(proba * weight)
        
        if not probabilities:
            raise ValueError("No models support probability prediction")
        
        # Sum weighted probabilities
        ensemble_proba = np.sum(probabilities, axis=0)
        
        # Normalize to ensure probabilities sum to 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba
