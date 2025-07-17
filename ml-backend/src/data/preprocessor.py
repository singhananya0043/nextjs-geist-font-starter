import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger

class LaLigaPreprocessor:
    """Preprocesses La Liga data for ML training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.team_stats = {}
        self.feature_columns = []
        
    def preprocess_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess training data and create features
        
        Args:
            df: Raw match data DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Basic data cleaning
        data = self.clean_data(data)
        
        # Create team statistics
        self.team_stats = self.calculate_team_statistics(data)
        
        # Engineer features
        features_df = self.engineer_features(data)
        
        # Create target variables
        target_outcome = self.create_outcome_target(data)
        target_goals = self.create_goals_target(data)
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        logger.info(f"Preprocessing complete. Features shape: {features_df.shape}")
        
        return features_df, target_outcome, target_goals
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data"""
        logger.info("Cleaning data...")
        
        # Remove rows with missing essential data
        essential_columns = ['home_team', 'away_team', 'home_goals', 'away_goals', 'result']
        df = df.dropna(subset=essential_columns)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            'home_goals', 'away_goals', 'home_shots', 'away_shots',
            'home_shots_on_target', 'away_shots_on_target', 'home_possession',
            'home_corners', 'away_corners', 'home_fouls', 'away_fouls'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill missing possession data
        if 'away_possession' in df.columns:
            df['away_possession'] = 100 - df['home_possession']
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def calculate_team_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate team-level statistics"""
        logger.info("Calculating team statistics...")
        
        team_stats = {}
        teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        for team in teams:
            # Home matches
            home_matches = df[df['home_team'] == team]
            # Away matches
            away_matches = df[df['away_team'] == team]
            
            # Calculate statistics
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                continue
                
            # Goals statistics
            home_goals_scored = home_matches['home_goals'].sum()
            home_goals_conceded = home_matches['away_goals'].sum()
            away_goals_scored = away_matches['away_goals'].sum()
            away_goals_conceded = away_matches['home_goals'].sum()
            
            total_goals_scored = home_goals_scored + away_goals_scored
            total_goals_conceded = home_goals_conceded + away_goals_conceded
            
            # Win/Draw/Loss statistics
            home_wins = len(home_matches[home_matches['result'] == 'H'])
            home_draws = len(home_matches[home_matches['result'] == 'D'])
            home_losses = len(home_matches[home_matches['result'] == 'A'])
            
            away_wins = len(away_matches[away_matches['result'] == 'A'])
            away_draws = len(away_matches[away_matches['result'] == 'D'])
            away_losses = len(away_matches[away_matches['result'] == 'H'])
            
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            total_losses = home_losses + away_losses
            
            team_stats[team] = {
                'total_matches': total_matches,
                'goals_per_match': total_goals_scored / total_matches,
                'goals_conceded_per_match': total_goals_conceded / total_matches,
                'win_rate': total_wins / total_matches,
                'draw_rate': total_draws / total_matches,
                'loss_rate': total_losses / total_matches,
                'home_win_rate': home_wins / len(home_matches) if len(home_matches) > 0 else 0,
                'away_win_rate': away_wins / len(away_matches) if len(away_matches) > 0 else 0,
                'home_goals_per_match': home_goals_scored / len(home_matches) if len(home_matches) > 0 else 0,
                'away_goals_per_match': away_goals_scored / len(away_matches) if len(away_matches) > 0 else 0,
                'home_goals_conceded_per_match': home_goals_conceded / len(home_matches) if len(home_matches) > 0 else 0,
                'away_goals_conceded_per_match': away_goals_conceded / len(away_matches) if len(away_matches) > 0 else 0,
            }
        
        logger.info(f"Calculated statistics for {len(team_stats)} teams")
        return team_stats
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model"""
        logger.info("Engineering features...")
        
        features = []
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get team statistics
            home_stats = self.team_stats.get(home_team, {})
            away_stats = self.team_stats.get(away_team, {})
            
            # Basic team strength features
            feature_dict = {
                'home_goals_per_match': home_stats.get('home_goals_per_match', 0),
                'home_goals_conceded_per_match': home_stats.get('home_goals_conceded_per_match', 0),
                'away_goals_per_match': away_stats.get('away_goals_per_match', 0),
                'away_goals_conceded_per_match': away_stats.get('away_goals_conceded_per_match', 0),
                'home_win_rate': home_stats.get('home_win_rate', 0),
                'away_win_rate': away_stats.get('away_win_rate', 0),
                'home_overall_win_rate': home_stats.get('win_rate', 0),
                'away_overall_win_rate': away_stats.get('win_rate', 0),
            }
            
            # Strength difference features
            feature_dict['goal_difference'] = (
                home_stats.get('goals_per_match', 0) - home_stats.get('goals_conceded_per_match', 0)
            ) - (
                away_stats.get('goals_per_match', 0) - away_stats.get('goals_conceded_per_match', 0)
            )
            
            feature_dict['win_rate_difference'] = (
                home_stats.get('win_rate', 0) - away_stats.get('win_rate', 0)
            )
            
            # Home advantage
            feature_dict['home_advantage'] = 1  # Always 1 for home team
            
            # Historical head-to-head (simplified)
            h2h_stats = self.get_head_to_head_stats(df, home_team, away_team, idx)
            feature_dict.update(h2h_stats)
            
            # Recent form (last 5 matches)
            home_form = self.get_recent_form(df, home_team, idx, 5)
            away_form = self.get_recent_form(df, away_team, idx, 5)
            
            feature_dict['home_recent_wins'] = home_form['wins']
            feature_dict['home_recent_draws'] = home_form['draws']
            feature_dict['home_recent_losses'] = home_form['losses']
            feature_dict['away_recent_wins'] = away_form['wins']
            feature_dict['away_recent_draws'] = away_form['draws']
            feature_dict['away_recent_losses'] = away_form['losses']
            
            # Match context features
            if 'round' in row:
                feature_dict['round'] = row['round']
                feature_dict['season_progress'] = row['round'] / 38  # Normalize round number
            
            # Team encoding (using strength as proxy)
            feature_dict['home_team_strength'] = home_stats.get('win_rate', 0.5)
            feature_dict['away_team_strength'] = away_stats.get('win_rate', 0.5)
            
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"Feature engineering complete. Features: {list(features_df.columns)}")
        return features_df
    
    def get_head_to_head_stats(self, df: pd.DataFrame, home_team: str, away_team: str, current_idx: int) -> Dict:
        """Get head-to-head statistics between two teams"""
        # Get matches before current match
        historical_df = df.iloc[:current_idx]
        
        # Find matches between these teams
        h2h_matches = historical_df[
            ((historical_df['home_team'] == home_team) & (historical_df['away_team'] == away_team)) |
            ((historical_df['home_team'] == away_team) & (historical_df['away_team'] == home_team))
        ]
        
        if len(h2h_matches) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_total_matches': 0,
                'h2h_avg_goals': 0
            }
        
        # Count results from home team's perspective
        home_wins = len(h2h_matches[
            ((h2h_matches['home_team'] == home_team) & (h2h_matches['result'] == 'H')) |
            ((h2h_matches['away_team'] == home_team) & (h2h_matches['result'] == 'A'))
        ])
        
        draws = len(h2h_matches[h2h_matches['result'] == 'D'])
        
        away_wins = len(h2h_matches) - home_wins - draws
        
        avg_goals = (h2h_matches['home_goals'] + h2h_matches['away_goals']).mean()
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_total_matches': len(h2h_matches),
            'h2h_avg_goals': avg_goals
        }
    
    def get_recent_form(self, df: pd.DataFrame, team: str, current_idx: int, num_matches: int = 5) -> Dict:
        """Get recent form for a team"""
        # Get matches before current match
        historical_df = df.iloc[:current_idx]
        
        # Find recent matches for this team
        team_matches = historical_df[
            (historical_df['home_team'] == team) | (historical_df['away_team'] == team)
        ].tail(num_matches)
        
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                if match['result'] == 'H':
                    wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    losses += 1
            else:  # away team
                if match['result'] == 'A':
                    wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    losses += 1
        
        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': wins * 3 + draws
        }
    
    def create_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for match outcome prediction"""
        # Map results to numeric values: Home Win = 2, Draw = 1, Away Win = 0
        outcome_mapping = {'H': 2, 'D': 1, 'A': 0}
        return df['result'].map(outcome_mapping)
    
    def create_goals_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for goals prediction"""
        goals_targets = pd.DataFrame({
            'home_goals': df['home_goals'],
            'away_goals': df['away_goals'],
            'total_goals': df['home_goals'] + df['away_goals']
        })
        return goals_targets
    
    def preprocess_prediction_data(self, home_team: str, away_team: str, 
                                 match_date: str = None) -> pd.DataFrame:
        """Preprocess data for making predictions"""
        # Create a single row DataFrame for prediction
        prediction_data = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date or pd.Timestamp.now(),
            'round': 1  # Default round
        }])
        
        # Engineer features using the same process
        features_df = self.engineer_features(prediction_data)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_columns]
        
        return features_df
    
    def get_feature_importance_names(self) -> List[str]:
        """Get human-readable feature names for importance analysis"""
        return self.feature_columns
