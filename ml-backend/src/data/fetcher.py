import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
from loguru import logger

class DataFetcher:
    """Fetches La Liga historical data for ML training"""
    
    def __init__(self):
        self.base_url = "https://fbref.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data_dir = "data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            os.makedirs(f"{self.data_dir}/raw")
            os.makedirs(f"{self.data_dir}/processed")
    
    async def fetch_historical_data(self, seasons: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical La Liga data for multiple seasons
        
        Args:
            seasons: List of seasons in format ['2023-24', '2022-23', ...]
        
        Returns:
            DataFrame with historical match data
        """
        if seasons is None:
            # Default to last 5 seasons
            current_year = datetime.now().year
            seasons = [f"{year}-{str(year+1)[2:]}" for year in range(current_year-5, current_year)]
        
        all_matches = []
        
        for season in seasons:
            logger.info(f"Fetching data for season {season}")
            try:
                season_data = await self.fetch_season_data(season)
                if season_data is not None:
                    all_matches.append(season_data)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching season {season}: {e}")
                continue
        
        if not all_matches:
            # Return sample data if fetching fails
            return self.get_sample_data()
        
        # Combine all seasons
        combined_data = pd.concat(all_matches, ignore_index=True)
        
        # Save raw data
        combined_data.to_csv(f"{self.data_dir}/raw/laliga_historical.csv", index=False)
        logger.info(f"Saved {len(combined_data)} matches to historical data")
        
        return combined_data
    
    async def fetch_season_data(self, season: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific season
        
        Args:
            season: Season in format '2023-24'
        
        Returns:
            DataFrame with season match data
        """
        try:
            # Check if we have cached data
            cache_file = f"{self.data_dir}/raw/laliga_{season}.csv"
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data for season {season}")
                return pd.read_csv(cache_file)
            
            # For demo purposes, generate realistic sample data
            # In production, this would fetch from actual APIs
            season_matches = self.generate_season_data(season)
            
            # Cache the data
            season_matches.to_csv(cache_file, index=False)
            
            return season_matches
            
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
            return None
    
    def generate_season_data(self, season: str) -> pd.DataFrame:
        """
        Generate realistic La Liga season data for demonstration
        In production, this would be replaced with actual data fetching
        """
        import random
        import numpy as np
        
        # La Liga teams (2023-24 season)
        teams = [
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Athletic Bilbao',
            'Real Sociedad', 'Real Betis', 'Villarreal', 'Valencia',
            'Getafe', 'Girona', 'Sevilla', 'Osasuna', 'Las Palmas',
            'Celta Vigo', 'Mallorca', 'Cadiz', 'Rayo Vallecano',
            'Alaves', 'Almeria', 'Granada'
        ]
        
        matches = []
        match_id = 1
        
        # Generate round-robin matches (each team plays each other twice)
        for round_num in range(1, 39):  # 38 rounds in La Liga
            round_matches = []
            
            # Shuffle teams for each round
            shuffled_teams = teams.copy()
            random.shuffle(shuffled_teams)
            
            # Create matches for this round
            for i in range(0, len(shuffled_teams), 2):
                if i + 1 < len(shuffled_teams):
                    home_team = shuffled_teams[i]
                    away_team = shuffled_teams[i + 1]
                    
                    # Generate realistic match data
                    match_data = self.generate_match_data(
                        match_id, home_team, away_team, round_num, season
                    )
                    matches.append(match_data)
                    match_id += 1
        
        return pd.DataFrame(matches)
    
    def generate_match_data(self, match_id: int, home_team: str, away_team: str, 
                          round_num: int, season: str) -> Dict:
        """Generate realistic match data"""
        import random
        import numpy as np
        
        # Team strength ratings (simplified)
        team_strength = {
            'Real Madrid': 0.85, 'Barcelona': 0.82, 'Atletico Madrid': 0.78,
            'Athletic Bilbao': 0.65, 'Real Sociedad': 0.68, 'Real Betis': 0.62,
            'Villarreal': 0.70, 'Valencia': 0.60, 'Getafe': 0.55,
            'Girona': 0.58, 'Sevilla': 0.64, 'Osasuna': 0.52,
            'Las Palmas': 0.48, 'Celta Vigo': 0.50, 'Mallorca': 0.49,
            'Cadiz': 0.45, 'Rayo Vallecano': 0.53, 'Alaves': 0.47,
            'Almeria': 0.42, 'Granada': 0.44
        }
        
        home_strength = team_strength.get(home_team, 0.5)
        away_strength = team_strength.get(away_team, 0.5)
        
        # Home advantage
        home_advantage = 0.1
        adjusted_home_strength = min(0.95, home_strength + home_advantage)
        
        # Calculate probabilities
        total_strength = adjusted_home_strength + away_strength
        home_win_prob = adjusted_home_strength / total_strength
        
        # Generate match outcome
        rand = random.random()
        if rand < home_win_prob * 0.7:  # Home win
            home_goals = random.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
            away_goals = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            result = 'H'
        elif rand < home_win_prob * 0.7 + 0.25:  # Draw
            goals = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
            home_goals = away_goals = goals
            result = 'D'
        else:  # Away win
            away_goals = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            home_goals = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            result = 'A'
        
        # Generate additional features
        total_goals = home_goals + away_goals
        
        return {
            'match_id': match_id,
            'season': season,
            'round': round_num,
            'date': f"2024-{random.randint(8, 12):02d}-{random.randint(1, 28):02d}",
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result,
            'total_goals': total_goals,
            'home_shots': random.randint(8, 20),
            'away_shots': random.randint(6, 18),
            'home_shots_on_target': random.randint(3, 8),
            'away_shots_on_target': random.randint(2, 7),
            'home_possession': random.randint(35, 75),
            'away_possession': lambda x: 100 - x,
            'home_corners': random.randint(2, 12),
            'away_corners': random.randint(2, 10),
            'home_fouls': random.randint(8, 20),
            'away_fouls': random.randint(8, 18),
            'home_yellow_cards': random.randint(0, 5),
            'away_yellow_cards': random.randint(0, 4),
            'home_red_cards': random.choices([0, 1], weights=[0.9, 0.1])[0],
            'away_red_cards': random.choices([0, 1], weights=[0.9, 0.1])[0],
            'home_team_strength': home_strength,
            'away_team_strength': away_strength,
            'venue': f"{home_team} Stadium"
        }
    
    def get_sample_data(self) -> pd.DataFrame:
        """Return sample data if fetching fails"""
        logger.info("Using sample data for demonstration")
        
        sample_matches = []
        for i in range(100):  # Generate 100 sample matches
            match_data = self.generate_match_data(
                i + 1, 'Real Madrid', 'Barcelona', 1, '2023-24'
            )
            sample_matches.append(match_data)
        
        return pd.DataFrame(sample_matches)
    
    async def fetch_current_season_fixtures(self) -> pd.DataFrame:
        """Fetch upcoming fixtures for predictions"""
        # This would fetch current season fixtures
        # For demo, return sample upcoming matches
        upcoming_matches = [
            {
                'match_id': 'upcoming_1',
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'date': '2024-12-15',
                'round': 16,
                'venue': 'Santiago Bernabeu'
            },
            {
                'match_id': 'upcoming_2',
                'home_team': 'Atletico Madrid',
                'away_team': 'Sevilla',
                'date': '2024-12-16',
                'round': 16,
                'venue': 'Wanda Metropolitano'
            }
        ]
        
        return pd.DataFrame(upcoming_matches)
    
    def get_team_recent_form(self, team_name: str, num_matches: int = 5) -> List[str]:
        """Get recent form for a team (W/D/L)"""
        # This would fetch actual recent results
        # For demo, return random form
        import random
        return random.choices(['W', 'D', 'L'], weights=[0.4, 0.3, 0.3], k=num_matches)
