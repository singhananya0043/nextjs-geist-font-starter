# La Liga ML Prediction Backend

A comprehensive machine learning system for predicting La Liga football match outcomes using historical data and advanced ML techniques.

## 🚀 Features

- **Match Outcome Prediction**: Predict win/draw/loss probabilities
- **Goals Prediction**: Predict exact scorelines and total goals
- **Team Form Analysis**: Analyze recent performance and trends
- **Season Simulation**: Predict final league standings
- **Real-time API**: FastAPI-based REST API for predictions
- **Ensemble Models**: Combines Random Forest, XGBoost, and Logistic Regression
- **Comprehensive Metrics**: Detailed performance tracking and evaluation

## 📊 ML Models

### Outcome Prediction
- **Ensemble Classifier**: Combines multiple algorithms for robust predictions
- **Classes**: Home Win (2), Draw (1), Away Win (0)
- **Features**: Team form, head-to-head, home advantage, goal statistics

### Goals Prediction
- **Poisson Regression**: For total goals prediction
- **Random Forest**: For individual team goals
- **Features**: Attack/defense strength, recent form, venue factors

## 🏗️ Architecture

```
ml-backend/
├── main.py              # FastAPI application
├── startup.py           # Initialization script
├── requirements.txt     # Python dependencies
├── src/
│   ├── data/           # Data processing
│   │   ├── fetcher.py  # Data collection
│   │   └── preprocessor.py  # Feature engineering
│   ├── models/         # ML models
│   │   └── ensemble_model.py  # Main predictor
│   └── utils/          # Utilities
│       └── metrics.py  # Performance metrics
├── data/               # Data storage
│   ├── raw/           # Raw historical data
│   └── processed/     # Processed datasets
└── models/            # Trained model files
```

## 🛠️ Installation

1. **Install Python dependencies**:
```bash
cd ml-backend
pip install -r requirements.txt
```

2. **Initialize the system**:
```bash
python startup.py
```

3. **Start the API server**:
```bash
python main.py
```

## 📡 API Endpoints

### Health Check
```http
GET /
```

### Match Prediction
```http
POST /api/predictions/match
Content-Type: application/json

{
  "home_team": "Real Madrid",
  "away_team": "Barcelona",
  "match_date": "2024-12-15"
}
```

### Round Predictions
```http
GET /api/predictions/round/{round_number}
```

### Team Form
```http
GET /api/teams/form/{team_name}
```

### Predicted Standings
```http
GET /api/standings/predicted
```

### Model Metrics
```http
GET /api/model/metrics
```

## 🎯 Usage Examples

### Python Client
```python
import requests

# Predict a match
response = requests.post('http://localhost:8001/api/predictions/match', json={
    'home_team': 'Real Madrid',
    'away_team': 'Barcelona'
})
prediction = response.json()

print(f"Prediction: {prediction['home_win_probability']:.1%} home win")
```

### cURL
```bash
curl -X POST http://localhost:8001/api/predictions/match \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Real Madrid", "away_team": "Barcelona"}'
```

## 📈 Sample Response

```json
{
  "home_team": "Real Madrid",
  "away_team": "Barcelona",
  "home_win_probability": 0.45,
  "draw_probability": 0.25,
  "away_win_probability": 0.30,
  "predicted_home_goals": 2.1,
  "predicted_away_goals": 1.8,
  "confidence_score": 78.5,
  "key_factors": [
    "Home advantage for Real Madrid",
    "Real Madrid has better recent form",
    "Strong attacking record at home"
  ],
  "prediction_date": "2024-12-15T10:30:00"
}
```

## 🏆 Supported Teams

All 20 La Liga teams are supported:
- Real Madrid, Barcelona, Atletico Madrid
- Athletic Bilbao, Real Sociedad, Real Betis
- Villarreal, Valencia, Getafe, Girona
- Sevilla, Osasuna, Las Palmas, Celta Vigo
- Mallorca, Cadiz, Rayo Vallecano, Alaves
- Almeria, Granada

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH`: Path to trained models (default: ./models)
- `DATA_PATH`: Path to data directory (default: ./data)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Parameters
- Training data: Last 5 seasons
- Feature count: 25+ engineered features
- Model types: Random Forest, XGBoost, Logistic Regression
- Cross-validation: 5-fold stratified

## 📊 Performance Metrics

### Current Model Performance
- **Outcome Accuracy**: ~65-70%
- **Goals RMSE**: ~0.8 goals
- **Confidence Calibration**: Well-calibrated probabilities

### Metrics Tracked
- Classification accuracy
- Precision/Recall/F1 scores
- Goals prediction error
- Betting ROI simulation
- Confidence calibration

## 🔄 Model Updates

### Automatic Retraining
```bash
# Trigger model retraining
curl -X POST http://localhost:8001/api/model/retrain
```

### Manual Training
```python
from src.models.ensemble_model import LaLigaPredictor
from src.data.fetcher import DataFetcher

predictor = LaLigaPredictor()
data_fetcher = DataFetcher()
data = await data_fetcher.fetch_historical_data()
predictor.train(data)
predictor.save_model()
```

## 🐳 Docker Support

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["python", "main.py"]
```

## 🧪 Testing

### Run Tests
```bash
# Test model initialization
python startup.py

# Test API endpoints
curl http://localhost:8001/

# Test predictions
curl -X POST http://localhost:8001/api/predictions/match \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Real Madrid", "away_team": "Barcelona"}'
```

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8001/
```

### Model Metrics
```bash
curl http://localhost:8001/api/model/metrics
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**
   ```bash
   python startup.py  # This will train and save models
   ```

3. **Port Already in Use**
   ```bash
   # Change port in main.py
   uvicorn.run("main:app", host="0.0.0.0", port=8002)
   ```

4. **Memory Issues**
   - Reduce training data size
   - Use smaller model parameters
   - Increase system memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect data sources and API usage limits.

## 📞 Support

For issues or questions:
- Check the logs in the console
- Review the API documentation at `/docs`
- Ensure all dependencies are installed
- Verify model files exist in the `models/` directory
