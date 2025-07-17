import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
from datetime import datetime

class ModelMetrics:
    """Comprehensive metrics calculation and tracking for La Liga ML models"""
    
    def __init__(self):
        self.metrics_history = []
        self.outcome_classes = ['Away Win', 'Draw', 'Home Win']
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics for match outcome prediction
        
        Args:
            y_true: True labels (0=Away Win, 1=Draw, 2=Home Win)
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all classification metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for each class
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Store per-class metrics
        for i, class_name in enumerate(self.outcome_classes):
            metrics[f'{class_name.lower().replace(" ", "_")}_precision'] = precision[i]
            metrics[f'{class_name.lower().replace(" ", "_")}_recall'] = recall[i]
            metrics[f'{class_name.lower().replace(" ", "_")}_f1'] = f1[i]
        
        # Macro and weighted averages
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        metrics['class_distribution'] = class_distribution
        
        # Prediction confidence metrics (if probabilities provided)
        if y_proba is not None:
            metrics.update(self._calculate_confidence_metrics(y_true, y_pred, y_proba))
        
        return metrics
    
    def _calculate_confidence_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate confidence-based metrics"""
        confidence_metrics = {}
        
        # Average prediction confidence
        max_probas = np.max(y_proba, axis=1)
        confidence_metrics['avg_prediction_confidence'] = np.mean(max_probas)
        confidence_metrics['min_prediction_confidence'] = np.min(max_probas)
        confidence_metrics['max_prediction_confidence'] = np.max(max_probas)
        
        # Confidence calibration - accuracy by confidence bins
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (max_probas >= confidence_bins[i]) & (max_probas < confidence_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = accuracy_score(y_true[bin_mask], y_pred[bin_mask])
                bin_confidence = np.mean(max_probas[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        if bin_accuracies:
            confidence_metrics['calibration_error'] = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        
        return confidence_metrics
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   target_name: str = "goals") -> Dict[str, float]:
        """
        Calculate regression metrics for goals prediction
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of the target variable
            
        Returns:
            Dictionary containing regression metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics[f'{target_name}_mse'] = mean_squared_error(y_true, y_pred)
        metrics[f'{target_name}_rmse'] = np.sqrt(metrics[f'{target_name}_mse'])
        metrics[f'{target_name}_mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{target_name}_r2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        non_zero_mask = y_true != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics[f'{target_name}_mape'] = mape
        
        # Prediction accuracy within tolerance
        tolerance_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)
        tolerance_10 = np.mean(np.abs(y_true - y_pred) <= 1.0)
        
        metrics[f'{target_name}_accuracy_05'] = tolerance_05  # Within 0.5 goals
        metrics[f'{target_name}_accuracy_10'] = tolerance_10  # Within 1.0 goals
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics[f'{target_name}_residual_mean'] = np.mean(residuals)
        metrics[f'{target_name}_residual_std'] = np.std(residuals)
        
        return metrics
    
    def calculate_betting_metrics(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                odds: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate betting-related metrics (profitability, ROI, etc.)
        
        Args:
            y_true: True outcomes
            y_proba: Predicted probabilities
            odds: Betting odds (optional)
            
        Returns:
            Dictionary containing betting metrics
        """
        betting_metrics = {}
        
        if odds is None:
            # Generate synthetic odds based on probabilities
            odds = 1 / y_proba
        
        # Kelly Criterion optimal bet sizing
        kelly_bets = []
        for i in range(len(y_proba)):
            for j in range(len(y_proba[i])):
                prob = y_proba[i][j]
                odd = odds[i][j] if len(odds.shape) > 1 else odds[i]
                
                # Kelly formula: f = (bp - q) / b
                # where b = odds - 1, p = probability, q = 1 - p
                b = odd - 1
                if b > 0:
                    kelly_fraction = (b * prob - (1 - prob)) / b
                    kelly_bets.append(max(0, kelly_fraction))  # No negative bets
        
        if kelly_bets:
            betting_metrics['avg_kelly_bet_size'] = np.mean(kelly_bets)
            betting_metrics['max_kelly_bet_size'] = np.max(kelly_bets)
        
        # Value betting opportunities (when model probability > implied probability)
        if len(odds.shape) > 1:
            implied_probs = 1 / odds
            value_bets = y_proba > implied_probs
            betting_metrics['value_bet_percentage'] = np.mean(value_bets) * 100
        
        return betting_metrics
    
    def calculate_seasonal_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate season-level metrics and trends
        
        Args:
            predictions_df: DataFrame with columns ['date', 'actual', 'predicted', 'confidence']
            
        Returns:
            Dictionary containing seasonal metrics
        """
        seasonal_metrics = {}
        
        if 'date' in predictions_df.columns:
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df['month'] = predictions_df['date'].dt.month
            predictions_df['week'] = predictions_df['date'].dt.isocalendar().week
            
            # Monthly accuracy trends
            monthly_accuracy = predictions_df.groupby('month').apply(
                lambda x: accuracy_score(x['actual'], x['predicted'])
            ).to_dict()
            seasonal_metrics['monthly_accuracy'] = monthly_accuracy
            
            # Weekly accuracy trends
            weekly_accuracy = predictions_df.groupby('week').apply(
                lambda x: accuracy_score(x['actual'], x['predicted']) if len(x) > 0 else 0
            ).to_dict()
            seasonal_metrics['weekly_accuracy'] = weekly_accuracy
        
        # Home vs Away prediction accuracy
        if 'venue' in predictions_df.columns:
            home_accuracy = accuracy_score(
                predictions_df[predictions_df['venue'] == 'home']['actual'],
                predictions_df[predictions_df['venue'] == 'home']['predicted']
            )
            away_accuracy = accuracy_score(
                predictions_df[predictions_df['venue'] == 'away']['actual'],
                predictions_df[predictions_df['venue'] == 'away']['predicted']
            )
            seasonal_metrics['home_prediction_accuracy'] = home_accuracy
            seasonal_metrics['away_prediction_accuracy'] = away_accuracy
        
        return seasonal_metrics
    
    def generate_performance_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray = None, 
                                  goals_true: np.ndarray = None, 
                                  goals_pred: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            y_true: True outcome labels
            y_pred: Predicted outcome labels
            y_proba: Prediction probabilities
            goals_true: True goals (optional)
            goals_pred: Predicted goals (optional)
            
        Returns:
            Complete performance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true)
        }
        
        # Classification metrics
        classification_metrics = self.calculate_classification_metrics(y_true, y_pred, y_proba)
        report['classification'] = classification_metrics
        
        # Goals prediction metrics (if provided)
        if goals_true is not None and goals_pred is not None:
            if len(goals_true.shape) > 1:
                # Multiple goal targets (home, away, total)
                goals_metrics = {}
                target_names = ['home_goals', 'away_goals', 'total_goals']
                for i, target_name in enumerate(target_names):
                    if i < goals_true.shape[1]:
                        target_metrics = self.calculate_regression_metrics(
                            goals_true[:, i], goals_pred[:, i], target_name
                        )
                        goals_metrics.update(target_metrics)
            else:
                # Single goal target
                goals_metrics = self.calculate_regression_metrics(goals_true, goals_pred, 'goals')
            
            report['goals_prediction'] = goals_metrics
        
        # Betting metrics (if probabilities provided)
        if y_proba is not None:
            betting_metrics = self.calculate_betting_metrics(y_true, y_proba)
            report['betting'] = betting_metrics
        
        # Store in history
        self.metrics_history.append(report)
        
        return report
    
    def save_metrics_report(self, report: Dict[str, Any], filepath: str):
        """Save metrics report to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Metrics report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics report: {e}")
    
    def load_metrics_history(self, filepath: str) -> List[Dict[str, Any]]:
        """Load metrics history from JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.metrics_history = json.load(f)
            logger.info(f"Loaded {len(self.metrics_history)} metrics reports")
            return self.metrics_history
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            return []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics in history"""
        if not self.metrics_history:
            return {}
        
        latest_report = self.metrics_history[-1]
        
        summary = {
            'latest_accuracy': latest_report.get('classification', {}).get('accuracy', 0),
            'total_evaluations': len(self.metrics_history),
            'latest_evaluation_date': latest_report.get('timestamp'),
            'sample_size': latest_report.get('sample_size', 0)
        }
        
        # Add goals prediction summary if available
        if 'goals_prediction' in latest_report:
            goals_metrics = latest_report['goals_prediction']
            summary['goals_rmse'] = goals_metrics.get('goals_rmse', 0)
            summary['goals_accuracy_05'] = goals_metrics.get('goals_accuracy_05', 0)
        
        return summary
    
    def compare_models(self, model_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance of multiple models
        
        Args:
            model_reports: Dictionary mapping model names to their performance reports
            
        Returns:
            Comparison summary
        """
        comparison = {
            'model_count': len(model_reports),
            'comparison_date': datetime.now().isoformat(),
            'metrics_comparison': {}
        }
        
        # Compare key metrics
        key_metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        
        for metric in key_metrics:
            metric_values = {}
            for model_name, report in model_reports.items():
                classification_metrics = report.get('classification', {})
                metric_values[model_name] = classification_metrics.get(metric, 0)
            
            if metric_values:
                best_model = max(metric_values, key=metric_values.get)
                comparison['metrics_comparison'][metric] = {
                    'values': metric_values,
                    'best_model': best_model,
                    'best_value': metric_values[best_model]
                }
        
        return comparison


class PredictionTracker:
    """Track and analyze prediction performance over time"""
    
    def __init__(self):
        self.predictions = []
    
    def add_prediction(self, prediction: Dict[str, Any], actual_result: str = None):
        """Add a prediction to tracking"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual_result': actual_result,
            'is_correct': None
        }
        
        if actual_result is not None:
            predicted_outcome = self._get_predicted_outcome(prediction)
            prediction_record['is_correct'] = predicted_outcome == actual_result
        
        self.predictions.append(prediction_record)
    
    def _get_predicted_outcome(self, prediction: Dict[str, Any]) -> str:
        """Extract predicted outcome from prediction dictionary"""
        probs = {
            'H': prediction.get('home_win_probability', 0),
            'D': prediction.get('draw_probability', 0),
            'A': prediction.get('away_win_probability', 0)
        }
        return max(probs, key=probs.get)
    
    def get_accuracy_over_time(self, window_size: int = 10) -> List[float]:
        """Calculate rolling accuracy over time"""
        if len(self.predictions) < window_size:
            return []
        
        accuracies = []
        for i in range(window_size, len(self.predictions) + 1):
            window_predictions = self.predictions[i-window_size:i]
            correct_predictions = sum(1 for p in window_predictions if p['is_correct'] is True)
            total_predictions = sum(1 for p in window_predictions if p['is_correct'] is not None)
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                accuracies.append(accuracy)
        
        return accuracies
    
    def export_predictions(self, filepath: str):
        """Export predictions to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.predictions, f, indent=2, default=str)
            logger.info(f"Predictions exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
