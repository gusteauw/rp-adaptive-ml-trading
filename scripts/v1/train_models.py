import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Callable
import json
import matplotlib.pyplot as plt
import seaborn as sns

from models.ml_models import (
    RandomForestModel,
    GradientBoostingModel,
    LogisticRegressionModel
)
from feature_engineering import FeatureEngineer
from walk_forward import WalkForwardAnalysis
from risk_analytics import RiskAnalytics

class ModelTrainer:
    """Class to handle training and evaluation of multiple models"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self.setup_directories()
        
        # Initialize components
        self.feature_engineer = None
        self.risk_analytics = None
        self.walk_forward = None
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            "data/processed",
            "models/trained",
            "models/metadata",
            "results/metrics",
            "results/plots",
            "results/walk_forward",
            "results/risk_analytics"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self) -> pd.DataFrame:
        """Prepare and engineer features"""
        # Load price data
        price_data = pd.read_csv(
            self.config['data']['price_path'],
            parse_dates=['date']
        )
        
        # Load sentiment data if available
        sentiment_data = None
        if self.config['data'].get('sentiment_path'):
            sentiment_data = pd.read_csv(
                self.config['data']['sentiment_path'],
                parse_dates=['date']
            )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            price_data=price_data,
            sentiment_data=sentiment_data,
            **self.config['feature_params']
        )
        
        # Engineer features
        return self.feature_engineer.engineer_features()
    
    def get_model_factory(self, model_type: str, params: Dict) -> Callable:
        """Create a model factory function"""
        def factory():
            if model_type == "random_forest":
                return RandomForestModel(
                    feature_columns=self.config['feature_columns'],
                    target_column=self.config['target_column'],
                    **params
                )
            elif model_type == "gradient_boosting":
                return GradientBoostingModel(
                    feature_columns=self.config['feature_columns'],
                    target_column=self.config['target_column'],
                    **params
                )
            elif model_type == "logistic_regression":
                return LogisticRegressionModel(
                    feature_columns=self.config['feature_columns'],
                    target_column=self.config['target_column'],
                    **params
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        return factory
    
    def train_and_evaluate_model(
        self,
        data: pd.DataFrame,
        model_config: Dict
    ) -> Dict:
        """Train and evaluate a single model using walk-forward analysis"""
        model_type = model_config['type']
        model_params = model_config['params']
        
        self.logger.info(f"Training and evaluating {model_type} model...")
        
        # Initialize walk-forward analysis
        walk_forward = WalkForwardAnalysis(
            data=data,
            model_factory=self.get_model_factory(model_type, model_params),
            feature_columns=self.config['feature_columns'],
            target_column=self.config['target_column'],
            **self.config['training']['walk_forward']
        )
        
        # Run walk-forward analysis
        results, predictions, positions = walk_forward.run_analysis()
        
        # Generate walk-forward report
        walk_forward.generate_report(
            results,
            predictions,
            positions,
            output_path=f"results/walk_forward/{model_type}"
        )
        
        # Initialize risk analytics
        risk_analytics = RiskAnalytics(
            returns_data=data,
            transaction_cost=self.config['training']['transaction_cost']
        )
        
        # Calculate overall returns
        strategy_returns = data[self.config['target_column']].loc[positions.index] * positions
        
        # Generate risk report
        risk_analytics.generate_risk_report(
            returns=strategy_returns,
            positions=positions,
            benchmark_returns=data.get('benchmark_returns'),
            output_path=f"results/risk_analytics/{model_type}_risk_report.html"
        )
        
        return {
            'model_type': model_type,
            'walk_forward_results': results,
            'predictions': predictions,
            'positions': positions
        }
    
    def run_training(self) -> Dict[str, Dict]:
        """Main training pipeline"""
        # Prepare data
        self.logger.info("Preparing data...")
        df = self.prepare_data()
        
        # Train and evaluate all models
        results = {}
        for model_config in self.config['models']:
            model_results = self.train_and_evaluate_model(df, model_config)
            results[model_config['type']] = model_results
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Dict]):
        """Generate report comparing all models"""
        # Prepare metrics for comparison
        comparison = {}
        for model_type, model_results in results.items():
            # Get last window metrics as final performance
            last_window = model_results['walk_forward_results'][-1]
            comparison[model_type] = last_window['val_metrics']
        
        # Create comparison plots
        self._plot_model_comparison(comparison)
        
        # Save comparison results
        output_path = f"results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    def _plot_model_comparison(self, comparison: Dict):
        """Create comparison plots for all models"""
        metrics = ['sharpe_ratio', 'information_ratio', 'alpha']
        
        # Create bar plot for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            values = [model_metrics[metric] for model_metrics in comparison.values()]
            plt.bar(comparison.keys(), values)
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/plots/comparison_{metric}.png")
            plt.close()
        
        # Create correlation heatmap of model predictions
        predictions = pd.DataFrame({
            model_type: results['predictions']
            for model_type, results in results.items()
        })
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            predictions.corr(),
            annot=True,
            cmap='RdYlBu',
            center=0
        )
        plt.title('Model Predictions Correlation')
        plt.tight_layout()
        plt.savefig("results/plots/prediction_correlation.png")
        plt.close()

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training() 