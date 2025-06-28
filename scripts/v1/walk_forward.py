import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from .risk_analytics import RiskAnalytics

class WalkForwardAnalysis:
    """Walk-forward analysis and optimization"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        model_factory: Callable[[], BaseEstimator],
        feature_columns: List[str],
        target_column: str,
        date_column: str = 'date',
        train_size: int = 252 * 2,  # 2 years
        validation_size: int = 126,  # 6 months
        step_size: int = 63,  # 3 months
        min_samples: int = 252,  # Minimum samples needed
        transaction_cost: float = 0.001  # 10 bps per trade
    ):
        self.data = data.sort_values(date_column).copy()
        self.model_factory = model_factory
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.date_column = date_column
        self.train_size = train_size
        self.validation_size = validation_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.transaction_cost = transaction_cost
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk analytics
        self.risk_analytics = RiskAnalytics(
            returns_data=self.data,
            transaction_cost=transaction_cost
        )
    
    def generate_windows(self) -> List[Dict[str, pd.DataFrame]]:
        """Generate walk-forward windows"""
        windows = []
        total_samples = len(self.data)
        
        # Calculate number of windows
        n_windows = (total_samples - self.train_size - self.validation_size) // self.step_size + 1
        
        for i in range(n_windows):
            start_idx = i * self.step_size
            train_end_idx = start_idx + self.train_size
            val_end_idx = train_end_idx + self.validation_size
            
            # Break if we don't have enough data
            if val_end_idx > total_samples:
                break
            
            # Get train and validation sets
            train_data = self.data.iloc[start_idx:train_end_idx]
            val_data = self.data.iloc[train_end_idx:val_end_idx]
            
            # Skip if we don't have enough samples
            if len(train_data) < self.min_samples:
                continue
            
            windows.append({
                'train': train_data,
                'validation': val_data,
                'window_id': i,
                'train_start': train_data[self.date_column].iloc[0],
                'train_end': train_data[self.date_column].iloc[-1],
                'val_start': val_data[self.date_column].iloc[0],
                'val_end': val_data[self.date_column].iloc[-1]
            })
        
        return windows
    
    def evaluate_window(
        self,
        model: BaseEstimator,
        window: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Evaluate model performance on a single window"""
        # Get predictions
        train_pred = model.predict_proba(window['train'][self.feature_columns])
        val_pred = model.predict_proba(window['validation'][self.feature_columns])
        
        # Convert to positions (-1 to 1 based on class probabilities)
        train_pos = 2 * train_pred[:, 1] - 1
        val_pos = 2 * val_pred[:, 1] - 1
        
        # Calculate returns
        train_rets = window['train'][self.target_column] * train_pos
        val_rets = window['validation'][self.target_column] * val_pos
        
        # Calculate metrics for both periods
        train_metrics = self.risk_analytics.calculate_risk_metrics(
            returns=train_rets,
            positions=pd.Series(train_pos, index=window['train'].index)
        )
        
        val_metrics = self.risk_analytics.calculate_risk_metrics(
            returns=val_rets,
            positions=pd.Series(val_pos, index=window['validation'].index)
        )
        
        # Calculate factor exposures for validation period
        val_exposures = self.risk_analytics.calculate_factor_exposures(
            returns=val_rets
        ).mean()
        
        return {
            'window_id': window['window_id'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_exposures': val_exposures.to_dict(),
            'period': {
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'val_start': window['val_start'],
                'val_end': window['val_end']
            }
        }
    
    def run_analysis(self) -> Tuple[List[Dict], pd.Series, pd.Series]:
        """Run walk-forward analysis"""
        windows = self.generate_windows()
        results = []
        all_predictions = []
        all_positions = []
        
        self.logger.info(f"Running walk-forward analysis on {len(windows)} windows...")
        
        for window in windows:
            self.logger.info(f"Processing window {window['window_id']}")
            
            # Initialize and train model
            model = self.model_factory()
            model.fit(
                window['train'][self.feature_columns],
                window['train'][self.target_column]
            )
            
            # Evaluate window
            window_results = self.evaluate_window(model, window)
            results.append(window_results)
            
            # Store validation predictions
            val_pred = model.predict_proba(window['validation'][self.feature_columns])
            val_pos = 2 * val_pred[:, 1] - 1
            
            all_predictions.extend(val_pred[:, 1])
            all_positions.extend(val_pos)
        
        # Combine all out-of-sample predictions
        val_dates = pd.concat([w['validation'].index for w in windows])
        predictions = pd.Series(all_predictions, index=val_dates)
        positions = pd.Series(all_positions, index=val_dates)
        
        return results, predictions, positions
    
    def generate_report(
        self,
        results: List[Dict],
        predictions: pd.Series,
        positions: pd.Series,
        output_path: str = "results/walk_forward"
    ):
        """Generate comprehensive walk-forward analysis report"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Calculate overall metrics
        overall_returns = self.data[self.target_column].loc[positions.index] * positions
        overall_metrics = self.risk_analytics.calculate_risk_metrics(
            returns=overall_returns,
            positions=positions
        )
        
        # Calculate factor contribution
        factor_contrib = self.risk_analytics.calculate_factor_contribution(
            returns=overall_returns,
            positions=positions
        )
        
        # Generate plots
        self._plot_window_metrics(results, output_path)
        self._plot_factor_exposures(results, output_path)
        self._plot_cumulative_performance(overall_returns, output_path)
        
        # Create summary report
        report = {
            'overall_metrics': overall_metrics,
            'factor_contribution': factor_contrib.to_dict(),
            'window_metrics': [
                {
                    'window_id': r['window_id'],
                    'period': r['period'],
                    'validation_metrics': r['val_metrics']
                }
                for r in results
            ]
        }
        
        # Save report
        with open(f"{output_path}/walk_forward_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report, output_path)
    
    def _plot_window_metrics(self, results: List[Dict], output_path: str):
        """Plot metrics across windows"""
        metrics = ['sharpe_ratio', 'information_ratio', 'alpha']
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [r['val_metrics'].get(metric, 0) for r in results]
            plt.plot(values, label=metric)
        
        plt.title('Validation Metrics Across Windows')
        plt.xlabel('Window')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/window_metrics.png")
        plt.close()
    
    def _plot_factor_exposures(self, results: List[Dict], output_path: str):
        """Plot factor exposures across windows"""
        exposures = pd.DataFrame([r['val_exposures'] for r in results])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=exposures)
        plt.title('Factor Exposures Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path}/factor_exposures.png")
        plt.close()
    
    def _plot_cumulative_performance(self, returns: pd.Series, output_path: str):
        """Plot cumulative performance"""
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + returns).cumprod()
        cum_returns.plot()
        plt.title('Cumulative Out-of-Sample Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.tight_layout()
        plt.savefig(f"{output_path}/cumulative_performance.png")
        plt.close()
    
    def _generate_html_report(self, report: Dict, output_path: str):
        """Generate HTML report"""
        html_content = f"""
        <html>
        <head>
            <title>Walk-Forward Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Walk-Forward Analysis Report</h1>
            
            <h2>Overall Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in report['overall_metrics'].items())}
            </table>
            
            <h2>Factor Analysis</h2>
            <table>
                <tr><th>Factor</th><th>Exposure</th><th>Contribution</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v['exposure']:.4f}</td><td>{v['contribution']:.4f}</td></tr>" for k, v in report['factor_contribution'].items())}
            </table>
            
            <h2>Window Analysis</h2>
            <table>
                <tr>
                    <th>Window</th>
                    <th>Period</th>
                    <th>Sharpe Ratio</th>
                    <th>Information Ratio</th>
                    <th>Alpha</th>
                </tr>
                {''.join(
                    f"<tr><td>{w['window_id']}</td>"
                    f"<td>{w['period']['val_start']} to {w['period']['val_end']}</td>"
                    f"<td>{w['validation_metrics'].get('sharpe_ratio', 0):.4f}</td>"
                    f"<td>{w['validation_metrics'].get('information_ratio', 0):.4f}</td>"
                    f"<td>{w['validation_metrics'].get('alpha', 0):.4f}</td></tr>"
                    for w in report['window_metrics']
                )}
            </table>
            
            <h2>Analysis Plots</h2>
            <img src="window_metrics.png" alt="Window Metrics">
            <img src="factor_exposures.png" alt="Factor Exposures">
            <img src="cumulative_performance.png" alt="Cumulative Performance">
        </body>
        </html>
        """
        
        with open(f"{output_path}/walk_forward_report.html", 'w') as f:
            f.write(html_content) 