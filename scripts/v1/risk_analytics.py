import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

class RiskAnalytics:
    """Risk analytics and factor analysis for trading strategies"""
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        factor_data_path: str = "data/factors/fama_french_factors_daily.csv",
        risk_free_rate: float = 0.02,  # Annual risk-free rate
        transaction_cost: float = 0.001,  # 10 bps per trade
        position_limits: Optional[Dict[str, float]] = None
    ):
        self.returns = returns_data
        self.rf_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.position_limits = position_limits or {'max_position': 1.0, 'min_position': -1.0}
        
        # Load factor data
        self.factors = pd.read_csv(factor_data_path, parse_dates=['date'])
        self.factors = self.factors.set_index('date')
        
        # Calculate daily risk-free rate
        self.daily_rf = (1 + self.rf_rate) ** (1/252) - 1
    
    def calculate_factor_exposures(
        self,
        returns: pd.Series,
        rolling_window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling factor exposures (betas)"""
        # Align returns with factor data
        aligned_data = pd.concat([
            returns,
            self.factors
        ], axis=1).dropna()
        
        # Initialize results
        betas = pd.DataFrame(index=aligned_data.index, columns=self.factors.columns)
        r_squared = pd.Series(index=aligned_data.index)
        
        # Calculate rolling regressions
        for i in range(rolling_window, len(aligned_data)):
            window_data = aligned_data.iloc[i-rolling_window:i]
            X = window_data[self.factors.columns]
            y = window_data[returns.name]
            
            model = LinearRegression()
            model.fit(X, y)
            
            betas.iloc[i] = model.coef_
            r_squared.iloc[i] = model.score(X, y)
        
        return pd.concat([betas, r_squared.rename('r_squared')], axis=1)
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        positions: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """Calculate comprehensive risk metrics"""
        # Adjust returns for transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        adjusted_returns = returns - transaction_costs
        
        # Calculate basic metrics
        total_return = (1 + adjusted_returns).prod() - 1
        annual_return = (1 + total_return) ** (252/len(returns)) - 1
        volatility = adjusted_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.rf_rate) / volatility
        
        # Calculate drawdown metrics
        cum_returns = (1 + adjusted_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate turnover
        annual_turnover = position_changes.mean() * 252
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'annual_turnover': annual_turnover,
            'avg_position': positions.mean(),
            'max_position': positions.max(),
            'min_position': positions.min()
        }
        
        # Calculate benchmark-relative metrics if provided
        if benchmark_returns is not None:
            excess_returns = adjusted_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Calculate beta and alpha
            benchmark_var = np.var(benchmark_returns)
            beta = np.cov(adjusted_returns, benchmark_returns)[0,1] / benchmark_var
            alpha = (adjusted_returns.mean() - self.daily_rf) - beta * (benchmark_returns.mean() - self.daily_rf)
            alpha = alpha * 252  # Annualize alpha
            
            metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha
            })
        
        return metrics
    
    def calculate_factor_contribution(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> pd.DataFrame:
        """Calculate factor contribution to returns"""
        # Calculate factor exposures
        exposures = self.calculate_factor_exposures(returns)
        
        # Calculate factor contributions
        factor_returns = exposures[self.factors.columns] * self.factors
        factor_contribution = factor_returns.mean() * 252  # Annualized
        
        # Calculate specific return (alpha)
        total_factor_return = factor_returns.sum(axis=1)
        specific_return = returns - total_factor_return
        
        # Create summary DataFrame
        contribution_summary = pd.DataFrame({
            'exposure': exposures[self.factors.columns].mean(),
            'contribution': factor_contribution,
            'contribution_pct': factor_contribution / returns.mean() * 100
        })
        
        # Add specific return
        contribution_summary.loc['specific'] = [
            1.0,  # Exposure
            specific_return.mean() * 252,  # Annualized specific return
            specific_return.mean() / returns.mean() * 100  # Contribution percentage
        ]
        
        return contribution_summary
    
    def plot_risk_analytics(
        self,
        returns: pd.Series,
        positions: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: str = "results/risk_analytics"
    ):
        """Generate risk analytics plots"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. Cumulative returns plot
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + returns).cumprod()
        cum_returns.plot(label='Strategy')
        if benchmark_returns is not None:
            cum_bench_returns = (1 + benchmark_returns).cumprod()
            cum_bench_returns.plot(label='Benchmark')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/cumulative_returns.png")
        plt.close()
        
        # 2. Rolling factor exposures
        exposures = self.calculate_factor_exposures(returns)
        plt.figure(figsize=(12, 6))
        exposures[self.factors.columns].plot()
        plt.title('Rolling Factor Exposures')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/factor_exposures.png")
        plt.close()
        
        # 3. Drawdown plot
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        drawdowns.plot(label='Strategy')
        plt.title('Drawdown')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/drawdowns.png")
        plt.close()
        
        # 4. Position and turnover analysis
        plt.figure(figsize=(12, 6))
        positions.plot(label='Position')
        plt.title('Strategy Positions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/positions.png")
        plt.close()
        
        # 5. Factor contribution plot
        contribution = self.calculate_factor_contribution(returns, positions)
        plt.figure(figsize=(12, 6))
        contribution['contribution'].plot(kind='bar')
        plt.title('Factor Return Contribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/factor_contribution.png")
        plt.close()
    
    def generate_risk_report(
        self,
        returns: pd.Series,
        positions: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        output_path: str = "results/risk_report.html"
    ):
        """Generate comprehensive risk report"""
        # Calculate all metrics
        metrics = self.calculate_risk_metrics(returns, positions, benchmark_returns)
        factor_contrib = self.calculate_factor_contribution(returns, positions)
        
        # Generate plots
        self.plot_risk_analytics(returns, positions, benchmark_returns)
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Risk Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Risk Analytics Report</h1>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items())}
            </table>
            
            <h2>Factor Analysis</h2>
            <table>
                <tr><th>Factor</th><th>Exposure</th><th>Contribution</th><th>Contribution %</th></tr>
                {''.join(f"<tr><td>{i}</td><td>{row['exposure']:.4f}</td><td>{row['contribution']:.4f}</td><td>{row['contribution_pct']:.2f}%</td></tr>" for i, row in factor_contrib.iterrows())}
            </table>
            
            <h2>Analytics Plots</h2>
            <img src="risk_analytics/cumulative_returns.png" alt="Cumulative Returns">
            <img src="risk_analytics/factor_exposures.png" alt="Factor Exposures">
            <img src="risk_analytics/drawdowns.png" alt="Drawdowns">
            <img src="risk_analytics/positions.png" alt="Positions">
            <img src="risk_analytics/factor_contribution.png" alt="Factor Contribution">
        </body>
        </html>
        """
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content) 