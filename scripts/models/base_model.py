from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
from pathlib import Path
import json
import logging
from datetime import datetime

class BaseModel(ABC, BaseEstimator):
    """Base class for all trading models"""
    
    def __init__(
        self,
        model_name: str,
        feature_columns: List[str],
        target_column: str,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(model_name)
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for model artifacts"""
        dirs = [
            "models/trained",
            "models/metadata",
            "results/metrics",
            "results/predictions"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Build and return the actual model"""
        pass
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features and target for modeling"""
        X = df[self.feature_columns].values
        y = df[self.target_column].values if is_training else None
        return X, y
    
    def get_cv_splits(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        return list(tscv.split(df))
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate model performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        return metrics
    
    def save_model(
        self,
        model: BaseEstimator,
        metrics: Dict,
        params: Dict,
        feature_importance: Optional[Dict] = None
    ):
        """Save model artifacts and metadata"""
        # Save model
        model_path = f"models/trained/{self.model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'features': self.feature_columns,
            'target': self.target_column,
            'parameters': params,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        metadata_path = f"models/metadata/{self.model_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self) -> BaseEstimator:
        """Load a trained model"""
        model_path = f"models/trained/{self.model_name}.joblib"
        return joblib.load(model_path)
    
    def get_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                return dict(zip(feature_names, abs(model.coef_)))
            else:
                # For multi-class, take mean of absolute coefficients
                return dict(zip(feature_names, abs(model.coef_).mean(axis=0)))
        return None
    
    def check_data_quality(self, df: pd.DataFrame) -> bool:
        """Check data quality before training/prediction"""
        # Check for missing values
        missing = df[self.feature_columns].isnull().sum()
        if missing.any():
            self.logger.warning(f"Missing values found: {missing[missing > 0]}")
            return False
        
        # Check for infinite values
        inf_count = np.isinf(df[self.feature_columns].values).sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values")
            return False
        
        # Check for constant columns
        constant_cols = [
            col for col in self.feature_columns
            if df[col].nunique() == 1
        ]
        if constant_cols:
            self.logger.warning(f"Constant columns found: {constant_cols}")
            return False
        
        return True
    
    def save_predictions(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        suffix: str = ""
    ):
        """Save model predictions"""
        pred_df = df.copy()
        pred_df[f'pred_{self.target_column}'] = predictions
        
        if probabilities is not None:
            for i, prob in enumerate(probabilities.T):
                pred_df[f'prob_class_{i}'] = prob
        
        output_path = f"results/predictions/{self.model_name}{suffix}_predictions.csv"
        pred_df.to_csv(output_path, index=False)
    
    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> Tuple[BaseEstimator, Dict]:
        """Train the model and return the trained model and metrics"""
        pass
    
    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the trained model"""
        pass 