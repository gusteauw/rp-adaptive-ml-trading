from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import optuna
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation with hyperparameter optimization"""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        n_trials: int = 100,
        **kwargs
    ):
        super().__init__(
            model_name="random_forest",
            feature_columns=feature_columns,
            target_column=target_column,
            **kwargs
        )
        self.n_trials = n_trials
    
    def build_model(self) -> BaseEstimator:
        """Build Random Forest pipeline"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=self.random_state))
        ])
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {
                'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'classifier__max_depth': trial.suggest_int('max_depth', 3, 30),
                'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            
            model = self.build_model()
            model.set_params(**params)
            
            # Use TimeSeriesSplit for validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params_
    
    def train(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> Tuple[BaseEstimator, Dict]:
        """Train the model with hyperparameter optimization"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Optimize hyperparameters
        self.logger.info("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(X, y)
        
        # Train final model
        model = self.build_model()
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        metrics = self.calculate_metrics(y, y_pred, y_prob)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(
            model.named_steps['classifier'],
            self.feature_columns
        )
        
        # Save model and metadata
        self.save_model(model, metrics, best_params, feature_importance)
        self.save_predictions(df, y_pred, y_prob)
        
        return model, metrics
    
    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        X, _ = self.prepare_data(df, is_training=False)
        model = self.load_model()
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        self.save_predictions(df, predictions, probabilities, suffix="_test")
        
        return predictions, probabilities


class GradientBoostingModel(BaseModel):
    """Gradient Boosting implementation with hyperparameter optimization"""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        n_trials: int = 100,
        **kwargs
    ):
        super().__init__(
            model_name="gradient_boosting",
            feature_columns=feature_columns,
            target_column=target_column,
            **kwargs
        )
        self.n_trials = n_trials
    
    def build_model(self) -> BaseEstimator:
        """Build Gradient Boosting pipeline"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=self.random_state))
        ])
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {
                'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'classifier__learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'classifier__max_depth': trial.suggest_int('max_depth', 3, 30),
                'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'classifier__subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
            }
            
            model = self.build_model()
            model.set_params(**params)
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params_
    
    def train(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> Tuple[BaseEstimator, Dict]:
        """Train the model with hyperparameter optimization"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Optimize hyperparameters
        self.logger.info("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(X, y)
        
        # Train final model
        model = self.build_model()
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        metrics = self.calculate_metrics(y, y_pred, y_prob)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(
            model.named_steps['classifier'],
            self.feature_columns
        )
        
        # Save model and metadata
        self.save_model(model, metrics, best_params, feature_importance)
        self.save_predictions(df, y_pred, y_prob)
        
        return model, metrics
    
    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        X, _ = self.prepare_data(df, is_training=False)
        model = self.load_model()
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        self.save_predictions(df, predictions, probabilities, suffix="_test")
        
        return predictions, probabilities


class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation with regularization"""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        n_trials: int = 50,
        **kwargs
    ):
        super().__init__(
            model_name="logistic_regression",
            feature_columns=feature_columns,
            target_column=target_column,
            **kwargs
        )
        self.n_trials = n_trials
    
    def build_model(self) -> BaseEstimator:
        """Build Logistic Regression pipeline"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=self.random_state))
        ])
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {
                'classifier__C': trial.suggest_loguniform('C', 1e-5, 1e5),
                'classifier__penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'classifier__solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'classifier__class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            
            model = self.build_model()
            model.set_params(**params)
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params_
    
    def train(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> Tuple[BaseEstimator, Dict]:
        """Train the model with hyperparameter optimization"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Optimize hyperparameters
        self.logger.info("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(X, y)
        
        # Train final model
        model = self.build_model()
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        metrics = self.calculate_metrics(y, y_pred, y_prob)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(
            model.named_steps['classifier'],
            self.feature_columns
        )
        
        # Save model and metadata
        self.save_model(model, metrics, best_params, feature_importance)
        self.save_predictions(df, y_pred, y_prob)
        
        return model, metrics
    
    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model"""
        if not self.check_data_quality(df):
            raise ValueError("Data quality checks failed")
        
        X, _ = self.prepare_data(df, is_training=False)
        model = self.load_model()
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        self.save_predictions(df, predictions, probabilities, suffix="_test")
        
        return predictions, probabilities 