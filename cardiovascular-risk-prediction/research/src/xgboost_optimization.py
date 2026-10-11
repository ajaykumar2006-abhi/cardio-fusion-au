"""
XGBoost Optimization with GridSearchCV
Addressing reference paper limitation: No hyperparameter optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score, matthews_corrcoef,
                            brier_score_loss, confusion_matrix)
import xgboost as xgb
import joblib
import json
from pathlib import Path
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class XGBoostOptimizer:
    """XGBoost hyperparameter optimization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        data_dir = Path.cwd() / 'research' / 'data'
        
        X_train = np.load(data_dir / 'X_train.npy', allow_pickle=True)
        X_val = np.load(data_dir / 'X_val.npy', allow_pickle=True)
        X_holdout = np.load(data_dir / 'X_holdout.npy', allow_pickle=True)
        y_train = np.load(data_dir / 'y_train.npy', allow_pickle=True)
        y_val = np.load(data_dir / 'y_val.npy', allow_pickle=True)
        y_holdout = np.load(data_dir / 'y_holdout.npy', allow_pickle=True)
        
        print(f"[OK] Data loaded")
        print(f"  Train: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Holdout: {X_holdout.shape}")
        
        return X_train, X_val, X_holdout, y_train, y_val, y_holdout
    
    def define_param_grid(self):
        """Define hyperparameter grid for XGBoost"""
        print("\n" + "="*70)
        print("XGBOOST HYPERPARAMETER GRID")
        print("="*70)
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1, 1.5, 2, 3]
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        print(f"\nTotal combinations: {total_combinations}")
        print("Using RandomizedSearchCV with 50 iterations for efficiency")
        
        return param_grid
    
    def optimize_xgboost(self, X_train, X_val, y_train, y_val, param_grid):
        """Optimize XGBoost using GridSearchCV"""
        print("\n" + "="*70)
        print("XGBOOST OPTIMIZATION (GridSearchCV)")
        print("="*70)
        
        # Combine train and val for grid search (still within development set)
        X_dev = np.vstack([X_train, X_val])
        y_dev = np.concatenate([y_train, y_val])
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # Handle class imbalance
        )
        
        # Create stratified 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Randomized search with ROC-AUC scoring (more efficient for large parameter space)
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,  # 50 random combinations
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )
        
        print("Starting RandomizedSearchCV (50 iterations)...")
        random_search.fit(X_dev, y_dev)
        
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
        
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        return self.best_model, self.best_params
    
    def evaluate_best_model(self, X_val, y_val):
        """Evaluate the best XGBoost model"""
        print("\n" + "="*70)
        print("EVALUATING BEST XGBOOST MODEL")
        print("="*70)
        
        # Predictions
        y_pred = self.best_model.predict(X_val)
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'specificity': None,
            'mcc': matthews_corrcoef(y_val, y_pred),
            'brier_score': brier_score_loss(y_val, y_pred_proba)
        }
        
        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        self.results = metrics
        
        return metrics
    
    def save_model(self):
        """Save optimized XGBoost model"""
        models_dir = Path.cwd() / 'research' / 'models'
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / 'optimized_xgboost.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"[OK] Saved optimized XGBoost to {model_path}")
        
        # Save parameters
        params_path = models_dir / 'optimized_xgboost_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"[OK] Saved parameters to {params_path}")
    
    def save_results(self):
        """Save optimization results"""
        reports_dir = Path.cwd() / 'research' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        results_dict = {
            'best_params': self.best_params,
            'validation_metrics': self.results,
            'random_state': self.random_state
        }
        
        results_path = reports_dir / 'xgboost_optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"[OK] Saved results to {results_path}")

def main():
    """Main optimization pipeline"""
    print("="*70)
    print("XGBOOST OPTIMIZATION WITH GRIDSEARCHCV")
    print("="*70)
    
    # Initialize optimizer
    optimizer = XGBoostOptimizer(random_state=42)
    
    # Load data
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = optimizer.load_data()
    
    # Define parameter grid
    param_grid = optimizer.define_param_grid()
    
    # Optimize XGBoost
    best_model, best_params = optimizer.optimize_xgboost(X_train, X_val, y_train, y_val, param_grid)
    
    # Evaluate best model
    metrics = optimizer.evaluate_best_model(X_val, y_val)
    
    # Save model and results
    optimizer.save_model()
    optimizer.save_results()
    
    print("\n" + "="*70)
    print("XGBOOST OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Best F1-Score: {metrics['f1']:.4f}")
    print(f"Best Recall: {metrics['recall']:.4f}")
    print("\nNext steps:")
    print("1. Compare with baseline models")
    print("2. Implement ensemble methods")
    print("3. Perform calibration experiments")
    print("4. Implement SHAP explainability")
    print("5. Evaluate final model on locked holdout")

if __name__ == "__main__":
    main()
