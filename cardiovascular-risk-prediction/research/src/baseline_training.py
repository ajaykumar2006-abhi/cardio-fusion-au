"""
Research Model Training with Baseline Models
Comprehensive baseline model comparison with proper cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
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

class ResearchModelTrainer:
    """Enhanced model training with comprehensive baseline comparison"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
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
    
    def define_models(self):
        """Define baseline models"""
        print("\n" + "="*70)
        print("DEFINING BASELINE MODELS")
        print("="*70)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=10
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'SVM': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        
        print(f"Defined {len(self.models)} baseline models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Evaluate a single model with comprehensive metrics"""
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'specificity': None,  # Will calculate from confusion matrix
            'mcc': matthews_corrcoef(y_val, y_pred),
            'brier_score': brier_score_loss(y_val, y_pred_proba)
        }
        
        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
        
        return model, metrics
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all baseline models"""
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            trained_model, metrics = self.evaluate_model(
                model, X_train, y_train, X_val, y_val, model_name
            )
            self.models[model_name] = trained_model
            self.results[model_name] = metrics
        
        return self.results
    
    def select_best_model(self):
        """Select best model based on multiple criteria"""
        print("\n" + "="*70)
        print("MODEL SELECTION")
        print("="*70)
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(self.results).T
        
        print("\nModel Comparison (Validation Set):")
        print(results_df[['accuracy', 'roc_auc', 'pr_auc', 'f1', 'recall', 'specificity', 'mcc']].round(4))
        
        # Select based on ROC-AUC (primary metric for imbalanced classification)
        best_model_name = results_df['roc_auc'].idxmax()
        best_roc_auc = results_df.loc[best_model_name, 'roc_auc']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best ROC-AUC: {best_roc_auc:.4f}")
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return best_model_name, results_df
    
    def save_models(self):
        """Save all trained models"""
        models_dir = Path.cwd() / 'research' / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            # Safe filename
            safe_name = model_name.lower().replace(' ', '_').replace('/', '_')
            model_path = models_dir / f'{safe_name}.pkl'
            joblib.dump(model, model_path)
            print(f"[OK] Saved {model_name} to {model_path}")
        
        # Save best model separately
        best_path = models_dir / 'best_baseline_model.pkl'
        joblib.dump(self.best_model, best_path)
        print(f"[OK] Saved best model ({self.best_model_name}) to {best_path}")
    
    def save_results(self, results_df):
        """Save training results"""
        reports_dir = Path.cwd() / 'research' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save CSV
        csv_path = reports_dir / 'baseline_model_comparison.csv'
        results_df.to_csv(csv_path)
        print(f"[OK] Saved results to {csv_path}")
        
        # Save JSON
        json_path = reports_dir / 'baseline_model_results.json'
        results_dict = results_df.to_dict(orient='index')
        
        # Convert numpy types for JSON
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json_results = convert_types(results_dict)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[OK] Saved JSON results to {json_path}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'best_roc_auc': float(results_df.loc[self.best_model_name, 'roc_auc']),
            'random_state': self.random_state,
            'model_count': len(self.models),
            'models_trained': list(self.models.keys())
        }
        
        metadata_path = reports_dir / 'baseline_training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Saved metadata to {metadata_path}")

def main():
    """Main training pipeline"""
    print("="*70)
    print("RESEARCH BASELINE MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ResearchModelTrainer(random_state=42)
    
    # Load data
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = trainer.load_data()
    
    # Define models
    trainer.define_models()
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Select best model
    best_model_name, results_df = trainer.select_best_model()
    
    # Save models and results
    trainer.save_models()
    trainer.save_results(results_df)
    
    print("\n" + "="*70)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Validation ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
    print("\nNext steps:")
    print("1. Optimize XGBoost with GridSearchCV")
    print("2. Implement ensemble methods")
    print("3. Perform calibration experiments")
    print("4. Implement SHAP explainability")
    print("5. Evaluate final model on locked holdout")

if __name__ == "__main__":
    main()
