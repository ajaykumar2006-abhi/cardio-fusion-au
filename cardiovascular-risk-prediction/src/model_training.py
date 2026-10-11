"""
Model Training Pipeline with Cross-Validation
Trains and compares multiple models for cardiovascular risk prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import joblib
from pathlib import Path
import json
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load preprocessed data"""
        data_dir = Path(__file__).parent.parent / 'data'
        
        X_train = np.load(data_dir / 'X_train.npy')
        X_val = np.load(data_dir / 'X_val.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        y_val = np.load(data_dir / 'y_val.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        print(f"[OK] Preprocessed data loaded")
        print(f"  Train: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_models(self):
        """Define models to train"""
        print("\n" + "="*70)
        print("MODELS TO TRAIN")
        print("="*70)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        # Try to add XGBoost if available
        try:
            from xgboost import XGBClassifier
            self.models['XGBoost'] = XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                scale_pos_weight=3,  # Handle class imbalance (approx 5.58:1 ratio)
                use_label_encoder=False,
                eval_metric='logloss'
            )
            print("  - Logistic Regression")
            print("  - Random Forest")
            print("  - XGBoost")
        except ImportError:
            print("  - Logistic Regression")
            print("  - Random Forest")
            print("  [XGBoost not available - skipping]")
        
        return self.models
    
    def cross_validate_model(self, model, X_train, y_train, model_name):
        """Perform cross-validation on training data"""
        print(f"\nCross-validating {model_name}...")
        
        # Use stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Cross-validation metrics
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        cv_results = {
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'roc_auc_mean': cv_roc_auc.mean(),
            'roc_auc_std': cv_roc_auc.std()
        }
        
        print(f"  Accuracy: {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})")
        print(f"  Precision: {cv_results['precision_mean']:.4f} (+/- {cv_results['precision_std']:.4f})")
        print(f"  Recall: {cv_results['recall_mean']:.4f} (+/- {cv_results['recall_std']:.4f})")
        print(f"  F1-score: {cv_results['f1_mean']:.4f} (+/- {cv_results['f1_std']:.4f})")
        print(f"  ROC-AUC: {cv_results['roc_auc_mean']:.4f} (+/- {cv_results['roc_auc_std']:.4f})")
        
        return cv_results
    
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"[OK] {model_name} trained")
        return model
    
    def evaluate_model(self, model, X, y, model_name, dataset_name="Validation"):
        """Evaluate a model on given dataset"""
        print(f"\nEvaluating {model_name} on {dataset_name}...")
        
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={tn}, FP={fp}")
        print(f"    FN={fn}, TP={tp}")
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        return results
    
    def train_and_evaluate_all(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("MODEL TRAINING AND EVALUATION")
        print("="*70)
        
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'='*70}")
            
            # Cross-validation on training data
            cv_results = self.cross_validate_model(model, X_train, y_train, model_name)
            
            # Train on full training data
            trained_model = self.train_model(model, X_train, y_train, model_name)
            
            # Evaluate on validation set
            val_results = self.evaluate_model(trained_model, X_val, y_val, model_name, "Validation")
            
            # Evaluate on test set
            test_results = self.evaluate_model(trained_model, X_test, y_test, model_name, "Test")
            
            # Store results
            all_results[model_name] = {
                'cv_results': cv_results,
                'validation_results': val_results,
                'test_results': test_results,
                'model': trained_model
            }
        
        return all_results
    
    def select_best_model(self, all_results):
        """Select best model based on multiple metrics"""
        print("\n" + "="*70)
        print("MODEL SELECTION")
        print("="*70)
        
        # Compare models based on test set ROC-AUC (primary metric)
        model_comparison = []
        
        for model_name, results in all_results.items():
            model_comparison.append({
                'model': model_name,
                'test_roc_auc': results['test_results']['roc_auc'],
                'test_accuracy': results['test_results']['accuracy'],
                'test_f1': results['test_results']['f1'],
                'test_recall': results['test_results']['recall'],
                'cv_roc_auc': results['cv_results']['roc_auc_mean']
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        
        print("\nModel Comparison (Test Set):")
        print(comparison_df.to_string(index=False))
        
        # Select best model based on ROC-AUC (primary) and F1 (secondary)
        best_model_idx = comparison_df['test_roc_auc'].idxmax()
        self.best_model_name = comparison_df.loc[best_model_idx, 'model']
        self.best_model = all_results[self.best_model_name]['model']
        
        print(f"\n[OK] Best model selected: {self.best_model_name}")
        print(f"  Test ROC-AUC: {comparison_df.loc[best_model_idx, 'test_roc_auc']:.4f}")
        print(f"  Test Accuracy: {comparison_df.loc[best_model_idx, 'test_accuracy']:.4f}")
        print(f"  Test F1-score: {comparison_df.loc[best_model_idx, 'test_f1']:.4f}")
        
        return self.best_model, self.best_model_name, comparison_df
    
    def save_model(self, model, model_name):
        """Save trained model"""
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        print(f"[OK] Model saved to {model_path}")
        
        return model_path
    
    def save_results(self, all_results, comparison_df):
        """Save training results"""
        results_dir = Path(__file__).parent.parent / 'reports'
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_summary = {}
        for model_name, results in all_results.items():
            results_summary[model_name] = {
                'cv_results': {k: float(v) for k, v in results['cv_results'].items()},
                'validation_results': {k: float(v) if not isinstance(v, dict) else v for k, v in results['validation_results'].items()},
                'test_results': {k: float(v) if not isinstance(v, dict) else v for k, v in results['test_results'].items()}
            }
        
        with open(results_dir / 'model_training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save comparison table
        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
        
        print(f"[OK] Results saved to {results_dir}")
        
        return results_summary

def main():
    """Main training pipeline"""
    print("="*70)
    print("CARDIOVASCULAR RISK PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    
    # Define models
    trainer.define_models()
    
    # Train and evaluate all models
    all_results = trainer.train_and_evaluate_all(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Select best model
    best_model, best_model_name, comparison_df = trainer.select_best_model(all_results)
    
    # Save best model
    model_path = trainer.save_model(best_model, best_model_name)
    
    # Save all results
    results_summary = trainer.save_results(all_results, comparison_df)
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {Path(__file__).parent.parent / 'reports'}")
    
    print("\nNext steps:")
    print("1. Review model comparison results")
    print("2. Implement explainable AI (SHAP)")
    print("3. Create model metadata")
    print("4. Build backend API")

if __name__ == "__main__":
    main()
