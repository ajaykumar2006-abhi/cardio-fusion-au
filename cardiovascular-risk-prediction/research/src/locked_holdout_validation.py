"""
Locked Holdout Validation
Final evaluation on locked holdout set (never used during development)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score, matthews_corrcoef,
                            brier_score_loss, confusion_matrix)
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

class LockedHoldoutValidator:
    """Final validation on locked holdout set"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
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
    
    def load_best_model(self):
        """Load the best model (optimized XGBoost)"""
        models_dir = Path.cwd() / 'research' / 'models'
        model_path = models_dir / 'optimized_xgboost.pkl'
        
        self.model = joblib.load(model_path)
        self.model_name = "Optimized XGBoost"
        
        print(f"[OK] Loaded {self.model_name}")
        
        return self.model
    
    def evaluate_on_holdout(self, X_holdout, y_holdout):
        """Evaluate model on locked holdout set"""
        print("\n" + "="*70)
        print("LOCKED HOLDOUT VALIDATION")
        print("="*70)
        print("IMPORTANT: This holdout set has NEVER been used during development")
        print("This provides an unbiased estimate of final model performance")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(X_holdout)
        y_pred_proba = self.model.predict_proba(X_holdout)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_holdout, y_pred),
            'precision': precision_score(y_holdout, y_pred, zero_division=0),
            'recall': recall_score(y_holdout, y_pred, zero_division=0),
            'f1': f1_score(y_holdout, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_holdout, y_pred_proba),
            'pr_auc': average_precision_score(y_holdout, y_pred_proba),
            'specificity': None,
            'mcc': matthews_corrcoef(y_holdout, y_pred),
            'brier_score': brier_score_loss(y_holdout, y_pred_proba)
        }
        
        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Calculate sensitivity (same as recall)
        metrics['sensitivity'] = metrics['recall']
        
        print(f"\nLOCKED HOLDOUT RESULTS ({len(y_holdout)} samples):")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall / Sensitivity: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives: {tp}")
        
        self.results = metrics
        self.results['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        self.results['holdout_samples'] = len(y_holdout)
        
        return metrics
    
    def compare_with_production(self):
        """Compare with current production model"""
        print("\n" + "="*70)
        print("COMPARISON WITH PRODUCTION MODEL")
        print("="*70)
        
        # Production model metrics (from metadata.json)
        production_metrics = {
            'accuracy': 0.6781,
            'roc_auc': 0.7023,
            'recall': 0.5969,
            'specificity': 0.6926,
            'f1': 0.3607
        }
        
        print(f"\nProduction Model (Logistic Regression):")
        print(f"  Accuracy: {production_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {production_metrics['roc_auc']:.4f}")
        print(f"  Recall: {production_metrics['recall']:.4f}")
        print(f"  Specificity: {production_metrics['specificity']:.4f}")
        print(f"  F1-Score: {production_metrics['f1']:.4f}")
        
        print(f"\nResearch Model (Optimized XGBoost):")
        print(f"  Accuracy: {self.results['accuracy']:.4f}")
        print(f"  ROC-AUC: {self.results['roc_auc']:.4f}")
        print(f"  Recall: {self.results['recall']:.4f}")
        print(f"  Specificity: {self.results['specificity']:.4f}")
        print(f"  F1-Score: {self.results['f1']:.4f}")
        
        print(f"\nIMPROVEMENTS:")
        accuracy_improvement = (self.results['accuracy'] - production_metrics['accuracy']) * 100
        roc_auc_improvement = self.results['roc_auc'] - production_metrics['roc_auc']
        recall_improvement = (self.results['recall'] - production_metrics['recall']) * 100
        specificity_improvement = (self.results['specificity'] - production_metrics['specificity']) * 100
        f1_improvement = self.results['f1'] - production_metrics['f1']
        
        print(f"  Accuracy: {accuracy_improvement:+.2f} percentage points")
        print(f"  ROC-AUC: {roc_auc_improvement:+.4f}")
        print(f"  Recall: {recall_improvement:+.2f} percentage points")
        print(f"  Specificity: {specificity_improvement:+.2f} percentage points")
        print(f"  F1-Score: {f1_improvement:+.4f}")
        
        self.results['production_comparison'] = {
            'accuracy_improvement_pp': accuracy_improvement,
            'roc_auc_improvement': roc_auc_improvement,
            'recall_improvement_pp': recall_improvement,
            'specificity_improvement_pp': specificity_improvement,
            'f1_improvement': f1_improvement
        }
        
        return self.results
    
    def compare_with_reference_paper(self):
        """Compare with reference paper"""
        print("\n" + "="*70)
        print("COMPARISON WITH REFERENCE PAPER")
        print("="*70)
        
        # Reference paper metrics
        reference_metrics = {
            'accuracy': 0.8607,
            'roc_auc': 0.91,
            'f1': 0.88
        }
        
        print(f"\nReference Paper (XGBoost on ~1,001 records):")
        print(f"  Test Accuracy: {reference_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {reference_metrics['roc_auc']:.4f}")
        print(f"  F1-Score: {reference_metrics['f1']:.4f}")
        
        print(f"\nResearch Model (Optimized XGBoost on 4,240 records):")
        print(f"  Holdout Accuracy: {self.results['accuracy']:.4f}")
        print(f"  Holdout ROC-AUC: {self.results['roc_auc']:.4f}")
        print(f"  Holdout F1-Score: {self.results['f1']:.4f}")
        
        print(f"\nIMPORTANT DISCLAIMER:")
        print(f"  Results are NOT directly comparable due to:")
        print(f"  - Different datasets (Dhaka vs Framingham)")
        print(f"  - Different target definitions")
        print(f"  - Different populations")
        print(f"  - Different evaluation protocols")
        
        self.results['reference_comparison'] = {
            'reference_accuracy': reference_metrics['accuracy'],
            'reference_roc_auc': reference_metrics['roc_auc'],
            'reference_f1': reference_metrics['f1'],
            'research_accuracy': self.results['accuracy'],
            'research_roc_auc': self.results['roc_auc'],
            'research_f1': self.results['f1'],
            'not_directly_comparable': True
        }
        
        return self.results
    
    def save_results(self):
        """Save holdout validation results"""
        reports_dir = Path.cwd() / 'research' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        results_path = reports_dir / 'locked_holdout_validation.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[OK] Saved holdout validation results to {results_path}")

def main():
    """Main validation pipeline"""
    print("="*70)
    print("LOCKED HOLDOUT VALIDATION")
    print("="*70)
    
    # Initialize validator
    validator = LockedHoldoutValidator()
    
    # Load data
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = validator.load_data()
    
    # Load best model
    validator.load_best_model()
    
    # Evaluate on holdout
    validator.evaluate_on_holdout(X_holdout, y_holdout)
    
    # Compare with production model
    validator.compare_with_production()
    
    # Compare with reference paper
    validator.compare_with_reference_paper()
    
    # Save results
    validator.save_results()
    
    print("\n" + "="*70)
    print("LOCKED HOLDOUT VALIDATION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"  - Research model ROC-AUC: {validator.results['roc_auc']:.4f}")
    print(f"  - Production model ROC-AUC: 0.7023")
    print(f"  - Improvement: {validator.results['production_comparison']['roc_auc_improvement']:+.4f}")
    print(f"\nNext steps:")
    print("1. Implement SHAP explainability")
    print("2. Create leakage audit")
    print("3. Generate final research reports")

if __name__ == "__main__":
    main()
