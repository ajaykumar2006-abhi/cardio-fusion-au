"""
Evaluate Production Logistic Regression on Locked Holdout
Fair comparison between production and research models
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

class ProductionModelValidator:
    """Evaluate production model on locked holdout"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.results = {}
        
    def load_production_model(self):
        """Load production model and preprocessor"""
        models_dir = Path.cwd() / 'models'
        
        self.model = joblib.load(models_dir / 'production_model.pkl')
        self.preprocessor = joblib.load(models_dir / 'production_preprocessor.pkl')
        
        print(f"[OK] Loaded production Logistic Regression model")
        print(f"[OK] Loaded production preprocessor")
        
        return self.model, self.preprocessor
    
    def load_holdout_data(self):
        """Load original (untransformed) holdout data"""
        # We need to load the original dataset and re-split it using the same seed
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        data_path = Path.cwd() / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(data_path)
        
        # Use the same split as research preprocessing (80/20, random_state=42)
        X = df.drop('TenYearCHD', axis=1)
        y = df['TenYearCHD']
        
        _, X_holdout, _, y_holdout = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        print(f"[OK] Loaded holdout data: {X_holdout.shape}")
        
        return X_holdout, y_holdout
    
    def evaluate_production_model(self, X_holdout, y_holdout):
        """Evaluate production model on holdout"""
        print("\n" + "="*70)
        print("PRODUCTION MODEL ON LOCKED HOLDOUT")
        print("="*70)
        
        # Transform using production preprocessor
        X_holdout_transformed = self.preprocessor.transform(X_holdout)
        
        # Predictions
        y_pred = self.model.predict(X_holdout_transformed)
        y_pred_proba = self.model.predict_proba(X_holdout_transformed)[:, 1]
        
        # Calculate metrics
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
        metrics['sensitivity'] = metrics['recall']
        
        print(f"\nPRODUCTION MODEL RESULTS ({len(y_holdout)} samples):")
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
    
    def compare_with_research_model(self):
        """Compare with research model"""
        print("\n" + "="*70)
        print("COMPARISON: PRODUCTION VS RESEARCH MODEL")
        print("="*70)
        
        # Load research model results
        reports_dir = Path.cwd() / 'research' / 'reports'
        with open(reports_dir / 'locked_holdout_validation.json', 'r') as f:
            research_results = json.load(f)
        
        print(f"\nProduction Model (Logistic Regression):")
        print(f"  Accuracy: {self.results['accuracy']:.4f}")
        print(f"  ROC-AUC: {self.results['roc_auc']:.4f}")
        print(f"  Recall: {self.results['recall']:.4f}")
        print(f"  Specificity: {self.results['specificity']:.4f}")
        print(f"  F1-Score: {self.results['f1']:.4f}")
        
        print(f"\nResearch Model (Optimized XGBoost):")
        print(f"  Accuracy: {research_results['accuracy']:.4f}")
        print(f"  ROC-AUC: {research_results['roc_auc']:.4f}")
        print(f"  Recall: {research_results['recall']:.4f}")
        print(f"  Specificity: {research_results['specificity']:.4f}")
        print(f"  F1-Score: {research_results['f1']:.4f}")
        
        print(f"\nDIFFERENCES (Research - Production):")
        accuracy_diff = (research_results['accuracy'] - self.results['accuracy']) * 100
        roc_auc_diff = research_results['roc_auc'] - self.results['roc_auc']
        recall_diff = (research_results['recall'] - self.results['recall']) * 100
        specificity_diff = (research_results['specificity'] - self.results['specificity']) * 100
        f1_diff = research_results['f1'] - self.results['f1']
        
        print(f"  Accuracy: {accuracy_diff:+.2f} percentage points")
        print(f"  ROC-AUC: {roc_auc_diff:+.4f}")
        print(f"  Recall: {recall_diff:+.2f} percentage points")
        print(f"  Specificity: {specificity_diff:+.2f} percentage points")
        print(f"  F1-Score: {f1_diff:+.4f}")
        
        # Determine winner
        if roc_auc_diff > 0:
            winner = "Research Model (XGBoost)"
        elif roc_auc_diff < 0:
            winner = "Production Model (Logistic Regression)"
        else:
            winner = "Tie"
        
        print(f"\nWINNER: {winner}")
        print(f"  Based on ROC-AUC")
        
        self.results['research_comparison'] = {
            'research_accuracy': research_results['accuracy'],
            'research_roc_auc': research_results['roc_auc'],
            'research_recall': research_results['recall'],
            'research_specificity': research_results['specificity'],
            'research_f1': research_results['f1'],
            'accuracy_diff_pp': accuracy_diff,
            'roc_auc_diff': roc_auc_diff,
            'recall_diff_pp': recall_diff,
            'specificity_diff_pp': specificity_diff,
            'f1_diff': f1_diff,
            'winner': winner
        }
        
        return self.results
    
    def save_results(self):
        """Save production model holdout results"""
        reports_dir = Path.cwd() / 'research' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        results_path = reports_dir / 'production_model_holdout_validation.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[OK] Saved production model holdout results to {results_path}")

def main():
    """Main validation pipeline"""
    print("="*70)
    print("PRODUCTION MODEL ON LOCKED HOLDOUT")
    print("="*70)
    
    # Initialize validator
    validator = ProductionModelValidator()
    
    # Load production model
    validator.load_production_model()
    
    # Load holdout data
    X_holdout, y_holdout = validator.load_holdout_data()
    
    # Evaluate production model
    validator.evaluate_production_model(X_holdout, y_holdout)
    
    # Compare with research model
    validator.compare_with_research_model()
    
    # Save results
    validator.save_results()
    
    print("\n" + "="*70)
    print("PRODUCTION MODEL HOLDOUT VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
