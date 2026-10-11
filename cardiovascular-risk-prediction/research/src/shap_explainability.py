"""
SHAP Explainability for Production Model
Global and patient-level SHAP explanations for the winning model
"""

import numpy as np
import pandas as pd
import shap
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

class SHAPExplainer:
    """SHAP explainability for the production model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def load_model(self):
        """Load production model and preprocessor"""
        models_dir = Path.cwd() / 'models'
        
        self.model = joblib.load(models_dir / 'production_model.pkl')
        self.preprocessor = joblib.load(models_dir / 'production_preprocessor.pkl')
        
        # Load feature names
        self.feature_names = joblib.load(models_dir / 'preprocessor_feature_names.pkl')
        
        print(f"[OK] Loaded production Logistic Regression model")
        print(f"[OK] Loaded production preprocessor")
        print(f"[OK] Loaded feature names: {len(self.feature_names)} features")
        
        return self.model, self.preprocessor
    
    def load_data(self):
        """Load data for SHAP analysis"""
        data_path = Path.cwd() / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(data_path)
        
        X = df.drop('TenYearCHD', axis=1)
        y = df['TenYearCHD']
        
        print(f"[OK] Loaded data: {X.shape}")
        
        return X, y
    
    def create_explainer(self, X_background):
        """Create SHAP explainer"""
        print("\n" + "="*70)
        print("CREATING SHAP EXPLAINER")
        print("="*70)
        
        # Transform background data
        X_background_transformed = self.preprocessor.transform(X_background)
        
        # Use a subset for background (100 samples)
        background_sample = X_background_transformed[:100]
        
        # Create explainer for Logistic Regression
        self.explainer = shap.LinearExplainer(self.model, background_sample)
        
        print(f"[OK] Created LinearExplainer for Logistic Regression")
        print(f"Background samples: {len(background_sample)}")
        
        return self.explainer
    
    def compute_shap_values(self, X_test):
        """Compute SHAP values"""
        print("\n" + "="*70)
        print("COMPUTING SHAP VALUES")
        print("="*70)
        
        # Transform test data
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Compute SHAP values (use subset for efficiency)
        test_sample = X_test_transformed[:200]
        self.shap_values = self.explainer.shap_values(test_sample)
        
        print(f"[OK] Computed SHAP values for {len(test_sample)} samples")
        print(f"SHAP values shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def analyze_global_importance(self):
        """Analyze global feature importance"""
        print("\n" + "="*70)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*70)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create feature importance ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        self.global_importance = importance_df.to_dict('records')
        
        return importance_df
    
    def analyze_patient_explanation(self, patient_idx=0):
        """Analyze individual patient explanation"""
        print("\n" + "="*70)
        print("PATIENT-LEVEL EXPLANATION")
        print("="*70)
        
        if patient_idx >= len(self.shap_values):
            patient_idx = 0
        
        # Get SHAP values for specific patient
        patient_shap = self.shap_values[patient_idx]
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # Get feature contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'contribution': patient_shap
        }).sort_values('contribution', ascending=False)
        
        print(f"Patient Index: {patient_idx}")
        print(f"Base Value (Expected Risk): {base_value:.4f}")
        print(f"\nTop 5 Positive Contributors (Increase Risk):")
        for i, row in contributions.head(5).iterrows():
            print(f"  {row['feature']}: +{row['contribution']:.4f}")
        
        print(f"\nTop 5 Negative Contributors (Decrease Risk):")
        for i, row in contributions.tail(5).iterrows():
            print(f"  {row['feature']}: {row['contribution']:.4f}")
        
        self.patient_explanation = {
            'patient_idx': patient_idx,
            'base_value': float(base_value),
            'positive_contributors': contributions.head(5).to_dict('records'),
            'negative_contributors': contributions.tail(5).to_dict('records')
        }
        
        return self.patient_explanation
    
    def save_results(self):
        """Save SHAP results"""
        reports_dir = Path.cwd() / 'research' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save global importance
        global_path = reports_dir / 'shap_global_importance.json'
        with open(global_path, 'w') as f:
            json.dump(self.global_importance, f, indent=2)
        print(f"\n[OK] Saved global importance to {global_path}")
        
        # Save patient explanation
        patient_path = reports_dir / 'shap_patient_explanation.json'
        with open(patient_path, 'w') as f:
            json.dump(self.patient_explanation, f, indent=2)
        print(f"[OK] Saved patient explanation to {patient_path}")
        
        # Save SHAP metadata
        metadata = {
            'model': 'Logistic Regression (Production)',
            'explainer_type': 'LinearExplainer',
            'background_samples': 100,
            'test_samples': len(self.shap_values),
            'feature_count': len(self.feature_names),
            'shap_values_shape': list(self.shap_values.shape)
        }
        
        metadata_path = reports_dir / 'shap_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Saved SHAP metadata to {metadata_path}")

def main():
    """Main SHAP pipeline"""
    print("="*70)
    print("SHAP EXPLAINABILITY FOR PRODUCTION MODEL")
    print("="*70)
    
    # Initialize explainer
    explainer = SHAPExplainer()
    
    # Load model
    explainer.load_model()
    
    # Load data
    X, y = explainer.load_data()
    
    # Create explainer
    explainer.create_explainer(X)
    
    # Compute SHAP values
    explainer.compute_shap_values(X)
    
    # Analyze global importance
    explainer.analyze_global_importance()
    
    # Analyze patient explanation
    explainer.analyze_patient_explanation(patient_idx=0)
    
    # Save results
    explainer.save_results()
    
    print("\n" + "="*70)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("  - Global feature importance computed")
    print("  - Patient-level explanations generated")
    print("  - SHAP values saved for analysis")
    print("\nIMPORTANT:")
    print("  - SHAP values represent model contributions, NOT medical causation")
    print("  - Feature contributions are statistical associations from the model")
    print("  - Do not interpret as direct causal relationships")
    print("\nNext steps:")
    print("1. Create leakage audit")
    print("2. Generate final research reports")

if __name__ == "__main__":
    main()
