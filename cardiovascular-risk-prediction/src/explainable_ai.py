"""
Explainable AI Implementation
Provides feature importance and individual prediction explanations using SHAP
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class ExplainableAI:
    """Explainable AI for cardiovascular risk prediction"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.shap_values = None
        self.expected_value = None
        
    def load_model_and_preprocessor(self):
        """Load trained model and preprocessor"""
        models_dir = Path(__file__).parent.parent / 'models'
        
        # Load model
        self.model = joblib.load(models_dir / 'model.pkl')
        print("[OK] Model loaded")
        
        # Load preprocessor
        self.preprocessor = joblib.load(models_dir / 'preprocessor.pkl')
        print("[OK] Preprocessor loaded")
        
        # Load feature names
        self.feature_names = joblib.load(models_dir / 'preprocessor_feature_names.pkl')
        print(f"[OK] Feature names loaded ({len(self.feature_names)} features)")
        
        return self.model, self.preprocessor, self.feature_names
    
    def get_feature_importance(self):
        """
        Get feature importance from logistic regression model
        For logistic regression, we use the absolute coefficient values
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Get coefficients from logistic regression
        coefficients = self.model.coef_[0]
        
        # Calculate absolute importance
        importance = np.abs(coefficients)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        print("\nFeature Importance Details:")
        print(f"  Most important: {feature_importance.iloc[0]['feature']} (coef: {feature_importance.iloc[0]['coefficient']:.4f})")
        print(f"  Least important: {feature_importance.iloc[-1]['feature']} (coef: {feature_importance.iloc[-1]['coefficient']:.4f})")
        
        return feature_importance
    
    def explain_prediction(self, input_data):
        """
        Explain an individual prediction
        Returns feature contributions to the prediction
        """
        # Preprocess input
        input_transformed = self.preprocessor.transform(input_data)
        
        # Get prediction and probability
        prediction = self.model.predict(input_transformed)[0]
        probability = self.model.predict_proba(input_transformed)[0, 1]
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # Calculate feature contributions
        # For logistic regression: contribution = coefficient * feature_value
        feature_contributions = []
        
        for i, (feature_name, feature_value) in enumerate(zip(self.feature_names, input_transformed[0])):
            contribution = coefficients[i] * feature_value
            feature_contributions.append({
                'feature': feature_name,
                'value': float(feature_value),
                'coefficient': float(coefficients[i]),
                'contribution': float(contribution)
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        explanation = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_category': self._get_risk_category(probability),
            'intercept': float(intercept),
            'feature_contributions': feature_contributions
        }
        
        return explanation
    
    def _get_risk_category(self, probability):
        """Categorize risk based on probability"""
        if probability < 0.10:
            return "Low Risk"
        elif probability < 0.20:
            return "Moderate Risk"
        elif probability < 0.30:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def generate_explanation_report(self, explanation):
        """Generate human-readable explanation report"""
        print("\n" + "="*70)
        print("PREDICTION EXPLANATION")
        print("="*70)
        
        print(f"\nPrediction: {'Positive (High CHD Risk)' if explanation['prediction'] == 1 else 'Negative (Low CHD Risk)'}")
        print(f"Probability: {explanation['probability']:.4f} ({explanation['probability']*100:.2f}%)")
        print(f"Risk Category: {explanation['risk_category']}")
        
        print(f"\nTop 5 Feature Contributions:")
        for i, feat in enumerate(explanation['feature_contributions'][:5], 1):
            direction = "increases" if feat['contribution'] > 0 else "decreases"
            print(f"  {i}. {feat['feature']}: {direction} risk")
            print(f"     Value: {feat['value']:.4f}, Coefficient: {feat['coefficient']:.4f}, Contribution: {feat['contribution']:.4f}")
        
        print("\n" + "="*70)
        print("IMPORTANT DISCLAIMER")
        print("="*70)
        print("This explanation shows which features contributed to the model's prediction.")
        print("These are statistical associations, NOT medical causation.")
        print("Feature contributions do not indicate that changing a feature will change")
        print("the actual risk. Consult healthcare professionals for medical advice.")
        print("="*70)
    
    def save_feature_importance(self, feature_importance):
        """Save feature importance to file"""
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        feature_importance.to_csv(reports_dir / 'feature_importance.csv', index=False)
        
        # Save as JSON
        importance_dict = feature_importance.to_dict('records')
        import json
        with open(reports_dir / 'feature_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        print(f"[OK] Feature importance saved to {reports_dir}")
    
    def generate_global_explanation(self):
        """Generate global feature importance explanation"""
        print("\n" + "="*70)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*70)
        
        feature_importance = self.get_feature_importance()
        
        print("\nInterpretation:")
        print("  - Positive coefficient: Higher feature value increases CHD risk")
        print("  - Negative coefficient: Higher feature value decreases CHD risk")
        print("  - Larger absolute value: Stronger influence on prediction")
        
        print("\nKey Risk Factors (Positive Coefficients):")
        positive_features = feature_importance[feature_importance['coefficient'] > 0].head(5)
        for _, row in positive_features.iterrows():
            print(f"  • {row['feature']}: +{row['coefficient']:.4f}")
        
        print("\nProtective Factors (Negative Coefficients):")
        negative_features = feature_importance[feature_importance['coefficient'] < 0].head(5)
        for _, row in negative_features.iterrows():
            print(f"  • {row['feature']}: {row['coefficient']:.4f}")
        
        print("\n" + "="*70)
        print("MEDICAL DISCLAIMER")
        print("="*70)
        print("These coefficients represent statistical associations in the training data.")
        print("They do NOT represent causal relationships or medical recommendations.")
        print("Feature importance should be interpreted as model behavior, not clinical fact.")
        print("="*70)
        
        self.save_feature_importance(feature_importance)
        
        return feature_importance

def main():
    """Main explainable AI pipeline"""
    print("="*70)
    print("EXPLAINABLE AI - CARDIOVASCULAR RISK PREDICTION")
    print("="*70)
    
    # Initialize explainable AI
    xai = ExplainableAI()
    
    # Load model and preprocessor
    xai.load_model_and_preprocessor()
    
    # Generate global feature importance
    feature_importance = xai.generate_global_explanation()
    
    # Example: Explain a sample prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION EXPLANATION")
    print("="*70)
    
    # Load raw dataset for example
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
    df = pd.read_csv(data_path)
    
    # Use first sample from dataset
    sample_input = df.drop('TenYearCHD', axis=1).iloc[0:1]
    explanation = xai.explain_prediction(sample_input)
    xai.generate_explanation_report(explanation)
    
    print("\n" + "="*70)
    print("EXPLAINABLE AI COMPLETE")
    print("="*70)
    print("\nFeature importance saved to: reports/feature_importance.csv")
    print("Feature importance saved to: reports/feature_importance.json")

if __name__ == "__main__":
    main()
