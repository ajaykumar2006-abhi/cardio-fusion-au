"""
Create Model Metadata
Saves comprehensive metadata for the trained model
"""

import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

def create_metadata():
    """Create comprehensive model metadata"""
    
    print("="*70)
    print("CREATING MODEL METADATA")
    print("="*70)
    
    # Load training results
    results_path = Path(__file__).parent.parent / 'reports' / 'model_training_results.json'
    with open(results_path, 'r') as f:
        training_results = json.load(f)
    
    # Load preprocessing metadata
    preprocessing_path = Path(__file__).parent.parent / 'data' / 'preprocessing_metadata.json'
    with open(preprocessing_path, 'r') as f:
        preprocessing_metadata = json.load(f)
    
    # Get best model info (Logistic Regression based on training results)
    best_model_name = "Logistic Regression"
    best_model_results = training_results[best_model_name]
    
    # Create comprehensive metadata
    metadata = {
        "model_info": {
            "model_name": best_model_name,
            "model_type": "LogisticRegression",
            "model_version": "1.0",
            "training_date": datetime.now().isoformat(),
            "random_state": 42
        },
        "dataset_info": {
            "dataset_name": "Framingham Heart Study Teaching Dataset",
            "dataset_source": "Kaggle/GitHub (derived from NHLBI Framingham Heart Study)",
            "dataset_url": "https://raw.githubusercontent.com/theleadio/datascience_demo/master/framingham.csv",
            "total_samples": 4240,
            "feature_count": 15,
            "target_definition": "TenYearCHD - 10-year risk of coronary heart disease (0=No, 1=Yes)"
        },
        "data_split": {
            "train_samples": preprocessing_metadata['train_samples'],
            "validation_samples": preprocessing_metadata['val_samples'],
            "test_samples": preprocessing_metadata['test_samples'],
            "train_ratio": f"{preprocessing_metadata['train_samples']/4240*100:.1f}%",
            "validation_ratio": f"{preprocessing_metadata['val_samples']/4240*100:.1f}%",
            "test_ratio": f"{preprocessing_metadata['test_samples']/4240*100:.1f}%",
            "stratified_split": True,
            "data_leakage_prevention": True,
            "no_data_leakage_verified": preprocessing_metadata['no_data_leakage']
        },
        "target_distribution": {
            "train": preprocessing_metadata['target_distribution_train'],
            "validation": preprocessing_metadata['target_distribution_val'],
            "test": preprocessing_metadata['target_distribution_test'],
            "class_imbalance_ratio": 5.58,
            "imbalance_handling": "class_weight='balanced'"
        },
        "preprocessing": {
            "version": "1.0",
            "missing_value_handling": {
                "binary_features": "mode imputation",
                "categorical_features": "mode imputation",
                "continuous_features": "median imputation"
            },
            "feature_encoding": {
                "binary_features": "no encoding (already 0/1)",
                "categorical_features": "one-hot encoding",
                "continuous_features": "standard scaling"
            },
            "transformed_feature_count": preprocessing_metadata['n_features'],
            "feature_names": preprocessing_metadata['feature_names']
        },
        "model_parameters": {
            "LogisticRegression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs"  # default solver
            }
        },
        "cross_validation_results": {
            "method": "Stratified 5-Fold Cross-Validation",
            "accuracy": {
                "mean": best_model_results['cv_results']['accuracy_mean'],
                "std": best_model_results['cv_results']['accuracy_std']
            },
            "precision": {
                "mean": best_model_results['cv_results']['precision_mean'],
                "std": best_model_results['cv_results']['precision_std']
            },
            "recall": {
                "mean": best_model_results['cv_results']['recall_mean'],
                "std": best_model_results['cv_results']['recall_std']
            },
            "f1_score": {
                "mean": best_model_results['cv_results']['f1_mean'],
                "std": best_model_results['cv_results']['f1_std']
            },
            "roc_auc": {
                "mean": best_model_results['cv_results']['roc_auc_mean'],
                "std": best_model_results['cv_results']['roc_auc_std']
            }
        },
        "validation_results": {
            "accuracy": best_model_results['validation_results']['accuracy'],
            "precision": best_model_results['validation_results']['precision'],
            "recall": best_model_results['validation_results']['recall'],
            "specificity": best_model_results['validation_results']['specificity'],
            "f1_score": best_model_results['validation_results']['f1'],
            "roc_auc": best_model_results['validation_results']['roc_auc'],
            "confusion_matrix": best_model_results['validation_results']['confusion_matrix']
        },
        "test_results": {
            "accuracy": best_model_results['test_results']['accuracy'],
            "precision": best_model_results['test_results']['precision'],
            "recall": best_model_results['test_results']['recall'],
            "specificity": best_model_results['test_results']['specificity'],
            "f1_score": best_model_results['test_results']['f1'],
            "roc_auc": best_model_results['test_results']['roc_auc'],
            "confusion_matrix": best_model_results['test_results']['confusion_matrix']
        },
        "model_comparison": {
            "models_tested": ["Logistic Regression", "Random Forest", "XGBoost"],
            "selection_criteria": "Highest ROC-AUC on test set",
            "selection_reason": "Logistic Regression achieved the highest ROC-AUC (0.7023) on the independent test set, indicating better discriminative ability for cardiovascular risk prediction",
            "all_models_performance": {
                "Logistic Regression": {
                    "test_roc_auc": best_model_results['test_results']['roc_auc'],
                    "test_accuracy": best_model_results['test_results']['accuracy'],
                    "test_f1": best_model_results['test_results']['f1']
                },
                "Random Forest": {
                    "test_roc_auc": training_results['Random Forest']['test_results']['roc_auc'],
                    "test_accuracy": training_results['Random Forest']['test_results']['accuracy'],
                    "test_f1": training_results['Random Forest']['test_results']['f1']
                },
                "XGBoost": {
                    "test_roc_auc": training_results['XGBoost']['test_results']['roc_auc'],
                    "test_accuracy": training_results['XGBoost']['test_results']['accuracy'],
                    "test_f1": training_results['XGBoost']['test_results']['f1']
                }
            }
        },
        "limitations": [
            "Historical data (1956-1968) may not reflect modern risk profiles",
            "Geographic bias - participants from Framingham, Massachusetts only",
            "Class imbalance (15.2% positive class) affects model performance",
            "Missing values handled through imputation (may introduce bias)",
            "Model not clinically validated - for research/educational purposes only",
            "Risk estimates are probabilistic, not diagnostic",
            "Does not account for all cardiovascular risk factors (e.g., family history, stress)",
            "Dataset represents teaching subset, not complete Framingham cohort"
        ],
        "intended_use": {
            "purpose": "Research and educational demonstration of machine learning for cardiovascular risk prediction",
            "not_intended_for": [
                "Clinical diagnosis",
                "Medical decision making",
                "Patient treatment decisions",
                "Regulatory approval submissions"
            ],
            "required_disclaimers": [
                "This is an AI-based research tool, not a medical device",
                "Predictions are risk estimates, not medical diagnoses",
                "Results should be reviewed by qualified healthcare professionals",
                "Model trained on historical data - may not reflect current medical practices",
                "Do not use for clinical decision making without proper validation"
            ]
        },
        "ethical_considerations": {
            "data_source": "Anonymized historical clinical data",
            "privacy": "No personally identifiable information in dataset",
            "bias_risks": [
                "Historical demographic bias (Framingham, MA population)",
                "Potential racial/ethnic underrepresentation",
                "Medical practice changes since data collection"
            ],
            "fairness_monitoring": "Recommended for deployment in diverse populations"
        },
        "maintenance": {
            "retraining_schedule": "Recommended when new data becomes available",
            "performance_monitoring": "Monitor ROC-AUC, calibration, and feature importance drift",
            "update_procedure": "Retrain with new data following same preprocessing pipeline"
        },
        "files": {
            "model_file": "models/logistic_regression.pkl",
            "preprocessor_file": "models/preprocessor.pkl",
            "feature_names_file": "models/preprocessor_feature_names.pkl",
            "training_results": "reports/model_training_results.json",
            "model_comparison": "reports/model_comparison.csv"
        }
    }
    
    # Save metadata
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    metadata_path = models_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Model metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL METADATA SUMMARY")
    print("="*70)
    print(f"Model: {metadata['model_info']['model_name']}")
    print(f"Test ROC-AUC: {metadata['test_results']['roc_auc']:.4f}")
    print(f"Test Accuracy: {metadata['test_results']['accuracy']:.4f}")
    print(f"Test Recall: {metadata['test_results']['recall']:.4f}")
    print(f"Test Specificity: {metadata['test_results']['specificity']:.4f}")
    print(f"Test F1-score: {metadata['test_results']['f1_score']:.4f}")
    print(f"\nConfusion Matrix (Test):")
    cm = metadata['test_results']['confusion_matrix']
    print(f"  True Negatives: {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")
    print(f"  True Positives: {cm['tp']}")
    
    print("\n" + "="*70)
    print("METADATA CREATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    create_metadata()
