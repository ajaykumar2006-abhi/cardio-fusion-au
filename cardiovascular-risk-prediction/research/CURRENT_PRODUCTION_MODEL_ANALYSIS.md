# Current Production Model Analysis

## Existing Project Inspection

### Current Production Model Details

**Model Type:** Logistic Regression  
**Model Version:** 1.0  
**Training Date:** 2026-09-22  
**Random State:** 42

### Dataset Information
- **Dataset Name:** Framingham Heart Study Teaching Dataset
- **Source:** Kaggle/GitHub (derived from NHLBI Framingham Heart Study)
- **Total Samples:** 4,240
- **Feature Count:** 15 (17 after one-hot encoding)
- **Target Definition:** TenYearCHD - 10-year risk of coronary heart disease (0=No, 1=Yes)
- **Class Distribution:** 84.81% negative, 15.19% positive (5.58:1 imbalance)

### Current Performance Metrics

**Test Set Performance (848 samples):**
- **Accuracy:** 0.6781 (67.81%)
- **Precision:** 0.2584 (25.84%)
- **Recall (Sensitivity):** 0.5969 (59.69%)
- **Specificity:** 0.6926 (69.26%)
- **F1-Score:** 0.3607 (36.07%)
- **ROC-AUC:** 0.7023

**Cross-Validation Performance (5-Fold Stratified):**
- **CV Accuracy:** 0.6851 ± 0.0097
- **CV Precision:** 0.2786 ± 0.0066
- **CV Recall:** 0.6762 ± 0.0352
- **CV F1-Score:** 0.3944 ± 0.0103
- **CV ROC-AUC:** 0.7412 ± 0.0141

### Model Comparison (Test Set)
| Model | Test ROC-AUC | Test Accuracy | Test F1-Score |
|-------|-------------|---------------|--------------|
| **Logistic Regression (Selected)** | **0.7023** | **0.6781** | **0.3607** |
| Random Forest | 0.6460 | 0.8125 | 0.2167 |
| XGBoost | 0.6356 | 0.8101 | 0.2441 |

### Features Used
1. sex (binary)
2. age (continuous)
3. education (categorical, one-hot encoded)
4. currentSmoker (binary)
5. cigsPerDay (continuous)
6. BPMeds (binary)
7. prevalentStroke (binary)
8. prevalentHyp (binary)
9. diabetes (binary)
10. totChol (continuous)
11. sysBP (continuous)
12. diaBP (continuous)
13. BMI (continuous)
14. heartRate (continuous)
15. glucose (continuous)

### Preprocessing Pipeline
- **Missing Value Handling:** Mode imputation (binary/categorical), Median imputation (continuous)
- **Feature Encoding:** One-hot encoding for education, no encoding for binary features
- **Scaling:** Standard scaling for continuous features
- **Class Imbalance:** class_weight='balanced'

### Data Split
- **Training:** 2,544 samples (60.0%)
- **Validation:** 848 samples (20.0%)
- **Test:** 848 samples (20.0%)
- **Split Method:** Stratified random split
- **Data Leakage Prevention:** Verified

### Existing System Components
- **Backend:** Flask API with prediction endpoint
- **Frontend:** Multi-step form interface
- **Database:** SQLite with patients, assessments, clinical_data, lifestyle_data tables
- **Testing:** 26 automated tests (all passing)
- **Explainability:** Feature importance via logistic regression coefficients

### Limitations of Current Implementation
1. **Single Dataset:** Only uses Framingham dataset (geographic/temporal bias)
2. **No External Validation:** No independent dataset validation
3. **Limited Feature Engineering:** No derived features (pulse pressure, age groups, etc.)
4. **Basic Model Selection:** Only 3 models compared
5. **No Hyperparameter Optimization:** Default parameters used
6. **No Ensemble Methods:** Single model only
7. **No Calibration:** Probability calibration not implemented
8. **Basic Explainability:** Only coefficient-based importance, no SHAP
9. **No Threshold Analysis:** Default 0.5 threshold used
10. **Limited Error Analysis:** No detailed error subgroup analysis

### Comparison with Reference Paper

**Reference Paper (2025):**
- Dataset: ~1,001 unique records from Dhaka, Bangladesh
- Features: 8 attributes (Age, Gender, SBP, DBP, Cholesterol, Glucose, Smoking, Physical Activity)
- XGBoost Training Accuracy: 97.12%
- XGBoost Testing Accuracy: 86.07%
- XGBoost F1: 0.88
- XGBoost AUC: 0.91

**Current Production Model:**
- Dataset: 4,240 records from Framingham, Massachusetts
- Features: 15 attributes (comprehensive cardiovascular risk factors)
- Logistic Regression Test Accuracy: 67.81%
- Logistic Regression ROC-AUC: 0.7023
- Logistic Regression F1: 0.3607

**Note:** Results are not directly comparable due to different datasets, target definitions, populations, and features.

### Research Improvement Opportunities

Based on the reference paper limitations and current implementation gaps:

1. **Dataset Enhancement:** Find larger, more diverse cardiovascular datasets
2. **External Validation:** Implement independent dataset validation
3. **Advanced Modeling:** Implement proper XGBoost with hyperparameter optimization
4. **Ensemble Methods:** Test ensemble approaches (voting, stacking)
5. **SHAP Explainability:** Implement both global and patient-level SHAP
6. **Calibration:** Implement probability calibration
7. **Feature Engineering:** Add clinically relevant derived features
8. **Threshold Optimization:** Optimize decision threshold based on clinical priorities
9. **Comprehensive Error Analysis:** Detailed error subgroup analysis
10. **Leakage Audit:** Comprehensive leakage prevention verification

### Target Improvements

**Current Production Model Performance:**
- Accuracy: 67.81%
- ROC-AUC: 0.7023

**Research Targets:**
- **Target 1:** Improve over current production model
- **Target 2:** Reach ≥80% if scientifically achievable
- **Target 3:** Attempt ≥85% (reference paper baseline)
- **Target 4:** Explore ≥90% only if supported by legitimate data and validation

### Preservation Plan

The existing production model will be preserved:
- Original model file: `models/model.pkl` (copy to `models/production_model.pkl`)
- Original preprocessor: `models/preprocessor.pkl` (copy to `models/production_preprocessor.pkl`)
- Original metadata: `models/metadata.json` (copy to `models/production_metadata.json`)
- Original backend endpoint: `/api/v1/predict` (keep as production)
- New research endpoint: `/api/research/predict` (separate research model)

### Next Steps

1. Create research directory structure
2. Search for additional cardiovascular datasets
3. Perform dataset compatibility analysis
4. Select primary and external validation datasets
5. Implement enhanced research pipeline
