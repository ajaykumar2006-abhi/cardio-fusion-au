# Data Quality Analysis Report

## Dataset Overview

- **Total Records**: 4240
- **Total Features**: 15
- **Target Variable**: TenYearCHD
- **Data Source**: Framingham Heart Study Teaching Dataset

## Duplicate Records

- **Exact Duplicates**: 0

[OK] No exact duplicates found

## Missing Values

| Column | Missing Count | Missing Percentage |
|--------|---------------|-------------------|
| glucose | 388 | 9.15% |
| education | 105 | 2.48% |
| BPMeds | 53 | 1.25% |
| totChol | 50 | 1.18% |
| cigsPerDay | 29 | 0.68% |
| BMI | 19 | 0.45% |
| heartRate | 1 | 0.02% |

## Invalid Values

[OK] No invalid values detected

## Outlier Analysis

Outliers detected using IQR method (1.5 × IQR):

| Variable | Outlier Count | Percentage | Lower Bound | Upper Bound | Min | Max |
|----------|---------------|------------|-------------|-------------|-----|-----|
| age | 0 | 0.00% | 21.00 | 77.00 | 32.00 | 70.00 |
| totChol | 56 | 1.34% | 120.50 | 348.50 | 107.00 | 696.00 |
| sysBP | 126 | 2.97% | 76.50 | 184.50 | 83.50 | 295.00 |
| diaBP | 77 | 1.82% | 52.50 | 112.50 | 48.00 | 142.50 |
| BMI | 97 | 2.30% | 15.62 | 35.49 | 15.54 | 56.80 |
| heartRate | 76 | 1.79% | 45.50 | 105.50 | 44.00 | 143.00 |
| glucose | 188 | 4.88% | 47.00 | 111.00 | 40.00 | 394.00 |
| cigsPerDay | 12 | 0.28% | -30.00 | 50.00 | 0.00 | 70.00 |

## Target Variable Analysis

- **Variable**: TenYearCHD
- **Type**: binary
- **Negative Class (No CHD)**: 3596 (84.81%)
- **Positive Class (CHD)**: 644 (15.19%)
- **Imbalance Ratio**: 5.58:1
- **Missing Values**: 0

## Class Imbalance

- **Total Samples**: 4240
- **Negative Class**: 3596 (84.81%)
- **Positive Class**: 644 (15.19%)
- **Imbalance Ratio**: 5.58:1

**Recommended Handling Strategies:**
1. Class weighting in model training
2. SMOTE or other oversampling techniques
3. Undersampling majority class
4. Use appropriate evaluation metrics (ROC-AUC, PR-AUC)
5. Stratified sampling for train/test split

## Feature Types

**Binary Features (6):**
sex, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes

**Categorical Features (1):**
education

**Continuous Features (8):**
age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose

**Target Variable:**
TenYearCHD

## Preprocessing Recommendations

1. **Missing Value Handling:**
   - Glucose: Median imputation (high missing rate)
   - Education: Mode imputation or separate category
   - BMI, heartRate: Median imputation
   - cigsPerDay: If currentSmoker=0, set to 0; otherwise median imputation

2. **Feature Scaling:**
   - Standard scaling for continuous variables
   - No scaling needed for binary variables

3. **Categorical Encoding:**
   - Education: One-hot encoding or ordinal encoding
   - Binary variables: Already encoded (0/1)

4. **Class Imbalance:**
   - Use class weights in model training
   - Consider SMOTE for training data only
   - Use stratified train/validation/test split

5. **Outlier Handling:**
   - Review outliers on case-by-case basis
   - Most outliers appear biologically plausible
   - Consider capping extreme values if needed

## Data Quality Assessment

**Overall Quality: GOOD**

[OK] No duplicate records
[OK] No invalid values detected
[OK] Target variable complete and well-defined
[OK] Features clinically relevant and well-documented
[WARNING] Missing values present (manageable)
[WARNING] Class imbalance present (manageable)
[WARNING] Some outliers present (appear biologically plausible)

## Conclusion

The dataset is suitable for machine learning after appropriate preprocessing. Missing values and class imbalance need to be addressed. Outliers appear to be biologically plausible rather than data errors. The dataset contains comprehensive cardiovascular risk factors suitable for risk prediction modeling.
