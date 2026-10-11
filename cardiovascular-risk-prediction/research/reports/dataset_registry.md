# Dataset Registry

## Discovered Cardiovascular Datasets

### 1. Framingham Heart Study Teaching Dataset (Current Primary)
- **Source:** Kaggle/GitHub (derived from NHLBI Framingham Heart Study)
- **URL:** https://raw.githubusercontent.com/theleadio/datascience_demo/master/framingham.csv
- **Records:** 4,240
- **Features:** 15 (17 after one-hot encoding)
- **Target:** TenYearCHD (10-year coronary heart disease risk, binary)
- **Population:** Framingham, Massachusetts (historical 1956-1968)
- **License:** NHLBI teaching dataset (research/educational use permitted)
- **Current Use:** Primary development dataset
- **Status:** Already downloaded and processed

### 2. UCI Heart Disease Dataset (Cleveland Subset)
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Records:** 303 (Cleveland subset)
- **Features:** 14 commonly used
- **Target:** Heart disease presence (0-4 severity scale, often binarized)
- **Population:** Cleveland Clinic Foundation (1980s)
- **License:** CC BY 4.0
- **Potential Use:** External validation dataset
- **Status:** Compatible for external validation

### 3. Statlog Heart Dataset
- **Source:** UCI Machine Learning Repository
- **URL:** http://archive.ics.uci.edu/dataset/145/statlog+heart
- **Records:** 270
- **Features:** 13
- **Target:** Absence (1) or presence (2) of heart disease (binary)
- **Population:** Not specified (similar to UCI heart disease)
- **License:** CC BY 4.0
- **Potential Use:** External validation dataset
- **Status:** Compatible for external validation

### 4. BRFSS 2020 Heart Disease Dataset
- **Source:** CDC Behavioral Risk Factor Surveillance System
- **URL:** https://zenodo.org/records/15336526
- **Records:** 401,958 (large dataset)
- **Features:** 18 relevant features (cleaned from 279 original)
- **Target:** HeartDiseaseorAttack (binary)
- **Population:** U.S. residents (50 states + territories)
- **License:** Public domain (CDC data)
- **Potential Use:** Primary development dataset (alternative)
- **Status:** Very promising for large-scale training

### 5. Whitehall II Phase 9 Dataset
- **Source:** CLOSER Discovery (UK civil servants)
- **URL:** https://discovery.closer.ac.uk/item/uk.whitehall2/78709461-88f2-4a83-9da3-e3bf3848734a
- **Records:** Not specified in search results
- **Features:** Physical health, cardiovascular system data
- **Target:** Framingham 10-year risk of CVD
- **Population:** UK civil servants
- **License:** Not specified in search results
- **Potential Use:** External validation dataset
- **Status:** Requires further investigation

### 6. NHANES Cardiovascular Data
- **Source:** CDC National Health and Nutrition Examination Survey
- **URL:** https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- **Records:** Large (annual survey with thousands of participants)
- **Features:** Comprehensive cardiovascular variables
- **Target:** Various cardiovascular outcomes
- **Population:** U.S. nationally representative
- **License:** Public domain (CDC data)
- **Potential Use:** Large-scale development dataset
- **Status:** Complex, requires data access and processing

## Feature Compatibility Analysis

### Common Features Across Datasets

| Feature | Framingham | UCI Heart | Statlog | BRFSS | NHANES |
|---------|------------|-----------|---------|-------|--------|
| Age | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sex | ✓ | ✓ | ✓ | ✓ | ✓ |
| Blood Pressure (Systolic) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Blood Pressure (Diastolic) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cholesterol | ✓ | ✓ | ✓ | ✓ | ✓ |
| Glucose/Diabetes | ✓ | ✓ | ✓ | ✓ | ✓ |
| Smoking | ✓ | ✓ | ✓ | ✓ | ✓ |
| BMI | ✓ | ✗ | ✗ | ✓ | ✓ |
| Physical Activity | ✗ | ✗ | ✗ | ✓ | ✓ |
| Heart Rate | ✓ | ✓ | ✓ | ✗ | ✓ |

### Target Compatibility

| Dataset | Target Definition | Type | Compatible with Framingham |
|---------|------------------|------|---------------------------|
| Framingham | 10-year CHD risk | Binary (0/1) | N/A (baseline) |
| UCI Heart | Heart disease presence | Binary (0/1) | ❌ Different definition |
| Statlog | Heart disease presence | Binary (1/2) | ❌ Different definition |
| BRFSS | Heart disease/attack | Binary (Yes/No) | ❌ Different definition |
| Whitehall II | 10-year CVD risk | Numeric | ⚠️ Potentially compatible |
| NHANES | Various CVD outcomes | Mixed | ❌ Multiple outcomes |

## Target Incompatibility Analysis

### Critical Finding: Targets Are NOT Interchangeable

The datasets have fundamentally different target definitions:

1. **Framingham:** 10-year coronary heart disease risk (prospective)
2. **UCI Heart:** Presence of heart disease at examination time (diagnostic)
3. **Statlog:** Presence of heart disease (diagnostic)
4. **BRFSS:** Ever had heart disease/attack (prevalence)
5. **Whitehall II:** 10-year CVD risk (potentially compatible)
6. **NHANES:** Various cardiovascular outcomes (mixed)

**Conclusion:** Most targets are semantically incompatible:
- **Diagnostic vs. Risk Prediction:** UCI/Statlog are diagnostic (current disease), Framingham is risk prediction (future disease)
- **Prevalence vs. Incidence:** BRFSS measures prevalence (ever had), Framingham measures incidence (will develop)
- **Different Disease Definitions:** Heart disease, CHD, CVD are not the same

## Recommended Dataset Strategy

### Primary Development Dataset: Framingham Heart Study
**Rationale:**
- Well-documented and medically authoritative
- Clear 10-year risk prediction target
- Comprehensive cardiovascular risk factors
- Compatible with clinical risk assessment
- Already processed and validated

### External Validation Dataset: None (CURRENTLY)
**Rationale:**
- UCI Heart Disease: Incompatible target (diagnostic vs. risk prediction)
- Statlog: Incompatible target (diagnostic vs. risk prediction)
- BRFSS: Incompatible target (prevalence vs. incidence)
- Whitehall II: Insufficient information, potential compatibility unknown
- NHANES: Complex, requires data access, mixed outcomes

**Decision:** EXTERNAL VALIDATION UNAVAILABLE because no compatible independent dataset was identified.

### Alternative Strategy: Internal Cross-Validation Enhancement
Since no compatible external dataset exists, we will:
1. Use the Framingham dataset with enhanced methodology
2. Implement strict train/validation/test split with locked holdout
3. Use stratified k-fold cross-validation for robust evaluation
4. Perform comprehensive overfitting analysis
5. Implement proper calibration and threshold optimization
6. Acknowledge the limitation in the final report

## Dataset Quality Summary

### Framingham Dataset (Primary)
- **Size:** 4,240 records (moderate, larger than reference paper's ~1,001)
- **Features:** 15 comprehensive cardiovascular risk factors
- **Target:** 10-year CHD risk (clinically relevant)
- **Quality:** High (well-documented, cleaned teaching dataset)
- **Missing Values:** 645 total (manageable)
- **Class Imbalance:** 5.58:1 (addressed with class weighting)

### Reference Paper Dataset Comparison
- **Reference Paper:** ~1,001 unique records from Dhaka, Bangladesh
- **Our Dataset:** 4,240 records from Framingham, Massachusetts
- **Size Improvement:** 4.2x larger than reference paper
- **Feature Count:** 15 vs. 8 (more comprehensive)
- **Geographic Diversity:** U.S. vs. Bangladesh (different populations)

## Next Steps

1. **Proceed with Framingham as primary dataset** (enhanced methodology)
2. **Implement advanced feature engineering** (pulse pressure, age groups, etc.)
3. **Use proper train/validation/test split** (80/10/10 with locked holdout)
4. **Implement comprehensive model comparison** (multiple algorithms)
5. **Optimize XGBoost with GridSearchCV** (address reference paper limitation)
6. **Implement SHAP explainability** (global and patient-level)
7. **Perform rigorous overfitting analysis** (address reference paper limitation)
8. **Implement ensemble methods** (address reference paper limitation)
9. **External validation:** Report as unavailable due to target incompatibility
