# Dataset Research and Selection Report

## Executive Summary

After researching multiple publicly available cardiovascular disease datasets, the **Framingham Heart Study Teaching Dataset** was selected for this cardiovascular risk prediction project. This decision was based on the dataset's clinical relevance, feature completeness, target clarity, and widespread use in cardiovascular research.

## Datasets Considered

### 1. Framingham Heart Study (Kaggle/GitHub) - **SELECTED**
- **Source**: Multiple public repositories (Kaggle, GitHub)
- **Download URL**: https://raw.githubusercontent.com/theleadio/datascience_demo/master/framingham.csv
- **Records**: 4,240
- **Features**: 15 (plus target)
- **Target**: TenYearCHD (10-year coronary heart disease risk)
- **License**: NHLBI teaching dataset (research/educational use permitted)
- **Status**: ✅ SELECTED

### 2. UCI Heart Disease Dataset (Cleveland Subset)
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/45/heart+disease
- **Records**: 303 (Cleveland subset)
- **Features**: 14 commonly used
- **Target**: Heart disease presence (0-4 severity scale, often binarized)
- **License**: CC BY 4.0
- **Status**: Not selected (smaller sample size, diagnostic focus vs. risk prediction)

### 3. Combined Cardiovascular Disease Dataset (IEEE DataPort)
- **Source**: IEEE DataPort
- **URL**: https://ieee-dataport.org/documents/cardiovascular-disease-dataset
- **Records**: 70,000+ (combined from multiple sources)
- **Features**: 11
- **Target**: Heart disease presence
- **License**: IEEE DataPort access required
- **Status**: Not selected (access restrictions, combined dataset may have inconsistencies)

### 4. Full Framingham Heart Study (BioLINCC)
- **Source**: NHLBI BioLINCC
- **URL**: https://biolincc.nhlbi.nih.gov/studies/framcohort/
- **Records**: 5,209+ (multiple cohorts)
- **Features**: 39+ variables
- **Target**: Multiple cardiovascular outcomes
- **License**: Requires application and approval
- **Status**: Not selected (access restrictions, complex approval process)

## Selection Rationale

### Why Framingham Teaching Dataset Was Chosen

#### 1. **Clinical Relevance and Authority**
- The Framingham Heart Study is the landmark study that established the concept of cardiovascular risk factors
- Widely recognized and validated in medical literature
- Forms the basis for the Framingham Risk Score used in clinical practice
- Peer-reviewed and extensively studied

#### 2. **Target Variable Clarity**
- Clear binary target: TenYearCHD (10-year coronary heart disease risk)
- Well-defined outcome: CHD event within 10-year follow-up
- Clinically meaningful prediction task
- Aligns with real-world cardiovascular risk assessment

#### 3. **Feature Completeness**
- Contains all major cardiovascular risk factors:
  - Demographics: age, sex, education
  - Clinical measurements: blood pressure, cholesterol, BMI, heart rate, glucose
  - Medical history: diabetes, hypertension, stroke
  - Lifestyle: smoking status, cigarettes per day
  - Medications: blood pressure medications
- Features are standard in cardiovascular risk assessment
- No need for feature engineering from raw clinical data

#### 4. **Adequate Sample Size**
- 4,240 records sufficient for machine learning
- Large enough for train/validation/test split
- Supports cross-validation
- Reasonable for computational resources

#### 5. **Data Quality and Documentation**
- Well-documented dataset with data dictionary available
- Clear variable definitions and clinical context
- Missing values documented and manageable
- Extensive research literature provides benchmarks

#### 6. **Accessibility and License**
- Publicly available without requiring special access
- Permission from NHLBI for teaching/research use
- No restrictive licensing barriers
- Can be downloaded directly

#### 7. **Research Benchmarking**
- Used in numerous published studies
- Performance benchmarks available in literature
- Enables comparison with published results
- SHAP-based interpretability studies available

### Why Other Datasets Were Not Selected

#### UCI Heart Disease Dataset
- **Sample size**: Only 303 records (too small for robust model development)
- **Focus**: Diagnostic classification (presence/absence) rather than risk prediction
- **Context**: Older dataset with different clinical context
- **Severity scale**: Multi-class target (0-4) often requires binarization

#### Combined Cardiovascular Dataset
- **Access**: Requires IEEE DataPort subscription
- **Consistency**: Combined from multiple sources may have inconsistencies
- **Documentation**: Less comprehensive documentation
- **Validation**: Less extensively validated in literature

#### Full Framingham Study (BioLINCC)
- **Access**: Requires formal application and approval process
- **Complexity**: Multiple cohorts, complex data structure
- **Timeline**: Approval process could delay project
- **Scope**: More data than needed for this specific project

## Dataset Details

### Selected Dataset: Framingham Heart Study Teaching Dataset

**Basic Information**
- **Name**: Framingham Heart Study Teaching Dataset
- **Source**: Kaggle/GitHub (derived from NHLBI Framingham Heart Study)
- **Original Study**: NHLBI Project No. N01-HC-25195
- **Download URL**: https://raw.githubusercontent.com/theleadio/datascience_demo/master/framingham.csv
- **File Size**: ~196 KB
- **Format**: CSV

**Dataset Statistics**
- **Total Records**: 4,240
- **Total Features**: 15 (excluding target)
- **Target Variable**: TenYearCHD (binary)
- **Data Collection Period**: 1956-1968 (examination periods)
- **Follow-up Period**: 24 years for outcomes

**Features Summary**
| Category | Features |
|----------|----------|
| Demographic | sex, age, education |
| Lifestyle | currentSmoker, cigsPerDay |
| Medical History | BPMeds, prevalentStroke, prevalentHyp, diabetes |
| Clinical Measurements | totChol, sysBP, diaBP, BMI, heartRate, glucose |
| Target | TenYearCHD |

**Target Definition**
- **TenYearCHD = 0**: No coronary heart disease event within 10-year follow-up
- **TenYearCHD = 1**: Coronary heart disease event within 10-year follow-up

**License and Usage Terms**
- Provided with permission from NHLBI for teaching and research purposes
- Suitable for educational machine learning projects
- Results should not be used for clinical diagnosis without proper validation
- Must acknowledge NHLBI Framingham Heart Study in any publications

## Dataset Limitations

### Known Limitations
1. **Historical Data**: Collected 1956-1968, may not reflect modern risk profiles
2. **Geographic Bias**: Participants from Framingham, Massachusetts only
3. **Population Subset**: Teaching dataset is not the complete Framingham cohort
4. **Demographic Changes**: Population demographics and risk factors have evolved
5. **Medical Practice Changes**: Diagnostic criteria and treatments have changed
6. **Missing Values**: Some variables contain missing data requiring handling
7. **Age Range**: Limited to original cohort age range at time of examination

### Mitigation Strategies
- Acknowledge limitations in model documentation
- Do not claim clinical validation
- Clearly present model as research/educational tool
- Include appropriate disclaimers in the application
- Focus on demonstrating ML methodology rather than clinical utility

## Class Distribution

Based on published research on this dataset:
- Approximately 15-20% positive class (TenYearCHD = 1)
- Approximately 80-85% negative class (TenYearCHD = 0)
- Class imbalance will need to be addressed in model training

*Note: Exact distribution will be calculated during data quality analysis phase.*

## Missing Values

Based on initial inspection:
- Glucose has notable missing values
- Education, BMI, heartRate may have missing values
- CigsPerDay may have missing values for smokers
- Missing value strategy will be determined during preprocessing

## Conclusion

The Framingham Heart Study Teaching Dataset was selected because it:
1. ✅ Is from a medically authoritative source
2. ✅ Has a clear, clinically relevant target variable
3. ✅ Contains comprehensive cardiovascular risk factors
4. ✅ Has adequate sample size for model development
5. ✅ Is publicly accessible with appropriate license
6. ✅ Is well-documented and widely used in research
7. ✅ Provides benchmarks for model comparison

The dataset limitations are acknowledged and will be documented in the final model metadata. The project will focus on demonstrating proper ML methodology, data leakage prevention, and explainable AI rather than claiming clinical utility.

## References

1. Framingham Heart Study: https://www.framinghamheartstudy.org/
2. NHLBI BioLINCC: https://biolincc.nhlbi.nih.gov/studies/framcohort/
3. Kaggle Dataset: https://www.kaggle.com/datasets/palakdoshijain/coronary-heart-disease-prediction-in-ten-years
4. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/45/heart+disease
5. Dawber TR, et al. Epidemiological approaches to heart disease: the Framingham Study. Am J Public Health. 1951.
6. Kannel WB, et al. Factors of risk in the development of coronary heart disease. Ann Intern Med. 1961.
