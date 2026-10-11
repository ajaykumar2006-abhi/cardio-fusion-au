# Framingham Heart Study Dataset

## Dataset Information

**Dataset Name:** Framingham Heart Study Teaching Dataset  
**Source:** Kaggle (derived from the Framingham Heart Study, NHLBI Project No. N01-HC-25195)  
**Download URL:** https://raw.githubusercontent.com/theleadio/datascience_demo/master/framingham.csv  
**License:** Provided with permission from the National Heart, Lung, and Blood Institute (NHLBI) for teaching and research purposes  
**Original Study:** Framingham Heart Study - Long-term prospective study of cardiovascular disease etiology

## Dataset Overview

The Framingham Heart Study is a landmark epidemiological study that began in 1948 and identified the concept of risk factors for cardiovascular disease. This teaching dataset is a subset of the full study data, containing 4,240 observations with 16 variables.

**Target Variable:** TenYearCHD - 10-year risk of coronary heart disease (CHD)  
**Number of Records:** 4,240  
**Number of Features:** 15 (excluding target)  
**Time Period:** Data collected from examination periods approximately 6 years apart (1956-1968)  
**Follow-up Period:** 24 years for cardiovascular outcomes

## Features

### Demographic Information
- **sex**: Sex of participant (0 = Female, 1 = Male)
- **age**: Age at examination (years)
- **education**: Education level (1 = Some high school, 2 = High school/GED, 3 = Some college/vocational school, 4 = College graduate)

### Clinical Information
- **currentSmoker**: Current cigarette smoking at exam (0 = No, 1 = Yes)
- **cigsPerDay**: Number of cigarettes smoked per day (0-90)
- **BPMeds**: Use of anti-hypertensive medication at exam (0 = No, 1 = Yes)
- **prevalentStroke**: Prevalent stroke (0 = No, 1 = Yes)
- **prevalentHyp**: Prevalent hypertension (0 = No, 1 = Yes)
- **diabetes**: Diabetic according to criteria (0 = No, 1 = Yes)

### Measurements
- **totChol**: Total cholesterol level (mg/dL)
- **sysBP**: Systolic blood pressure (mmHg)
- **diaBP**: Diastolic blood pressure (mmHg)
- **BMI**: Body Mass Index (kg/m²)
- **heartRate**: Heart rate (beats/minute)
- **glucose**: Blood glucose level (mg/dL)

### Target Variable
- **TenYearCHD**: 10-year risk of coronary heart disease (0 = No, 1 = Yes)

## Data Quality Notes

Based on initial inspection:
- Missing values are present (noted as "NA" in the dataset)
- Dataset includes standard cardiovascular risk factors
- Binary target variable (CHD event within 10 years)
- Mix of categorical and continuous variables

## Dataset Limitations

1. **Historical Data**: Collected from 1956-1968, may not reflect modern risk profiles
2. **Geographic Bias**: Participants from Framingham, Massachusetts only
3. **Population Subset**: Teaching dataset, not the complete Framingham cohort
4. **Age Range**: Limited to original cohort age range
5. **Missing Values**: Some fields contain missing data that need handling
6. **Temporal Changes**: Medical practices and risk factor definitions have evolved

## Intended Use

This dataset is suitable for:
- Educational purposes and machine learning demonstrations
- Research on cardiovascular risk prediction
- Developing and testing risk prediction models
- Studying feature importance in cardiovascular disease

## Ethical Considerations

- This is real patient data (anonymized)
- Results should not be used for clinical diagnosis without proper validation
- Model predictions are risk estimates, not medical diagnoses
- Results should be interpreted by qualified healthcare professionals
- Dataset reflects historical medical practices and may have demographic biases

## References

- Framingham Heart Study: https://www.framinghamheartstudy.org/
- NHLBI BioLINCC: https://biolincc.nhlbi.nih.gov/studies/framcohort/
- Original Study: Dawber TR, Meadors GF, Moore FE Jr. Epidemiological approaches to heart disease: the Framingham Study. Am J Public Health. 1951;41(3):279-286.
