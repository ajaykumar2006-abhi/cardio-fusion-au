# Enhanced Data Quality Analysis Report

## Dataset Overview

- **Total Records:** 4240
- **Total Features:** 16
- **Memory Usage:** 0.52 MB

## Target Analysis

- **Target:** TenYearCHD
- **Type:** binary
- **Negative Class:** 3596 (84.81%)
- **Positive Class:** 644 (15.19%)
- **Imbalance Ratio:** 5.58:1
- **Missing Values:** 0

## Feature Engineering Opportunities

- **pulse_pressure:** Pulse pressure (difference between systolic and diastolic BP)
  Formula: sysBP - diaBP
  Clinical Relevance: Strong predictor of cardiovascular risk

- **age_groups:** Age group categorization
  Formula: age categories (<40, 40-49, 50-59, 60+)
  Clinical Relevance: Risk varies significantly by age group

- **bmi_categories:** BMI categorization
  Formula: BMI categories (underweight, normal, overweight, obese)
  Clinical Relevance: WHO BMI categories for risk assessment

- **bp_categories:** Blood pressure categorization
  Formula: BP categories (normal, elevated, hypertension stage 1, stage 2)
  Clinical Relevance: ACC/AHA BP categories for risk assessment

- **cholesterol_categories:** Cholesterol categorization
  Formula: Cholesterol categories (desirable, borderline high, high)
  Clinical Relevance: NCEP cholesterol categories for risk assessment

- **metabolic_syndrome_indicators:** Metabolic syndrome risk indicators
  Formula: Combined metabolic risk factors
  Clinical Relevance: Metabolic syndrome is a strong CVD risk factor

