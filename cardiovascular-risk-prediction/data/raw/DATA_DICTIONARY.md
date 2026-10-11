# Framingham Heart Study Data Dictionary

## Variable Descriptions

| Variable Name | Type | Description | Values | Missing Notes |
|---------------|------|-------------|--------|---------------|
| sex | Binary | Sex of participant | 0 = Female, 1 = Male | None expected |
| age | Continuous | Age at examination | Years (integer) | None expected |
| education | Categorical | Highest education level achieved | 1 = Some high school (0-11 years)<br>2 = High school diploma/GED<br>3 = Some college/vocational school<br>4 = College graduate | May have missing values |
| currentSmoker | Binary | Current cigarette smoking at exam | 0 = No, 1 = Yes | None expected |
| cigsPerDay | Continuous | Number of cigarettes smoked per day | 0-90 (0 if non-smoker) | May have missing values |
| BPMeds | Binary | Use of anti-hypertensive medication | 0 = No, 1 = Yes | May have missing values |
| prevalentStroke | Binary | History of stroke | 0 = No, 1 = Yes | None expected |
| prevalentHyp | Binary | Prevalent hypertension | 0 = No, 1 = Yes<br>Defined as treated or SBP ≥140 or DBP ≥90 | None expected |
| diabetes | Binary | Diabetic status | 0 = No, 1 = Yes<br>Defined as treated or casual glucose ≥200 mg/dL | None expected |
| totChol | Continuous | Total cholesterol level | mg/dL | May have missing values |
| sysBP | Continuous | Systolic blood pressure | mmHg (mean of last 2 of 3 measurements) | None expected |
| diaBP | Continuous | Diastolic blood pressure | mmHg (mean of last 2 of 3 measurements) | None expected |
| BMI | Continuous | Body Mass Index | kg/m² (weight/height²) | May have missing values |
| heartRate | Continuous | Heart rate | Beats per minute (ventricular rate) | May have missing values |
| glucose | Continuous | Blood glucose level | mg/dL (casual serum glucose) | May have missing values |
| TenYearCHD | Binary | Target variable - 10-year CHD risk | 0 = No CHD event within 10 years<br>1 = CHD event within 10 years | None expected |

## Clinical Context

### Blood Pressure Definitions
- **Systolic BP (sysBP)**: Pressure when heart beats
- **Diastolic BP (diaBP)**: Pressure when heart rests between beats
- **Hypertension**: Generally defined as SBP ≥140 mmHg or DBP ≥90 mmHg

### Cholesterol Definitions
- **Total Cholesterol (totChol)**: Combined measure of all cholesterol types
- Normal range: <200 mg/dL
- Borderline high: 200-239 mg/dL
- High: ≥240 mg/dL

### BMI Categories
- Underweight: <18.5 kg/m²
- Normal weight: 18.5-24.9 kg/m²
- Overweight: 25-29.9 kg/m²
- Obese: ≥30 kg/m²

### Glucose Definitions
- Normal fasting glucose: <100 mg/dL
- Prediabetes: 100-125 mg/dL
- Diabetes: ≥126 mg/dL (fasting) or ≥200 mg/dL (casual)

### CHD Events
Coronary Heart Disease includes:
- Angina pectoris
- Myocardial infarction (heart attack)
- Coronary insufficiency
- Sudden cardiac death

## Variable Categories

### Demographic Variables
- sex
- age
- education

### Behavioral/Lifestyle Variables
- currentSmoker
- cigsPerDay

### Medical History Variables
- BPMeds
- prevalentStroke
- prevalentHyp
- diabetes

### Clinical Measurements
- totChol
- sysBP
- diaBP
- BMI
- heartRate
- glucose

### Outcome Variable
- TenYearCHD (target)

## Data Preprocessing Considerations

### Missing Value Handling
Variables with potential missing values:
- education
- cigsPerDay
- BPMeds
- totChol
- BMI
- heartRate
- glucose

Missing value strategies to consider:
- For clinical measurements: Imputation with median or regression-based imputation
- For categorical variables: Mode imputation or separate category
- For smoking: If currentSmoker=0, cigsPerDay should be 0

### Feature Engineering Considerations
- Age categories (e.g., <40, 40-49, 50-59, 60+)
- BMI categories
- Blood pressure categories
- Cholesterol categories
- Interaction terms (e.g., age × smoking)
- Framingham risk score calculation (for comparison)

### Scaling Considerations
Continuous variables may require scaling:
- age
- totChol
- sysBP
- diaBP
- BMI
- heartRate
- glucose
- cigsPerDay

### Encoding Considerations
Categorical variables requiring encoding:
- education (ordinal - can be treated as numeric)
- Binary variables already encoded (0/1)

## Target Variable Details

**TenYearCHD**: Binary classification target
- 0 = No coronary heart disease event within 10-year follow-up
- 1 = Coronary heart disease event within 10-year follow-up

**Note**: This is a binary outcome representing whether the patient experienced a CHD event during the 10-year follow-up period after their baseline examination.
