"""
Research Preprocessing Pipeline with Feature Engineering
Leakage-safe preprocessing with advanced feature engineering for cardiovascular risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import json
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class ResearchPreprocessor:
    """Enhanced preprocessing pipeline with feature engineering"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self.feature_names_out = None
        self.engineered_features = {}
        
    def load_data(self):
        """Load the Framingham dataset"""
        current_dir = Path.cwd()
        data_path = current_dir / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(data_path)
        print(f"[OK] Dataset loaded: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """Engineer clinically relevant features"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        df_engineered = df.copy()
        
        # 1. Pulse pressure (sysBP - diaBP)
        df_engineered['pulse_pressure'] = df_engineered['sysBP'] - df_engineered['diaBP']
        print("[FEATURE] pulse_pressure = sysBP - diaBP")
        
        # 2. Age groups
        df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                               bins=[0, 40, 50, 60, 100],
                                               labels=['<40', '40-49', '50-59', '60+'])
        print("[FEATURE] age_group: <40, 40-49, 50-59, 60+")
        
        # 3. BMI categories
        df_engineered['bmi_category'] = pd.cut(df_engineered['BMI'],
                                                bins=[0, 18.5, 25, 30, 100],
                                                labels=['underweight', 'normal', 'overweight', 'obese'])
        print("[FEATURE] bmi_category: underweight, normal, overweight, obese")
        
        # 4. BP categories (ACC/AHA guidelines)
        def categorize_bp(sys_bp, dia_bp):
            if sys_bp < 120 and dia_bp < 80:
                return 'normal'
            elif 120 <= sys_bp < 130 or 80 <= dia_bp < 85:
                return 'elevated'
            elif 130 <= sys_bp < 140 or 85 <= dia_bp < 90:
                return 'hypertension_stage1'
            else:
                return 'hypertension_stage2'
        
        df_engineered['bp_category'] = df_engineered.apply(
            lambda row: categorize_bp(row['sysBP'], row['diaBP']), axis=1)
        print("[FEATURE] bp_category: normal, elevated, hypertension_stage1, hypertension_stage2")
        
        # 5. Cholesterol categories (NCEP guidelines)
        def categorize_cholesterol(chol):
            if chol < 200:
                return 'desirable'
            elif 200 <= chol < 240:
                return 'borderline_high'
            else:
                return 'high'
        
        df_engineered['cholesterol_category'] = df_engineered['totChol'].apply(categorize_cholesterol)
        print("[FEATURE] cholesterol_category: desirable, borderline_high, high")
        
        # 6. Metabolic syndrome indicators
        # High BP, high glucose, abnormal cholesterol are key components
        df_engineered['high_bp'] = (df_engineered['sysBP'] >= 130) | (df_engineered['diaBP'] >= 85).astype(int)
        df_engineered['high_glucose'] = (df_engineered['glucose'] >= 100).astype(int)
        df_engineered['high_cholesterol'] = (df_engineered['totChol'] >= 240).astype(int)
        print("[FEATURE] Metabolic syndrome indicators: high_bp, high_glucose, high_cholesterol")
        
        self.engineered_features = {
            'pulse_pressure': 'sysBP - diaBP',
            'age_group': 'Age categorization',
            'bmi_category': 'BMI categorization',
            'bp_category': 'BP categorization',
            'cholesterol_category': 'Cholesterol categorization',
            'high_bp': 'Systolic >=130 or Diastolic >=85',
            'high_glucose': 'Glucose >= 100',
            'high_cholesterol': 'Cholesterol >= 240'
        }
        
        print(f"\n[OK] Engineered {len(self.engineered_features)} new features")
        return df_engineered
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """
        Split data into development (80%) and locked holdout (20%)
        Development split further into train/validation
        """
        print("\n" + "="*70)
        print("DATA SPLIT (Development + Locked Holdout)")
        print("="*70)
        
        # Separate features and target
        X = df.drop('TenYearCHD', axis=1)
        y = df['TenYearCHD']
        
        print(f"Total samples: {len(df)}")
        print(f"Features: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # First split: development (80%) vs locked holdout (20%)
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # Stratified split to maintain class distribution
        )
        
        # Second split: development data into train (89% of dev) vs val (11% of dev)
        # This gives us 71.2% train, 8.8% val, 20% holdout of original data
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev,
            test_size=val_size / (1 - test_size),  # Adjust for first split
            random_state=self.random_state,
            stratify=y_dev
        )
        
        print(f"\nSplit ratios:")
        print(f"  Development: {len(X_dev)} ({len(X_dev)/len(df)*100:.1f}%)")
        print(f"    Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"    Validation: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Locked Holdout: {len(X_holdout)} ({len(X_holdout)/len(df)*100:.1f}%)")
        
        print(f"\nClass distribution in splits:")
        print(f"  Train - Negative: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%), Positive: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"  Validation - Negative: {(y_val==0).sum()} ({(y_val==0).sum()/len(y_val)*100:.1f}%), Positive: {(y_val==1).sum()} ({(y_val==1).sum()/len(y_val)*100:.1f}%)")
        print(f"  Holdout - Negative: {(y_holdout==0).sum()} ({(y_holdout==0).sum()/len(y_holdout)*100:.1f}%), Positive: {(y_holdout==1).sum()} ({(y_holdout==1).sum()/len(y_holdout)*100:.1f}%)")
        
        return X_train, X_val, X_holdout, y_train, y_val, y_holdout
    
    def create_preprocessor(self):
        """Create preprocessing pipeline with feature engineering"""
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
        
        # Define feature categories (after feature engineering)
        binary_features = ['sex', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                         'high_bp', 'high_glucose', 'high_cholesterol']
        categorical_features = ['education', 'age_group', 'bmi_category', 'bp_category', 'cholesterol_category']
        continuous_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 
                            'pulse_pressure']
        
        # Create transformers for each feature type
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
        
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine all transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('binary', binary_transformer, binary_features),
                ('categorical', categorical_transformer, categorical_features),
                ('continuous', continuous_transformer, continuous_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        print("Preprocessing steps:")
        print("  Binary features: Mode imputation")
        print("  Categorical features: Mode imputation + One-hot encoding")
        print("  Continuous features: Median imputation + Standard scaling")
        print(f"  Total features: {len(binary_features + categorical_features + continuous_features)}")
        
        return self.preprocessor
    
    def fit_preprocessor(self, X_train):
        """Fit preprocessor on training data only (leakage prevention)"""
        print("\n" + "="*70)
        print("FITTING PREPROCESSOR (Training Data Only)")
        print("="*70)
        
        # Store original feature names
        self.feature_names = X_train.columns.tolist()
        print(f"Fitting on {len(X_train)} training samples")
        print(f"Features: {self.feature_names}")
        
        # Fit preprocessor
        self.preprocessor.fit(X_train)
        
        # Get feature names after transformation
        self.feature_names_out = self.get_feature_names_out()
        print(f"Transformed features: {len(self.feature_names_out)}")
        
        print("[OK] Preprocessor fitted on training data only")
        
        return self.preprocessor
    
    def get_feature_names_out(self):
        """Get feature names after preprocessing"""
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'binary':
                feature_names.extend(features)
            elif name == 'categorical':
                encoder = transformer.named_steps['onehot']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(features)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend([f"{feat}_{i}" for feat in features for i in range(len(encoder.categories_[0])-1)])
            elif name == 'continuous':
                feature_names.extend(features)
        
        return feature_names
    
    def transform_data(self, X):
        """Transform data using fitted preprocessor"""
        X_transformed = self.preprocessor.transform(X)
        return X_transformed
    
    def save_preprocessor(self, path):
        """Save fitted preprocessor"""
        joblib.dump(self.preprocessor, path)
        joblib.dump(self.feature_names_out, path.replace('.pkl', '_feature_names.pkl'))
        joblib.dump(self.engineered_features, path.replace('.pkl', '_engineered_features.json'))
        print(f"[OK] Preprocessor saved to {path}")
    
    def save_split_data(self, X_train, X_val, X_holdout, y_train, y_val, y_holdout,
                        X_train_transformed, X_val_transformed, X_holdout_transformed):
        """Save split data"""
        data_dir = Path.cwd() / 'research' / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # Save transformed data as numpy arrays
        np.save(data_dir / 'X_train.npy', X_train_transformed)
        np.save(data_dir / 'X_val.npy', X_val_transformed)
        np.save(data_dir / 'X_holdout.npy', X_holdout_transformed)
        np.save(data_dir / 'y_train.npy', y_train.values)
        np.save(data_dir / 'y_val.npy', y_val.values)
        np.save(data_dir / 'y_holdout.npy', y_holdout.values)
        
        print(f"[OK] Split data saved to {data_dir}")
        
        # Save metadata
        metadata = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'holdout_samples': len(X_holdout),
            'total_samples': len(X_train) + len(X_val) + len(X_holdout),
            'feature_names': X_train.columns.tolist(),
            'engineered_features': self.engineered_features,
            'target_distribution_train': {
                'negative': int((y_train==0).sum()),
                'positive': int((y_train==1).sum())
            },
            'target_distribution_val': {
                'negative': int((y_val==0).sum()),
                'positive': int((y_val==1).sum())
            },
            'target_distribution_holdout': {
                'negative': int((y_holdout==0).sum()),
                'positive': int((y_holdout==1).sum())
            },
            'random_state': self.random_state
        }
        
        with open(data_dir / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Split metadata saved")

def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("RESEARCH PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = ResearchPreprocessor(random_state=42)
    
    # Load data
    df = preprocessor.load_data()
    
    # Engineer features
    df_engineered = preprocessor.engineer_features(df)
    
    # Split data (development + locked holdout)
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = preprocessor.split_data(df_engineered)
    
    # Create preprocessing pipeline
    preprocessor.create_preprocessor()
    
    # Fit preprocessor on training data only (leakage prevention)
    preprocessor.fit_preprocessor(X_train)
    
    # Transform all datasets
    print("\n" + "="*70)
    print("TRANSFORMING DATA")
    print("="*70)
    
    X_train_transformed = preprocessor.transform_data(X_train)
    X_val_transformed = preprocessor.transform_data(X_val)
    X_holdout_transformed = preprocessor.transform_data(X_holdout)
    
    print(f"Train transformed shape: {X_train_transformed.shape}")
    print(f"Validation transformed shape: {X_val_transformed.shape}")
    print(f"Holdout transformed shape: {X_holdout_transformed.shape}")
    
    # Save preprocessor
    models_dir = Path.cwd() / 'research' / 'models'
    models_dir.mkdir(exist_ok=True)
    preprocessor_path = models_dir / 'research_preprocessor.pkl'
    preprocessor.save_preprocessor(str(preprocessor_path))
    
    # Save split data
    preprocessor.save_split_data(X_train, X_val, X_holdout, y_train, y_val, y_holdout,
                                  X_train_transformed, X_val_transformed, X_holdout_transformed)
    
    print("\n" + "="*70)
    print("RESEARCH PREPROCESSING COMPLETE")
    print("="*70)
    print("\nKey improvements over production:")
    print("  1. Enhanced feature engineering (6 new features)")
    print("  2. Development/Holdout split (instead of Train/Val/Test)")
    print("  3. Locked holdout for final validation")
    print("  4. More clinical categorization features")
    print("  5. Metabolic syndrome indicators")
    print("\nNext steps:")
    print("1. Train baseline models on development data")
    print("2. Optimize XGBoost with GridSearchCV")
    print("3. Evaluate on locked holdout only once")
    print("4. Implement SHAP explainability")

if __name__ == "__main__":
    main()
