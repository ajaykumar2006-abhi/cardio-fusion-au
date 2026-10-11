"""
Data Preprocessing Pipeline with Data Leakage Prevention
Implements proper train/validation/test split and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class CardioPreprocessor:
    """Preprocessing pipeline for cardiovascular risk prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self.feature_names_out = None
        
    def load_data(self):
        """Load the Framingham dataset"""
        data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(data_path)
        print(f"[OK] Dataset loaded: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean data and handle inconsistencies"""
        print("\n" + "="*70)
        print("DATA CLEANING")
        print("="*70)
        
        df_clean = df.copy()
        
        # Handle cigsPerDay: if currentSmoker=0, set cigsPerDay to 0
        smoker_zero_mask = (df_clean['currentSmoker'] == 0) & (df_clean['cigsPerDay'].notna())
        if smoker_zero_mask.sum() > 0:
            df_clean.loc[smoker_zero_mask, 'cigsPerDay'] = 0
            print(f"[OK] Set cigsPerDay=0 for {smoker_zero_mask.sum()} non-smokers with cigarette data")
        
        # Check for logical inconsistencies
        smoker_positive_mask = (df_clean['currentSmoker'] == 1) & (df_clean['cigsPerDay'] == 0)
        if smoker_positive_mask.sum() > 0:
            print(f"[INFO] {smoker_positive_mask.sum()} smokers report 0 cigarettes/day (kept as-is)")
        
        return df_clean
    
    def split_data(self, df, test_size=0.2, val_size=0.2):
        """
        Split data into train/validation/test sets with stratification
        Prevents data leakage by ensuring proper split before any preprocessing
        """
        print("\n" + "="*70)
        print("DATA SPLIT (Data Leakage Prevention)")
        print("="*70)
        
        # Separate features and target
        X = df.drop('TenYearCHD', axis=1)
        y = df['TenYearCHD']
        
        print(f"Total samples: {len(df)}")
        print(f"Features: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # First split: train + val vs test (80% vs 20%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # Stratified split to maintain class distribution
        )
        
        # Second split: train vs val (80% of remaining vs 20% of remaining)
        # This gives us 64% train, 16% val, 20% test of original data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (1 - test_size),  # Adjust for first split
            random_state=self.random_state,
            stratify=y_train_val
        )
        
        print(f"\nSplit ratios:")
        print(f"  Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
        
        print(f"\nClass distribution in splits:")
        print(f"  Train - Negative: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%), Positive: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"  Validation - Negative: {(y_val==0).sum()} ({(y_val==0).sum()/len(y_val)*100:.1f}%), Positive: {(y_val==1).sum()} ({(y_val==1).sum()/len(y_val)*100:.1f}%)")
        print(f"  Test - Negative: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%), Positive: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_preprocessor(self):
        """
        Create preprocessing pipeline
        - Fitted only on training data to prevent leakage
        - Applied to validation/test data using fitted parameters
        """
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
        
        # Define feature categories
        binary_features = ['sex', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
        categorical_features = ['education']
        continuous_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        # Create transformers for each feature type
        # Binary features: No scaling, impute with mode
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Categorical features: One-hot encode, impute with mode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
        
        # Continuous features: Standard scaling, impute with median
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
        
        return self.preprocessor
    
    def fit_preprocessor(self, X_train):
        """
        Fit preprocessor on training data only
        This prevents data leakage from validation/test sets
        """
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
        # Get feature names from ColumnTransformer
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'binary':
                feature_names.extend(features)
            elif name == 'categorical':
                # Get one-hot encoded feature names
                encoder = transformer.named_steps['onehot']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(features)
                    feature_names.extend(cat_names)
                else:
                    # Fallback for older sklearn versions
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
        print(f"[OK] Preprocessor saved to {path}")
    
    def load_preprocessor(self, path):
        """Load fitted preprocessor"""
        self.preprocessor = joblib.load(path)
        self.feature_names_out = joblib.load(path.replace('.pkl', '_feature_names.pkl'))
        print(f"[OK] Preprocessor loaded from {path}")
        return self.preprocessor
    
    def verify_no_leakage(self, X_train, X_val, X_test):
        """Verify no data leakage between splits"""
        print("\n" + "="*70)
        print("DATA LEAKAGE VERIFICATION")
        print("="*70)
        
        # Check for overlapping indices
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices
        
        if train_val_overlap:
            print(f"[X] WARNING: {len(train_val_overlap)} samples overlap between train and validation")
        else:
            print("[OK] No overlap between train and validation")
        
        if train_test_overlap:
            print(f"[X] WARNING: {len(train_test_overlap)} samples overlap between train and test")
        else:
            print("[OK] No overlap between train and test")
        
        if val_test_overlap:
            print(f"[X] WARNING: {len(val_test_overlap)} samples overlap between validation and test")
        else:
            print("[OK] No overlap between validation and test")
        
        # Check for duplicate patients (if any ID exists)
        # Since we don't have patient IDs, we check for exact row duplicates
        train_duplicates = X_train.duplicated().sum()
        val_duplicates = X_val.duplicated().sum()
        test_duplicates = X_test.duplicated().sum()
        
        print(f"\nDuplicate records within splits:")
        print(f"  Train: {train_duplicates}")
        print(f"  Validation: {val_duplicates}")
        print(f"  Test: {test_duplicates}")
        
        if train_duplicates == 0 and val_duplicates == 0 and test_duplicates == 0:
            print("[OK] No duplicate records within splits")
        
        return len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0

def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("CARDIOVASCULAR RISK PREDICTION - PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = CardioPreprocessor(random_state=42)
    
    # Load data
    df = preprocessor.load_data()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Split data (prevents leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_clean)
    
    # Create preprocessing pipeline
    preprocessor.create_preprocessor()
    
    # Fit preprocessor on training data only (prevents leakage)
    preprocessor.fit_preprocessor(X_train)
    
    # Transform all datasets
    print("\n" + "="*70)
    print("TRANSFORMING DATA")
    print("="*70)
    
    X_train_transformed = preprocessor.transform_data(X_train)
    X_val_transformed = preprocessor.transform_data(X_val)
    X_test_transformed = preprocessor.transform_data(X_test)
    
    print(f"Train transformed shape: {X_train_transformed.shape}")
    print(f"Validation transformed shape: {X_val_transformed.shape}")
    print(f"Test transformed shape: {X_test_transformed.shape}")
    
    # Verify no data leakage
    no_leakage = preprocessor.verify_no_leakage(X_train, X_val, X_test)
    
    if no_leakage:
        print("\n[OK] Data leakage prevention verified")
    else:
        print("\n[X] WARNING: Potential data leakage detected")
    
    # Save preprocessor
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    preprocessor_path = models_dir / 'preprocessor.pkl'
    preprocessor.save_preprocessor(str(preprocessor_path))
    
    # Save split data for model training
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    np.save(data_dir / 'X_train.npy', X_train_transformed)
    np.save(data_dir / 'X_val.npy', X_val_transformed)
    np.save(data_dir / 'X_test.npy', X_test_transformed)
    np.save(data_dir / 'y_train.npy', y_train.values)
    np.save(data_dir / 'y_val.npy', y_val.values)
    np.save(data_dir / 'y_test.npy', y_test.values)
    
    print(f"\n[OK] Preprocessed data saved to {data_dir}")
    
    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'n_features': X_train_transformed.shape[1],
        'feature_names': preprocessor.feature_names_out,
        'target_distribution_train': {
            'negative': int((y_train==0).sum()),
            'positive': int((y_train==1).sum())
        },
        'target_distribution_val': {
            'negative': int((y_val==0).sum()),
            'positive': int((y_val==1).sum())
        },
        'target_distribution_test': {
            'negative': int((y_test==0).sum()),
            'positive': int((y_test==1).sum())
        },
        'no_data_leakage': no_leakage
    }
    
    import json
    with open(data_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Preprocessing metadata saved")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Train models using X_train.npy and y_train.npy")
    print("2. Validate using X_val.npy and y_val.npy")
    print("3. Final evaluation using X_test.npy and y_test.npy")
    print("4. Load preprocessor.pkl for inference on new data")

if __name__ == "__main__":
    main()
