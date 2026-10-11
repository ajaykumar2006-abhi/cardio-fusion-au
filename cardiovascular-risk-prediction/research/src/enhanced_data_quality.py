"""
Enhanced Data Quality Analysis for Research
Comprehensive quality checks with additional analysis for research improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedDataQuality:
    """Enhanced data quality analysis for research"""
    
    def __init__(self):
        # Use current working directory (project root)
        current_dir = Path.cwd()
        self.dataset_path = current_dir / 'data' / 'raw' / 'framingham.csv'
        self.df = None
        self.quality_report = {}
    
    def load_data(self):
        """Load the Framingham dataset"""
        self.df = pd.read_csv(self.dataset_path)
        print(f"[OK] Dataset loaded: {self.df.shape}")
        return self.df
    
    def comprehensive_analysis(self):
        """Perform comprehensive data quality analysis"""
        print("="*70)
        print("ENHANCED DATA QUALITY ANALYSIS")
        print("="*70)
        
        # Basic statistics
        self.quality_report['basic_stats'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Target analysis
        self.quality_report['target_analysis'] = self.analyze_target()
        
        # Feature analysis
        self.quality_report['feature_analysis'] = self.analyze_features()
        
        # Missing value analysis
        self.quality_report['missing_analysis'] = self.analyze_missing_values()
        
        # Outlier analysis
        self.quality_report['outlier_analysis'] = self.analyze_outliers()
        
        # Duplicate analysis
        self.quality_report['duplicate_analysis'] = self.analyze_duplicates()
        
        # Feature correlations
        self.quality_report['correlation_analysis'] = self.analyze_correlations()
        
        # Clinical range validation
        self.quality_report['clinical_validation'] = self.validate_clinical_ranges()
        
        # Feature engineering opportunities
        self.quality_report['feature_engineering_opportunities'] = self.identify_feature_engineering_opportunities()
        
        self.save_report()
        
        return self.quality_report
    
    def analyze_target(self):
        """Analyze target variable"""
        print("\n" + "="*70)
        print("TARGET VARIABLE ANALYSIS")
        print("="*70)
        
        target = self.df['TenYearCHD']
        
        analysis = {
            'name': 'TenYearCHD',
            'type': 'binary',
            'total': len(target),
            'negative': int((target == 0).sum()),
            'positive': int((target == 1).sum()),
            'negative_pct': float((target == 0).sum() / len(target) * 100),
            'positive_pct': float((target == 1).sum() / len(target) * 100),
            'imbalance_ratio': float((target == 0).sum() / (target == 1).sum()),
            'missing': int(target.isnull().sum())
        }
        
        print(f"Target: {analysis['name']}")
        print(f"Type: {analysis['type']}")
        print(f"Total: {analysis['total']}")
        print(f"Negative (No CHD): {analysis['negative']} ({analysis['negative_pct']:.2f}%)")
        print(f"Positive (CHD): {analysis['positive']} ({analysis['positive_pct']:.2f}%)")
        print(f"Imbalance Ratio: {analysis['imbalance_ratio']:.2f}:1")
        print(f"Missing: {analysis['missing']}")
        
        if analysis['imbalance_ratio'] > 3:
            print("[WARNING] Significant class imbalance detected")
        
        return analysis
    
    def analyze_features(self):
        """Analyze feature types and distributions"""
        print("\n" + "="*70)
        print("FEATURE ANALYSIS")
        print("="*70)
        
        binary_features = ['sex', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
        categorical_features = ['education']
        continuous_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        analysis = {
            'binary': binary_features,
            'categorical': categorical_features,
            'continuous': continuous_features,
            'target': 'TenYearCHD'
        }
        
        print(f"Binary features ({len(binary_features)}): {', '.join(binary_features)}")
        print(f"Categorical features ({len(categorical_features)}): {', '.join(categorical_features)}")
        print(f"Continuous features ({len(continuous_features)}): {', '.join(continuous_features)}")
        
        # Continuous feature statistics
        print("\nContinuous Feature Statistics:")
        for feature in continuous_features:
            if feature in self.df.columns:
                stats = self.df[feature].describe()
                print(f"  {feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        return analysis
    
    def analyze_missing_values(self):
        """Analyze missing values in detail"""
        print("\n" + "="*70)
        print("MISSING VALUE ANALYSIS")
        print("="*70)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        analysis = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': int((missing > 0).sum()),
            'details': {}
        }
        
        for col in missing[missing > 0].index:
            analysis['details'][col] = {
                'count': int(missing[col]),
                'percentage': float(missing_pct[col])
            }
            print(f"{col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        
        print(f"\nTotal missing values: {analysis['total_missing']}")
        print(f"Columns with missing: {analysis['columns_with_missing']}")
        
        return analysis
    
    def analyze_outliers(self):
        """Analyze outliers using IQR method"""
        print("\n" + "="*70)
        print("OUTLIER ANALYSIS (IQR Method)")
        print("="*70)
        
        continuous_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        analysis = {}
        
        for feature in continuous_features:
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                analysis[feature] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': data.min(),
                    'max': data.max()
                }
                
                print(f"{feature}: {len(outliers)} outliers ({len(outliers)/len(data)*100:.2f}%)")
        
        return analysis
    
    def analyze_duplicates(self):
        """Analyze duplicate records"""
        print("\n" + "="*70)
        print("DUPLICATE RECORD ANALYSIS")
        print("="*70)
        
        exact_duplicates = self.df.duplicated().sum()
        
        analysis = {
            'exact_duplicates': int(exact_duplicates),
            'duplicate_percentage': float(exact_duplicates / len(self.df) * 100)
        }
        
        print(f"Exact duplicates: {exact_duplicates}")
        print(f"Percentage: {exact_duplicates / len(self.df) * 100:.2f}%")
        
        if exact_duplicates == 0:
            print("[OK] No exact duplicates found")
        
        return analysis
    
    def analyze_correlations(self):
        """Analyze feature correlations"""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Calculate correlations with target
        correlations = self.df.corr()['TenYearCHD'].sort_values(ascending=False)
        
        analysis = {
            'top_positive': correlations[1:6].to_dict(),  # Exclude target itself
            'top_negative': correlations[-5:].to_dict()
        }
        
        print("Top 5 Positive Correlations with Target:")
        for feature, corr in analysis['top_positive'].items():
            print(f"  {feature}: {corr:.4f}")
        
        print("\nTop 5 Negative Correlations with Target:")
        for feature, corr in analysis['top_negative'].items():
            print(f"  {feature}: {corr:.4f}")
        
        return analysis
    
    def validate_clinical_ranges(self):
        """Validate clinical ranges"""
        print("\n" + "="*70)
        print("CLINICAL RANGE VALIDATION")
        print("="*70)
        
        validation_rules = {
            'age': (0, 120),
            'totChol': (50, 800),
            'sysBP': (50, 300),
            'diaBP': (30, 200),
            'BMI': (10, 100),
            'heartRate': (30, 250),
            'glucose': (20, 1000),
            'cigsPerDay': (0, 100)
        }
        
        analysis = {
            'violations': {},
            'all_valid': True
        }
        
        for feature, (min_val, max_val) in validation_rules.items():
            if feature in self.df.columns:
                violations = self.df[(self.df[feature] < min_val) | (self.df[feature] > max_val)]
                if len(violations) > 0:
                    analysis['violations'][feature] = len(violations)
                    analysis['all_valid'] = False
                    print(f"[X] {feature}: {len(violations)} violations (range: {min_val}-{max_val})")
                else:
                    print(f"[OK] {feature}: All values in valid range ({min_val}-{max_val})")
        
        if analysis['all_valid']:
            print("\n[OK] All clinical ranges valid")
        
        return analysis
    
    def identify_feature_engineering_opportunities(self):
        """Identify feature engineering opportunities"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING OPPORTUNITIES")
        print("="*70)
        
        opportunities = []
        
        # Pulse pressure
        if 'sysBP' in self.df.columns and 'diaBP' in self.df.columns:
            opportunities.append({
                'name': 'pulse_pressure',
                'formula': 'sysBP - diaBP',
                'description': 'Pulse pressure (difference between systolic and diastolic BP)',
                'clinical_relevance': 'Strong predictor of cardiovascular risk'
            })
            print("[OPPORTUNITY] Pulse pressure: sysBP - diaBP")
        
        # Age groups
        if 'age' in self.df.columns:
            opportunities.append({
                'name': 'age_groups',
                'formula': 'age categories (<40, 40-49, 50-59, 60+)',
                'description': 'Age group categorization',
                'clinical_relevance': 'Risk varies significantly by age group'
            })
            print("[OPPORTUNITY] Age groups: <40, 40-49, 50-59, 60+")
        
        # BMI categories
        if 'BMI' in self.df.columns:
            opportunities.append({
                'name': 'bmi_categories',
                'formula': 'BMI categories (underweight, normal, overweight, obese)',
                'description': 'BMI categorization',
                'clinical_relevance': 'WHO BMI categories for risk assessment'
            })
            print("[OPPORTUNITY] BMI categories: underweight, normal, overweight, obese")
        
        # BP categories
        if 'sysBP' in self.df.columns:
            opportunities.append({
                'name': 'bp_categories',
                'formula': 'BP categories (normal, elevated, hypertension stage 1, stage 2)',
                'description': 'Blood pressure categorization',
                'clinical_relevance': 'ACC/AHA BP categories for risk assessment'
            })
            print("[OPPORTUNITY] BP categories: normal, elevated, hypertension stage 1, stage 2")
        
        # Cholesterol categories
        if 'totChol' in self.df.columns:
            opportunities.append({
                'name': 'cholesterol_categories',
                'formula': 'Cholesterol categories (desirable, borderline high, high)',
                'description': 'Cholesterol categorization',
                'clinical_relevance': 'NCEP cholesterol categories for risk assessment'
            })
            print("[OPPORTUNITY] Cholesterol categories: desirable, borderline high, high")
        
        # Metabolic syndrome indicators
        bp_available = 'sysBP' in self.df.columns and 'diaBP' in self.df.columns
        glucose_available = 'glucose' in self.df.columns
        chol_available = 'totChol' in self.df.columns
        bmi_available = 'BMI' in self.df.columns
        
        if bp_available and glucose_available and chol_available and bmi_available:
            opportunities.append({
                'name': 'metabolic_syndrome_indicators',
                'formula': 'Combined metabolic risk factors',
                'description': 'Metabolic syndrome risk indicators',
                'clinical_relevance': 'Metabolic syndrome is a strong CVD risk factor'
            })
            print("[OPPORTUNITY] Metabolic syndrome indicators: Combined metabolic risk factors")
        
        analysis = {
            'opportunities': opportunities,
            'count': len(opportunities)
        }
        
        print(f"\nTotal feature engineering opportunities: {len(opportunities)}")
        
        return analysis
    
    def save_report(self):
        """Save comprehensive quality report"""
        report_path = Path(__file__).parent.parent / 'reports' / 'enhanced_data_quality_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Data Quality Analysis Report\n\n")
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Total Records:** {self.quality_report['basic_stats']['rows']}\n")
            f.write(f"- **Total Features:** {self.quality_report['basic_stats']['columns']}\n")
            f.write(f"- **Memory Usage:** {self.quality_report['basic_stats']['memory_usage_mb']:.2f} MB\n\n")
            
            f.write("## Target Analysis\n\n")
            target = self.quality_report['target_analysis']
            f.write(f"- **Target:** {target['name']}\n")
            f.write(f"- **Type:** {target['type']}\n")
            f.write(f"- **Negative Class:** {target['negative']} ({target['negative_pct']:.2f}%)\n")
            f.write(f"- **Positive Class:** {target['positive']} ({target['positive_pct']:.2f}%)\n")
            f.write(f"- **Imbalance Ratio:** {target['imbalance_ratio']:.2f}:1\n")
            f.write(f"- **Missing Values:** {target['missing']}\n\n")
            
            f.write("## Feature Engineering Opportunities\n\n")
            opportunities = self.quality_report['feature_engineering_opportunities']['opportunities']
            for opp in opportunities:
                f.write(f"- **{opp['name']}:** {opp['description']}\n")
                f.write(f"  Formula: {opp['formula']}\n")
                f.write(f"  Clinical Relevance: {opp['clinical_relevance']}\n\n")
        
        print(f"\n[OK] Enhanced quality report saved to {report_path}")

def main():
    """Main function"""
    print("="*70)
    print("ENHANCED DATA QUALITY ANALYSIS")
    print("="*70)
    
    analyzer = EnhancedDataQuality()
    analyzer.load_data()
    analyzer.comprehensive_analysis()
    
    print("\n" + "="*70)
    print("ENHANCED DATA QUALITY ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
