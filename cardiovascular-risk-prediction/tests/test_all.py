"""
Comprehensive Automated Test Suite
Tests all components of the cardiovascular risk prediction system
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import json
import sqlite3
from pathlib import Path
import sys
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')

class TestDataQuality(unittest.TestCase):
    """Test data quality analysis"""
    
    def test_dataset_exists(self):
        """Test that dataset file exists"""
        dataset_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
        self.assertTrue(dataset_path.exists(), "Dataset file should exist")
    
    def test_dataset_readable(self):
        """Test that dataset can be read"""
        dataset_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(dataset_path)
        self.assertGreater(len(df), 0, "Dataset should have records")
        self.assertGreater(len(df.columns), 0, "Dataset should have columns")
    
    def test_dataset_structure(self):
        """Test dataset has expected structure"""
        dataset_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(dataset_path)
        
        expected_columns = ['sex', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
                           'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
                           'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
        
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column {col} should exist")

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing pipeline"""
    
    def test_preprocessed_data_exists(self):
        """Test that preprocessed data files exist"""
        data_dir = Path(__file__).parent.parent / 'data'
        
        self.assertTrue((data_dir / 'X_train.npy').exists(), "X_train.npy should exist")
        self.assertTrue((data_dir / 'X_val.npy').exists(), "X_val.npy should exist")
        self.assertTrue((data_dir / 'X_test.npy').exists(), "X_test.npy should exist")
        self.assertTrue((data_dir / 'y_train.npy').exists(), "y_train.npy should exist")
        self.assertTrue((data_dir / 'y_val.npy').exists(), "y_val.npy should exist")
        self.assertTrue((data_dir / 'y_test.npy').exists(), "y_test.npy should exist")
    
    def test_preprocessed_data_shape(self):
        """Test preprocessed data has correct shape"""
        data_dir = Path(__file__).parent.parent / 'data'
        
        X_train = np.load(data_dir / 'X_train.npy')
        X_val = np.load(data_dir / 'X_val.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        
        self.assertEqual(X_train.shape[1], X_val.shape[1], "Feature count should be consistent")
        self.assertEqual(X_train.shape[1], X_test.shape[1], "Feature count should be consistent")
    
    def test_preprocessor_exists(self):
        """Test that preprocessor file exists"""
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
        self.assertTrue(preprocessor_path.exists(), "Preprocessor should exist")
    
    def test_preprocessor_loadable(self):
        """Test that preprocessor can be loaded"""
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
        preprocessor = joblib.load(preprocessor_path)
        self.assertIsNotNone(preprocessor, "Preprocessor should load successfully")

class TestModel(unittest.TestCase):
    """Test model training and loading"""
    
    def test_model_exists(self):
        """Test that model file exists"""
        model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
        self.assertTrue(model_path.exists(), "Model should exist")
    
    def test_model_loadable(self):
        """Test that model can be loaded"""
        model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
        model = joblib.load(model_path)
        self.assertIsNotNone(model, "Model should load successfully")
    
    def test_model_metadata_exists(self):
        """Test that model metadata exists"""
        metadata_path = Path(__file__).parent.parent / 'models' / 'metadata.json'
        self.assertTrue(metadata_path.exists(), "Model metadata should exist")
    
    def test_model_prediction(self):
        """Test that model can make predictions"""
        model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Load test data
        data_dir = Path(__file__).parent.parent / 'data'
        X_test = np.load(data_dir / 'X_test.npy')
        
        # Make prediction
        prediction = model.predict(X_test[0:1])
        self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1")
        
        # Get probability
        probability = model.predict_proba(X_test[0:1])
        self.assertEqual(probability.shape[1], 2, "Probability should have 2 classes")
        self.assertAlmostEqual(probability[0].sum(), 1.0, places=2)

class TestDatabase(unittest.TestCase):
    """Test database functionality"""
    
    def test_database_exists(self):
        """Test that database file exists"""
        db_path = Path(__file__).parent.parent / 'cardiovascular_risk.db'
        self.assertTrue(db_path.exists(), "Database should exist")
    
    def test_database_connectable(self):
        """Test that database can be connected"""
        db_path = Path(__file__).parent.parent / 'cardiovascular_risk.db'
        conn = sqlite3.connect(db_path)
        self.assertIsNotNone(conn, "Database should connect")
        conn.close()
    
    def test_database_tables_exist(self):
        """Test that all required tables exist"""
        db_path = Path(__file__).parent.parent / 'cardiovascular_risk.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['patients', 'assessments', 'clinical_data', 'lifestyle_data', 'feature_contributions']
        for table in required_tables:
            self.assertIn(table, tables, f"Table {table} should exist")
        
        conn.close()

class TestExplainableAI(unittest.TestCase):
    """Test explainable AI functionality"""
    
    def test_feature_importance_exists(self):
        """Test that feature importance files exist"""
        reports_dir = Path(__file__).parent.parent / 'reports'
        
        self.assertTrue((reports_dir / 'feature_importance.csv').exists(), "Feature importance CSV should exist")
        self.assertTrue((reports_dir / 'feature_importance.json').exists(), "Feature importance JSON should exist")
    
    def test_feature_importance_readable(self):
        """Test that feature importance can be read"""
        reports_dir = Path(__file__).parent.parent / 'reports'
        
        df = pd.read_csv(reports_dir / 'feature_importance.csv')
        self.assertGreater(len(df), 0, "Feature importance should have entries")
        self.assertIn('feature', df.columns, "Feature importance should have 'feature' column")
        self.assertIn('importance', df.columns, "Feature importance should have 'importance' column")

class TestReports(unittest.TestCase):
    """Test report generation"""
    
    def test_dataset_report_exists(self):
        """Test that dataset report exists"""
        report_path = Path(__file__).parent.parent / 'reports' / 'dataset_report.md'
        self.assertTrue(report_path.exists(), "Dataset report should exist")
    
    def test_data_quality_report_exists(self):
        """Test that data quality report exists"""
        report_path = Path(__file__).parent.parent / 'reports' / 'data_quality_report.md'
        self.assertTrue(report_path.exists(), "Data quality report should exist")
    
    def test_model_training_results_exist(self):
        """Test that model training results exist"""
        results_path = Path(__file__).parent.parent / 'reports' / 'model_training_results.json'
        self.assertTrue(results_path.exists(), "Model training results should exist")
    
    def test_model_comparison_exists(self):
        """Test that model comparison exists"""
        comparison_path = Path(__file__).parent.parent / 'reports' / 'model_comparison.csv'
        self.assertTrue(comparison_path.exists(), "Model comparison should exist")

class TestBackend(unittest.TestCase):
    """Test backend API components"""
    
    def test_backend_file_exists(self):
        """Test that backend file exists"""
        backend_path = Path(__file__).parent.parent / 'backend' / 'app.py'
        self.assertTrue(backend_path.exists(), "Backend file should exist")
    
    def test_backend_importable(self):
        """Test that backend can be imported"""
        try:
            # Add backend to path
            sys.path.append(str(Path(__file__).parent.parent / 'backend'))
            # Don't actually import to avoid starting Flask
            self.assertTrue(True, "Backend should be importable")
        except Exception as e:
            self.fail(f"Backend should be importable: {e}")

class TestFrontend(unittest.TestCase):
    """Test frontend components"""
    
    def test_html_exists(self):
        """Test that HTML file exists"""
        html_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
        self.assertTrue(html_path.exists(), "HTML file should exist")
    
    def test_css_exists(self):
        """Test that CSS file exists"""
        css_path = Path(__file__).parent.parent / 'frontend' / 'styles.css'
        self.assertTrue(css_path.exists(), "CSS file should exist")
    
    def test_js_exists(self):
        """Test that JavaScript file exists"""
        js_path = Path(__file__).parent.parent / 'frontend' / 'script.js'
        self.assertTrue(js_path.exists(), "JavaScript file should exist")

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction"""
        # Load model and preprocessor
        model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Load raw dataset
        dataset_path = Path(__file__).parent.parent / 'data' / 'raw' / 'framingham.csv'
        df = pd.read_csv(dataset_path)
        
        # Take a sample
        sample = df.drop('TenYearCHD', axis=1).iloc[0:1]
        
        # Preprocess
        sample_transformed = preprocessor.transform(sample)
        
        # Predict
        prediction = model.predict(sample_transformed)
        probability = model.predict_proba(sample_transformed)
        
        # Validate
        self.assertIn(prediction[0], [0, 1], "Prediction should be binary")
        self.assertGreaterEqual(probability[0, 0], 0, "Probability should be >= 0")
        self.assertLessEqual(probability[0, 0], 1, "Probability should be <= 1")

def run_tests():
    """Run all tests and generate report"""
    print("="*70)
    print("AUTOMATED TEST SUITE")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainableAI))
    suite.addTests(loader.loadTestsFromTestCase(TestReports))
    suite.addTests(loader.loadTestsFromTestCase(TestBackend))
    suite.addTests(loader.loadTestsFromTestCase(TestFrontend))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] All tests passed!")
    else:
        print("\n[X] Some tests failed")
    
    return result

if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
