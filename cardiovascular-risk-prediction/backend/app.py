"""
Backend API for Cardiovascular Risk Prediction
Flask API with prediction and health endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import uuid
import warnings
import sys
import io

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from database import CardioDatabase

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None
model_metadata = None
database = None

def load_model_and_dependencies():
    """Load model, preprocessor, metadata, and database"""
    global model, preprocessor, feature_names, model_metadata, database
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        # Load model
        model = joblib.load(models_dir / 'model.pkl')
        
        # Load preprocessor
        preprocessor = joblib.load(models_dir / 'preprocessor.pkl')
        
        # Load feature names
        feature_names = joblib.load(models_dir / 'preprocessor_feature_names.pkl')
        
        # Load metadata
        with open(models_dir / 'metadata.json', 'r') as f:
            model_metadata = json.load(f)
        
        # Initialize database
        database = CardioDatabase()
        database.connect()
        
        print("[OK] Model and dependencies loaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model dependencies: {e}")
        return False

def validate_input(data):
    """Validate input data"""
    required_fields = [
        'sex', 'age', 'education', 'currentSmoker', 'cigsPerDay',
        'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
        'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    
    errors = []
    
    # Check for missing fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if 'age' in data:
        try:
            age = float(data['age'])
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number")
    
    if 'sex' in data:
        if data['sex'] not in [0, 1]:
            errors.append("Sex must be 0 (female) or 1 (male)")
    
    if 'totChol' in data:
        try:
            chol = float(data['totChol'])
            if chol < 50 or chol > 800:
                errors.append("Total cholesterol must be between 50 and 800 mg/dL")
        except (ValueError, TypeError):
            errors.append("Total cholesterol must be a valid number")
    
    if 'sysBP' in data:
        try:
            sbp = float(data['sysBP'])
            if sbp < 50 or sbp > 300:
                errors.append("Systolic BP must be between 50 and 300 mmHg")
        except (ValueError, TypeError):
            errors.append("Systolic BP must be a valid number")
    
    if 'diaBP' in data:
        try:
            dbp = float(data['diaBP'])
            if dbp < 30 or dbp > 200:
                errors.append("Diastolic BP must be between 30 and 200 mmHg")
        except (ValueError, TypeError):
            errors.append("Diastolic BP must be a valid number")
    
    if 'BMI' in data:
        try:
            bmi = float(data['BMI'])
            if bmi < 10 or bmi > 100:
                errors.append("BMI must be between 10 and 100")
        except (ValueError, TypeError):
            errors.append("BMI must be a valid number")
    
    if 'glucose' in data:
        try:
            glucose = float(data['glucose'])
            if glucose < 20 or glucose > 1000:
                errors.append("Glucose must be between 20 and 1000 mg/dL")
        except (ValueError, TypeError):
            errors.append("Glucose must be a valid number")
    
    return errors

def get_risk_category(probability):
    """Categorize risk based on probability"""
    if probability < 0.10:
        return "Low Risk"
    elif probability < 0.20:
        return "Moderate Risk"
    elif probability < 0.30:
        return "High Risk"
    else:
        return "Very High Risk"

def explain_prediction(input_data, probability):
    """Generate feature-level explanation for prediction"""
    # Preprocess input
    input_transformed = preprocessor.transform(input_data)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Calculate feature contributions
    feature_contributions = []
    
    for i, (feature_name, feature_value) in enumerate(zip(feature_names, input_transformed[0])):
        contribution = coefficients[i] * feature_value
        feature_contributions.append({
            'feature': feature_name,
            'value': float(feature_value),
            'coefficient': float(coefficients[i]),
            'contribution': float(contribution)
        })
    
    # Sort by absolute contribution
    feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return feature_contributions

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None or preprocessor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    db_status = 'connected' if database else 'not connected'
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_name': model_metadata['model_info']['model_name'] if model_metadata else 'Unknown',
        'model_version': model_metadata['model_info']['model_version'] if model_metadata else 'Unknown',
        'database_status': db_status
    }), 200

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None or preprocessor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        # Validate input
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({
                'status': 'error',
                'message': 'Validation failed',
                'errors': validation_errors
            }), 400
        
        # Create DataFrame for preprocessing
        input_df = pd.DataFrame([data])
        
        # Preprocess input
        input_transformed = preprocessor.transform(input_df)
        
        # Get prediction and probability
        prediction = int(model.predict(input_transformed)[0])
        probability = float(model.predict_proba(input_transformed)[0, 1])
        
        # Get risk category
        risk_category = get_risk_category(probability)
        
        # Generate explanation
        feature_contributions = explain_prediction(input_df, probability)
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'class': prediction,
                'class_label': 'High CHD Risk' if prediction == 1 else 'Low CHD Risk',
                'probability': probability,
                'probability_percentage': round(probability * 100, 2),
                'risk_category': risk_category
            },
            'explanation': {
                'feature_contributions': feature_contributions[:10],  # Top 10 features
                'interpretation': 'Positive contributions increase risk, negative contributions decrease risk'
            },
            'model_info': {
                'model_name': model_metadata['model_info']['model_name'] if model_metadata else 'Unknown',
                'model_version': model_metadata['model_info']['model_version'] if model_metadata else 'Unknown',
                'training_date': model_metadata['model_info']['training_date'] if model_metadata else 'Unknown'
            },
            'disclaimers': [
                'This is an AI-based research tool, not a medical device',
                'Predictions are risk estimates, not medical diagnoses',
                'Results should be reviewed by qualified healthcare professionals',
                'Model trained on historical data (1956-1968)',
                'Do not use for clinical decision making without proper validation'
            ]
        }
        
        # Save to database
        try:
            # Generate IDs
            patient_id = str(uuid.uuid4())
            assessment_id = str(uuid.uuid4())
            
            # Insert patient
            database.insert_patient(patient_id)
            
            # Insert assessment
            database.insert_assessment(
                assessment_id,
                patient_id,
                model_metadata['model_info']['model_version'] if model_metadata else '1.0',
                response
            )
            
            # Insert clinical data
            clinical_data = {
                'sex': data.get('sex'),
                'age': data.get('age'),
                'education': data.get('education'),
                'totChol': data.get('totChol'),
                'sysBP': data.get('sysBP'),
                'diaBP': data.get('diaBP'),
                'BMI': data.get('BMI'),
                'heartRate': data.get('heartRate'),
                'glucose': data.get('glucose'),
                'BPMeds': data.get('BPMeds', 0),
                'prevalentStroke': data.get('prevalentStroke', 0),
                'prevalentHyp': data.get('prevalentHyp', 0),
                'diabetes': data.get('diabetes', 0)
            }
            database.insert_clinical_data(assessment_id, clinical_data)
            
            # Insert lifestyle data
            lifestyle_data = {
                'currentSmoker': data.get('currentSmoker', 0),
                'cigsPerDay': data.get('cigsPerDay', 0)
            }
            database.insert_lifestyle_data(assessment_id, lifestyle_data)
            
            # Insert feature contributions
            database.insert_feature_contributions(assessment_id, feature_contributions)
            
            # Add IDs to response
            response['assessment_id'] = assessment_id
            response['patient_id'] = patient_id
            
        except Exception as db_error:
            print(f"[WARNING] Failed to save to database: {db_error}")
            # Continue without database save
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_metadata is None:
        return jsonify({
            'status': 'error',
            'message': 'Model metadata not loaded'
        }), 500
    
    # Return safe subset of metadata
    safe_metadata = {
        'model_info': model_metadata['model_info'],
        'dataset_info': {
            'dataset_name': model_metadata['dataset_info']['dataset_name'],
            'dataset_source': model_metadata['dataset_info']['dataset_source'],
            'total_samples': model_metadata['dataset_info']['total_samples'],
            'feature_count': model_metadata['dataset_info']['feature_count']
        },
        'test_results': model_metadata['test_results'],
        'limitations': model_metadata['limitations'][:5],  # First 5 limitations
        'intended_use': model_metadata['intended_use']
    }
    
    return jsonify({
        'status': 'success',
        'metadata': safe_metadata
    }), 200

@app.route('/api/v1/assessments', methods=['GET'])
def get_assessments():
    """Get assessment history"""
    if database is None:
        return jsonify({
            'status': 'error',
            'message': 'Database not connected'
        }), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        assessments = database.get_all_assessments(limit)
        
        # Convert to list of dicts
        assessment_list = [dict(assessment) for assessment in assessments]
        
        return jsonify({
            'status': 'success',
            'count': len(assessment_list),
            'assessments': assessment_list
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve assessments: {str(e)}'
        }), 500

@app.route('/api/v1/assessments/<assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
    """Get specific assessment by ID"""
    if database is None:
        return jsonify({
            'status': 'error',
            'message': 'Database not connected'
        }), 500
    
    try:
        assessment = database.get_assessment(assessment_id)
        
        if assessment is None:
            return jsonify({
                'status': 'error',
                'message': 'Assessment not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'assessment': dict(assessment)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve assessment: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Load model and dependencies
    if load_model_and_dependencies():
        print("[OK] Starting API server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("[ERROR] Failed to start API server")
        sys.exit(1)
