// API Configuration
const API_BASE_URL = 'http://localhost:5000/api/v1';

// Current step tracking
let currentStep = 1;
const totalSteps = 6;

// Form data storage
let formData = {};

// Navigation
function nextStep(step) {
    if (validateCurrentStep()) {
        saveCurrentStepData();
        showStep(step);
    }
}

function prevStep(step) {
    showStep(step);
}

function showStep(step) {
    // Hide all steps
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    
    // Show target step
    document.getElementById(`step${step}`).classList.add('active');
    
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (parseInt(btn.dataset.step) === step) {
            btn.classList.add('active');
        }
    });
    
    currentStep = step;
    
    // Generate review if on review step
    if (step === 5) {
        generateReview();
    }
}

// Form validation
function validateCurrentStep() {
    const currentStepEl = document.getElementById(`step${currentStep}`);
    const requiredFields = currentStepEl.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value) {
            field.style.borderColor = '#dc3545';
            isValid = false;
        } else {
            field.style.borderColor = '#ddd';
        }
    });
    
    if (!isValid) {
        alert('Please fill in all required fields');
    }
    
    return isValid;
}

// Save form data
function saveCurrentStepData() {
    const currentStepEl = document.getElementById(`step${currentStep}`);
    const inputs = currentStepEl.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        if (input.type === 'checkbox') {
            formData[input.name] = input.checked ? 1 : 0;
        } else if (input.value) {
            formData[input.name] = parseFloat(input.value);
        }
    });
}

// Toggle cigarettes per day field
function toggleCigsPerDay() {
    const smokerCheckbox = document.getElementById('currentSmoker');
    const cigsPerDayGroup = document.getElementById('cigsPerDayGroup');
    
    if (smokerCheckbox.checked) {
        cigsPerDayGroup.style.display = 'block';
        document.getElementById('cigsPerDay').required = true;
    } else {
        cigsPerDayGroup.style.display = 'none';
        document.getElementById('cigsPerDay').required = false;
        document.getElementById('cigsPerDay').value = '';
    }
}

// Generate review
function generateReview() {
    const reviewContainer = document.getElementById('reviewContainer');
    
    const fieldLabels = {
        sex: 'Sex',
        age: 'Age',
        education: 'Education Level',
        totChol: 'Total Cholesterol',
        sysBP: 'Systolic Blood Pressure',
        diaBP: 'Diastolic Blood Pressure',
        BMI: 'BMI',
        heartRate: 'Heart Rate',
        glucose: 'Glucose',
        BPMeds: 'Blood Pressure Medication',
        prevalentStroke: 'History of Stroke',
        prevalentHyp: 'History of Hypertension',
        diabetes: 'History of Diabetes',
        currentSmoker: 'Current Smoker',
        cigsPerDay: 'Cigarettes per Day'
    };
    
    const valueMappings = {
        sex: { 0: 'Female', 1: 'Male' },
        education: { 1: 'Some High School', 2: 'High School Diploma / GED', 3: 'Some College / Vocational School', 4: 'College Graduate' },
        BPMeds: { 0: 'No', 1: 'Yes' },
        prevalentStroke: { 0: 'No', 1: 'Yes' },
        prevalentHyp: { 0: 'No', 1: 'Yes' },
        diabetes: { 0: 'No', 1: 'Yes' },
        currentSmoker: { 0: 'No', 1: 'Yes' }
    };
    
    let reviewHTML = '';
    
    for (const [key, value] of Object.entries(formData)) {
        if (value !== undefined && value !== null && value !== '') {
            const label = fieldLabels[key] || key;
            let displayValue = value;
            
            if (valueMappings[key]) {
                displayValue = valueMappings[key][value] || value;
            }
            
            reviewHTML += `
                <div class="review-item">
                    <div class="review-label">${label}</div>
                    <div class="review-value">${displayValue}</div>
                </div>
            `;
        }
    }
    
    reviewContainer.innerHTML = reviewHTML;
}

// Submit assessment
async function submitAssessment() {
    // Save final step data
    saveCurrentStepData();
    
    // Ensure cigsPerDay is 0 if not a smoker
    if (formData.currentSmoker === 0 && !formData.cigsPerDay) {
        formData.cigsPerDay = 0;
    }
    
    // Move to results step
    showStep(6);
    
    // Show loading
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = '<div class="loading">Analyzing data and calculating risk...</div>';
    
    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayResults(result);
        } else {
            displayError(result.message || 'Prediction failed');
        }
    } catch (error) {
        displayError('Failed to connect to server. Please ensure the backend is running.');
    }
}

// Display results
function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const prediction = result.prediction;
    const explanation = result.explanation;
    const modelInfo = result.model_info;
    
    // Risk level color
    const riskColors = {
        'Low Risk': '#4caf50',
        'Moderate Risk': '#ff9800',
        'High Risk': '#f44336',
        'Very High Risk': '#d32f2f'
    };
    
    const riskColor = riskColors[prediction.risk_category] || '#667eea';
    
    let resultsHTML = `
        <div class="risk-card" style="background: ${riskColor}">
            <h2>Risk Assessment Result</h2>
            <div class="risk-level">${prediction.risk_category}</div>
            <div class="probability">${prediction.probability_percentage}% Risk of CHD in 10 Years</div>
            <div>Prediction: ${prediction.class_label}</div>
        </div>
        
        <div class="prediction-details">
            <h3>Prediction Details</h3>
            <div class="detail-item">
                <span class="detail-label">Model Prediction</span>
                <span class="detail-value">${prediction.class_label}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Risk Probability</span>
                <span class="detail-value">${prediction.probability_percentage}%</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Risk Category</span>
                <span class="detail-value">${prediction.risk_category}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Model Used</span>
                <span class="detail-value">${modelInfo.model_name}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Model Version</span>
                <span class="detail-value">${modelInfo.model_version}</span>
            </div>
        </div>
        
        <div class="explanation-card">
            <h3>Feature Contributions</h3>
            <p style="margin-bottom: 15px; color: #666;">Top factors that influenced this prediction:</p>
    `;
    
    explanation.feature_contributions.forEach(feature => {
        const direction = feature.contribution > 0 ? 'increases' : 'decreases';
        const directionText = feature.contribution > 0 ? 'Increases Risk' : 'Decreases Risk';
        
        resultsHTML += `
            <div class="feature-contribution">
                <span class="feature-name">${formatFeatureName(feature.feature)}</span>
                <span class="feature-direction ${direction}">${directionText}</span>
                <span class="feature-value">(${feature.contribution.toFixed(4)})</span>
            </div>
        `;
    });
    
    resultsHTML += `
        </div>
        
        <div class="disclaimer-card">
            <h3>Important Disclaimers</h3>
            <ul>
    `;
    
    result.disclaimers.forEach(disclaimer => {
        resultsHTML += `<li>${disclaimer}</li>`;
    });
    
    resultsHTML += `
            </ul>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
}

// Format feature names for display
function formatFeatureName(featureName) {
    const nameMappings = {
        'sex': 'Sex',
        'age': 'Age',
        'education_2.0': 'Education: High School',
        'education_3.0': 'Education: Some College',
        'education_4.0': 'Education: College Graduate',
        'currentSmoker': 'Current Smoker',
        'cigsPerDay': 'Cigarettes per Day',
        'BPMeds': 'Blood Pressure Medication',
        'prevalentStroke': 'History of Stroke',
        'prevalentHyp': 'History of Hypertension',
        'diabetes': 'Diabetes',
        'totChol': 'Total Cholesterol',
        'sysBP': 'Systolic Blood Pressure',
        'diaBP': 'Diastolic Blood Pressure',
        'BMI': 'BMI',
        'heartRate': 'Heart Rate',
        'glucose': 'Glucose'
    };
    
    return nameMappings[featureName] || featureName;
}

// Display error
function displayError(message) {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = `
        <div class="error">
            <h3>Error</h3>
            <p>${message}</p>
        </div>
    `;
}

// Reset assessment
function resetAssessment() {
    formData = {};
    currentStep = 1;
    
    // Reset all forms
    document.querySelectorAll('form').forEach(form => form.reset());
    
    // Reset checkbox display
    document.getElementById('cigsPerDayGroup').style.display = 'none';
    document.getElementById('cigsPerDay').required = false;
    
    // Go to first step
    showStep(1);
}

// Navigation button click handlers
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const step = parseInt(this.dataset.step);
        if (step < currentStep) {
            showStep(step);
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    showStep(1);
});
