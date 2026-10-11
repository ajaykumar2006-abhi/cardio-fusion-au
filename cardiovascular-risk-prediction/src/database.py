"""
Database Implementation for Cardiovascular Risk Assessment
SQLite database with tables for patients, assessments, clinical data, and lifestyle data
"""

import sqlite3
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

class CardioDatabase:
    """Database manager for cardiovascular risk assessment"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            self.db_path = Path(__file__).parent.parent / 'cardiovascular_risk.db'
        else:
            self.db_path = Path(db_path)
        
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        self.cursor = self.conn.cursor()
        print(f"[OK] Connected to database: {self.db_path}")
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("[OK] Database connection closed")
    
    def create_tables(self):
        """Create all database tables"""
        print("\n" + "="*70)
        print("CREATING DATABASE TABLES")
        print("="*70)
        
        # Patients table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("[OK] Created patients table")
        
        # Assessments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT NOT NULL,
                prediction_class INTEGER NOT NULL,
                prediction_probability REAL NOT NULL,
                risk_category TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        ''')
        print("[OK] Created assessments table")
        
        # Clinical data table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clinical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT NOT NULL,
                sex INTEGER NOT NULL,
                age INTEGER NOT NULL,
                education REAL,
                totChol REAL,
                sysBP REAL,
                diaBP REAL,
                BMI REAL,
                heartRate REAL,
                glucose REAL,
                BPMeds INTEGER DEFAULT 0,
                prevalentStroke INTEGER DEFAULT 0,
                prevalentHyp INTEGER DEFAULT 0,
                diabetes INTEGER DEFAULT 0,
                FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
            )
        ''')
        print("[OK] Created clinical_data table")
        
        # Lifestyle data table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS lifestyle_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT NOT NULL,
                currentSmoker INTEGER DEFAULT 0,
                cigsPerDay REAL DEFAULT 0,
                FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
            )
        ''')
        print("[OK] Created lifestyle_data table")
        
        # Feature contributions table (for explainability)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                coefficient REAL NOT NULL,
                contribution REAL NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
            )
        ''')
        print("[OK] Created feature_contributions table")
        
        self.conn.commit()
        print("\n[OK] All database tables created successfully")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        print("\n" + "="*70)
        print("CREATING INDEXES")
        print("="*70)
        
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id)',
            'CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON assessments(patient_id)',
            'CREATE INDEX IF NOT EXISTS idx_assessments_assessment_id ON assessments(assessment_id)',
            'CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date)',
            'CREATE INDEX IF NOT EXISTS idx_clinical_assessment_id ON clinical_data(assessment_id)',
            'CREATE INDEX IF NOT EXISTS idx_lifestyle_assessment_id ON lifestyle_data(assessment_id)',
            'CREATE INDEX IF NOT EXISTS idx_contributions_assessment_id ON feature_contributions(assessment_id)'
        ]
        
        for index_sql in indexes:
            self.cursor.execute(index_sql)
        
        self.conn.commit()
        print("[OK] All indexes created successfully")
    
    def insert_patient(self, patient_id):
        """Insert a new patient"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO patients (patient_id, created_at, updated_at)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (patient_id,))
            self.conn.commit()
            print(f"[OK] Patient {patient_id} inserted/updated")
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to insert patient: {e}")
            return None
    
    def insert_assessment(self, assessment_id, patient_id, model_version, prediction_result):
        """Insert a new assessment"""
        try:
            self.cursor.execute('''
                INSERT INTO assessments 
                (assessment_id, patient_id, model_version, prediction_class, prediction_probability, risk_category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                assessment_id,
                patient_id,
                model_version,
                prediction_result['prediction']['class'],
                prediction_result['prediction']['probability'],
                prediction_result['prediction']['risk_category']
            ))
            self.conn.commit()
            print(f"[OK] Assessment {assessment_id} inserted")
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to insert assessment: {e}")
            return None
    
    def insert_clinical_data(self, assessment_id, clinical_data):
        """Insert clinical data for an assessment"""
        try:
            self.cursor.execute('''
                INSERT INTO clinical_data 
                (assessment_id, sex, age, education, totChol, sysBP, diaBP, BMI, heartRate, glucose, 
                 BPMeds, prevalentStroke, prevalentHyp, diabetes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment_id,
                clinical_data.get('sex'),
                clinical_data.get('age'),
                clinical_data.get('education'),
                clinical_data.get('totChol'),
                clinical_data.get('sysBP'),
                clinical_data.get('diaBP'),
                clinical_data.get('BMI'),
                clinical_data.get('heartRate'),
                clinical_data.get('glucose'),
                clinical_data.get('BPMeds', 0),
                clinical_data.get('prevalentStroke', 0),
                clinical_data.get('prevalentHyp', 0),
                clinical_data.get('diabetes', 0)
            ))
            self.conn.commit()
            print(f"[OK] Clinical data for assessment {assessment_id} inserted")
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to insert clinical data: {e}")
            return None
    
    def insert_lifestyle_data(self, assessment_id, lifestyle_data):
        """Insert lifestyle data for an assessment"""
        try:
            self.cursor.execute('''
                INSERT INTO lifestyle_data (assessment_id, currentSmoker, cigsPerDay)
                VALUES (?, ?, ?)
            ''', (
                assessment_id,
                lifestyle_data.get('currentSmoker', 0),
                lifestyle_data.get('cigsPerDay', 0)
            ))
            self.conn.commit()
            print(f"[OK] Lifestyle data for assessment {assessment_id} inserted")
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to insert lifestyle data: {e}")
            return None
    
    def insert_feature_contributions(self, assessment_id, feature_contributions):
        """Insert feature contributions for an assessment"""
        try:
            for feature in feature_contributions:
                self.cursor.execute('''
                    INSERT INTO feature_contributions 
                    (assessment_id, feature_name, feature_value, coefficient, contribution)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    assessment_id,
                    feature['feature'],
                    feature['value'],
                    feature['coefficient'],
                    feature['contribution']
                ))
            self.conn.commit()
            print(f"[OK] Feature contributions for assessment {assessment_id} inserted")
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to insert feature contributions: {e}")
            return False
    
    def get_assessment(self, assessment_id):
        """Get complete assessment by ID"""
        query = '''
            SELECT 
                a.*,
                c.*,
                l.*,
                p.patient_id
            FROM assessments a
            LEFT JOIN clinical_data c ON a.assessment_id = c.assessment_id
            LEFT JOIN lifestyle_data l ON a.assessment_id = l.assessment_id
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.assessment_id = ?
        '''
        
        self.cursor.execute(query, (assessment_id,))
        return self.cursor.fetchone()
    
    def get_patient_assessments(self, patient_id):
        """Get all assessments for a patient"""
        query = '''
            SELECT 
                a.*,
                c.*,
                l.*
            FROM assessments a
            LEFT JOIN clinical_data c ON a.assessment_id = c.assessment_id
            LEFT JOIN lifestyle_data l ON a.assessment_id = l.assessment_id
            WHERE a.patient_id = ?
            ORDER BY a.assessment_date DESC
        '''
        
        self.cursor.execute(query, (patient_id,))
        return self.cursor.fetchall()
    
    def get_all_assessments(self, limit=50):
        """Get all assessments (with limit)"""
        query = '''
            SELECT 
                a.*,
                c.*,
                l.*,
                p.patient_id
            FROM assessments a
            LEFT JOIN clinical_data c ON a.assessment_id = c.assessment_id
            LEFT JOIN lifestyle_data l ON a.assessment_id = l.assessment_id
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            ORDER BY a.assessment_date DESC
            LIMIT ?
        '''
        
        self.cursor.execute(query, (limit,))
        return self.cursor.fetchall()
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {}
        
        # Count patients
        self.cursor.execute('SELECT COUNT(*) FROM patients')
        stats['total_patients'] = self.cursor.fetchone()[0]
        
        # Count assessments
        self.cursor.execute('SELECT COUNT(*) FROM assessments')
        stats['total_assessments'] = self.cursor.fetchone()[0]
        
        # Count by risk category
        self.cursor.execute('''
            SELECT risk_category, COUNT(*) 
            FROM assessments 
            GROUP BY risk_category
        ''')
        stats['by_risk_category'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        # Count by model version
        self.cursor.execute('''
            SELECT model_version, COUNT(*) 
            FROM assessments 
            GROUP BY model_version
        ''')
        stats['by_model_version'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        return stats

def main():
    """Main database setup"""
    print("="*70)
    print("DATABASE SETUP")
    print("="*70)
    
    # Initialize database
    db = CardioDatabase()
    
    # Connect
    db.connect()
    
    # Create tables
    db.create_tables()
    
    # Create indexes
    db.create_indexes()
    
    # Get statistics
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    stats = db.get_statistics()
    print(f"Total patients: {stats['total_patients']}")
    print(f"Total assessments: {stats['total_assessments']}")
    print(f"Assessments by risk category: {stats['by_risk_category']}")
    print(f"Assessments by model version: {stats['by_model_version']}")
    
    # Close connection
    db.close()
    
    print("\n" + "="*70)
    print("DATABASE SETUP COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
