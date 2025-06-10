"""
ETL Pipeline for Medical Insurance Data
Handles data extraction, transformation, and loading processes
"""

import kagglehub
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalInsuranceETL:
    def __init__(self):
        """Initialize ETL pipeline with configuration"""
        self.RAW_DIR = "../data/raw/"
        self.PROCESSED_DIR = "../data/processed/"
        self.FINAL_CSV_NAME = "medical_insurance.csv"
        self.EDA_FILE_NAME = "medical_insurance_eda.csv"
        self.ML_FILE_NAME = "medical_insurance_ML.csv"
        
        # Ensure directories exist
        os.makedirs(self.RAW_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        self._setup_db_connection()
    
    def _setup_db_connection(self):
        """Setup database connection parameters"""
        try:
            self.username = os.getenv('DB_USERNAME')
            self.password = os.getenv('DB_PASSWORD')
            self.host = os.getenv('DB_HOST')
            self.port = int(os.getenv('DB_PORT', 3306))
            self.database = os.getenv('DB_DATABASE')
            
            self.engine = create_engine(
                f'mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'
            )
            logger.info("Database connection configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup database connection: {e}")
            raise
    
    def extract_data(self):
        """Extract data from Kaggle dataset"""
        try:
            logger.info("Starting data extraction...")
            
            # Download dataset from Kaggle
            dataset_path = kagglehub.dataset_download("rahulvyasm/medical-insurance-cost-prediction")
            logger.info(f"Downloaded dataset to: {dataset_path}")
            
            # Find and copy CSV file
            csv_found = False
            for file in os.listdir(dataset_path):
                if file.endswith(".csv"):
                    src_csv_path = os.path.join(dataset_path, file)
                    dest_csv_path = os.path.join(self.RAW_DIR, self.FINAL_CSV_NAME)
                    shutil.copy2(src_csv_path, dest_csv_path)
                    logger.info(f"Copied {file} to {dest_csv_path}")
                    csv_found = True
                    break
            
            if not csv_found:
                raise FileNotFoundError("No CSV file found in the KaggleHub dataset directory.")
            
            logger.info("Data extraction completed successfully")
            return os.path.join(self.RAW_DIR, self.FINAL_CSV_NAME)
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    def transform_data(self, csv_path):
        """Transform raw data for analysis and ML"""
        try:
            logger.info("Starting data transformation...")
            
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Check data quality
            missing_values = df.isnull().sum()
            duplicates = df.duplicated().sum()
            duplicate_percentage = (duplicates / len(df)) * 100
            
            logger.info(f"Missing values: {missing_values.sum()}")
            logger.info(f"Duplicate rows: {duplicates} ({duplicate_percentage:.2f}%)")
            
            # Feature engineering
            df['isparent'] = df['children'] > 0
            
            # Age categorization
            df['age_group'] = pd.cut(
                df['age'],
                bins=[17, 28, 48, np.inf],
                labels=['young_adult', 'adult', 'senior'],
                right=True,
                include_lowest=True
            )
            
            # Save EDA version
            eda_csv_path = os.path.join(self.PROCESSED_DIR, self.EDA_FILE_NAME)
            df.to_csv(eda_csv_path, index=False)
            logger.info(f"EDA data saved to {eda_csv_path}")
            
            # Prepare ML version
            df_ml = df.copy()
            
            # Label encoding for categorical columns
            categorical_columns = df_ml.select_dtypes(include=['object', 'category']).columns
            label_encoders = {}
            
            for col in categorical_columns:
                le = LabelEncoder()
                df_ml[col] = le.fit_transform(df_ml[col])
                label_encoders[col] = le
                logger.info(f"Encoded {col} with {len(le.classes_)} unique values")
            
            # Encode isparent column
            le_parent = LabelEncoder()
            df_ml['isparent'] = le_parent.fit_transform(df_ml['isparent'])
            
            # Scale numerical columns
            numerical_columns = df_ml.select_dtypes(include=['int64', 'float64']).columns
            scaler = StandardScaler()
            df_ml[numerical_columns] = scaler.fit_transform(df_ml[numerical_columns])
            logger.info(f"Scaled numerical columns: {list(numerical_columns)}")
            
            # Save ML version
            ml_csv_path = os.path.join(self.PROCESSED_DIR, self.ML_FILE_NAME)
            df_ml.to_csv(ml_csv_path, index=False)
            logger.info(f"ML data saved to {ml_csv_path}")
            
            logger.info("Data transformation completed successfully")
            return eda_csv_path, ml_csv_path, scaler, label_encoders
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
    
    def load_data(self, eda_csv_path, ml_csv_path):
        """Load processed data to database"""
        try:
            logger.info("Starting data loading to database...")
            
            # Load ML data to database
            df_ml = pd.read_csv(ml_csv_path)
            df_ml.to_sql('insurance_ml', con=self.engine, if_exists='replace', index=False)
            logger.info("ML data loaded to 'insurance_ml' table")
            
            # Load EDA data to database
            df_eda = pd.read_csv(eda_csv_path)
            df_eda.to_sql('insurance_bi', con=self.engine, if_exists='replace', index=False)
            logger.info("EDA data loaded to 'insurance_bi' table")
            
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def run_etl(self):
        """Run complete ETL pipeline"""
        try:
            logger.info("Starting ETL pipeline...")
            
            # Extract
            csv_path = self.extract_data()
            
            # Transform
            eda_csv_path, ml_csv_path, scaler, label_encoders = self.transform_data(csv_path)
            
            # Load
            self.load_data(eda_csv_path, ml_csv_path)
            
            logger.info("ETL pipeline completed successfully")
            return {
                'status': 'success',
                'eda_csv_path': eda_csv_path,
                'ml_csv_path': ml_csv_path,
                'scaler': scaler,
                'label_encoders': label_encoders
            }
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

if __name__ == "__main__":
    etl = MedicalInsuranceETL()
    result = etl.run_etl()
    print(f"ETL Result: {result['status']}")