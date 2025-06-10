"""
Machine Learning Model Pipeline for Medical Insurance Cost Prediction
Handles model training, evaluation, and persistence
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalInsuranceModel:
    def __init__(self, data_path="../data/processed/medical_insurance_ML.csv"):
        """Initialize model pipeline"""
        self.data_path = data_path
        self.model_dir = "../model"
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.model_configs = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def load_data(self):
        """Load and prepare data for training"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Separate features and target
            X = df.drop('charges', axis=1)
            y = df['charges']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        try:
            logger.info("Starting model training...")
            
            for name, model in self.model_configs.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                self.models[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all trained models"""
        try:
            logger.info("Starting model evaluation...")
            
            evaluation_results = {}
            
            for name, model_info in self.models.items():
                model = model_info['model']
                
                # Predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                evaluation_results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_mean': model_info['cv_mean'],
                    'cv_std': model_info['cv_std']
                }
                
                logger.info(f"\n{name} Results:")
                logger.info(f"  Train R²: {train_r2:.4f}")
                logger.info(f"  Test R²: {test_r2:.4f}")
                logger.info(f"  Test RMSE: {test_rmse:.4f}")
                logger.info(f"  Test MAE: {test_mae:.4f}")
            
            # Select best model based on test R² score
            best_model_name = max(evaluation_results.keys(), 
                                key=lambda k: evaluation_results[k]['test_r2'])
            
            self.best_model = self.models[best_model_name]['model']
            self.best_model_name = best_model_name
            
            logger.info(f"Best model selected: {best_model_name}")
            logger.info("Model evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def save_model(self, model_name=None):
        """Save the best model or specified model"""
        try:
            if model_name is None:
                model_name = self.best_model_name
                model = self.best_model
            else:
                model = self.models[model_name]['model']
            
            # Save using both pickle and joblib for compatibility
            pickle_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
            joblib_path = os.path.join(self.model_dir, f'{model_name}_model.joblib')
            
            # Save with pickle
            with open(pickle_path, 'wb') as file:
                pickle.dump(model, file)
            
            # Save with joblib
            joblib.dump(model, joblib_path)
            
            logger.info(f"Model saved successfully:")
            logger.info(f"  Pickle: {pickle_path}")
            logger.info(f"  Joblib: {joblib_path}")
            
            return pickle_path, joblib_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
            elif model_path.endswith('.joblib'):
                model = joblib.load(model_path)
            else:
                raise ValueError("Unsupported model file format. Use .pkl or .joblib")
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X, model=None):
        """Make predictions using the model"""
        try:
            if model is None:
                model = self.best_model
            
            if model is None:
                raise ValueError("No model available for prediction. Train a model first.")
            
            predictions = model.predict(X)
            logger.info(f"Predictions made for {len(X)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def run_pipeline(self):
        """Run complete model pipeline"""
        try:
            logger.info("Starting model pipeline...")
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Train models
            self.train_models(X_train, y_train)
            
            # Evaluate models
            results = self.evaluate_models(X_train, X_test, y_train, y_test)
            
            # Save best model
            model_paths = self.save_model()
            
            logger.info("Model pipeline completed successfully")
            
            return {
                'status': 'success',
                'best_model': self.best_model_name,
                'evaluation_results': results,
                'model_paths': model_paths
            }
            
        except Exception as e:
            logger.error(f"Model pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

if __name__ == "__main__":
    model_pipeline = MedicalInsuranceModel()
    result = model_pipeline.run_pipeline()
    print(f"Model Pipeline Result: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Best Model: {result['best_model']}")
        print("Evaluation Results:")
        for model_name, metrics in result['evaluation_results'].items():
            print(f"  {model_name}: Test R² = {metrics['test_r2']:.4f}")