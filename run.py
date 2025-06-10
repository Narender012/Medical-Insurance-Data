"""
Main execution script for Medical Insurance ML Pipeline
Orchestrates ETL and Model training processes
"""

import logging
import sys
import os
from datetime import datetime
import json

# Add current directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import MedicalInsuranceETL
from model import MedicalInsuranceModel

# Set up logging with timestamp
def setup_logging():
    """Setup logging configuration with proper file handling"""
    log_dir = os.path.abspath("./logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create separate log files for each run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"pipeline_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Also create a daily log file
    daily_log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    daily_log_path = os.path.join(log_dir, daily_log_filename)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging with multiple handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),  # Individual run log
            logging.FileHandler(daily_log_path, mode='a'),  # Daily cumulative log
            logging.StreamHandler(sys.stdout)  # Console output
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log files: {log_path}, {daily_log_path}")
    return logger

logger = setup_logging()

class PipelineOrchestrator:
    def __init__(self):
        """Initialize pipeline orchestrator"""
        self.results_dir = "../results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.etl = MedicalInsuranceETL()
        self.model_pipeline = None
        
    def run_etl(self):
        """Execute ETL pipeline"""
        logger.info("="*60)
        logger.info("STARTING ETL PIPELINE")
        logger.info("="*60)
        
        try:
            result = self.etl.run_etl()
            
            if result['status'] == 'success':
                logger.info("ETL pipeline completed successfully")
                return result
            else:
                logger.error(f"ETL pipeline failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"ETL pipeline crashed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_model_training(self, etl_result):
        """Execute model training pipeline"""
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Initialize model pipeline with the ML data path from ETL
            ml_csv_path = etl_result.get('ml_csv_path')
            if ml_csv_path:
                self.model_pipeline = MedicalInsuranceModel(data_path=ml_csv_path)
            else:
                self.model_pipeline = MedicalInsuranceModel()
            
            result = self.model_pipeline.run_pipeline()
            
            if result['status'] == 'success':
                logger.info("Model training pipeline completed successfully")
                logger.info(f"Best model: {result['best_model']}")
                return result
            else:
                logger.error(f"Model training pipeline failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"Model training pipeline crashed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def save_results(self, etl_result, model_result):
        """Save pipeline results to JSON file"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'etl_result': etl_result,
                'model_result': model_result
            }
            
            # Remove non-serializable objects
            if 'scaler' in etl_result:
                del etl_result['scaler']
            if 'label_encoders' in etl_result:
                del etl_result['label_encoders']
            
            results_filename = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_path = os.path.join(self.results_dir, results_filename)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run_full_pipeline(self):
        """Execute complete pipeline: ETL + Model Training"""
        start_time = datetime.now()
        logger.info("="*80)
        logger.info(f"STARTING FULL MEDICAL INSURANCE ML PIPELINE - {start_time}")
        logger.info("="*80)
        
        overall_status = 'success'
        
        try:
            # Step 1: Run ETL
            etl_result = self.run_etl()
            
            if etl_result['status'] != 'success':
                overall_status = 'failed'
                logger.error("Pipeline stopped due to ETL failure")
                return {
                    'status': overall_status,
                    'etl_result': etl_result,
                    'model_result': {'status': 'skipped', 'reason': 'ETL failed'}
                }
            
            # Step 2: Run Model Training
            model_result = self.run_model_training(etl_result)
            
            if model_result['status'] != 'success':
                overall_status = 'partial'  # ETL succeeded but model failed
            
            # Step 3: Save results
            self.save_results(etl_result, model_result)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*80)
            logger.info(f"PIPELINE COMPLETED - Status: {overall_status.upper()}")
            logger.info(f"Duration: {duration}")
            logger.info(f"End time: {end_time}")
            logger.info("="*80)
            
            return {
                'status': overall_status,
                'etl_result': etl_result,
                'model_result': model_result,
                'duration': str(duration),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline crashed with error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration': str(datetime.now() - start_time)
            }

def main():
    """Main execution function"""
    try:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run_full_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Overall Status: {result['status'].upper()}")
        
        if 'duration' in result:
            print(f"Duration: {result['duration']}")
        
        if result['status'] == 'success':
            print("✅ Pipeline completed successfully!")
            if 'model_result' in result and result['model_result']['status'] == 'success':
                best_model = result['model_result'].get('best_model', 'Unknown')
                print(f"✅ Best model trained: {best_model}")
        elif result['status'] == 'partial':
            print("⚠️ Pipeline partially completed (ETL successful, Model training failed)")
        else:
            print("❌ Pipeline failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("="*60)
        
        # Exit with appropriate code
        if result['status'] in ['success', 'partial']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()