"""
Test script to verify logging is working properly
"""

import os
import logging
from datetime import datetime

def test_logging():
    """Test logging configuration"""
    # Ensure logs directory exists
    logs_dir = os.path.abspath("../logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs directory: {logs_dir}")
    print(f"Logs directory exists: {os.path.exists(logs_dir)}")
    
    # Create log file path
    log_file = os.path.join(logs_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Log file path: {log_file}")
    
    # Clear any existing handlers
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    
    # Test logging
    logger.info("This is a test info message")
    logger.warning("This is a test warning message")
    logger.error("This is a test error message")
    
    # Verify file was created
    if os.path.exists(log_file):
        print(f"✅ Log file created successfully: {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"Log file content:\n{content}")
    else:
        print(f"❌ Log file was not created: {log_file}")
    
    # List all files in logs directory
    print(f"\nFiles in logs directory:")
    for file in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")

if __name__ == "__main__":
    test_logging()