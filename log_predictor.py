"""
log_predictor.py
Utility module providing the predict_log_severity function.
Can be imported in other scripts for programmatic usage.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def predict_log_severity(log_line: str, model_path: str = 'model.pkl') -> str:
    """
    Predict the severity of a log line using the trained model.
    
    This is the main prediction function that can be imported and used in other scripts.
    
    Args:
        log_line (str): A single log line to classify
        model_path (str): Path to the trained model pickle file (default: 'model.pkl')
    
    Returns:
        str: One of 'CRITICAL', 'WARNING', or 'NORMAL'
    
    Example:
        >>> from log_predictor import predict_log_severity
        >>> severity = predict_log_severity("ERROR: System failure detected")
        >>> print(severity)  # Output: CRITICAL
    """
    try:
        # Load model if not already cached
        if not hasattr(predict_log_severity, '_cached_model'):
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                predict_log_severity._cached_model = pickle.load(f)
        
        model_package = predict_log_severity._cached_model
        
        # Extract components
        vectorizer = model_package['vectorizer']
        ensemble = model_package['ensemble']
        
        # Transform and predict
        features = vectorizer.transform([log_line])
        prediction = ensemble.predict(features)[0]
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error predicting log severity: {e}")
        raise


def get_model_info(model_path: str = 'model.pkl') -> Dict[str, Any]:
    """
    Get information about the trained model.
    
    Args:
        model_path (str): Path to the trained model pickle file
    
    Returns:
        dict: Dictionary containing model metadata
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    return {
        'accuracy': model_package.get('accuracy', 'N/A'),
        'label_classes': model_package.get('label_classes', []),
        'model_file': model_path,
        'file_size_mb': Path(model_path).stat().st_size / 1024 / 1024
    }


def predict_batch(log_lines: list, model_path: str = 'model.pkl') -> list:
    """
    Predict severity for multiple log lines.
    
    Args:
        log_lines (list): List of log line strings
        model_path (str): Path to the trained model pickle file
    
    Returns:
        list: List of predictions corresponding to input log lines
    """
    predictions = [predict_log_severity(line, model_path) for line in log_lines]
    return predictions


if __name__ == '__main__':
    # Test the function
    test_logs = [
        "INFO: System started successfully",
        "ERROR: Database connection failed",
        "WARNING: Low memory available",
        "CRITICAL: Service crashed",
    ]
    
    print("Testing predict_log_severity function:")
    print("-" * 50)
    
    try:
        for log in test_logs:
            severity = predict_log_severity(log)
            print(f"[{severity:8}] {log}")
    
    except FileNotFoundError:
        print("Model not found. Please run train_model.py first.")
