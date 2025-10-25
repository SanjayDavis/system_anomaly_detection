"""
test_pipeline.py
Testing and validation script for the ML pipeline.
Run this to verify everything works before processing your 26GB file.
"""

import os
import logging
from pathlib import Path
import csv
from log_predictor import predict_log_severity, get_model_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_sample_predictions():
    """Test predictions on sample log lines"""
    logger.info("=" * 70)
    logger.info("TEST 1: Sample Predictions")
    logger.info("=" * 70)
    
    test_cases = [
        ("INFO: Application started successfully", "NORMAL"),
        ("ERROR: Database connection failed", "CRITICAL"),
        ("WARNING: Low disk space on C: drive", "WARNING"),
        ("CRITICAL: System crash detected", "CRITICAL"),
        ("User login successful", "NORMAL"),
        ("Failed to authenticate user", "CRITICAL"),
        ("Deprecation warning: API v1 will be removed", "WARNING"),
        ("Exception: OutOfMemoryException", "CRITICAL"),
        ("Heartbeat received from agent", "NORMAL"),
    ]
    
    try:
        passed = 0
        failed = 0
        
        for log_line, expected in test_cases:
            try:
                prediction = predict_log_severity(log_line)
                status = "✓ PASS" if prediction == expected else "✗ FAIL"
                
                if prediction == expected:
                    passed += 1
                else:
                    failed += 1
                
                print(f"{status}: {log_line[:50]:50} → {prediction:8} (expected: {expected})")
            
            except Exception as e:
                logger.error(f"Error predicting for '{log_line}': {e}")
                failed += 1
        
        logger.info(f"\nResults: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
    
    except FileNotFoundError:
        logger.error("Model file 'model.pkl' not found. Please run train_model.py first.")
        return False


def test_model_info():
    """Check model information"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Model Information")
    logger.info("=" * 70)
    
    try:
        info = get_model_info('model.pkl')
        logger.info(f"Model Accuracy: {info['accuracy']:.4f}")
        logger.info(f"Label Classes: {info['label_classes']}")
        logger.info(f"File Size: {info['file_size_mb']:.2f} MB")
        return True
    
    except FileNotFoundError:
        logger.error("Model file 'model.pkl' not found.")
        return False
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        return False


def test_csv_reading():
    """Test CSV reading from labeled_logs.csv"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: CSV Data Reading")
    logger.info("=" * 70)
    
    csv_file = 'labeled_logs.csv'
    
    if not Path(csv_file).exists():
        logger.warning(f"{csv_file} not found. Skipping test.")
        return True
    
    try:
        line_count = 0
        label_counts = {'CRITICAL': 0, 'WARNING': 0, 'NORMAL': 0}
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                line_count += 1
                label = row['label']
                if label in label_counts:
                    label_counts[label] += 1
                
                if line_count >= 1000:  # Sample first 1000 lines
                    break
        
        logger.info(f"Sampled {line_count:,} lines from {csv_file}")
        for label, count in label_counts.items():
            pct = 100 * count / line_count if line_count > 0 else 0
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
        
        return True
    
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return False


def test_file_exists():
    """Check if required files exist"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 0: File Requirements")
    logger.info("=" * 70)
    
    files_to_check = [
        ('windows.log', 'Input log file'),
        ('model.pkl', 'Trained model (optional for first run)'),
    ]
    
    all_exist = True
    for filename, description in files_to_check:
        exists = Path(filename).exists()
        status = "✓" if exists else "✗"
        logger.info(f"{status} {filename:20} - {description}")
        if filename != 'model.pkl':  # model.pkl is optional for first run
            all_exist = all_exist and exists
    
    return all_exist


def test_dependencies():
    """Check if all required packages are installed"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST -1: Dependencies")
    logger.info("=" * 70)
    
    dependencies = ['sklearn', 'numpy', 'csv', 'pickle']
    all_installed = True
    
    for package in dependencies:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
            elif package == 'csv':
                import csv
                version = "Built-in"
            elif package == 'pickle':
                import pickle
                version = "Built-in"
            
            logger.info(f"✓ {package:15} {version}")
        except ImportError:
            logger.error(f"✗ {package:15} NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        logger.error("\nPlease install missing packages:")
        logger.error("  pip install scikit-learn numpy")
    
    return all_installed


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "🧪 ML PIPELINE VALIDATION SUITE" + "\n")
    
    results = []
    
    # Check dependencies first
    logger.info("Checking dependencies...")
    results.append(("Dependencies", test_dependencies()))
    
    # Check files
    logger.info("\nChecking files...")
    results.append(("File Requirements", test_file_exists()))
    
    # Test CSV reading
    logger.info("\nTesting CSV reading...")
    results.append(("CSV Data Reading", test_csv_reading()))
    
    # Test model info
    logger.info("\nTesting model info...")
    results.append(("Model Information", test_model_info()))
    
    # Test predictions
    logger.info("\nTesting predictions...")
    results.append(("Sample Predictions", test_sample_predictions()))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ All tests passed! Pipeline is ready to use.")
        return True
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Check errors above.")
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
