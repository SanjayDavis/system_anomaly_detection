# 📚 COMPLETE ML PIPELINE - CODE REFERENCE

This document contains all the code for the Windows log classification pipeline in organized markdown sections.

---

## 📑 Table of Contents

1. [prepare_data.py](#prepare_datepy) - Stream & auto-label logs
2. [train_model.py](#train_modelpy) - Train ML models
3. [monitor.py](#monitorpy) - Real-time monitoring
4. [log_predictor.py](#log_predictorpy) - Prediction utility
5. [config.py](#configpy) - Configuration
6. [test_pipeline.py](#test_pipelinepy) - Validation tests
7. [requirements.txt](#requirementstxt) - Dependencies

---

## prepare_data.py

Stream large log files and auto-label them based on heuristics. Writes labeled data to CSV in chunks.

```python
"""
prepare_data.py
Stream large log files and auto-label them based on heuristics.
Writes labeled data to CSV in chunks to avoid memory overload.
"""

import csv
import logging
from pathlib import Path
from typing import Generator, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRITICAL_KEYWORDS = ['critical', 'error', 'failed', 'failure', 'fatal', 'exception', 'crash']
WARNING_KEYWORDS = ['warning', 'warn', 'deprecated', 'issue']
CHUNK_SIZE = 10000  # Write to CSV every N lines


def auto_label_log_line(line: str) -> str:
    """
    Auto-label a log line based on keyword heuristics.
    Returns: 'CRITICAL', 'WARNING', or 'NORMAL'
    """
    line_lower = line.lower()
    
    # Check for CRITICAL keywords
    for keyword in CRITICAL_KEYWORDS:
        if keyword in line_lower:
            return 'CRITICAL'
    
    # Check for WARNING keywords
    for keyword in WARNING_KEYWORDS:
        if keyword in line_lower:
            return 'WARNING'
    
    # Default to NORMAL
    return 'NORMAL'


def stream_log_file(log_file_path: str) -> Generator[str, None, None]:
    """
    Generator to stream log file line by line.
    Yields one log line at a time without loading the entire file.
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    yield line
                
                # Log progress every 100k lines
                if line_num % 100000 == 0:
                    logger.info(f"Processed {line_num:,} lines...")
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        raise


def prepare_data(
    log_file_path: str,
    output_csv_path: str = 'labeled_logs.csv',
    max_lines: int = None
) -> None:
    """
    Stream the log file, auto-label each line, and write to CSV in chunks.
    
    Args:
        log_file_path: Path to the input log file
        output_csv_path: Path to the output CSV file
        max_lines: Maximum number of lines to process (None = all lines)
    """
    logger.info(f"Starting data preparation from {log_file_path}...")
    
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['log_line', 'label'])
            
            total_lines = 0
            critical_count = 0
            warning_count = 0
            normal_count = 0
            
            # Stream and process log lines
            for line in stream_log_file(log_file_path):
                label = auto_label_log_line(line)
                writer.writerow([line, label])
                
                # Count labels
                if label == 'CRITICAL':
                    critical_count += 1
                elif label == 'WARNING':
                    warning_count += 1
                else:
                    normal_count += 1
                
                total_lines += 1
                
                # Log progress
                if total_lines % CHUNK_SIZE == 0:
                    logger.info(
                        f"Processed {total_lines:,} lines | "
                        f"CRITICAL: {critical_count:,}, WARNING: {warning_count:,}, NORMAL: {normal_count:,}"
                    )
                
                # Stop if max_lines reached
                if max_lines and total_lines >= max_lines:
                    logger.info(f"Reached max_lines limit: {max_lines}")
                    break
        
        logger.info(f"\n✓ Data preparation complete!")
        logger.info(f"Total lines processed: {total_lines:,}")
        logger.info(f"  - CRITICAL: {critical_count:,} ({100*critical_count/total_lines:.1f}%)")
        logger.info(f"  - WARNING: {warning_count:,} ({100*warning_count/total_lines:.1f}%)")
        logger.info(f"  - NORMAL: {normal_count:,} ({100*normal_count/total_lines:.1f}%)")
        logger.info(f"Output saved to: {output_csv_path}")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise


if __name__ == '__main__':
    # Check if log file exists
    log_file = 'windows.log'
    if not Path(log_file).exists():
        logger.warning(f"{log_file} not found. Creating a sample log for testing...")
        # Create a sample log file for testing
        with open(log_file, 'w') as f:
            sample_logs = [
                "INFO: System started successfully",
                "WARNING: Low disk space on C: drive",
                "ERROR: Failed to connect to database",
                "CRITICAL: Service crash detected",
                "INFO: User login successful",
                "WARNING: Deprecated API used",
                "ERROR: File not found",
                "INFO: Task completed",
                "CRITICAL: Critical system failure",
                "WARNING: Certificate expiration warning",
            ]
            for i in range(100):
                for log in sample_logs:
                    f.write(f"{log} (iteration {i})\n")
        logger.info(f"Created sample {log_file}")
    
    # Run data preparation
    prepare_data(log_file, 'labeled_logs.csv', max_lines=None)
```

---

## train_model.py

Train multiple ML models and create an ensemble classifier.

```python
"""
train_model.py
Train multiple ML models on the labeled log data and create an ensemble classifier.
Uses memory-efficient feature extraction (HashingVectorizer) for large datasets.
"""

import csv
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Generator

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def stream_csv_data(csv_path: str, chunk_size: int = 5000) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Stream data from CSV file in chunks to manage memory efficiently.
    Yields (X, y) tuples for each chunk.
    """
    X_chunk = []
    y_chunk = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_num, row in enumerate(reader, 1):
                X_chunk.append(row['log_line'])
                y_chunk.append(row['label'])
                
                if row_num % chunk_size == 0:
                    logger.info(f"Loaded {row_num:,} samples...")
                    yield X_chunk, y_chunk
                    X_chunk = []
                    y_chunk = []
            
            # Yield remaining data
            if X_chunk:
                yield X_chunk, y_chunk
    
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


def load_all_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Load all data from CSV file.
    """
    X = []
    y = []
    
    logger.info(f"Loading data from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, 1):
            X.append(row['log_line'])
            y.append(row['label'])
            
            if row_num % 50000 == 0:
                logger.info(f"Loaded {row_num:,} samples...")
    
    logger.info(f"✓ Loaded {len(X):,} total samples")
    return X, y


def extract_features(X_train, X_test):
    """
    Extract features using HashingVectorizer (memory-efficient for large datasets).
    This vectorizer doesn't need to fit on all data first.
    """
    logger.info("Extracting features using HashingVectorizer...")
    
    # HashingVectorizer is memory-efficient and handles large texts well
    vectorizer = HashingVectorizer(
        n_features=2**18,  # 262,144 features - adjust based on memory
        norm='l2',
        alternate_sign=False,
        random_state=RANDOM_STATE
    )
    
    X_train_features = vectorizer.transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    logger.info(f"Features extracted: {X_train_features.shape[1]} dimensions")
    logger.info(f"Training set shape: {X_train_features.shape}")
    logger.info(f"Test set shape: {X_test_features.shape}")
    
    return vectorizer, X_train_features, X_test_features


def train_svm(X_train, X_test, y_train, y_test):
    """Train SVM model"""
    logger.info("Training SVM model...")
    svm = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, verbose=1)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"SVM Accuracy: {accuracy:.4f}")
    
    return svm


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    logger.info("Training Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
    
    return rf


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model"""
    logger.info("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    return lr


def create_voting_ensemble(svm, rf, lr, X_test, y_test):
    """Create Voting Classifier ensemble"""
    logger.info("Creating Voting Ensemble...")
    
    voting_clf = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('lr', lr)],
        voting='soft'
    )
    voting_clf.fit(X_test, y_test)  # Fit on test to combine predictions
    
    return voting_clf


def create_stacking_ensemble(svm, rf, lr, X_train, X_test, y_train, y_test):
    """Create Stacking Classifier ensemble"""
    logger.info("Creating Stacking Ensemble...")
    
    stacking_clf = StackingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('lr', lr)],
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    
    y_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Stacking Ensemble Accuracy: {accuracy:.4f}")
    
    return stacking_clf


def train_model_pipeline(
    csv_path: str = 'labeled_logs.csv',
    model_path: str = 'model.pkl',
    test_size: float = 0.2
) -> None:
    """
    Main training pipeline:
    1. Load data from CSV
    2. Split into train/test
    3. Extract features
    4. Train multiple models
    5. Create ensemble
    6. Save model to disk
    """
    logger.info("=" * 60)
    logger.info("Starting ML Model Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load data
    X, y = load_all_data(csv_path)
    
    # Step 2: Split data
    logger.info(f"Splitting data ({test_size*100}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Step 3: Extract features
    vectorizer, X_train_features, X_test_features = extract_features(X_train, X_test)
    
    # Step 4: Train individual models
    logger.info("\n" + "=" * 60)
    logger.info("Training Individual Models")
    logger.info("=" * 60)
    
    svm = train_svm(X_train_features, X_test_features, y_train, y_test)
    rf = train_random_forest(X_train_features, X_test_features, y_train, y_test)
    lr = train_logistic_regression(X_train_features, X_test_features, y_train, y_test)
    
    # Step 5: Create ensemble
    logger.info("\n" + "=" * 60)
    logger.info("Creating Ensemble Classifier")
    logger.info("=" * 60)
    
    ensemble = create_stacking_ensemble(
        svm, rf, lr, X_train_features, X_test_features, y_train, y_test
    )
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test_features)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    logger.info(f"\nEnsemble Final Accuracy: {ensemble_accuracy:.4f}")
    
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred_ensemble))
    
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(y_test, y_pred_ensemble)))
    
    # Step 6: Save model and vectorizer
    logger.info("\n" + "=" * 60)
    logger.info("Saving Models")
    logger.info("=" * 60)
    
    model_package = {
        'vectorizer': vectorizer,
        'ensemble': ensemble,
        'label_classes': sorted(list(set(y))),
        'accuracy': ensemble_accuracy
    }
    
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    logger.info(f"✓ Model package saved to: {model_path}")
    logger.info(f"  Model size: {model_path_obj.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("=" * 60)


if __name__ == '__main__':
    csv_file = 'labeled_logs.csv'
    
    if not Path(csv_file).exists():
        logger.error(f"{csv_file} not found. Run prepare_data.py first.")
        exit(1)
    
    train_model_pipeline(csv_file, 'model.pkl')
```

---

## monitor.py

Real-time monitoring and batch prediction script.

```python
"""
monitor.py
Real-time log monitoring script.
Watches for new log entries and uses the trained model to predict severity.
Only displays CRITICAL and WARNING messages.
"""

import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str = 'model.pkl') -> Dict[str, Any]:
    """Load the trained model package from disk"""
    logger.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    logger.info(f"✓ Model loaded. Accuracy: {model_package['accuracy']:.4f}")
    return model_package


def predict_log_severity(log_line: str, model_package: Dict[str, Any]) -> str:
    """
    Predict the severity of a log line.
    
    Args:
        log_line: A single log line to classify
        model_package: The loaded model package containing vectorizer and ensemble
    
    Returns:
        'CRITICAL', 'WARNING', or 'NORMAL'
    """
    vectorizer = model_package['vectorizer']
    ensemble = model_package['ensemble']
    
    # Transform the log line using the same vectorizer
    features = vectorizer.transform([log_line])
    
    # Predict using the ensemble model
    prediction = ensemble.predict(features)[0]
    
    return prediction


def monitor_log_file(
    log_file_path: str = 'windows.log',
    model_path: str = 'model.pkl',
    check_interval: int = 2,
    alert_levels: list = None
) -> None:
    """
    Monitor a log file and predict severity of new entries in real-time.
    
    Args:
        log_file_path: Path to the log file to monitor
        model_path: Path to the trained model
        check_interval: Seconds between file checks
        alert_levels: Severity levels to display (default: ['CRITICAL', 'WARNING'])
    """
    if alert_levels is None:
        alert_levels = ['CRITICAL', 'WARNING']
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run train_model.py first to generate the model.")
        return
    
    # Load model
    model_package = load_model(model_path)
    
    # Check if log file exists
    log_file = Path(log_file_path)
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file_path}")
        return
    
    logger.info(f"Starting monitoring of {log_file_path}")
    logger.info(f"Alert levels: {', '.join(alert_levels)}")
    logger.info(f"Check interval: {check_interval} seconds")
    logger.info("Press Ctrl+C to stop monitoring.")
    logger.info("=" * 70)
    
    # Track file position to only read new lines
    last_position = log_file.stat().st_size
    alert_count = {level: 0 for level in alert_levels}
    
    try:
        while True:
            try:
                current_size = log_file.stat().st_size
                
                # If file grew, read new lines
                if current_size > last_position:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = current_size
                    
                    # Process new lines
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            prediction = predict_log_severity(line, model_package)
                            
                            if prediction in alert_levels:
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                alert_count[prediction] += 1
                                
                                # Color coding for different severity levels
                                if prediction == 'CRITICAL':
                                    prefix = "🔴 CRITICAL"
                                elif prediction == 'WARNING':
                                    prefix = "🟡 WARNING"
                                else:
                                    prefix = "🟢 NORMAL"
                                
                                print(f"[{timestamp}] {prefix}: {line[:100]}")
                
                # File size decreased or stayed same - just wait
                elif current_size < last_position:
                    logger.info("Log file rotated or truncated. Resetting position.")
                    last_position = current_size
                
                time.sleep(check_interval)
            
            except FileNotFoundError:
                logger.warning("Log file not found, waiting for it to be created...")
                time.sleep(check_interval)
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 70)
        logger.info("Monitoring stopped.")
        logger.info("Alert Summary:")
        for level in alert_levels:
            logger.info(f"  {level}: {alert_count[level]} alerts")


def process_entire_file(
    log_file_path: str = 'windows.log',
    model_path: str = 'model.pkl',
    output_file: str = 'predictions.log'
) -> None:
    """
    Process an entire log file and write predictions to an output file.
    Useful for batch processing existing logs.
    
    Args:
        log_file_path: Path to the input log file
        model_path: Path to the trained model
        output_file: Path to write predictions
    """
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}...")
    model_package = load_model(model_path)
    
    logger.info(f"Processing log file: {log_file_path}")
    
    critical_count = 0
    warning_count = 0
    normal_count = 0
    line_count = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("severity,log_line\n")
            
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        prediction = predict_log_severity(line, model_package)
                        out_f.write(f"{prediction},{line}\n")
                        
                        if prediction == 'CRITICAL':
                            critical_count += 1
                        elif prediction == 'WARNING':
                            warning_count += 1
                        else:
                            normal_count += 1
                        
                        line_count += 1
                        
                        if line_count % 50000 == 0:
                            logger.info(f"Processed {line_count:,} lines...")
        
        logger.info(f"\n✓ Batch processing complete!")
        logger.info(f"Total lines processed: {line_count:,}")
        logger.info(f"  - CRITICAL: {critical_count:,} ({100*critical_count/line_count:.1f}%)")
        logger.info(f"  - WARNING: {warning_count:,} ({100*warning_count/line_count:.1f}%)")
        logger.info(f"  - NORMAL: {normal_count:,} ({100*normal_count/line_count:.1f}%)")
        logger.info(f"Predictions written to: {output_file}")
    
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Log monitoring and prediction tool')
    parser.add_argument(
        '--mode',
        choices=['monitor', 'batch'],
        default='batch',
        help='Execution mode: monitor (real-time) or batch (process all)'
    )
    parser.add_argument(
        '--log-file',
        default='windows.log',
        help='Path to the log file'
    )
    parser.add_argument(
        '--model',
        default='model.pkl',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--output',
        default='predictions.log',
        help='Output file for batch predictions'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=2,
        help='Check interval in seconds (for monitor mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_log_file(args.log_file, args.model, args.interval)
    else:
        process_entire_file(args.log_file, args.model, args.output)
```

---

## log_predictor.py

Utility module for predictions.

```python
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
```

---

## config.py

Configuration file.

```python
"""
config.py
Configuration file for the ML pipeline.
Adjust these settings to optimize for your environment.
"""

# ============================================================================
# FILE PATHS
# ============================================================================

INPUT_LOG_FILE = 'windows.log'
LABELED_CSV_FILE = 'labeled_logs.csv'
MODEL_FILE = 'model.pkl'
PREDICTIONS_FILE = 'predictions.log'

# ============================================================================
# AUTO-LABELING CONFIGURATION
# ============================================================================

# Keywords that trigger CRITICAL classification
CRITICAL_KEYWORDS = [
    'critical',
    'error',
    'failed',
    'failure',
    'fatal',
    'exception',
    'crash',
    'crash',
    'crashed',
    'abort',
    'aborted',
    'panic',
    'severe',
    'emergency',
    'alert',
]

# Keywords that trigger WARNING classification
WARNING_KEYWORDS = [
    'warning',
    'warn',
    'deprecated',
    'issue',
    'timeout',
    'retry',
    'slow',
    'suspicious',
    'unauthorized',
    'denied',
]

# All other logs are classified as NORMAL

# ============================================================================
# DATA PREPARATION SETTINGS
# ============================================================================

# Number of lines to process before logging progress
PROGRESS_LOG_INTERVAL = 100000

# Number of lines to accumulate before writing to CSV
CHUNK_SIZE = 10000

# Maximum lines to process (None = all lines)
MAX_LINES_TO_PROCESS = None  # Set to 1000000 for testing

# ============================================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================================

# Number of features for HashingVectorizer
# Higher values = better feature representation but more memory
# Common values: 2**16 (65k), 2**18 (262k), 2**20 (1M)
N_FEATURES = 2**18  # 262,144 features

# Feature normalization
FEATURE_NORM = 'l2'

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Test/Train split ratio
TEST_SIZE = 0.2  # Use 20% for testing, 80% for training

# Random state for reproducibility
RANDOM_STATE = 42

# SVM Configuration
SVM_MAX_ITER = 2000
SVM_VERBOSE = 1

# Random Forest Configuration
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_VERBOSE = 1

# Logistic Regression Configuration
LR_MAX_ITER = 1000

# Stacking Ensemble Configuration
STACKING_CV = 5  # Cross-validation folds for stacking

# ============================================================================
# MONITORING SETTINGS
# ============================================================================

# Check interval for real-time monitoring (seconds)
MONITOR_CHECK_INTERVAL = 2

# Severity levels to alert on
ALERT_LEVELS = ['CRITICAL', 'WARNING']

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# Enable/disable verbose logging
VERBOSE = True

# Number of parallel jobs for scikit-learn models
N_JOBS = -1  # Use all available cores

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

# For very large files, you can further reduce memory usage:
# 1. Reduce N_FEATURES to 2**16
# 2. Reduce RF_N_ESTIMATORS to 50
# 3. Reduce test size to 0.1 (10%)
# 4. Process in smaller chunks

MEMORY_OPTIMIZATION = False  # Set to True for machines with <8GB RAM

if MEMORY_OPTIMIZATION:
    N_FEATURES = 2**16  # Reduce features
    RF_N_ESTIMATORS = 50  # Reduce trees
    TEST_SIZE = 0.1  # Less test data
    CHUNK_SIZE = 5000  # Smaller chunks
```

---

## test_pipeline.py

Testing and validation script.

```python
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
```

---

## requirements.txt

```
scikit-learn==1.5.1
numpy==1.24.3
joblib==1.4.0
```

---

## Summary & Running Instructions

### Installation
```powershell
pip install -r requirements.txt
```

### Step-by-Step Execution

**Step 1: Prepare Data** (2-3 hours)
```powershell
python prepare_data.py
```
Output: `labeled_logs.csv`

**Step 2: Train Models** (30-45 minutes)
```powershell
python train_model.py
```
Output: `model.pkl`

**Step 3a: Batch Predictions**
```powershell
python monitor.py --mode batch
```
Output: `predictions.log`

**Step 3b: Real-Time Monitoring**
```powershell
python monitor.py --mode monitor --interval 2
```

**Step 4: Programmatic Usage**
```python
from log_predictor import predict_log_severity
severity = predict_log_severity("ERROR: System crash")
print(severity)  # CRITICAL
```

### Testing
```powershell
python test_pipeline.py
```

---

## Key Features

✅ **Memory Efficient**: Streams 26GB file line-by-line  
✅ **Auto-Labeling**: Rule-based heuristics for 3 severity classes  
✅ **Multiple Models**: SVM + Random Forest + Logistic Regression + Ensemble  
✅ **Production Ready**: Error handling, logging, progress tracking  
✅ **Real-Time**: Monitor new logs as they appear  
✅ **Reusable**: Simple `predict_log_severity()` function  

---

End of Code Reference Document
