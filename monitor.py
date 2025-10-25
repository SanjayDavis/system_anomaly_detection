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
