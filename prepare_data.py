
import csv
import logging
from pathlib import Path
from typing import Generator, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRITICAL_KEYWORDS = ['critical', 'error', 'failed', 'failure', 'fatal', 'exception', 'crash']
WARNING_KEYWORDS = ['warning', 'warn', 'deprecated', 'issue']
CHUNK_SIZE = 10000

def auto_label_log_line(line: str) -> str:
    line_lower = line.lower()

    for keyword in CRITICAL_KEYWORDS:
        if keyword in line_lower:
            return 'CRITICAL'

    for keyword in WARNING_KEYWORDS:
        if keyword in line_lower:
            return 'WARNING'

    return 'NORMAL'

def stream_log_file(log_file_path: str) -> Generator[str, None, None]:
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    yield line

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
    logger.info(f"Starting data preparation from {log_file_path}...")
    
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['log_line', 'label'])
            
            total_lines = 0
            critical_count = 0
            warning_count = 0
            normal_count = 0

            for line in stream_log_file(log_file_path):
                label = auto_label_log_line(line)
                writer.writerow([line, label])

                if label == 'CRITICAL':
                    critical_count += 1
                elif label == 'WARNING':
                    warning_count += 1
                else:
                    normal_count += 1
                
                total_lines += 1

                if total_lines % CHUNK_SIZE == 0:
                    logger.info(
                        f"Processed {total_lines:,} lines | "
                        f"CRITICAL: {critical_count:,}, WARNING: {warning_count:,}, NORMAL: {normal_count:,}"
                    )

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

    log_file = 'windows.log'
    if not Path(log_file).exists():
        logger.warning(f"{log_file} not found. Creating a sample log for testing...")

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

    prepare_data(log_file, 'labeled_logs.csv', max_lines=None)
