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
# FEATURE EXTRACTION SETTINGS (HIGH ACCURACY MODE)
# ============================================================================

# Using TfidfVectorizer instead of HashingVectorizer for MAXIMUM ACCURACY
# TfidfVectorizer provides better text representation with IDF weighting

MAX_FEATURES = 5000         # Learn from 5000 most important features
MIN_DF = 5                  # Ignore terms that appear in < 5 documents
MAX_DF = 0.8                # Ignore terms that appear in > 80% of documents
NGRAM_RANGE = (1, 2)        # Use unigrams and bigrams for better context
SUBLINEAR_TF = True         # Sublinear TF scaling for better accuracy
USE_IDF = True              # Inverse document frequency weighting
FEATURE_NORM = 'l2'         # L2 normalization

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Test/Train split ratio
TEST_SIZE = 0.2  # Use 20% for testing, 80% for training

# Random state for reproducibility
RANDOM_STATE = 42

# SVM Configuration (HIGH ACCURACY)
SVM_KERNEL = 'rbf'          # RBF kernel for non-linear classification
SVM_C = 100                 # High C value for better fit
SVM_GAMMA = 'scale'         # Auto gamma scaling
SVM_PROBABILITY = True      # Enable probability estimates
SVM_CLASS_WEIGHT = 'balanced'  # Handle class imbalance

# Random Forest Configuration (HIGH ACCURACY)
RF_N_ESTIMATORS = 500       # More trees = better accuracy (was 100)
RF_MAX_DEPTH = 30           # Deeper trees (was 20)
RF_MIN_SAMPLES_SPLIT = 5    # Split more aggressively
RF_MIN_SAMPLES_LEAF = 2     # Allow small leaf nodes
RF_CLASS_WEIGHT = 'balanced'  # Handle class imbalance
RF_VERBOSE = 1

# Gradient Boosting Configuration (NEW - High Accuracy)
GB_N_ESTIMATORS = 300       # More boosting rounds
GB_LEARNING_RATE = 0.05     # Lower learning rate for better accuracy
GB_MAX_DEPTH = 7            # Medium depth for complex interactions
GB_MIN_SAMPLES_SPLIT = 5
GB_MIN_SAMPLES_LEAF = 2
GB_SUBSAMPLE = 0.8          # Stochastic boosting
GB_VERBOSE = 1

# Logistic Regression Configuration (HIGH ACCURACY)
LR_MAX_ITER = 5000          # More iterations (was 1000)
LR_C = 0.1                  # Lower C for stronger regularization
LR_SOLVER = 'lbfgs'         # Better for multiclass
LR_CLASS_WEIGHT = 'balanced'  # Handle class imbalance

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
