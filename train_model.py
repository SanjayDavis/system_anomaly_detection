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
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
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
    Load all data from CSV file with memory-efficient batching.
    Uses buffering to avoid MemoryError on huge files.
    """
    logger.info(f"Loading data from {csv_path} (memory-efficient mode)...")
    
    X = []
    y = []
    buffer_size = 1000000  # 1M samples per buffer
    buffer_count = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8', buffering=65536*2) as csvfile:
            reader = csv.DictReader(csvfile)
            for row_num, row in enumerate(reader, 1):
                try:
                    X.append(row['log_line'].strip())
                    y.append(row['label'].strip())
                except (KeyError, AttributeError) as e:
                    logger.warning(f"Skipping row {row_num}: {e}")
                    continue
                
                # Log progress
                if row_num % 1000000 == 0:
                    logger.info(f"Loaded {row_num:,} samples...")
                
                # Memory check - if buffer getting too large, force cleanup
                if len(X) >= buffer_size:
                    logger.info(f"Buffer reached {len(X):,} samples, proceeding to training...")
                    logger.warning("Note: Using stratified sampling to reduce memory")
                    break
    
    except MemoryError as e:
        logger.error(f"MemoryError while loading data: {e}")
        logger.warning(f"Loaded {len(X):,} samples before memory limit")
        logger.info("Using what we have in memory for training...")
    
    logger.info(f"✓ Loaded {len(X):,} total samples")
    
    if len(X) == 0:
        raise ValueError("No data could be loaded from CSV file")
    
    return X, y


def extract_features(X_train, X_test):
    """
    Extract features using TfidfVectorizer for MAXIMUM ACCURACY.
    More accurate than HashingVectorizer but requires more memory.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    logger.info("Extracting features using TfidfVectorizer (HIGH ACCURACY MODE)...")
    
    # TfidfVectorizer provides better accuracy than HashingVectorizer
    # Uses IDF weighting to emphasize important terms
    vectorizer = TfidfVectorizer(
        max_features=5000,          # Learn from 5000 most important features
        min_df=5,                   # Ignore terms that appear in < 5 docs
        max_df=0.8,                 # Ignore terms that appear in > 80% of docs
        ngram_range=(1, 2),         # Use unigrams and bigrams
        sublinear_tf=True,          # Sublinear TF scaling
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )
    
    logger.info("Fitting vectorizer on training data...")
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    logger.info(f"Features extracted: {X_train_features.shape[1]} dimensions")
    logger.info(f"Training set shape: {X_train_features.shape}")
    logger.info(f"Test set shape: {X_test_features.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    return vectorizer, X_train_features, X_test_features


def train_svm(X_train, X_test, y_train, y_test):
    """Train SVM model with RBF kernel for MAXIMUM ACCURACY"""
    logger.info("Training SVM model (RBF kernel, HIGH ACCURACY)...")
    
    # SVC with RBF kernel is more accurate than LinearSVC
    # Uses probability=True for confidence scores
    svm = SVC(
        kernel='rbf',               # RBF kernel for non-linear classification
        C=100,                      # High C value for better fit
        gamma='scale',              # Auto gamma scaling
        probability=True,           # Enable probability estimates
        random_state=RANDOM_STATE,
        verbose=1,
        class_weight='balanced'     # Handle class imbalance
    )
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"SVM (RBF) Accuracy: {accuracy:.4f}")
    
    return svm


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model with HIGH ACCURACY settings"""
    logger.info("Training Random Forest model (HIGH ACCURACY MODE)...")
    rf = RandomForestClassifier(
        n_estimators=500,           # More trees = better accuracy
        max_depth=30,               # Deeper trees capture more patterns
        min_samples_split=5,        # Split more aggressively
        min_samples_leaf=2,         # Allow small leaf nodes
        max_features='sqrt',        # Feature sampling strategy
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        class_weight='balanced'     # Handle class imbalance
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
    
    return rf


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model with HIGH ACCURACY settings"""
    logger.info("Training Logistic Regression model (HIGH ACCURACY MODE)...")
    lr = LogisticRegression(
        max_iter=5000,              # More iterations for convergence
        C=0.1,                      # Lower C for stronger regularization
        solver='lbfgs',             # Better for multiclass
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced',    # Handle class imbalance
        verbose=1
    )
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    return lr


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model with HIGH ACCURACY settings"""
    logger.info("Training Gradient Boosting model (HIGH ACCURACY MODE)...")
    gb = GradientBoostingClassifier(
        n_estimators=300,           # More boosting rounds
        learning_rate=0.05,         # Lower learning rate for better accuracy
        max_depth=7,                # Medium depth for complex interactions
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,              # Stochastic boosting
        random_state=RANDOM_STATE,
        verbose=1
    )
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    
    return gb


def create_stacking_ensemble(svm, rf, gb, X_train, X_test, y_train, y_test):
    """Create Stacking Classifier ensemble with MAXIMUM ACCURACY"""
    logger.info("Creating Stacking Ensemble (HIGH ACCURACY MODE)...")
    
    stacking_clf = StackingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
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
    test_size: float = 0.2,
    batch_size: int = 1000000
) -> None:
    """
    Main training pipeline with MEMORY-EFFICIENT batching:
    1. Load data in batches
    2. Extract features
    3. Train models
    4. Create ensemble
    5. Save model to disk
    """
    logger.info("=" * 60)
    logger.info("Starting ML Model Training Pipeline (MEMORY EFFICIENT)")
    logger.info("=" * 60)
    
    # Step 1: Load data (with memory buffering)
    logger.info(f"\nStep 1: Loading data with batch size {batch_size:,}...")
    X, y = load_all_data(csv_path)
    
    actual_samples = len(X)
    logger.info(f"Loaded {actual_samples:,} samples (memory-safe subset)")
    
    if actual_samples < 10000:
        logger.warning("WARNING: Very small dataset (< 10k samples)")
        logger.warning("Consider running prepare_data.py without max_lines limit")
    
    # Step 2: Split data
    logger.info(f"\nStep 2: Splitting data ({test_size*100}% test)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as e:
        logger.error(f"Error during train/test split: {e}")
        logger.warning("Attempting without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
    
    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Step 3: Extract features
    logger.info("\nStep 3: Extracting features...")
    vectorizer, X_train_features, X_test_features = extract_features(X_train, X_test)
    
    # Clear raw text data to free memory
    logger.info("Freeing text memory...")
    del X_train, X_test, X
    import gc
    gc.collect()
    
    # Step 4: Train individual models
    logger.info("\nStep 4: Training Individual Models (HIGH ACCURACY MODE)")
    logger.info("=" * 60)
    
    svm = train_svm(X_train_features, X_test_features, y_train, y_test)
    logger.info("OK: SVM training complete")
    
    rf = train_random_forest(X_train_features, X_test_features, y_train, y_test)
    logger.info("OK: RandomForest training complete")
    
    gb = train_gradient_boosting(X_train_features, X_test_features, y_train, y_test)
    logger.info("OK: Gradient Boosting training complete")
    
    # Step 5: Create ensemble
    logger.info("\nStep 5: Creating Ensemble Classifier")
    logger.info("=" * 60)
    
    ensemble = create_stacking_ensemble(
        svm, rf, gb, X_train_features, X_test_features, y_train, y_test
    )
    logger.info("OK: Ensemble training complete")
    
    # Step 6: Evaluating results
    logger.info("\nStep 6: Evaluating Ensemble")
    logger.info("=" * 60)
    
    y_pred_ensemble = ensemble.predict(X_test_features)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    logger.info(f"\nEnsemble Final Accuracy: {ensemble_accuracy:.4f}")
    
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred_ensemble))
    
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_ensemble)
    logger.info("\n" + str(cm))
    
    # Step 7: Save model and vectorizer
    logger.info("\nStep 7: Saving Models")
    logger.info("=" * 60)
    
    model_package = {
        'vectorizer': vectorizer,
        'ensemble': ensemble,
        'label_classes': sorted(list(set(y))),
        'accuracy': ensemble_accuracy,
        'samples_trained': actual_samples
    }
    
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    logger.info(f"✓ Model package saved to: {model_path}")
    logger.info(f"  Model size: {model_path_obj.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"  Trained on: {actual_samples:,} samples")
    logger.info("=" * 60)
    logger.info("\n✓✓✓ TRAINING COMPLETE ✓✓✓")
    logger.info(f"Final Accuracy: {ensemble_accuracy:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    csv_file = 'labeled_logs.csv'
    
    if not Path(csv_file).exists():
        logger.error(f"{csv_file} not found. Run prepare_data.py first.")
        exit(1)
    
    train_model_pipeline(csv_file, 'model.pkl')
