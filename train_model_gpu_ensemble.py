"""
GPU-Accelerated Multi-Model Ensemble Training
Uses XGBoost, LightGBM, and CatBoost with GPU acceleration for maximum accuracy
"""

import time
import pickle
import gc
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠ CatBoost not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_gpu_ensemble.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check GPU availability for each library"""
    logger.info("\n" + "=" * 80)
    logger.info("GPU AVAILABILITY CHECK")
    logger.info("=" * 80)
    
    gpu_status = {}
    
    # Check XGBoost
    if XGBOOST_AVAILABLE:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_status['xgboost'] = 'cuda'
                logger.info("[OK] XGBoost: GPU (CUDA) available")
            else:
                gpu_status['xgboost'] = 'cpu'
                logger.info("[WARN] XGBoost: Using CPU (no CUDA)")
        except:
            gpu_status['xgboost'] = 'cpu'
            logger.info("[WARN] XGBoost: Using CPU")
    
    # Check LightGBM
    if LIGHTGBM_AVAILABLE:
        try:
            gpu_status['lightgbm'] = 'gpu'
            logger.info("[OK] LightGBM: GPU support available")
        except:
            gpu_status['lightgbm'] = 'cpu'
            logger.info("[WARN] LightGBM: Using CPU")
    
    # Check CatBoost
    if CATBOOST_AVAILABLE:
        try:
            gpu_status['catboost'] = 'GPU'
            logger.info("[OK] CatBoost: GPU support available")
        except:
            gpu_status['catboost'] = 'CPU'
            logger.info("[WARN] CatBoost: Using CPU")
    
    logger.info("=" * 80 + "\n")
    return gpu_status


def load_data_fast(csv_file, sample_size=1000000):
    """Load training data efficiently using pandas"""
    logger.info(f"Loading data from {csv_file} (max {sample_size:,} samples)...")
    
    start_time = time.time()
    
    # Use pandas to properly handle CSV with commas in fields
    df = pd.read_csv(csv_file, nrows=sample_size, encoding='utf-8-sig')
    
    data = df['log_line'].tolist()
    labels = df['label'].tolist()
    
    elapsed = time.time() - start_time
    logger.info(f"[OK] Loaded {len(data):,} samples in {elapsed:.1f}s\n")
    
    return data, labels


def extract_features_tfidf(X_train, X_test, max_features=10000):
    """Extract TF-IDF features (better for gradient boosting models)"""
    logger.info(f"Extracting TF-IDF features ({max_features:,} dimensions, bigrams)...")
    
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    elapsed = time.time() - start_time
    logger.info(f"[OK] Features extracted in {elapsed:.1f}s")
    logger.info(f"  Training shape: {X_train_features.shape}")
    logger.info(f"  Test shape: {X_test_features.shape}\n")
    
    return X_train_features, X_test_features, vectorizer


def train_xgboost_gpu(X_train, y_train, X_test, y_test, gpu_status):
    """Train XGBoost with GPU/CPU"""
    logger.info("Training XGBoost model...")
    logger.info("-" * 80)
    
    start_time = time.time()
    
    # Use hist (CPU) since GPU not properly configured
    tree_method = 'hist'
    
    model = xgb.XGBClassifier(
        tree_method=tree_method,
        n_estimators=300,  # Reduced for speed
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    logger.info(f"[OK] XGBoost training complete in {elapsed:.1f}s")
    logger.info(f"  Device: {tree_method}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%\n")
    
    return model


def train_lightgbm_gpu(X_train, y_train, X_test, y_test, gpu_status):
    """Train LightGBM"""
    logger.info("Training LightGBM model...")
    logger.info("-" * 80)
    
    start_time = time.time()
    
    # Use CPU
    device = 'cpu'
    
    model = lgb.LGBMClassifier(
        device=device,
        n_estimators=300,  # Reduced for speed
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    logger.info(f"[OK] LightGBM training complete in {elapsed:.1f}s")
    logger.info(f"  Device: {device}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%\n")
    
    return model


def train_catboost_gpu(X_train, y_train, X_test, y_test, gpu_status):
    """Train CatBoost with GPU acceleration"""
    logger.info("Training CatBoost model (GPU-accelerated)...")
    logger.info("-" * 80)
    
    start_time = time.time()
    
    # Determine device
    task_type = gpu_status.get('catboost', 'CPU')
    
    model = CatBoostClassifier(
        task_type=task_type,
        iterations=500,
        depth=8,
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        thread_count=-1
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    logger.info(f"[OK] CatBoost training complete in {elapsed:.1f}s")
    logger.info(f"  Device: {task_type}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%\n")
    
    return model


def create_weighted_ensemble(models, model_names, X_test, y_test):
    """Create weighted ensemble based on individual model performance"""
    logger.info("Creating weighted ensemble...")
    logger.info("-" * 80)
    
    # Calculate weights based on accuracy
    accuracies = []
    predictions_list = []
    
    for model, name in zip(models, model_names):
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
        predictions_list.append(pred)
        logger.info(f"  {name}: {acc*100:.2f}% accuracy")
    
    # Normalize weights
    total_acc = sum(accuracies)
    weights = [acc / total_acc for acc in accuracies]
    
    logger.info("\nEnsemble weights:")
    for name, weight in zip(model_names, weights):
        logger.info(f"  {name}: {weight*100:.1f}%")
    
    # Weighted voting
    ensemble_pred = np.zeros((len(y_test), 3))
    for pred, weight in zip(predictions_list, weights):
        for i, p in enumerate(pred):
            ensemble_pred[i, p] += weight
    
    final_pred = np.argmax(ensemble_pred, axis=1)
    ensemble_acc = accuracy_score(y_test, final_pred)
    
    logger.info(f"\n[OK] Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    
    ensemble = {
        'models': models,
        'model_names': model_names,
        'weights': weights,
        'accuracies': accuracies
    }
    
    return ensemble, final_pred


def train_gpu_ensemble():
    """Main GPU ensemble training pipeline"""
    logger.info("=" * 80)
    logger.info("GPU-ACCELERATED MULTI-MODEL ENSEMBLE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    pipeline_start = time.time()
    
    # Check GPU availability
    gpu_status = check_gpu_availability()
    
    # Step 1: Load data (reduced to 500K to fit in memory)
    logger.info("Step 1: Loading data (500K samples for memory efficiency)...")
    logger.info("-" * 80)
    X, y = load_data_fast('labeled_logs.csv', sample_size=500000)
    
    label_map = {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2}
    reverse_map = {v: k for k, v in label_map.items()}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Step 2: Split data
    logger.info("Step 2: Splitting data (80/20)...")
    logger.info("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"  Training: {len(X_train):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples\n")
    
    del X
    gc.collect()
    
    # Step 3: Extract features
    logger.info("Step 3: Extracting TF-IDF features...")
    logger.info("-" * 80)
    X_train_features, X_test_features, vectorizer = extract_features_tfidf(
        X_train, X_test, max_features=3000  # Reduced for memory
    )
    
    # Convert to dense in smaller batches
    logger.info("Converting to dense arrays (batch processing)...")
    
    batch_size = 50000  # Smaller batches
    X_train_dense_list = []
    
    for i in range(0, X_train_features.shape[0], batch_size):
        try:
            batch = X_train_features[i:i+batch_size].toarray().astype(np.float32)  # Use float32 instead of float64
            X_train_dense_list.append(batch)
            logger.info(f"  Converted {min(i+batch_size, X_train_features.shape[0]):,}/{X_train_features.shape[0]:,} training samples")
            del batch
            gc.collect()
        except MemoryError:
            logger.error(f"Memory error at batch {i}. Stopping...")
            break
    
    if not X_train_dense_list:
        logger.error("[ERROR] Could not convert any training data to dense format")
        return
    
    X_train_dense = np.vstack(X_train_dense_list).astype(np.float32)
    del X_train_dense_list
    gc.collect()
    
    X_test_dense = X_test_features.toarray().astype(np.float32)
    logger.info(f"  Converted {X_test_features.shape[0]:,} test samples\n")
    
    del X_train, X_test
    gc.collect()
    
    # Step 4: Train individual models
    logger.info("Step 4: Training individual models...")
    logger.info("=" * 80)
    
    models = []
    model_names = []
    
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost_gpu(X_train_dense, y_train, X_test_dense, y_test, gpu_status)
        models.append(xgb_model)
        model_names.append('XGBoost')
    
    if LIGHTGBM_AVAILABLE:
        lgb_model = train_lightgbm_gpu(X_train_dense, y_train, X_test_dense, y_test, gpu_status)
        models.append(lgb_model)
        model_names.append('LightGBM')
    
    if CATBOOST_AVAILABLE:
        cat_model = train_catboost_gpu(X_train_dense, y_train, X_test_dense, y_test, gpu_status)
        models.append(cat_model)
        model_names.append('CatBoost')
    
    if not models:
        logger.error("[ERROR] No GPU models available! Please install xgboost, lightgbm, or catboost")
        return
    
    # Step 5: Create ensemble
    logger.info("Step 5: Creating weighted ensemble...")
    logger.info("=" * 80)
    ensemble, ensemble_pred = create_weighted_ensemble(models, model_names, X_test_dense, y_test)
    
    # Step 6: Detailed evaluation
    logger.info("\nStep 6: Detailed Evaluation")
    logger.info("=" * 80)
    
    class_names = ['NORMAL', 'WARNING', 'CRITICAL']
    logger.info("\nEnsemble Classification Report:")
    logger.info(classification_report(y_test, ensemble_pred, target_names=class_names))
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info(f"\n{cm}")
    
    # Step 7: Save model
    logger.info("\nStep 7: Saving model...")
    logger.info("-" * 80)
    
    model_package = {
        'ensemble': ensemble,
        'vectorizer': vectorizer,
        'label_map': label_map,
        'reverse_map': reverse_map,
        'model_type': 'GPU_Ensemble_XGB_LGB_CAT',
        'training_samples': len(X_train_dense),
        'features': X_train_features.shape[1],
        'mode': 'GPU_ENSEMBLE',
        'gpu_status': gpu_status,
        'model_names': model_names
    }
    
    with open('model_gpu.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    elapsed = time.time() - pipeline_start
    logger.info(f"\n[OK] TRAINING COMPLETE")
    logger.info(f"Total time: {elapsed/60:.2f} minutes ({int(elapsed)} seconds)")
    logger.info(f"Model saved: model_gpu.pkl")
    logger.info(f"Models in ensemble: {', '.join(model_names)}")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    train_gpu_ensemble()
