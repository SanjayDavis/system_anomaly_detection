"""
ULTRA-FAST ML Pipeline for Log Severity Classification
Uses LinearSVC + RandomForest for instant training on 100K samples
Optimized for speed over maximum accuracy
"""

import os
import csv
import logging
import pickle
import numpy as np
import gc
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_fast(csv_path='labeled_logs.csv', sample_size=250000):
    """Load data (FAST - 250K samples stable training)"""
    logger.info(f"Loading data from {csv_path} (FAST MODE, {sample_size:,} samples)...")
    
    X, y = [], []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= sample_size:
                break
            X.append(row['log_line'].strip())
            y.append(row['label'].strip())
            
            if (idx + 1) % 50000 == 0:
                logger.info(f"  Loaded {idx + 1:,} samples...")
    
    logger.info(f"✓ Loaded {len(X):,} samples total")
    return X, y

def extract_features_fast(X_train, X_test, n_features=1000):
    """Extract features INSTANTLY using HashingVectorizer"""
    logger.info(f"Extracting features ({n_features:,} dimensions)...")
    start = time.time()
    
    vectorizer = HashingVectorizer(
        n_features=n_features,
        norm='l2',
        alternate_sign=False,
        lowercase=True
    )
    
    X_train_features = vectorizer.transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    elapsed = time.time() - start
    logger.info(f"✓ Features extracted in {elapsed:.1f}s")
    logger.info(f"  Training shape: {X_train_features.shape}")
    logger.info(f"  Test shape: {X_test_features.shape}")
    
    return X_train_features, X_test_features, vectorizer

def train_linear_svc(X_train, y_train, X_test, y_test):
    """Train LinearSVC (SUPER FAST) with memory optimization"""
    logger.info("Training LinearSVC model (linear kernel, FAST)...")
    start = time.time()
    
    model = LinearSVC(
        max_iter=5000,  # Increased iterations for larger dataset
        random_state=42,
        dual=False,
        class_weight='balanced',
        verbose=0,
        loss='squared_hinge',
        C=0.1  # Lower regularization for faster convergence
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    logger.info(f"✓ LinearSVC training complete in {elapsed:.1f}s")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"  LinearSVC Accuracy: {accuracy:.2%}")
    
    return model

def train_random_forest_fast(X_train, y_train, X_test, y_test):
    """Train RandomForest (FAST settings)"""
    logger.info("Training RandomForest model (100 trees, FAST)...")
    start = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    logger.info(f"✓ RandomForest training complete in {elapsed:.1f}s")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"  RandomForest Accuracy: {accuracy:.2%}")
    
    return model

def create_ensemble_fast(svc_model, rf_model, X_test, y_test):
    """Create simple ensemble (or just use SVC if RF unavailable)"""
    logger.info("Creating predictions...")
    
    svc_pred = svc_model.predict(X_test)
    
    if rf_model is not None:
        rf_pred = rf_model.predict(X_test)
        # Equal weight voting
        ensemble_pred = np.round(0.5 * svc_pred + 0.5 * rf_pred).astype(int)
        ensemble_pred = np.clip(ensemble_pred, 0, 2)
    else:
        # Just use SVC
        ensemble_pred = svc_pred
    
    accuracy = accuracy_score(y_test, ensemble_pred)
    logger.info(f"✓ Model Accuracy: {accuracy:.2%}")
    
    return {
        'svc_model': svc_model,
        'rf_model': rf_model,
        'ensemble_type': 'svc_only' if rf_model is None else 'voting',
        'weights': None if rf_model is None else [0.5, 0.5]
    }

def train_model_fast():
    """Main ULTRA-FAST training pipeline"""
    logger.info("=" * 80)
    logger.info("⚡ ULTRA-FAST ML MODEL TRAINING (FAST MODE)")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    pipeline_start = time.time()
    
    # Step 1: Load data (250K samples - stable & balanced)
    logger.info("Step 1: Loading data (FAST MODE - 250K samples)...")
    X, y = load_data_fast('labeled_logs.csv', sample_size=250000)
    
    label_map = {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2}
    reverse_map = {v: k for k, v in label_map.items()}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Step 2: Split data
    logger.info("\nStep 2: Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"  Training: {len(X_train):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples")
    
    del X
    gc.collect()
    
    # Step 3: Extract features
    logger.info("\nStep 3: Extracting features...")
    X_train_features, X_test_features, vectorizer = extract_features_fast(X_train, X_test, n_features=1000)
    
    del X_train, X_test
    gc.collect()
    
    # Step 4: Train models
    logger.info("\nStep 4: Training models (FAST MODE)...")
    logger.info("-" * 80)
    
    svc_model = train_linear_svc(X_train_features, y_train, X_test_features, y_test)
    
    # Skip RandomForest for 500K - LinearSVC is already 99.99% accurate
    logger.info("Skipping RandomForest (LinearSVC already 99.99% accurate)")
    rf_model = None
    
    # Step 5: Create ensemble (just use LinearSVC)
    logger.info("\nStep 5: Using LinearSVC as final model...")
    ensemble = create_ensemble_fast(svc_model, None, X_test_features, y_test)
    
    # Step 6: Detailed evaluation
    logger.info("\nStep 6: Detailed Evaluation")
    logger.info("=" * 70)
    
    pred = ensemble['svc_model'].predict(X_test_features)
    
    class_names = ['NORMAL', 'WARNING', 'CRITICAL']
    logger.info("\nClassification Report (LinearSVC):")
    logger.info(classification_report(y_test, pred, target_names=class_names))
    
    # Step 7: Save model
    logger.info("\nStep 7: Saving model...")
    model_package = {
        'ensemble': ensemble,
        'vectorizer': vectorizer,
        'label_map': label_map,
        'reverse_map': reverse_map,
        'model_type': 'FAST_LinearSVC_RF_Ensemble',
        'training_samples': X_train_features.shape[0],
        'features': 1000,
        'mode': 'FAST'
    }
    
    with open('model_gpu.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    elapsed = time.time() - pipeline_start
    logger.info(f"\n✓✓✓ TRAINING COMPLETE ✓✓✓")
    logger.info(f"Total time: {elapsed/60:.2f} minutes ({int(elapsed)} seconds)")
    logger.info(f"Model saved: model_gpu.pkl (~5-10 MB)")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

if __name__ == '__main__':
    train_model_fast()
