# Model Training & Loading Pipeline

## How Training Works (train_model_fast.py)

### Step 1: Data Loading
```python
X, y = load_data_fast('labeled_logs.csv', sample_size=500000)
# Loads 500K log lines from CSV
# X = list of log strings
# y = list of labels ('NORMAL', 'WARNING', 'CRITICAL')
```

### Step 2: Feature Extraction
```python
vectorizer = HashingVectorizer(n_features=1000)
X_train_features = vectorizer.transform(X_train)  # (400K, 1000) sparse matrix
X_test_features = vectorizer.transform(X_test)    # (100K, 1000) sparse matrix
```

### Step 3: Model Training
```python
svc_model = LinearSVC()        # Fast linear classifier
rf_model = RandomForestClassifier()  # Ensemble method
```

### Step 4: Save Everything to Pickle
```python
model_package = {
    'svc_model': svc_model,                    # Trained LinearSVC
    'rf_model': rf_model,                      # Trained RandomForest
    'vectorizer': vectorizer,                  # HashingVectorizer
    'label_map': {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2},
    'reverse_map': {0: 'NORMAL', 1: 'WARNING', 2: 'CRITICAL'},
    'model_type': 'FAST_LinearSVC_RF_Ensemble',
    'training_samples': 500000,
    'features': 1000
}

# Save to file
with open('model_gpu.pkl', 'wb') as f:
    pickle.dump(model_package, f)
```

File size: ~5-10 MB containing all trained models

---

## How log_checker.py Loads & Uses the Model

### Step 1: Load Pickle File
```python
class LogScanner:
    def __init__(self, model_path='model_gpu.pkl'):
        # Load entire model package from disk
        with open(model_path, 'rb') as f:
            self.model_package = pickle.load(f)
        
        # Extract components
        self.vectorizer = self.model_package['vectorizer']
        self.ensemble = self.model_package['ensemble']
        self.reverse_map = self.model_package['reverse_map']
```

### Step 2: For Each Log Line
```python
def predict_severity(self, log_line):
    # Step 1: Transform using SAME vectorizer used during training
    features = self.vectorizer.transform([log_line])  # (1, 1000) sparse matrix
    
    # Step 2: Predict using trained models
    svc_pred = self.ensemble['svc_model'].predict(features)
    rf_pred = self.ensemble['rf_model'].predict(features)
    
    # Step 3: Combine predictions
    ensemble_pred = (svc_pred + rf_pred) / 2
    
    # Step 4: Convert to label
    severity = self.reverse_map[int(ensemble_pred)]  # 'CRITICAL', 'WARNING', or 'NORMAL'
    
    return severity
```

### Step 3: Scan System
```python
# Example: Scan Windows Event Logs
for event in windows_event_logs:
    msg = event.message
    severity = predict_severity(msg)
    if severity == 'CRITICAL':
        problems_found['CRITICAL'].append(msg)
```

---

## Data Flow Diagram

```
TRAINING PHASE:
================
labeled_logs.csv (500K lines)
    ↓
load_data_fast()
    ↓
X = [log1, log2, ..., log500k]
y = [NORMAL, WARNING, CRITICAL, ...]
    ↓
train_test_split (80/20)
    ↓
X_train, X_test, y_train, y_test
    ↓
extract_features_fast()
    ↓
X_train_features (400K, 1000)  ← vectorizer learns how to transform
X_test_features (100K, 1000)
    ↓
train models:
  - LinearSVC
  - RandomForest
    ↓
Save model_package with:
  - Trained LinearSVC (weights)
  - Trained RandomForest (tree structure)
  - Vectorizer (hash features configuration)
    ↓
model_gpu.pkl (5-10 MB)


PREDICTION PHASE (log_checker.py):
====================================
model_gpu.pkl (load from disk)
    ↓
Extract:
  - vectorizer
  - svc_model (trained)
  - rf_model (trained)
    ↓
For each new log line:
    ↓
log_line = "ERROR: System crashed"
    ↓
features = vectorizer.transform(log_line)  ← SAME vectorizer, uses same hash function
    ↓
features shape: (1, 1000)  ← Same 1000 dimensions as training
    ↓
svc_pred = svc_model.predict(features)     ← Uses trained weights
rf_pred = rf_model.predict(features)       ← Uses trained trees
    ↓
Combine → severity = 'CRITICAL'
    ↓
Report to user
```

---

## Key Points

1. **Same Vectorizer**: The SAME `vectorizer` is used in both training and prediction
   - During training: learns how to hash log text into 1000 dimensions
   - During prediction: applies the same hashing rules

2. **Trained Models Stored**: Model weights/trees are frozen in pickle
   - LinearSVC: stores coefficient weights
   - RandomForest: stores complete tree structures
   - These don't change after training

3. **No Retraining Needed**: `log_checker.py` just loads and uses, no training

4. **Pickle Format**: Everything is in ONE file for easy distribution
   - Can copy `model_gpu.pkl` to any machine
   - `log_checker.py` will work as long as sklearn is installed

5. **Scaling**: Works for ANY number of new predictions
   - Trained on 500K samples
   - Can predict on millions of new logs
   - Speed: ~100-1000 logs per second depending on system

---

## Run Commands

```powershell
# Train the model (one time)
python train_model_fast.py
# Creates: model_gpu.pkl (5-10 MB)

# Use the model for scanning (any time after)
python log_checker.py
# Loads: model_gpu.pkl
# Scans: Windows Event Logs + system log files
# Reports: All CRITICAL and WARNING issues found
```

---

## File Structure

```
Windows.tar/
├── train_model_fast.py        (training script - RUN ONCE)
├── labeled_logs.csv           (training data - 500K samples)
├── model_gpu.pkl              (trained model - CREATED AFTER TRAINING)
├── log_checker.py             (prediction/scanning script - RUN MANY TIMES)
└── system_log_analysis_*.txt  (output reports - CREATED AFTER SCANNING)
```
