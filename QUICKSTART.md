# 🚀 QUICK START GUIDE

## ⚡ TL;DR (30 seconds)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (auto-label logs from windows.log)
python prepare_data.py

# 3. Train models
python train_model.py

# 4. Start monitoring
python monitor.py --mode batch
```

---

## 📊 Full Step-by-Step

### Step 1️⃣: Setup Environment (2 minutes)

```powershell
# Open PowerShell in your project directory

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
```

**Expected:** Should print version number without errors.

---

### Step 2️⃣: Prepare Data (2-3 hours for 26GB)

```powershell
python prepare_data.py
```

**What happens:**
- Reads `windows.log` line-by-line
- Auto-labels each line as CRITICAL, WARNING, or NORMAL
- Creates `labeled_logs.csv` with ~100M labeled examples

**Expected output:**
```
Starting data preparation from windows.log...
Processed 100,000 lines...
Processed 200,000 lines...
...
✓ Data preparation complete!
Total lines processed: 100,000,000
  - CRITICAL: 15,234,567 (15.2%)
  - WARNING: 28,456,789 (28.5%)
  - NORMAL: 56,308,644 (56.3%)
Output saved to: labeled_logs.csv
```

---

### Step 3️⃣: Train Models (30-45 minutes)

```powershell
python train_model.py
```

**What happens:**
- Loads labeled data from CSV
- Trains 4 models: SVM, Random Forest, Logistic Regression, + Ensemble
- Evaluates accuracy on test set
- Saves `model.pkl` (1.4GB)

**Expected output:**
```
Loading data from labeled_logs.csv...
Loaded 100,000,000 total samples

Splitting data (20% test)...
Training set: 80,000,000 samples
Test set: 20,000,000 samples

Training SVM model...
SVM Accuracy: 0.8876

Training Random Forest model...
Random Forest Accuracy: 0.8923

Training Logistic Regression model...
Logistic Regression Accuracy: 0.8856

Creating Stacking Ensemble...
Ensemble Final Accuracy: 0.8945

✓ Model package saved to: model.pkl
  Model size: 1456.78 MB
```

---

### Step 4️⃣: Use the Model

#### Option A: Batch Process All Logs
```powershell
python monitor.py --mode batch --log-file windows.log --output predictions.log
```

Creates `predictions.log` with severity predictions for every line.

#### Option B: Real-Time Monitoring
```powershell
python monitor.py --mode monitor
```

Watches for new lines and alerts on CRITICAL/WARNING.

#### Option C: Use in Python Code
```python
from log_predictor import predict_log_severity

severity = predict_log_severity("ERROR: System crash detected")
print(severity)  # Output: CRITICAL
```

---

## 🧪 Test Everything First

Before running on your 26GB file, test with a sample:

```powershell
python test_pipeline.py
```

This runs 5 validation tests to ensure everything works.

---

## 🎯 Commands Cheat Sheet

```powershell
# Full pipeline
python prepare_data.py
python train_model.py
python monitor.py --mode batch

# Real-time monitoring
python monitor.py --mode monitor --interval 2

# Batch processing
python monitor.py --mode batch --output my_predictions.log

# Test everything
python test_pipeline.py

# Test with limited samples
# (Edit config.py, set MAX_LINES_TO_PROCESS = 100000)
python prepare_data.py
python train_model.py
python test_pipeline.py
```

---

## ⏱️ Timeline

| Step | Time | Output |
|------|------|--------|
| Setup | 2 min | Dependencies installed |
| Prepare Data | 2-3 hrs | `labeled_logs.csv` |
| Train | 30-45 min | `model.pkl` |
| Total | ~3 hrs | Ready to use! |

---

## 🐛 Common Issues & Fixes

### ❌ "ModuleNotFoundError: scikit-learn"
```powershell
pip install scikit-learn numpy
```

### ❌ "windows.log not found"
Create a sample: `prepare_data.py` auto-creates one if missing

### ❌ "Out of memory" error
Edit `config.py`, set `MEMORY_OPTIMIZATION = True`

### ❌ Low accuracy (< 80%)
Review the auto-labeling keywords in `config.py`
- Add more specific keywords to CRITICAL_KEYWORDS
- Adjust WARNING_KEYWORDS for your log format

### ❌ Very slow on batch processing
- Use `--interval 5` for monitoring mode
- Reduce `N_FEATURES` in config.py to 2**16

---

## 📖 Next Steps

1. **Read the README.md** for detailed documentation
2. **Adjust config.py** for your specific log format
3. **Review predictions.log** to check accuracy
4. **Integrate with your monitoring system** using `log_predictor.py`

---

## 💡 Pro Tips

1. **Test with sample first:**
   ```python
   # In config.py, set:
   MAX_LINES_TO_PROCESS = 100000  # Test with 100k lines
   ```

2. **Monitor real-time logs:**
   ```powershell
   # Terminal 1: Run monitor
   python monitor.py --mode monitor
   
   # Terminal 2: Append test logs
   Add-Content windows.log "ERROR: Test critical alert"
   ```

3. **Check prediction confidence:**
   ```python
   from monitor import load_model, predict_log_severity
   model_package = load_model('model.pkl')
   prediction = predict_log_severity("WARNING: Low disk", model_package)
   # Add model.predict_proba() for confidence scores
   ```

---

## 📞 Getting Help

1. Check errors in `test_pipeline.py` output
2. Review logs in the console
3. Ensure `windows.log` format matches expectations
4. Read detailed docs in `README.md`

---

**Ready? Let's go!**

```powershell
pip install -r requirements.txt
python prepare_data.py
python train_model.py
python monitor.py --mode batch
```
