# Windows Log Analysis System - Quick Start Guide# 🚀 QUICK START GUIDE



Get your log analyzer running in under 5 minutes!## ⚡ TL;DR (30 seconds)



---```powershell

# 1. Install dependencies

## ⚡ **60-Second Start** (Using Pre-trained Model)pip install -r requirements.txt



```bash# 2. Prepare data (auto-label logs from windows.log)

# 1. Install dependencies (30 seconds)python prepare_data.py

pip install -r requirements.txt

# 3. Train models

# 2. Scan your logs (5 seconds)python train_model.py

python log_checker.py Windows.log

# 4. Start monitoring

# 3. Check the reportpython monitor.py --mode batch

# Look in reports/ directory for detailed analysis```

```

---

**Done!** Your logs are analyzed.

## 📊 Full Step-by-Step

---

### Step 1️⃣: Setup Environment (2 minutes)

## 🔄 **Training Custom Model** (Optional - 20 minutes)

```powershell

Only needed if you want to train on YOUR specific logs:# Open PowerShell in your project directory



```bash# Install required packages

# Step 1: Prepare training data (5 min)pip install -r requirements.txt

python prepare_data.py

# Verify installation

# Step 2: Train ensemble model (15 min)python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

python train_model_gpu_ensemble.py```



# Step 3: Validate accuracy (30 sec)**Expected:** Should print version number without errors.

python validate_model_quality.py

```---



---### Step 2️⃣: Prepare Data (2-3 hours for 26GB)



## 📊 **What You'll See**```powershell

python prepare_data.py

### Console Output:```

```

🔴 CRITICAL PROBLEMS FOUND: 21**What happens:**

🟡 WARNINGS FOUND: 7- Reads `windows.log` line-by-line

✓ NORMAL: 1,972- Auto-labels each line as CRITICAL, WARNING, or NORMAL

Total: 2,000 lines scanned- Creates `labeled_logs.csv` with ~100M labeled examples

```

**Expected output:**

### Report File (reports/):```

```Starting data preparation from windows.log...

[1] CRITICAL ISSUE:Processed 100,000 lines...

    Message: Failed to start upload...Processed 200,000 lines...

    Line Number: 11...

    File: Windows.log✓ Data preparation complete!

```Total lines processed: 100,000,000

  - CRITICAL: 15,234,567 (15.2%)

---  - WARNING: 28,456,789 (28.5%)

  - NORMAL: 56,308,644 (56.3%)

## 🎯 **Common Commands**Output saved to: labeled_logs.csv

```

```bash

# Scan a log file---

python log_checker.py my_logfile.log

### Step 3️⃣: Train Models (30-45 minutes)

# Validate model quality

python validate_model_quality.py```powershell

python train_model.py

# Retrain on new data```

python train_model_gpu_ensemble.py

```**What happens:**

- Loads labeled data from CSV

---- Trains 4 models: SVM, Random Forest, Logistic Regression, + Ensemble

- Evaluates accuracy on test set

## 🔧 **Quick Customization**- Saves `model.pkl` (1.4GB)



### Change Detection Keywords**Expected output:**

Edit `prepare_data.py`:```

```pythonLoading data from labeled_logs.csv...

CRITICAL_KEYWORDS = ['error', 'failed', 'crash', 'fatal']Loaded 100,000,000 total samples

WARNING_KEYWORDS = ['warning', 'deprecated', 'timeout']

```Splitting data (20% test)...

Training set: 80,000,000 samples

### Adjust Training SizeTest set: 20,000,000 samples

Edit `train_model_gpu_ensemble.py`:

```pythonTraining SVM model...

sample_size = 500000  # Reduce if low on memorySVM Accuracy: 0.8876

max_features = 3000    # Increase for better accuracy

```Training Random Forest model...

Random Forest Accuracy: 0.8923

---

Training Logistic Regression model...

## ❓ **Troubleshooting**Logistic Regression Accuracy: 0.8856



| Problem | Solution |Creating Stacking Ensemble...

|---------|----------|Ensemble Final Accuracy: 0.8945

| ModuleNotFoundError | `pip install -r requirements.txt` |

| Model not found | Run `python train_model_gpu_ensemble.py` first |✓ Model package saved to: model.pkl

| Out of memory | Reduce sample_size to 100000 |  Model size: 1456.78 MB

| Slow processing | Normal for large files - be patient |```



------



## 📚 **Learn More**### Step 4️⃣: Use the Model



- **README.md** - Complete documentation#### Option A: Batch Process All Logs

- **log_checker.py** - Main analysis tool```powershell

- **train_model_gpu_ensemble.py** - Training scriptpython monitor.py --mode batch --log-file windows.log --output predictions.log

```

---

Creates `predictions.log` with severity predictions for every line.

**Ready to analyze logs! 🚀**

#### Option B: Real-Time Monitoring
```powershell
python monitor.py --mode monitor
```

Watches for new lines and alerts on CRITICAL/WARNING.

#### Option C: Use in Python Code
```python
from log_checker import predict_log_severity

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
4. **Integrate with your monitoring system** using `log_checker.py`

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
