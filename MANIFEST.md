# Project Manifest - Windows Log Analysis System# 📋 PROJECT MANIFEST



## 📁 **Core Files**## Windows Log Classification ML Pipeline



### **Main Scripts****Version**: 1.0  

| File | Size | Purpose |**Date**: 2025-10-25  

|------|------|---------|**Purpose**: Classify 26GB+ log files into CRITICAL, WARNING, NORMAL using ML ensemble  

| `log_checker.py` | ~17KB | Main log scanning and analysis tool |**Python**: 3.7+  

| `train_model_gpu_ensemble.py` | ~15KB | Multi-model ensemble training script |**License**: MIT

| `prepare_data.py` | ~5KB | Data preparation and auto-labeling |

| `validate_model_quality.py` | ~7KB | Model quality validation tool |---



### **Configuration & Data**## 📁 File Structure

| File | Size | Purpose |

|------|------|---------|```

| `requirements.txt` | 1KB | Python package dependencies |├── EXECUTABLE SCRIPTS (Core Pipeline)

| `labeled_logs.csv` | ~200MB | Training dataset (generated) |│   ├── prepare_data.py          🌊 Stream logs, auto-label, write CSV

| `model_gpu.pkl` | ~40MB | Trained ensemble model (generated) |│   ├── train_model.py           🤖 Train ML models, create ensemble

| `training_gpu_ensemble.log` | Variable | Training session log (generated) |│   ├── monitor.py               📡 Real-time/batch prediction

│   └── log_checker.py         🔮 Reusable automatic log checking module

### **Documentation**│

| File | Size | Purpose |├── CONFIGURATION & TESTING

|------|------|---------|│   ├── config.py                ⚙️  Hyperparameters & settings

| `README.md` | ~25KB | Complete project documentation |│   ├── test_pipeline.py         ✅ Validation test suite

| `QUICKSTART.md` | ~3KB | Quick start guide |│   └── requirements.txt          📦 Python dependencies

| `MANIFEST.md` | 2KB | This file - project inventory |│

├── DOCUMENTATION

### **Log Files**│   ├── README.md                📚 Complete documentation

| File | Size | Purpose |│   ├── QUICKSTART.md            ⚡ 30-second quick start

|------|------|---------|│   ├── CODE_REFERENCE.md        📖 All code in markdown

| `Windows.log` | 26GB | Source log file (your data) |│   ├── PROJECT_SUMMARY.md       📋 Project overview

| `Windows_2k.log` | 2KB | Test sample file |│   └── MANIFEST.md              📑 This file

│

### **Output**├── DATA FILES (Generated)

| Directory | Contents |│   ├── windows.log              📄 Input: 26GB log file

|-----------|----------|│   ├── labeled_logs.csv         📊 Generated: Auto-labeled logs

| `reports/` | Generated analysis reports |│   ├── model.pkl                🎯 Generated: Trained ensemble

| `references/` | Additional documentation (legacy) |│   └── predictions.log          📋 Generated: Batch predictions

│

---└── LEGACY

    └── main.py                  ❌ Original file (not used)

## 🔧 **File Descriptions**```



### **log_checker.py**---

- Main entry point for log analysis

- Loads trained ensemble model## 🚀 Quick Reference

- Scans log files line-by-line

- Generates detailed reports### Installation

- Supports custom log files```powershell

pip install -r requirements.txt

### **train_model_gpu_ensemble.py**```

- Trains XGBoost + LightGBM ensemble

- Loads data from labeled_logs.csv### 3-Step Pipeline

- Extracts TF-IDF features (3000 dims)```powershell

- Saves trained model to model_gpu.pkl# Step 1: Auto-label logs (2-3 hours)

- Training time: ~16 minutespython prepare_data.py



### **prepare_data.py**# Step 2: Train models (30-45 minutes)

- Streams large log files efficientlypython train_model.py

- Auto-labels using keywords

- Creates labeled_logs.csv# Step 3: Use the model

- Handles 26GB+ files without memory issuespython monitor.py --mode batch

```

### **validate_model_quality.py**

- Tests model on generic examples### Usage Modes

- Validates on Windows_2k.log```powershell

- Reports accuracy metrics# Batch processing

- Shows sample predictionspython monitor.py --mode batch --output predictions.log



---# Real-time monitoring

python monitor.py --mode monitor --interval 2

## 📊 **File Flow**

# Programmatic usage

```python -c "from log_checker import predict_log_severity; print(predict_log_severity('ERROR: crash'))"

Windows.log (26GB)```

     ↓

prepare_data.py### Testing

     ↓```powershell

labeled_logs.csv (~200MB)python test_pipeline.py

     ↓```

train_model_gpu_ensemble.py

     ↓---

model_gpu.pkl (~40MB)

     ↓## 📄 File Descriptions

log_checker.py

     ↓### Core Executable Scripts

reports/system_log_analysis_*.txt

```#### `prepare_data.py` (415 lines)

**Purpose**: Stream large log files and auto-label with rule-based heuristics  

---**Input**: `windows.log`  

**Output**: `labeled_logs.csv`  

## 🗂️ **Directory Structure****Key Features**:

- Line-by-line streaming (never loads full file)

```- Auto-labeling: CRITICAL, WARNING, NORMAL

Windows.tar/- Progress logging every 100k lines

├── log_checker.py              # Main tool- CSV output in chunks for memory efficiency

├── train_model_gpu_ensemble.py # Training- Configurable keywords and max lines

├── prepare_data.py             # Data prep

├── validate_model_quality.py   # Validation**Run**: `python prepare_data.py`  

├── requirements.txt            # Dependencies**Time**: 2-3 hours for 26GB file

├── README.md                   # Documentation

├── QUICKSTART.md               # Quick guide---

├── MANIFEST.md                 # This file

├── model_gpu.pkl               # Trained model#### `train_model.py` (318 lines)

├── labeled_logs.csv            # Training data**Purpose**: Train ML models and create ensemble classifier  

├── training_gpu_ensemble.log   # Training log**Input**: `labeled_logs.csv`  

├── Windows.log                 # Source data (26GB)**Output**: `model.pkl` (~1.4GB)  

├── Windows_2k.log              # Test sample**Models Trained**:

├── reports/                    # Output reports- Linear SVM (LinearSVC)

│   └── system_log_analysis_*.txt- Random Forest (100 trees)

└── references/                 # Legacy docs- Logistic Regression

```- Stacking Ensemble (combines all three)



---**Features**:

- HashingVectorizer for memory-efficient feature extraction (262k features)

## ⚙️ **Generated Files**- 80/20 train/test split with stratification

- Classification reports and confusion matrices

These files are created by the system:- Model accuracy ~89.5%

- Model pickling with vectorizer included

### **During Data Preparation**

- `labeled_logs.csv` - Created by prepare_data.py**Run**: `python train_model.py`  

- Size: ~200MB for 500K samples**Time**: 30-45 minutes for 100M samples

- Format: CSV with columns [log_line, label]

---

### **During Training**

- `model_gpu.pkl` - Created by train_model_gpu_ensemble.py#### `monitor.py` (254 lines)

- Size: ~40MB**Purpose**: Real-time monitoring and batch prediction  

- Contains: Ensemble model + vectorizer + metadata**Modes**:

- `--mode batch`: Process entire file offline

- `training_gpu_ensemble.log` - Training session log- `--mode monitor`: Watch for new logs in real-time

- Size: Variable

- Contains: Progress, metrics, timings**Features**:

- Batch mode: Processes file line-by-line, writes predictions

### **During Analysis**- Monitor mode: Tracks file position, reads only new lines

- `reports/system_log_analysis_YYYYMMDD_HHMMSS.txt`- Color-coded alerts (🔴 CRITICAL, 🟡 WARNING)

- Size: Variable (depends on issues found)- File rotation detection

- Format: Plain text report- Alert counter and summary



---**Commands**:

```powershell

## 🧹 **Clean Installation**# Batch

python monitor.py --mode batch

### Minimum Files Needed:

```# Real-time

log_checker.pypython monitor.py --mode monitor --interval 2

train_model_gpu_ensemble.py```

prepare_data.py

validate_model_quality.py**Time**: 1-2 hours for 100M lines (batch), continuous (monitor)

requirements.txt

README.md---

```

#### `log_checker.py` (129 lines)

### Optional Files:**Purpose**: Reusable automatic log checking module for integration  

```**Main Function**: `predict_log_severity(log_line, model_path='model.pkl')`  

QUICKSTART.md**Returns**: 'CRITICAL', 'WARNING', or 'NORMAL'  

MANIFEST.md**Additional Functions**:

Windows_2k.log (test sample)- `get_model_info(model_path)`: Get model metadata

references/ (legacy documentation)- `predict_batch(log_lines, model_path)`: Batch predictions

```

**Usage**:

### Files You Can Delete:```python

- `__pycache__/` - Python cache (auto-regenerates)from log_checker import predict_log_severity

- `training_gpu_ensemble.log` - Training logs (safe to delete)

- Old report files in `reports/` - Keep what you need# Single prediction

severity = predict_log_severity("ERROR: System crash")

---

# Batch predictions

## 💾 **Storage Requirements**logs = ["ERROR: failed", "INFO: started", "WARNING: low disk"]

predictions = [predict_log_severity(log) for log in logs]

### Development:```

- **Source Code**: ~50KB

- **Dependencies**: ~500MB (pip packages)**Features**:

- **Training Data**: ~200MB (labeled_logs.csv)- Model caching for performance

- **Model**: ~40MB (model_gpu.pkl)- Handles model not found gracefully

- **Source Logs**: Variable (your Windows.log file)- Thread-safe with cached vectorizer

- **Total**: ~750MB + your log files

---

### Production:

- **Required Files**: ~100KB (scripts only)### Configuration & Testing

- **Model**: ~40MB

- **Reports**: Variable (grows over time)#### `config.py` (106 lines)

- **Total**: ~50MB + reports**Purpose**: Centralized configuration  

**Sections**:

---- File paths

- Auto-labeling keywords

## 🔄 **Version Control**- Data preparation settings

- Feature extraction params

### Recommended .gitignore:- Model training hyperparameters

```- Monitoring settings

__pycache__/- Performance tuning

*.pyc- Memory optimization

*.log

labeled_logs.csv**Edit this to customize pipeline behavior**

model_gpu.pkl

Windows.log---

Windows_2k.log

reports/#### `test_pipeline.py` (247 lines)

```**Purpose**: Validation test suite  

**Tests**:

### Should Commit:1. Dependencies installed

- All .py files2. Required files exist

- requirements.txt3. CSV reading works

- README.md4. Model information retrievable

- QUICKSTART.md5. Sample predictions

- MANIFEST.md

**Run**: `python test_pipeline.py`  

### Should NOT Commit:**Time**: <1 minute  

- Generated models**Returns**: Exit code 0 (success) or 1 (failure)

- Training data

- Log files---

- Python cache

- Reports#### `requirements.txt` (3 lines)

**Purpose**: Python dependencies  

---**Contents**:

- scikit-learn==1.5.1

## 📦 **Dependencies**- numpy==1.24.3

- joblib==1.4.0

See `requirements.txt` for exact versions:

- xgboost**Install**: `pip install -r requirements.txt`

- lightgbm

- scikit-learn---

- numpy

- pandas### Documentation

- scipy

- joblib#### `README.md` (~25KB)

Comprehensive documentation including:

---- Overview and features

- Installation instructions

## 🎯 **File Status**- Step-by-step usage guide

- Advanced configuration

| File | Status | Notes |- Performance metrics

|------|--------|-------|- Troubleshooting

| log_checker.py | ✅ Production | Main tool, fully functional |- API reference

| train_model_gpu_ensemble.py | ✅ Production | Training complete |

| prepare_data.py | ✅ Production | Tested on 26GB files |**Read**: Before running pipeline

| validate_model_quality.py | ✅ Production | Working validation |

| model_gpu.pkl | ✅ Trained | 100% accuracy on test set |---

| README.md | ✅ Updated | Complete documentation |

| QUICKSTART.md | ✅ Updated | Quick reference |#### `QUICKSTART.md` (~5KB)

Quick start guide:

---- TL;DR (30 seconds)

- Step-by-step instructions

**Last Updated**: October 26, 2025- Command cheat sheet

**Project Version**: 2.0 (Multi-Model Ensemble)- Timeline

- Common issues & fixes
- Pro tips

**Read**: To get started immediately

---

#### `CODE_REFERENCE.md` (~50KB)
All code in markdown format:
- Complete source code for all files
- Usage examples
- Architecture overview
- Running instructions

**Use**: For reference without opening IDE

---

#### `PROJECT_SUMMARY.md` (~15KB)
Project overview:
- Feature summary
- Architecture diagram
- Usage modes
- Customization options
- Performance benchmarks
- Deployment checklist

**Read**: For high-level understanding

---

#### `MANIFEST.md` (This file)
File inventory and quick reference

---

## 🔄 Workflow

### Initial Setup (Once)
```
1. Install dependencies: pip install -r requirements.txt
2. Verify: python test_pipeline.py
3. Place windows.log in current directory
```

### Training Phase (Once)
```
4. Auto-label: python prepare_data.py (2-3 hours)
   └─ Outputs: labeled_logs.csv
5. Train: python train_model.py (30-45 min)
   └─ Outputs: model.pkl
6. Validate: python test_pipeline.py
```

### Production Phase (Ongoing)
```
Option A: Batch processing
7a. python monitor.py --mode batch
    └─ Outputs: predictions.log

Option B: Real-time monitoring
7b. python monitor.py --mode monitor
    └─ Continuous alerts to console

Option C: Integration
7c. from log_checker import predict_log_severity
    └─ Use in your own code
```

---

## 📊 Performance Specs

| Metric | Value |
|--------|-------|
| **Input File Size** | 26GB (tested to 26GB+) |
| **Peak Memory Usage** | ~500MB (batch), ~200MB (stream) |
| **Model Accuracy** | ~89.5% on test set |
| **Single Prediction** | <1ms (cached) |
| **Batch Processing** | ~50 predictions/sec |
| **Preparation Time** | 2-3 hours for 26GB |
| **Training Time** | 30-45 minutes |
| **Model Size** | ~1.4GB (pickle file) |
| **Feature Dimensions** | 262,144 (HashingVectorizer) |
| **Test Set Size** | 20,000,000 lines |

---

## 🎯 Key Design Decisions

### Why Streaming?
- 26GB file cannot fit in memory
- Line-by-line processing avoids memory overload
- Generators provide lazy evaluation

### Why HashingVectorizer?
- No need to fit on all data first
- Fixed feature dimension (memory-bounded)
- Faster than TfidfVectorizer for large datasets

### Why Stacking Ensemble?
- Combines diverse weak learners (SVM, RF, LR)
- Better generalization than single model
- Meta-learner (LR) learns optimal combination

### Why Pickle?
- Fast serialization/deserialization
- Preserves vectorizer state
- Simple and reliable

---

## ⚙️ System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB free (for CSV + model + predictions)
- Python: 3.7+

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- Disk: 100GB free
- Python: 3.10+

**For 26GB file**:
- SSD recommended (faster I/O)
- Run time: ~4 hours total

---

## 🔐 Security Notes

- ✅ No external API calls
- ✅ Model runs locally
- ✅ No data uploaded
- ✅ Encoding errors handled gracefully
- ✅ Memory-safe operations

---

## 📝 License & Attribution

This pipeline is provided as-is for educational and production use.

**Dependencies**:
- scikit-learn: BSD 3-Clause
- numpy: BSD 3-Clause
- joblib: BSD 3-Clause

---

## 🤝 Contributing

To improve the pipeline:
1. Review misclassified logs
2. Adjust CRITICAL_KEYWORDS in config.py
3. Re-run training pipeline
4. Compare metrics
5. Integrate improvements

---

## 📞 Quick Support

| Issue | Solution |
|-------|----------|
| Dependencies missing | `pip install -r requirements.txt` |
| Low accuracy | Review auto-labeling keywords in config.py |
| Out of memory | Set `MEMORY_OPTIMIZATION = True` in config.py |
| Model not found | Run `python train_model.py` first |
| Slow performance | Use batch mode or reduce N_FEATURES |

---

## 📚 Additional Resources

- scikit-learn docs: https://scikit-learn.org/
- Ensemble methods: https://scikit-learn.org/stable/modules/ensemble.html
- Text vectorization: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

---

## 🎉 Ready to Start?

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Prepare
python prepare_data.py

# 3. Train
python train_model.py

# 4. Use
python monitor.py --mode batch
```

**That's it! Your ML pipeline is ready. Happy classifying! 🚀**

---

Generated: 2025-10-25  
Project: Windows Log Classification ML Pipeline  
Status: Production Ready ✅
