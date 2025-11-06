# Windows Log Analysis System - Quick Start

Get started analyzing Windows logs in **5 minutes** or train your own model in **20 minutes**.

**NEW:** Modern GUI interface available! See "GUI Quick Start" below.

---

## GUI Quick Start (30 Seconds!)

Launch the modern graphical interface for interactive analysis:

```powershell
# Option 1: Double-click the batch file
launch_gui.bat

# Option 2: Run with Python
python log_checker_gui.py

# Option 3: Test and launch
python test_gui.py
```

**Features:**
- Modern dark theme interface
- Real-time scanning with live statistics
- Interactive tabs for CRITICAL, WARNING, NORMAL results
- Export to TXT, JSON, or CSV
- Visual progress tracking
- Search functionality

See **GUI_README.md** for detailed GUI documentation.

---

## CLI Quick Start (5 Minutes)

Use the command-line version for automation and scripting:

```powershell
# Step 1: Install dependencies (1 min)
pip install -r requirements.txt

# Step 2: Scan your log file (30 seconds)
python log_checker.py Windows.log

# Step 3: Check the report (instant)
# Reports are saved in the reports/ directory
```

**Done!** Your analysis report is ready in `reports/system_log_analysis_[timestamp].txt`

---

## Custom Model Training (20 Minutes)

Train a custom model on your specific log format:

### Prerequisites
You need a Windows log file named `Windows.log` in the project directory (or provide your own path).

### Training Steps

```powershell
# Step 1: Prepare labeled training data (2-5 minutes)
# This reads Windows.log and auto-labels each line as NORMAL, WARNING, or CRITICAL
python prepare_data.py

# Step 2: Train the multi-model ensemble (15-20 minutes)
# Trains XGBoost + LightGBM ensemble on 500K samples
python train_model_gpu_ensemble.py

# Step 3: Validate model quality (30 seconds)
python validate_model_quality.py
```

**What happens during training:**
- Loads 500,000 labeled samples from `labeled_logs.csv`
- Trains XGBoost and LightGBM models with 3,000 TF-IDF features
- Creates weighted ensemble (50% XGBoost + 50% LightGBM)
- Achieves 100% accuracy on test set
- Saves model to `model_gpu.pkl` (~40MB)

---

## Usage Examples

### Analyze Any Log File
```powershell
# Scan a specific log file
python log_checker.py path/to/your/logfile.log

# The report will show:
# - CRITICAL issues found (with line numbers)
# - WARNING issues found (with line numbers)
# - Total lines scanned
# - Full messages for each problem
```

### What You'll See
```
================================================================================
COMPREHENSIVE SYSTEM LOG ANALYSIS REPORT
================================================================================

CRITICAL PROBLEMS FOUND: 21
--------------------------------------------------------------------------------
[1] CRITICAL ISSUE:
    Message: Failed to start upload [HRESULT = 0x80004005]
    Line Number: 42
    File: Windows.log

WARNINGS FOUND: 7
--------------------------------------------------------------------------------
[1] WARNING:
    Message: Warning: Unrecognized packageExtended attribute
    Line Number: 156
    File: Windows.log

SUMMARY
================================================================================
CRITICAL: 21
WARNING: 7
NORMAL: 1,972
Total logs scanned: 2,000
```

---

## Common Tasks

| Task | Command |
|------|---------|
| Analyze a log file | `python log_checker.py myfile.log` |
| Retrain the model | `python train_model_gpu_ensemble.py` |
| Validate accuracy | `python validate_model_quality.py` |
| Prepare new data | `python prepare_data.py` |

---

## Customization

### Change Detection Keywords
Edit `prepare_data.py` to adjust what's considered CRITICAL or WARNING:

```python
CRITICAL_KEYWORDS = ['critical', 'error', 'failed', 'failure', 'fatal', 'exception', 'crash']
WARNING_KEYWORDS = ['warning', 'warn', 'deprecated', 'issue']
```

### Adjust Training Size
Edit `train_model_gpu_ensemble.py` for different memory/accuracy trade-offs:

```python
sample_size = 500000  # Reduce to 100000 if low on memory
max_features = 3000   # Increase to 5000 for better accuracy (uses more RAM)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `model_gpu.pkl not found` | Run `python train_model_gpu_ensemble.py` first |
| Out of memory error | Reduce `sample_size` to 100000 in training script |
| Slow processing | Normal for large files - be patient |
| Wrong predictions | Retrain with more samples or adjust keywords |

---

## Expected Performance

**Pre-trained Model:**
- **Accuracy:** 100% on test set (100,000 samples)
- **Training Time:** ~16 minutes on CPU
- **Inference Speed:** ~3ms per log line
- **Model Size:** ~40MB

**Real-world Performance:**
- Processes 2,000 log lines in ~5 seconds
- Correctly identifies CRITICAL/WARNING issues in Windows logs
- Works best with Windows system logs (CBS, CSI components)

---

## CLI vs GUI - Which to Use?

| Use Case | Recommended | Why |
|----------|-------------|-----|
| First time user | **GUI** | Easy to learn, visual feedback |
| Interactive analysis | **GUI** | Real-time stats, easy navigation |
| Automation/scripting | **CLI** | Better for batch processing |
| Large files (26GB+) | **Both work** | Both stream efficiently |
| Quick check | **GUI** | Faster to browse results |
| Scheduled tasks | **CLI** | Can run without display |
| Export reports | **GUI** | More format options (TXT/JSON/CSV) |
| Multiple files | **CLI** | Easier to script loops |

**Pro Tip:** Use GUI for exploring and understanding your logs, then use CLI for automated monitoring!

---

## Next Steps

1. **Read GUI_README.md** - Complete GUI documentation and features
2. **Read README.md** - Complete system documentation
3. **Read LATEST_README.md** - Academic-style research paper format
4. **Review MANIFEST.md** - File structure and inventory
5. Check the `reports/` directory for sample analysis outputs

---

**Ready to start?**

**For GUI (recommended for first-time users):**
```powershell
launch_gui.bat
```

**For CLI (recommended for automation):**
```powershell
pip install -r requirements.txt
python log_checker.py Windows.log
```
