"""
Comprehensive Model Quality Validation
Tests model accuracy, consistency, and real-world performance
"""
import pickle
import numpy as np
from collections import Counter

print("="*80)
print("MODEL QUALITY VALIDATION")
print("="*80)

# Load model
print("\n1. Loading model...")
with open('model_gpu.pkl', 'rb') as f:
    model = pickle.load(f)

vectorizer = model['vectorizer']
ensemble = model['ensemble']
reverse_map = model.get('reverse_map', {0: 'NORMAL', 1: 'WARNING', 2: 'CRITICAL'})

print(f"   ✓ Model type: {model.get('model_type', 'Unknown')}")
print(f"   ✓ Training samples: {model.get('training_samples', 'Unknown')}")
print(f"   ✓ Features: {model.get('features', 'Unknown')}")
print(f"   ✓ Ensemble type: {ensemble.get('ensemble_type', 'Unknown')}")

# Test with known examples
print("\n2. Testing with known examples...")
print("-"*80)

test_cases = [
    # Should be CRITICAL (score 2)
    ("ERROR: System failed to start", 2),
    ("CRITICAL: Database connection lost", 2),
    ("FATAL: Application crash detected", 2),
    ("Exception: OutOfMemoryError", 2),
    ("Failed to connect to server", 2),
    ("ERROR: Authentication failed", 2),
    
    # Should be WARNING (score 1)
    ("WARNING: Disk space low", 1),
    ("WARN: Deprecated API usage", 1),
    ("Warning: Connection timeout", 1),
    ("Deprecated function call", 1),
    
    # Should be NORMAL (score 0)
    ("INFO: System started successfully", 0),
    ("User login successful", 0),
    ("Service is running normally", 0),
    ("Configuration loaded", 0),
]

correct = 0
total = len(test_cases)
incorrect_predictions = []

# Handle different ensemble types
if isinstance(ensemble, dict) and 'models' in ensemble:
    # GPU Ensemble: Weighted voting
    models = ensemble['models']
    weights = ensemble['weights']
    
    def predict(message):
        features = vectorizer.transform([message]).toarray().astype(np.float32)
        votes = np.zeros(3)
        for model, weight in zip(models, weights):
            pred = model.predict(features)[0]
            votes[pred] += weight
        return np.argmax(votes)
        
elif 'svc_model' in ensemble:
    # SVC model
    svc_model = ensemble['svc_model']
    def predict(message):
        features = vectorizer.transform([message])
        return svc_model.predict(features)[0]
else:
    print("❌ Unknown ensemble type!")
    exit(1)

for message, expected_score in test_cases:
    pred = predict(message)
    expected_label = reverse_map[expected_score]
    predicted_label = reverse_map[pred]
    
    is_correct = pred == expected_score
    correct += is_correct
    
    status = "✓" if is_correct else "✗"
    print(f"{status} '{message[:50]:50}' → {predicted_label:8} (expected: {expected_label})")
    
    if not is_correct:
        incorrect_predictions.append((message, expected_label, predicted_label))

accuracy = (correct / total) * 100
print(f"\n   Accuracy on test cases: {correct}/{total} = {accuracy:.1f}%")

if incorrect_predictions:
    print("\n   ❌ Incorrect predictions:")
    for msg, expected, predicted in incorrect_predictions:
        print(f"      '{msg}' → Got {predicted}, Expected {expected}")

# Test on Windows_2k.log
print("\n3. Testing on Windows_2k.log (2,000 lines)...")
print("-"*80)

predictions = []
with open('Windows_2k.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if not line.strip():
            continue
        pred = predict(line)
        predictions.append(pred)

prediction_counts = Counter(predictions)
print(f"   NORMAL (0):   {prediction_counts.get(0, 0):4} ({prediction_counts.get(0, 0)/len(predictions)*100:5.1f}%)")
print(f"   WARNING (1):  {prediction_counts.get(1, 0):4} ({prediction_counts.get(1, 0)/len(predictions)*100:5.1f}%)")
print(f"   CRITICAL (2): {prediction_counts.get(2, 0):4} ({prediction_counts.get(2, 0)/len(predictions)*100:5.1f}%)")
print(f"   Total:        {len(predictions):4}")

# Check for reasonable distribution
critical_pct = prediction_counts.get(2, 0) / len(predictions) * 100
warning_pct = prediction_counts.get(1, 0) / len(predictions) * 100
normal_pct = prediction_counts.get(0, 0) / len(predictions) * 100

print(f"\n4. Distribution Analysis...")
print("-"*80)

if normal_pct < 50:
    print("   ⚠ WARNING: Less than 50% NORMAL logs - model might be too sensitive")
elif normal_pct > 99:
    print("   ⚠ WARNING: More than 99% NORMAL logs - model might be too lenient")
else:
    print("   ✓ Distribution looks reasonable")

if critical_pct < 0.1:
    print("   ⚠ WARNING: Very few CRITICAL logs detected")
elif critical_pct > 50:
    print("   ⚠ WARNING: Too many CRITICAL logs - model might be too aggressive")
else:
    print("   ✓ Critical detection rate looks good")

# Sample some actual predictions
print("\n5. Sample predictions from Windows_2k.log...")
print("-"*80)

sample_critical = []
sample_warning = []
sample_normal = []

with open('Windows_2k.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        pred = predict(line)
        
        if pred == 2 and len(sample_critical) < 3:
            sample_critical.append((line_num, line.strip()))
        elif pred == 1 and len(sample_warning) < 3:
            sample_warning.append((line_num, line.strip()))
        elif pred == 0 and len(sample_normal) < 3:
            sample_normal.append((line_num, line.strip()))

if sample_critical:
    print("\nSample CRITICAL predictions:")
    for line_num, line in sample_critical:
        print(f"  Line {line_num}: {line[:100]}")

if sample_warning:
    print("\nSample WARNING predictions:")
    for line_num, line in sample_warning:
        print(f"  Line {line_num}: {line[:100]}")

if sample_normal:
    print("\nSample NORMAL predictions:")
    for line_num, line in sample_normal:
        print(f"  Line {line_num}: {line[:100]}")

# Overall assessment
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

score = 0
max_score = 4

if accuracy >= 80:
    score += 1
    print("✓ Test case accuracy: GOOD")
else:
    print("✗ Test case accuracy: POOR")

if 50 <= normal_pct <= 99:
    score += 1
    print("✓ Distribution: GOOD")
else:
    print("✗ Distribution: PROBLEMATIC")

if 0.1 <= critical_pct <= 50:
    score += 1
    print("✓ Critical detection: GOOD")
else:
    print("✗ Critical detection: NEEDS TUNING")

if len(predictions) == 2000:
    score += 1
    print("✓ All lines processed: GOOD")
else:
    print("✗ Some lines not processed")

print("\n" + "="*80)
if score == max_score:
    print("🎉 MODEL QUALITY: EXCELLENT")
    print("The model is working correctly and making sensible predictions.")
elif score >= 3:
    print("✓ MODEL QUALITY: GOOD")
    print("The model is generally working well with minor issues.")
elif score >= 2:
    print("⚠ MODEL QUALITY: ACCEPTABLE")
    print("The model works but could use retraining with better data.")
else:
    print("❌ MODEL QUALITY: POOR")
    print("The model needs retraining. Data quality or hyperparameters need adjustment.")

print(f"Score: {score}/{max_score}")
print("="*80)
