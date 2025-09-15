# Training Scripts Overview

## 🧹 **Cleaned Up Training Scripts**

After removing redundant and superseded scripts, here are the **3 essential training approaches**:

---

## 🚀 **1. LATEST: Fixed Training (MOST RECOMMENDED)**
**File:** `train_fixed_double_dqn.py`
**Command:** `python train_fixed_double_dqn.py`

### ✅ **Key Features:**
- **Fixes Q-value learning issues** identified in diagnosis
- **Gradient clipping** (prevents exploding gradients: 215+ → <10 norm)  
- **Enhanced state encoding** (better positional awareness for empty boards)
- **Huber loss** instead of MSE (more stable training)
- **30%+ heuristic preservation** (never drops below guaranteed minimum)
- **Optimized hyperparameters** (reduced learning rate, smaller batches)
- **Comprehensive monitoring** with improved plots in `logs_fixed/`

### 🎯 **Best For:**
- **New training runs** where you want the highest quality Q-values
- **Addressing the Q-value plot issues** you identified
- **Most stable and reliable training**

---

## 🟢 **2. Advanced Training (Heuristic Preservation Focus)**  
**File:** `train/double_dqn_train_advanced.py`
**Command:** `python train/double_dqn_train_advanced.py`

### ✅ **Key Features:**
- **All 7 expert suggestions** you provided implemented
- **30%+ heuristic preservation** (never below minimum)
- **10x learning rate reduction** for self-play fine-tuning
- **Gradual self-play introduction** (30% → 60% over 20K episodes)
- **Historical opponent pool** (7 diverse agents prevent overfitting)
- **Emergency early stopping** (90% threshold with 5-episode patience)
- **L2 regularization** and comprehensive monitoring

### 🎯 **Best For:**
- **Heuristic preservation experiments**
- **Testing curriculum learning approaches**
- **When you want all advanced continual learning techniques**

---

## 🔴 **3. Original Training (Reference Only)**
**File:** `train/double_dqn_train.py`  
**Command:** `python train/double_dqn_train.py`

### ❌ **Issues:**
- **Catastrophic forgetting** (100% → 40% heuristic performance drop)
- **No heuristic preservation** in self-play
- **Buffer clearing** at curriculum transitions
- **Exploding gradients** and training instability

### 🎯 **Keep For:**
- **Historical reference** and comparison
- **Understanding what went wrong** in original approach
- **Baseline for measuring improvement**

---

## 🎮 **Easy Access: Interactive Launcher**
**File:** `run_advanced_training.py`
**Command:** `python run_advanced_training.py`

Provides a user-friendly menu to choose between all training methods with explanations.

---

## 📊 **Recommended Usage**

### **For New Training:**
```bash
# Use the latest fixed version (addresses Q-value issues)
python train_fixed_double_dqn.py
```

### **For Experimentation:**
```bash
# Use the advanced version (focuses on heuristic preservation)
python train/double_dqn_train_advanced.py
```

### **For Convenience:**
```bash
# Use the interactive launcher
python run_advanced_training.py
```

---

## 🗂️ **Removed Scripts**
- ❌ `train/double_dqn_train_improved.py` - Superseded by advanced version
- ❌ `train/double_dqn_train_working.py` - Old developmental version

## 📁 **Output Directories**
- `train_fixed_double_dqn.py` → `models_fixed/` + `logs_fixed/`
- `train_advanced.py` → `models_advanced/` + `logs_advanced/`  
- `train.py` → `models/` + `logs/`

This keeps only the essential scripts while maintaining all the advanced features you need! 🚀