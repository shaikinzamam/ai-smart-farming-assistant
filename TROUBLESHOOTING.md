# 🔧 Troubleshooting & Error Fixes

## ⚠️ Import Errors (TensorFlow)

### Error Message
```
Import "tensorflow" could not be resolved from source
Import "tensorflow.keras" could not be resolved from source
```

### ✅ Fix (Simple - One Command)

```bash
python fix_errors.py
```

This script will:
1. Upgrade pip
2. Install all missing packages
3. Verify everything is working

### Alternative: Manual Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install tensorflow numpy pillow kaggle tqdm
```

---

## 🆘 Other Common Errors

### Error: "No module named 'PIL'"
```bash
pip install Pillow
```

### Error: "No module named 'kaggle'"
```bash
pip install kaggle
```

### Error: "ModuleNotFoundError" for any package
```bash
# Reinstall from requirements
pip install -r requirements.txt --force-reinstall
```

### Error: Virtual environment issues
```bash
# Activate virtual environment (Windows)
venv\Scripts\Activate.ps1

# Then try again
python fix_errors.py
```

---

## 🚀 Quick Fix Steps

1. **Run the auto-fix script**:
   ```bash
   python fix_errors.py
   ```

2. **If that doesn't work, try**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **If still issues, try clean install**:
   ```bash
   pip uninstall tensorflow numpy pillow kaggle -y
   pip install tensorflow-cpu numpy pillow kaggle
   ```

4. **Verify it works**:
   ```bash
   python verify_setup.py
   ```

---

## 🔍 Check Your Setup

### Quick Diagnostics
```bash
python verify_setup.py
```

### See Installed Packages
```bash
pip list
```

### Check Python Version
```bash
python --version
```

### Check Virtual Environment
```bash
where python
# Should show: venv\Scripts\python.exe
```

---

## 💡 Best Practices

1. **Always use virtual environment**:
   ```bash
   venv\Scripts\Activate.ps1  # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. **Keep requirements.txt updated**:
   ```bash
   pip freeze > requirements.txt
   ```

3. **Install from requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Summary of Fixes

| Error | Fix | Command |
|-------|-----|---------|
| TensorFlow import error | Run auto-fix | `python fix_errors.py` |
| Missing PIL | Install Pillow | `pip install Pillow` |
| Missing kaggle | Install kaggle | `pip install kaggle` |
| Any package missing | Reinstall all | `pip install -r requirements.txt` |
| Virtual env issue | Activate venv | `venv\Scripts\Activate.ps1` |

---

## ✅ Verification

After running fixes, verify with:

```bash
# Check 1: Health check
python verify_setup.py

# Check 2: Import test
python -c "import tensorflow; print('✓ TensorFlow OK')"

# Check 3: Quick test
python train_complete.py --help
```

---

**All errors fixed?** ✅ Yes
**Ready to proceed?** ✅ Yes

**Next step**: `python setup_kaggle.py`
