# Social Media Reputation Monitoring (SMRM) - Setup Guide

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your computer:

1. **Python 3.8 or higher**
   - Check version: `python3 --version`
   - Download from: https://www.python.org/downloads/

2. **pip (Python package manager)**
   - Usually comes with Python
   - Check version: `pip3 --version`

3. **Git** (optional, for cloning)
   - Check version: `git --version`

---

## 🚀 Installation Steps

### Step 1: Copy/Download the Project

**Option A: If you have the folder**
- Copy the entire `Sentiment_Analysis_RP-main` folder to your computer

**Option B: If using Git**
```bash
git clone <repository-url>
cd Sentiment_Analysis_RP-main
```

---

### Step 2: Navigate to Project Directory

Open Terminal (Mac/Linux) or Command Prompt (Windows) and navigate to the project folder:

```bash
cd /path/to/Sentiment_Analysis_RP-main
```

---

### Step 3: Create Virtual Environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

---

### Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- nltk (natural language processing)
- And other dependencies

**Expected installation time:** 2-5 minutes

---

### Step 5: Verify Models Exist

Check if the `models` folder contains the trained models:

```bash
ls -l models/
```

You should see:
- `vectoriser.pickle`
- `model_lr.pickle`
- `model_bnb.pickle`
- `model_svc.pickle`

✅ **If these files exist, skip to Step 7**

❌ **If models are missing, proceed to Step 6**

---

### Step 6: Train Models (Only if models are missing)

If you don't have the pre-trained models, you need to:

1. **Download the Sentiment140 dataset:**
   - URL: https://www.kaggle.com/datasets/kazanova/sentiment140
   - File: `training.1600000.processed.noemoticon.csv`
   - Place it in the project root directory

2. **Run the training script:**
   ```bash
   python train_model.py
   ```
   
   ⏱️ **This may take 10-30 minutes** depending on your computer's speed.

---

### Step 7: Start the Application

**For macOS/Linux:**
```bash
PORT=5001 python app.py
```

**For Windows:**
```cmd
set PORT=5001
python app.py
```

**Why port 5001?** 
- macOS uses port 5000 for AirPlay Receiver
- Using 5001 avoids conflicts

---

### Step 8: Access the Web Interface

Once you see:
```
✓ All models loaded successfully!
* Running on http://127.0.0.1:5001
```

Open your web browser and go to:
```
http://127.0.0.1:5001
```

or

```
http://localhost:5001
```

---

## 🎯 Usage

1. **Enter text** in the input box (tweets, social media posts, reviews, etc.)
2. **Select a model:**
   - LR (Logistic Regression) - Best accuracy
   - BNB (Bernoulli Naive Bayes) - Fastest
   - SVC (Support Vector Classifier) - Balanced
3. **Click "Analyze Sentiment"**
4. **View results** showing Positive/Negative sentiment with confidence score

---

## 🛠️ Troubleshooting

### Problem 1: Port Already in Use
**Error:** `Address already in use`

**Solution:**
```bash
# Try a different port
PORT=5002 python app.py
```

Or disable AirPlay Receiver on Mac:
- System Preferences → General → AirDrop & Handoff → Uncheck AirPlay Receiver

---

### Problem 2: Models Not Loading
**Error:** `Models not loaded. Please train the models first.`

**Solution:**
- Verify models folder exists: `ls models/`
- If missing, run: `python train_model.py`
- If dataset missing, download from Kaggle (see Step 6)

---

### Problem 3: Module Not Found
**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

---

### Problem 4: Python Version Issues
**Error:** Compatibility issues with pandas or scikit-learn

**Solution:**
```bash
# Check Python version (needs 3.8+)
python3 --version

# If using Python 3.14+, you may need:
pip install --upgrade pip
pip install pandas scikit-learn --upgrade
```

---

### Problem 5: Virtual Environment Issues (macOS)
**Error:** `externally-managed-environment`

**Solution:**
Always use virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Project Structure

```
Sentiment_Analysis_RP-main/
├── app.py                  # Main Flask application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── models/                 # Trained ML models
│   ├── vectoriser.pickle
│   ├── model_lr.pickle
│   ├── model_bnb.pickle
│   └── model_svc.pickle
├── templates/              # HTML templates
│   └── index.html         # Web interface
└── venv/                  # Virtual environment (created in Step 3)
```

---

## 🔄 Stopping the Application

To stop the Flask server:
- Press `CTRL + C` in the terminal

To deactivate virtual environment:
```bash
deactivate
```

---

## 💡 Quick Start Commands (Summary)

```bash
# 1. Navigate to project
cd /path/to/Sentiment_Analysis_RP-main

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
PORT=5001 python app.py

# 6. Open browser
# Go to http://127.0.0.1:5001
```

---

## 📊 Model Performance

The application uses three machine learning models:

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| Logistic Regression (LR) | ~78% | Medium | General use |
| Bernoulli Naive Bayes (BNB) | ~76% | Fast | Quick analysis |
| Linear SVC | ~77% | Medium | Balanced results |

---

## 🌐 API Endpoints

The application provides REST API endpoints:

### Single Prediction
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "model": "lr"}'
```

### Batch Prediction
```bash
curl -X POST http://127.0.0.1:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!"], "model": "lr"}'
```

---

## 📝 System Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** ~500MB for project and dependencies
- **Internet:** Required for initial package installation

---

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check that all prerequisites are installed
2. Verify virtual environment is activated
3. Ensure all model files exist in `models/` folder
4. Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`
5. Check Python version compatibility

---

## ✅ Verification Checklist

Before running the application, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages from requirements.txt installed
- [ ] Models folder contains 4 .pickle files
- [ ] Port 5001 (or chosen port) is available
- [ ] Terminal shows "Models loaded successfully"

---

## 🎉 Success!

If you see the SMRM web interface with the heading "📱 Social Media Reputation Monitoring", you're all set!

You can now analyze sentiment in social media posts, tweets, reviews, and any text content.

---

**Last Updated:** November 2025
**Version:** 1.0
