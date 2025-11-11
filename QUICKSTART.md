# 🚀 Quick Start Guide

Follow these simple steps to get the Twitter Sentiment Analysis project running on your local machine.

## ⚡ Quick Setup (5 Steps)

### Step 1: Install Python Packages
```bash
pip install -r requirements.txt
```

### Step 2: Get the Dataset
- Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
- Extract `training.1600000.processed.noemoticon.csv`
- Place it in this folder (same location as app.py)

### Step 3: Train the Models (10-30 minutes)
```bash
python train_model.py
```
Wait for it to complete. You'll see "Training completed successfully!" when done.

### Step 4: Start the Web Server
```bash
python app.py
```

### Step 5: Open Your Browser
Go to: **http://127.0.0.1:5000**

That's it! 🎉

## 📝 What to Expect

### During Training
You'll see progress messages like:
```
1. Loading dataset...
2. Preprocessing text data...
3. Splitting data...
4. Vectorizing text...
5. Training Logistic Regression model...
   Accuracy: 0.7892
...
8. Saving models...
   ✓ All models saved
```

### Web Interface
The interface allows you to:
- Type or paste any text
- Choose from 3 different ML models
- Get instant sentiment predictions
- See confidence scores
- Try example tweets

## 🎯 Try These Example Tweets

**Positive:**
- "I absolutely love this new phone! Best purchase ever! 😊"
- "Amazing customer service, thank you so much!"

**Negative:**
- "This is the worst product I've ever bought. Total waste of money."
- "Very disappointed with the quality. Would not recommend."

## ⚠️ Common Issues

**Problem:** "Dataset not found"
**Solution:** Make sure the CSV file is in the correct location

**Problem:** "Models not loaded"
**Solution:** Run `python train_model.py` first

**Problem:** "Port already in use"
**Solution:** Close other applications or change the port in app.py

## 💡 Tips

1. **First time?** Use the sample examples to test quickly
2. **Want faster training?** The script uses 20% of data by default
3. **Testing models?** Try the same text with different models
4. **Need help?** Check README.md for detailed documentation

## 📱 Screenshots Preview

### Home Page
- Clean, modern interface
- Purple gradient design
- Easy-to-use text input

### Results
- Clear positive/negative indication
- Confidence percentage
- Color-coded display (green/red)

## 🎓 Next Steps

After running the basic setup:
1. Try different types of text (formal, informal, with emojis)
2. Compare predictions across different models
3. Explore the preprocessing functions in the code
4. Modify parameters in train_model.py for experimentation

---

Need more details? See **README.md** for complete documentation!
