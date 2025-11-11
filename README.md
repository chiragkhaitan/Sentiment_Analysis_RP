# Twitter Sentiment Analysis - Local Implementation

A complete machine learning project for analyzing sentiment in tweets using Natural Language Processing and various ML algorithms.

## 🎯 Project Overview

This project analyzes sentiment in tweets using the Sentiment140 dataset (1.6 million tweets). It includes:
- Text preprocessing with NLP techniques
- Multiple ML models (Logistic Regression, BernoulliNB, LinearSVC)
- A user-friendly web interface built with Flask
- Real-time sentiment prediction

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Sentiment140 dataset (download link below)

## 🚀 Setup Instructions

### Step 1: Install Dependencies

Open Command Prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

Download the Sentiment140 dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/kazanova/sentiment140
- File: `training.1600000.processed.noemoticon.csv`
- Place it in the project root directory

Alternatively, if you have the zip file:
```bash
# Extract the CSV file to the current directory
```

### Step 3: Train the Models

Run the training script (this may take 10-30 minutes depending on your hardware):

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train three different ML models
- Save the trained models to the `models/` directory
- Display accuracy metrics for each model

**Note:** The script uses 20% of the data for faster training. For production use, modify `train_model.py` to use the full dataset.

### Step 4: Start the Web Application

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

### Step 5: Access the Interface

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## 🎨 Features

### Web Interface
- **Real-time Analysis**: Enter any text and get instant sentiment prediction
- **Multiple Models**: Choose between Logistic Regression, BernoulliNB, or LinearSVC
- **Confidence Scores**: See how confident the model is in its prediction
- **Example Tweets**: Try pre-loaded examples to test the system
- **Responsive Design**: Works on desktop and mobile devices

### Model Performance
Based on the Sentiment140 dataset:
- Logistic Regression: ~78% accuracy
- BernoulliNB: ~76% accuracy  
- LinearSVC: ~77% accuracy

### Text Preprocessing
- URL removal and normalization
- Emoji detection and replacement
- Username handling (@mentions)
- Special character normalization
- Repeated character reduction

## 📁 Project Structure

```
GYAN_RP/
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── training.1600000.processed.noemoticon.csv  # Dataset (not included)
├── models/                         # Saved models (created after training)
│   ├── vectoriser.pickle
│   ├── model_lr.pickle
│   ├── model_bnb.pickle
│   └── model_svc.pickle
├── templates/                      # HTML templates
│   └── index.html                  # Main interface
└── Social-Media-Sentiment-Analysis/  # Original project files
    ├── README.md
    ├── delivery4/
    └── ...
```

## 🔧 Usage Examples

### Single Tweet Analysis
1. Open the web interface
2. Enter a tweet in the text box
3. Select a model
4. Click "Analyze Sentiment"

### Programmatic Usage (Python)

```python
import pickle

# Load models
with open('models/vectoriser.pickle', 'rb') as f:
    vectoriser = pickle.load(f)

with open('models/model_lr.pickle', 'rb') as f:
    model = pickle.load(f)

# Preprocess and predict
from train_model import preprocess

text = ["I love this product!"]
processed = preprocess(text)
vectorized = vectoriser.transform(processed)
prediction = model.predict(vectorized)

print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```

### API Endpoints

The Flask app provides REST API endpoints:

**Predict Single Text:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!", "model": "lr"}'
```

**Batch Prediction:**
```bash
curl -X POST http://127.0.0.1:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!"], "model": "lr"}'
```

## 📊 Dataset Information

- **Source**: Sentiment140 (Stanford University)
- **Size**: 1.6 million tweets
- **Classes**: Binary (Negative: 0, Positive: 1)
- **Features**: Tweet text
- **Language**: English

## 🛠️ Troubleshooting

### Models not loading
```bash
# Re-train the models
python train_model.py
```

### NLTK data not found
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Port already in use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Memory errors during training
Reduce the sample size in `train_model.py`:
```python
sample_size = int(len(text) * 0.1)  # Use 10% instead of 20%
```

## 🎓 Model Details

### Preprocessing Pipeline
1. Convert to lowercase
2. Replace URLs with 'URL'
3. Replace emojis with text descriptors
4. Replace @mentions with 'USER'
5. Remove special characters
6. Normalize repeated characters

### Feature Extraction
- TF-IDF Vectorization
- Bigram features (1,2)
- 50,000 features (adjustable)

### Models Implemented
1. **Logistic Regression**: Best overall performance
2. **BernoulliNB**: Fastest training time
3. **LinearSVC**: Good balance of speed and accuracy

## 📝 Future Improvements

- [ ] Add LSTM/CNN deep learning models
- [ ] Implement real-time Twitter API integration
- [ ] Add multi-class sentiment (positive, negative, neutral)
- [ ] Create batch upload feature for CSV files
- [ ] Add visualization dashboard
- [ ] Deploy to cloud platform

## 🤝 Contributing

This is a learning project. Feel free to:
- Report issues
- Suggest improvements
- Add new features
- Optimize performance


## 🙏 Acknowledgments

- Sentiment140 dataset by Stanford University
- Kaggle notebook references

## 📧 Contact

For questions or issues, please create an issue in the repository.

---

**Happy Analyzing! 🎉**
