# Twitter Sentiment Analysis - Model Training Script
import re
import pickle
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# NLTK
from nltk.stem import WordNetLemmatizer

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download required NLTK data
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

print("="*50)
print("Twitter Sentiment Analysis - Model Training")
print("="*50)

# Defining emojis dictionary
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def preprocess(textdata):
    """Preprocess text data"""
    processedText = []
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = r'@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText

def main():
    # Load the dataset
    print("\n1. Loading dataset...")
    try:
        DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
        DATASET_ENCODING = "ISO-8859-1"
        
        # Try to find the dataset
        dataset_paths = [
            'training.1600000.processed.noemoticon.csv',
            'Social-Media-Sentiment-Analysis/data/training.1600000.processed.noemoticon.csv',
            'data/training.1600000.processed.noemoticon.csv'
        ]
        
        dataset = None
        for path in dataset_paths:
            try:
                dataset = pd.read_csv(path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
                print(f"   Dataset loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if dataset is None:
            print("\n   ERROR: Dataset not found!")
            print("   Please ensure 'training.1600000.processed.noemoticon.csv' is in the current directory.")
            print("   Download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
            return
        
        # Keep only sentiment and text
        dataset = dataset[['sentiment', 'text']]
        
        # Replace 4 with 1 for binary classification (0=negative, 1=positive)
        dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
        
        print(f"   Dataset shape: {dataset.shape}")
        print(f"   Class distribution:\n{dataset['sentiment'].value_counts()}")
        
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        return
    
    # Preprocess text
    print("\n2. Preprocessing text data...")
    t = time.time()
    
    # For faster training, use a sample (remove this for full training)
    print("   Note: Using smaller sample for deployment. Remove sampling for production.")
    # Sample equally from both classes (reduced for deployment memory constraints)
    sample_per_class = 80000  # 80k from each class = 160k total (optimized for Render free tier)
    df_negative = dataset[dataset['sentiment'] == 0].head(sample_per_class)
    df_positive = dataset[dataset['sentiment'] == 1].head(sample_per_class)
    dataset_sample = pd.concat([df_negative, df_positive]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    text = list(dataset_sample['text'])
    sentiment = list(dataset_sample['sentiment'])
    
    processedtext = preprocess(text)
    print(f"   Preprocessing complete. Time taken: {round(time.time()-t)} seconds")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        processedtext, sentiment, test_size=0.05, random_state=42, stratify=sentiment
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Vectorization
    print("\n4. Vectorizing text...")
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=50000)  # Reduced features for faster training
    vectoriser.fit(X_train)
    print(f"   Number of features: {len(vectoriser.get_feature_names_out())}")
    
    X_train_vec = vectoriser.transform(X_train)
    X_test_vec = vectoriser.transform(X_test)
    
    # Train models
    models = {}
    
    # 1. Logistic Regression
    print("\n5. Training Logistic Regression model...")
    t = time.time()
    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    LRmodel.fit(X_train_vec, y_train)
    y_pred = LRmodel.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Time taken: {round(time.time()-t)} seconds")
    print(f"   Accuracy: {accuracy:.4f}")
    models['LR'] = LRmodel
    
    # 2. BernoulliNB
    print("\n6. Training BernoulliNB model...")
    t = time.time()
    BNBmodel = BernoulliNB(alpha=2)
    BNBmodel.fit(X_train_vec, y_train)
    y_pred = BNBmodel.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Time taken: {round(time.time()-t)} seconds")
    print(f"   Accuracy: {accuracy:.4f}")
    models['BNB'] = BNBmodel
    
    # 3. LinearSVC
    print("\n7. Training LinearSVC model...")
    t = time.time()
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train_vec, y_train)
    y_pred = SVCmodel.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Time taken: {round(time.time()-t)} seconds")
    print(f"   Accuracy: {accuracy:.4f}")
    models['SVC'] = SVCmodel
    
    # Save models
    print("\n8. Saving models...")
    
    # Save vectorizer
    with open('models/vectoriser.pickle', 'wb') as f:
        pickle.dump(vectoriser, f)
    print("   ✓ Vectorizer saved")
    
    # Save Logistic Regression
    with open('models/model_lr.pickle', 'wb') as f:
        pickle.dump(LRmodel, f)
    print("   ✓ Logistic Regression model saved")
    
    # Save BernoulliNB
    with open('models/model_bnb.pickle', 'wb') as f:
        pickle.dump(BNBmodel, f)
    print("   ✓ BernoulliNB model saved")
    
    # Save LinearSVC
    with open('models/model_svc.pickle', 'wb') as f:
        pickle.dump(SVCmodel, f)
    print("   ✓ LinearSVC model saved")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("Models saved in 'models/' directory")
    print("="*50)

if __name__ == "__main__":
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    main()
