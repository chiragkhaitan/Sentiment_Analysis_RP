# Flask Web Application for Twitter Sentiment Analysis
from flask import Flask, render_template, request, jsonify
import pickle
import re
import os

app = Flask(__name__)

# Load models
print("Loading models...")
try:
    with open('models/vectoriser.pickle', 'rb') as f:
        vectoriser = pickle.load(f)
    
    with open('models/model_lr.pickle', 'rb') as f:
        model_lr = pickle.load(f)
    
    with open('models/model_bnb.pickle', 'rb') as f:
        model_bnb = pickle.load(f)
        
    with open('models/model_svc.pickle', 'rb') as f:
        model_svc = pickle.load(f)
    
    print("✓ All models loaded successfully!")
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run 'python train_model.py' first to train the models.")
    MODELS_LOADED = False

# Emojis dictionary
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

def predict_sentiment(text, model_name='lr'):
    """Predict sentiment of given text"""
    if not MODELS_LOADED:
        return None, "Models not loaded. Please train the models first."
    
    # Select model
    if model_name == 'lr':
        model = model_lr
    elif model_name == 'bnb':
        model = model_bnb
    elif model_name == 'svc':
        model = model_svc
    else:
        model = model_lr
    
    # Preprocess and predict
    processed = preprocess([text])
    vectorized = vectoriser.transform(processed)
    prediction = model.predict(vectorized)[0]
    
    # Get probability if available
    try:
        proba = model.predict_proba(vectorized)[0]
        confidence = float(max(proba) * 100)
    except:
        # For models without predict_proba (like LinearSVC)
        try:
            decision = model.decision_function(vectorized)[0]
            confidence = float(min(abs(decision) * 10, 100))
        except:
            confidence = 75.0  # Default confidence
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if not MODELS_LOADED:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.',
            'success': False
        })
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_name = data.get('model', 'lr')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'success': False
            })
        
        sentiment, confidence = predict_sentiment(text, model_name)
        
        return jsonify({
            'success': True,
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 2) if confidence else None,
            'model': model_name.upper()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    if not MODELS_LOADED:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.',
            'success': False
        })
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_name = data.get('model', 'lr')
        
        if not texts:
            return jsonify({
                'error': 'No texts provided',
                'success': False
            })
        
        results = []
        for text in texts:
            sentiment, confidence = predict_sentiment(text, model_name)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 2) if confidence else None
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'model': model_name.upper()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Twitter Sentiment Analysis Web App")
    print("="*50)
    if MODELS_LOADED:
        print("✓ Models loaded successfully!")
        print("\nStarting Flask server...")
        print("Open http://127.0.0.1:5000 in your browser")
    else:
        print("\n⚠ WARNING: Models not loaded!")
        print("Please run 'python train_model.py' first")
    print("="*50 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
