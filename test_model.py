# Test Script for Twitter Sentiment Analysis
# This script tests the sentiment analysis without the web interface

import pickle
import sys
import os

def load_models():
    """Load the trained models"""
    try:
        print("Loading models...")
        
        with open('models/vectoriser.pickle', 'rb') as f:
            vectoriser = pickle.load(f)
        print("✓ Vectoriser loaded")
        
        with open('models/model_lr.pickle', 'rb') as f:
            model_lr = pickle.load(f)
        print("✓ Logistic Regression model loaded")
        
        with open('models/model_bnb.pickle', 'rb') as f:
            model_bnb = pickle.load(f)
        print("✓ BernoulliNB model loaded")
        
        with open('models/model_svc.pickle', 'rb') as f:
            model_svc = pickle.load(f)
        print("✓ LinearSVC model loaded")
        
        return vectoriser, model_lr, model_bnb, model_svc
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nModels not found! Please run 'python train_model.py' first.")
        sys.exit(1)

def test_predictions(vectoriser, model_lr, model_bnb, model_svc):
    """Test predictions with sample texts"""
    
    # Import preprocessing function
    from train_model import preprocess
    
    print("\n" + "="*60)
    print("Testing Sentiment Analysis Models")
    print("="*60)
    
    # Test cases
    test_cases = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. I hate it so much.",
        "The weather is nice today.",
        "Worst experience ever. Very disappointed.",
        "Best day of my life! So happy! 😊",
        "@Apple Your new iPhone is fantastic!",
        "Check out this link: https://example.com"
    ]
    
    models = {
        'Logistic Regression': model_lr,
        'BernoulliNB': model_bnb,
        'LinearSVC': model_svc
    }
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: \"{text}\"")
        print("-" * 60)
        
        # Preprocess
        processed = preprocess([text])
        vectorized = vectoriser.transform(processed)
        
        # Get predictions from all models
        for model_name, model in models.items():
            prediction = model.predict(vectorized)[0]
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
            
            # Try to get confidence
            try:
                proba = model.predict_proba(vectorized)[0]
                confidence = max(proba) * 100
                print(f"   {model_name:20s}: {sentiment:15s} (Confidence: {confidence:.1f}%)")
            except:
                print(f"   {model_name:20s}: {sentiment}")
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)

def main():
    print("="*60)
    print("Twitter Sentiment Analysis - Test Script")
    print("="*60)
    print()
    
    # Check if models exist
    if not os.path.exists('models/vectoriser.pickle'):
        print("❌ ERROR: Models not found!")
        print("\nPlease run one of the following:")
        print("  1. Double-click 'setup.bat' (recommended)")
        print("  2. Run 'python train_model.py'")
        print()
        sys.exit(1)
    
    # Load models
    vectoriser, model_lr, model_bnb, model_svc = load_models()
    
    # Run tests
    test_predictions(vectoriser, model_lr, model_bnb, model_svc)
    
    print("\n💡 Next steps:")
    print("  1. Run 'python app.py' to start the web interface")
    print("  2. Open http://127.0.0.1:5000 in your browser")
    print()

if __name__ == "__main__":
    main()
