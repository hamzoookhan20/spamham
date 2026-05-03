from transformers import pipeline
import os

# Path to your model folder
model_path = "spam_model"

print("--- Local Model Test ---")

if not os.path.exists(model_path):
    print(f"ERROR: Folder '{model_path}' not found!")
else:
    try:
        # Load the local model
        pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)
        
        # Test inputs
        test_spam = "WINNER! You won a prize. Click http://scam.com"
        test_ham = "Hello Hamza, let's meet at the university library."

        print(f"Testing Spam: {pipe(test_spam)}")
        print(f"Testing Ham: {pipe(test_ham)}")
        
    except Exception as e:
        print(f"FAILED to load model: {e}")
