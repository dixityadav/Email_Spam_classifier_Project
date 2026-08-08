import sys
from pathlib import Path

import joblib

MODEL_FILE = Path("model/spam_detector.joblib")

if not MODEL_FILE.exists():
    print("Trained model not found.")
    print("Run: python3 model_training.py")
    raise SystemExit(1)

text = " ".join(sys.argv[1:]).strip()

if not text:
    text = input("Enter a message: ").strip()

if not text:
    print("No message entered.")
    raise SystemExit(1)

detector = joblib.load(MODEL_FILE)
prediction = detector.predict([text])[0]

print("Result:", "SPAM" if prediction == "spam" else "HAM (NORMAL MESSAGE)")
