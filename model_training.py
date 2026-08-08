import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = Path("spam.csv")
OUTPUT_DIR = Path("documentation")
MODEL_DIR = Path("model")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError("spam.csv was not found.")

df = pd.read_csv(DATA_FILE, encoding="latin-1")
df = df.iloc[:, :2].copy()
df.columns = ["category", "message"]
df.dropna(inplace=True)

df["category"] = df["category"].str.lower().str.strip()
df["message"] = df["message"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["category"],
    test_size=0.20,
    random_state=42,
    stratify=df["category"]
)

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
matrix = confusion_matrix(y_test, predictions).tolist()

joblib.dump(pipeline, MODEL_DIR / "spam_detector.joblib")

results = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": matrix,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

(OUTPUT_DIR / "evaluation.json").write_text(
    json.dumps(results, indent=4),
    encoding="utf-8"
)

print("=" * 58)
print("              MODEL TRAINING COMPLETE")
print("=" * 58)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Accuracy        : {accuracy:.2%}\n")
print(classification_report(y_test, predictions, zero_division=0))
print("\nSaved model:", MODEL_DIR / "spam_detector.joblib")
