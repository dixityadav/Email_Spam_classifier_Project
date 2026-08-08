import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "spam.csv"


def prepare_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "spam.csv is missing. Place the dataset in the project folder."
        )

    data = pd.read_csv(DATA_FILE, encoding="latin-1")
    data = data.iloc[:, :2].copy()
    data.columns = ["category", "message"]
    data.dropna(inplace=True)

    data["category"] = data["category"].str.lower().str.strip()
    data["message"] = data["message"].astype(str)

    return data


def create_classifier():
    return Pipeline([
        (
            "text_features",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])


def run():
    print("=" * 58)
    print("             SMS SPAM DETECTION SYSTEM")
    print("=" * 58)

    dataset = prepare_dataset()

    messages_train, messages_test, labels_train, labels_test = train_test_split(
        dataset["message"],
        dataset["category"],
        test_size=0.20,
        random_state=42,
        stratify=dataset["category"]
    )

    classifier = create_classifier()
    classifier.fit(messages_train, labels_train)

    predictions = classifier.predict(messages_test)
    score = accuracy_score(labels_test, predictions)

    print(f"\nModel accuracy: {score:.2%}\n")
    print(classification_report(labels_test, predictions, zero_division=0))

    print("\nType a message to check it.")
    print("Enter 'exit' when you are finished.")

    while True:
        text = input("\nMessage: ").strip()

        if text.lower() == "exit":
            print("Program finished.")
            break

        if not text:
            print("Please enter some text.")
            continue

        result = classifier.predict([text])[0]

        if result == "spam":
            print("Result: SPAM")
        else:
            print("Result: HAM (NORMAL MESSAGE)")


if __name__ == "__main__":
    run()
