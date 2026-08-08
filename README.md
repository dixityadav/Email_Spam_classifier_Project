# 📱 SMS Spam Detection System

A Python-based machine learning project that automatically identifies text messages as **Spam** or **Ham (Normal)**.

## Project Summary

The system uses natural-language text as input and converts it into numerical features using **TF-IDF**. A supervised classification model then predicts whether the message is spam.

### Main concepts

- Text preprocessing
- TF-IDF feature extraction
- Supervised classification
- Train/test evaluation
- Precision, recall and F1-score
- Interactive command-line prediction

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- joblib

No Jupyter Notebook or Streamlit is required.

## Workflow

```text
SMS Message
    ↓
Data Cleaning
    ↓
Train/Test Split
    ↓
TF-IDF Feature Extraction
    ↓
Classification Model
    ↓
Evaluation
    ↓
Spam / Ham Prediction
```

## Files

```text
sms-spam-detection/
├── main.py
├── model_training.py
├── classify.py
├── spam.csv
├── requirements.txt
├── README.md
├── .gitignore
├── model/
└── documentation/
    └── Project_Report.docx
```

## Dataset

This project uses the **SMS Spam Collection** dataset.

Place the dataset file in the project root with the name:

```text
spam.csv
```

The CSV should contain the message category and message text in its first two columns.

## Installation

```bash
git clone https://github.com/dixityadav/Email_Spam_classifier_Project.git
cd Email_Spam_classifier_Project
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

## Train and Evaluate

Run:

```bash
python3 model_training.py
```

The script evaluates the classifier and creates:

```text
model/spam_detector.joblib
documentation/evaluation.json
```

The generated model and evaluation file are ignored by Git so the repository remains lightweight.

## Run the Interactive Program

```bash
python3 main.py
```

Example:

```text
Message: Congratulations! You won a free prize. Claim now!
Result: SPAM

Message: Are we meeting at 5 pm today?
Result: HAM (NORMAL MESSAGE)
```

## Direct Prediction

After training:

```bash
python3 classify.py "You have won a free prize. Claim it now!"
```

## Evaluation

The training program reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Accuracy alone is not enough for spam detection because normal messages are usually much more common than spam messages. Precision and recall for the spam class are therefore also considered.

## Limitations

This is an educational spam-classification project. It should not be considered a complete production email-security solution.

## Future Improvements

- Add a larger email dataset.
- Compare Naive Bayes, Logistic Regression and SVM.
- Add URL and sender-related features.
- Improve text preprocessing.
- Deploy the classifier as an API.

## Internship Project

Developed as an Artificial Intelligence / Machine Learning internship project.
