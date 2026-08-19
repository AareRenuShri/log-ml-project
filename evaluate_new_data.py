import joblib
import pandas as pd

import re

def normalize_log(log):
    log = re.sub(r'user_\d+', 'user_ID', log)
    log = re.sub(r'object_\d+', 'object_ID', log)
    log = re.sub(r'\b\d+ms\b', 'TIME_MS', log)
    log = re.sub(r'\b\d+s\b', 'TIME_S', log)
    log = re.sub(r'\b\d+\b', 'NUMBER', log)

    return log

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = joblib.load("log_classified.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

data = pd.read_csv("new_test_logs.csv")

X_new = data["log_text"].apply(normalize_log)
y_new = data["label"]

X_new_tfidf = vectorizer.transform(X_new)
y_new_pred = model.predict(X_new_tfidf)

accuracy = accuracy_score(y_new, y_new_pred)
print("Accuracy on new data:", accuracy)

cm = confusion_matrix(y_new, y_new_pred)
print("Confusion Matrix on new data:")
print(cm)

print("\nLabels:")
print(model.classes_)

print("\nClassification Report on new data:")
print(classification_report(y_new, y_new_pred))

print("\nMisclassified Logs:")

for log, actual, predicted in zip(X_new, y_new, y_new_pred):
    if actual != predicted:
        print("-----------------------------")
        print("Log:", log)
        print("Actual:", actual)
        print("Predicted:", predicted)