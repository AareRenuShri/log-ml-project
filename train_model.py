import joblib

import re

def normalize_log(log):
    log = re.sub(r'user_\d+', 'user_ID', log)
    log = re.sub(r'object_\d+', 'object_ID', log)
    log = re.sub(r'\b\d+ms\b', 'TIME_MS', log)
    log = re.sub(r'\b\d+s\b', 'TIME_S', log)
    log = re.sub(r'\b\d+\b', 'NUMBER', log)

    return log

import pandas as pd

df = pd.read_csv("logs_combined_v2.csv")
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split

X = df["log_text"].apply(normalize_log)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 1000)
model.fit(X_train_tfidf, y_train)

feature_names = vectorizer.get_feature_names_out()

for i, label in enumerate(model.classes_):
    print(f"\nTop features for {label}:")

    coefficients = model.coef_[i]

    top_indices = coefficients.argsort()[-10:][::-1]

    for index in top_indices:
        print(
            feature_names[index],
            "->",
            round(coefficients[index], 3)
        )

joblib.dump(model, "log_classified.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

y_pred = model.predict(X_test_tfidf)
print("Predictions:", y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Labels:", model.classes_)

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))

new_log = ["User 4821 completed the request successfully"]
new_log_normalized = [normalize_log(log) for log in new_log]
new_log_tfidf = vectorizer.transform(new_log_normalized)
prediction = model.predict(new_log_tfidf)
print("New log:", new_log[0])
print("Prediction:", prediction[0])