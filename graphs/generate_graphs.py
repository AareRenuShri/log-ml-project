import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

import re

pastel_colors = [
    "#F8B4C8",  # pastel pink
    "#B8D8F0",  # pastel blue
    "#C8E6C9",  # pastel green
    "#DCC6E0"   # pastel lavender
]

# Log normalization
def normalize_log(log):
    log = re.sub(r'user_\d+', 'user_ID', log)
    log = re.sub(r'object_\d+', 'object_ID', log)
    log = re.sub(r'\b\d+ms\b', 'TIME_MS', log)
    log = re.sub(r'\b\d+s\b', 'TIME_S', log)
    log = re.sub(r'\b\d+\b', 'NUMBER', log)

    return log

# Create graphs folder
import os

os.makedirs("graphs", exist_ok=True)

# Load final dataset
df = pd.read_csv("logs_combined_v2.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# GRAPH 1 — CLASS DISTRIBUTION
class_counts = df["label"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(class_counts.index, class_counts.values, color=pastel_colors)

plt.title("Log Class Distribution")
plt.xlabel("Log Category")
plt.ylabel("Number of Logs")

plt.tight_layout()

plt.savefig("graphs/class_distribution.png", dpi=300)

plt.close()

print("\nCreated: graphs/class_distribution.png")

# Prepare data
X = df["log_text"].apply(normalize_log)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# GRAPH 2 — CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(ax=ax, cmap="Pastel1")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "graphs/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Created: graphs/confusion_matrix.png")

# GRAPH 3 — ROC CURVE
classes = model.classes_

y_test_binary = label_binarize(
    y_test,
    classes=classes
)

y_scores = model.predict_proba(X_test_tfidf)

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        y_scores[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        color=pastel_colors[i],
        linewidth=2,
        label=f"{class_name} (AUC = {roc_auc:.2f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves for Log Classification")

plt.legend()

plt.tight_layout()

plt.savefig(
    "graphs/roc_curve.png",
    dpi=300
)

plt.close()

print("Created: graphs/roc_curve.png")

# GRAPH 4 — TOP FEATURES
feature_names = vectorizer.get_feature_names_out()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

axes = axes.flatten()


for i, label in enumerate(model.classes_):

    coefficients = model.coef_[i]

    top_indices = coefficients.argsort()[-10:][::-1]

    top_features = feature_names[top_indices]
    top_values = coefficients[top_indices]

    axes[i].barh(
        top_features[::-1],
        top_values[::-1],
        color=pastel_colors[i]
    )

    axes[i].set_title(
        f"Top Features - {label}"
    )

    axes[i].set_xlabel(
        "Model Coefficient"
    )
    
plt.tight_layout()

plt.savefig(
    "graphs/top_features.png",
    dpi=300
)

plt.close()

print("Created: graphs/top_features.png")

# FINISHED
print("\nAll graphs generated successfully!")

print("\nFiles created:")

print("graphs/class_distribution.png")
print("graphs/confusion_matrix.png")
print("graphs/roc_curve.png")
print("graphs/top_features.png")