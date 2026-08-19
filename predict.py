import joblib

model = joblib.load("log_classified.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")    

print("Model loaded successfully")
print("Type 'exit' to quit.\n")

while True:
    log = input("Enter a log (or 'exit' to quit): ")
    if log.lower() == "exit":
        print("Exiting...")
        break
    log_tfidf = vectorizer.transform([log])
    prediction = model.predict(log_tfidf)
    print("Prediction label:", prediction[0])
    print()