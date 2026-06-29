from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("models/svm_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        comment = request.form["comment"]

        cleaned = clean_text(comment)

        vector = tfidf.transform([cleaned])

        result = model.predict(vector)

        if result[0] == 1:
            prediction = "🚨 Cyberbullying Detected"
        else:
            prediction = "✅ Non-Cyberbullying"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)