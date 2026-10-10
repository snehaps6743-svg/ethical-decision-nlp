from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel("ethical_decision_nlp_dataset.xlsx")


# ============================================================
# TF-IDF MODEL FOR BIAS DETECTION
# ============================================================

X = df["Decision_Statement"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)


bias_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

bias_model.fit(
    X_tfidf,
    df["Bias_Type"]
)


# ============================================================
# LOAD TRAINED DISTILBERT MODEL
# ============================================================

transformer_path = "transformer_ethical_model"

print("\nLoading Transformer model...")

tokenizer = AutoTokenizer.from_pretrained(
    transformer_path
)

transformer_model = AutoModelForSequenceClassification.from_pretrained(
    transformer_path
)

transformer_model.eval()

print("Transformer model loaded successfully!")


# ============================================================
# ETHICAL STATUS PREDICTION
# ============================================================

def predict_ethical_status(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = transformer_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][prediction].item()

    if prediction == 0:
        result = "Ethical"
    else:
        result = "Unethical"

    return result, confidence


# ============================================================
# FAIR RECOMMENDATION
# ============================================================

def get_recommendation(result, bias):

    if result == "Unethical":

        if bias == "Age Bias":

            return (
                "Do not judge applicants based on age. "
                "Evaluate their skills, experience, and ability to learn."
            )

        elif bias == "Gender Bias":

            return (
                "Do not make decisions based on gender. "
                "Evaluate all candidates using the same qualifications."
            )

        elif bias == "Disability Bias":

            return (
                "Do not discriminate based on disability. "
                "Consider the person's skills and provide reasonable support."
            )

        elif bias == "Cultural/Ethnic Bias":

            return (
                "Do not judge people based on cultural or ethnic background. "
                "Evaluate candidates using relevant skills and qualifications."
            )

        elif bias == "Language Bias":

            return (
                "Do not judge a person based on accent or language background. "
                "Evaluate their actual communication skills and job requirements."
            )

        else:

            return (
                "Review the decision and avoid discrimination. "
                "Evaluate the person based on relevant skills and qualifications."
            )

    else:

        return (
            "The decision appears fair and does not show major bias."
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    bias = None
    recommendation = None
    confidence = None
    user_input = ""

    if request.method == "POST":

        user_input = request.form["decision"]

        # ----------------------------------------------------
        # Transformer prediction
        # ----------------------------------------------------

        result, confidence = predict_ethical_status(
            user_input
        )

        # ----------------------------------------------------
        # TF-IDF bias prediction
        # ----------------------------------------------------

        user_tfidf = vectorizer.transform(
            [user_input]
        )

        bias = bias_model.predict(
            user_tfidf
        )[0]

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        recommendation = get_recommendation(
            result,
            bias
        )

    return render_template(
        "index.html",
        result=result,
        bias=bias,
        recommendation=recommendation,
        confidence=confidence,
        user_input=user_input
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)