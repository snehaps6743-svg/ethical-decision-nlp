from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Training examples
training_texts = [
    "Reject older applicants because they cannot learn technology",
    "Older people are not suitable for modern jobs",
    "Do not hire older employees",
    "Reject candidates because they are too old",
    "Evaluate candidates fairly regardless of age",
    "Women cannot perform technical jobs",
    "Men are better leaders than women",
    "Reject women because they are women",
    "Evaluate candidates regardless of gender",
    "Hire candidates based on skills and qualifications",
    "Evaluate applicants using technical skills",
    "Provide equal opportunities to all candidates",
    "Select candidates based on experience and ability",
    "Judge every applicant using the same criteria",
    "Reject candidates because of their disability",
    "Disabled people cannot do office work",
    "Consider applicants based on their abilities",
    "People from different cultures are not trustworthy",
    "Reject people because of their accent",
    "Evaluate applicants based on relevant qualifications",
    "Choose candidates based on their knowledge",
    "Reject applicants because of their religion",
    "Give everyone an equal opportunity",
    "Reject candidates because of their nationality",
    "Select employees based on job-related skills"
]

training_labels = [
    "Age Bias",
    "Age Bias",
    "Age Bias",
    "Age Bias",
    "No Major Bias",
    "Gender Bias",
    "Gender Bias",
    "Gender Bias",
    "No Major Bias",
    "No Major Bias",
    "No Major Bias",
    "No Major Bias",
    "No Major Bias",
    "No Major Bias",
    "Disability Bias",
    "Disability Bias",
    "No Major Bias",
    "Cultural/Ethnic Bias",
    "Language Bias",
    "No Major Bias",
    "No Major Bias",
    "Religious Bias",
    "No Major Bias",
    "Nationality Bias",
    "No Major Bias"
]

# Train the lightweight text classification model
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(training_texts)

bias_model = LogisticRegression(max_iter=300)
bias_model.fit(X, training_labels)


# Words and phrases that may indicate unfair decisions
unfair_terms = [
    "reject older",
    "too old",
    "because of age",
    "older applicants",
    "do not hire older",
    "because they are women",
    "because she is a woman",
    "because he is a man",
    "because of gender",
    "because of disability",
    "because they are disabled",
    "because of their accent",
    "because of their culture",
    "because of their religion",
    "because of their nationality",
    "not trustworthy because",
    "reject women",
    "reject candidates because",
    "reject applicants because"
]


def analyze_statement(text):
    normalized = text.lower()

    features = vectorizer.transform([text])
    bias = bias_model.predict(features)[0]

    # Check for potentially unfair wording
    is_unfair = any(term in normalized for term in unfair_terms)

    # Produce a result
    if is_unfair:
        status = "Unethical"
        confidence = 90
    else:
        status = "Needs Human Review"
        confidence = 70

    recommendations = {
        "Age Bias": (
            "Evaluate applicants based on skills, experience, and "
            "ability to learn rather than age."
        ),
        "Gender Bias": (
            "Evaluate every candidate using the same job-related "
            "qualifications regardless of gender."
        ),
        "Disability Bias": (
            "Avoid disability discrimination and consider reasonable "
            "accommodations and individual abilities."
        ),
        "Cultural/Ethnic Bias": (
            "Assess relevant qualifications and skills rather than "
            "cultural or ethnic background."
        ),
        "Language Bias": (
            "Assess actual job-related communication requirements "
            "rather than a person's accent."
        ),
        "Religious Bias": (
            "Evaluate candidates using relevant job criteria, "
            "not religious beliefs."
        ),
        "Nationality Bias": (
            "Apply fair, job-related selection criteria without "
            "unjustified nationality-based discrimination."
        ),
        "No Major Bias": (
            "Use consistent, transparent, job-related criteria "
            "and review the decision carefully."
        )
    }

    if is_unfair:
        detected_bias = bias
    elif bias != "No Major Bias":
        detected_bias = bias + " - Needs Review"
    else:
        detected_bias = "No Obvious Bias Detected"

    recommendation = recommendations.get(
        bias,
        "Review the statement carefully and apply fair, consistent criteria."
    )

    return status, confidence, detected_bias, recommendation


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    bias = None
    recommendation = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("decision", "").strip()

        if user_input:
            result, confidence, bias, recommendation = analyze_statement(
                user_input
            )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        bias=bias,
        recommendation=recommendation,
        user_input=user_input
    )


if __name__ == "__main__":
    app.run()
