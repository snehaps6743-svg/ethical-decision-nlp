from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Small built-in dataset for the live demonstration

training_texts = [
"Reject older applicants because they cannot learn technology",
"Older people are not suitable for modern jobs",
"Women cannot perform technical jobs",
"Men are better leaders than women",
"Reject candidates because of their disability",
"People from different cultures are not trustworthy",
"Reject people because of their accent",
"Hire candidates based on skills and qualifications",
"Evaluate applicants using technical skills",
"Provide equal opportunities to all candidates",
"Select candidates based on experience and ability",
"Judge every applicant using the same criteria",
"Reject women because they are women",
"Do not hire older employees",
"Evaluate candidates fairly regardless of age"
]

training_labels = [
"Age Bias",
"Age Bias",
"Gender Bias",
"Gender Bias",
"Disability Bias",
"Cultural/Ethnic Bias",
"Language Bias",
"No Major Bias",
"No Major Bias",
"No Major Bias",
"No Major Bias",
"No Major Bias",
"Gender Bias",
"Age Bias",
"No Major Bias"
]

# Train the lightweight bias detection model

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(training_texts)

bias_model = LogisticRegression(max_iter=300)
bias_model.fit(X, training_labels)

# Simple rules for identifying potentially unfair statements

unfair_terms = [
"reject older",
"too old",
"because of age",
"older applicants",
"because they are women",
"because she is a woman",
"because he is a man",
"because of gender",
"because of disability",
"because they are disabled",
"because of their accent",
"because of their culture",
"not suitable because",
"not trustworthy because"
]

def analyze_statement(text):
normalized = text.lower()

```
features = vectorizer.transform([text])
bias = bias_model.predict(features)[0]

is_unfair = any(
    term in normalized for term in unfair_terms
)

if is_unfair:
    status = "Unethical"
    confidence = 0.90
else:
    status = "Ethical"
    confidence = 0.70

recommendations = {
    "Age Bias": (
        "Evaluate applicants based on skills, experience "
        "and ability to learn, not age."
    ),
    "Gender Bias": (
        "Evaluate every candidate using the same qualifications "
        "regardless of gender."
    ),
    "Disability Bias": (
        "Avoid disability discrimination and consider reasonable support."
    ),
    "Cultural/Ethnic Bias": (
        "Evaluate relevant qualifications rather than cultural background."
    ),
    "Language Bias": (
        "Assess actual communication requirements rather than accent."
    ),
    "No Major Bias": (
        "Review the decision carefully and apply consistent criteria."
    )
}

if not is_unfair and bias != "No Major Bias":
    bias = "Needs Review"

recommendation = recommendations.get(
    bias,
    "Review the statement and apply fair, consistent decision criteria."
)

return status, confidence, bias, recommendation
```

@app.route("/", methods=["GET", "POST"])
def home():
result = None
confidence = None
bias = None
recommendation = None
user_input = ""

```
if request.method == "POST":
    user_input = request.form.get("decision", "").strip()

    if user_input:
        result, confidence, bias, recommendation = (
            analyze_statement(user_input)
        )

return render_template(
    "index.html",
    result=result,
    confidence=confidence,
    bias=bias,
    recommendation=recommendation,
    user_input=user_input
)
```

if **name** == "**main**":
app.run()
