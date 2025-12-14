from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI(title="Sentiment Analysis System")

# ================= LOAD MODELS =================
vectorizer = joblib.load("tfidf_vectorizer.pkl")
manual_model = joblib.load("manual_squared_hinge.pkl")
library_model = joblib.load("library_logistic.pkl")

w = manual_model["weights"]
b = manual_model["bias"]

# ================= MANUAL PREDICTION =================
def manual_predict(text):
    X = vectorizer.transform([text])
    score = X.dot(w) + b
    return "Positive" if score > 0 else "Negative"

# ================= HOME PAGE =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Movie Review Sentiment Analysis</title>
        <style>
            body {
                font-family: Segoe UI, sans-serif;
                background: linear-gradient(120deg, #84fab0, #8fd3f4);
                padding: 50px;
            }
            .container {
                background: white;
                width: 750px;
                margin: auto;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            }
            textarea {
                width: 100%;
                height: 140px;
                font-size: 16px;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #ccc;
            }
            button {
                padding: 12px 25px;
                font-size: 16px;
                border-radius: 8px;
                border: none;
                margin-top: 15px;
                cursor: pointer;
                color: white;
            }
            .manual { background: #ff9800; }
            .library { background: #4caf50; }
            h2 { margin-bottom: 5px; }
            .section {
                margin-top: 25px;
                padding: 15px;
                background: #f7f7f7;
                border-radius: 10px;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                font-size: 13px;
                color: #555;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h2>🎬 Movie Review Sentiment Analysis</h2>
            <p>Enter a movie review to predict whether it is positive or negative.</p>

            <form action="/predict" method="post">
                <textarea name="text" placeholder="Paste your movie review here..." required></textarea>
                <br><br>
                <button class="manual" name="model" value="manual">Manual SVM Model</button>
                <button class="library" name="model" value="library">Library Logistic Model</button>
            </form>

            <div class="section">
                <h3>📊 Dataset Information</h3>
                <p>
                    The system is trained on the IMDB Movie Review Dataset consisting of
                    50,000 labeled movie reviews divided into positive and negative classes.
                </p>
            </div>

            <div class="section">
                <h3>⚙️ Model Comparison</h3>
                <ul>
                    <li><b>Manual Model:</b> Support Vector Machine implemented from scratch using Squared Hinge Loss.</li>
                    <li><b>Library Model:</b> Logistic Regression implemented using Scikit-learn.</li>
                </ul>
            </div>

            <div class="section">
                <h3>🔄 Prediction Workflow</h3>
                <ol>
                    <li>User enters a movie review.</li>
                    <li>Text is converted into numerical features using TF-IDF.</li>
                    <li>The selected model computes a decision score.</li>
                    <li>If the score is positive → Positive sentiment, otherwise Negative.</li>
                </ol>
            </div>

            <div class="section">
                <h3>⚠️ Error & Limitation Discussion</h3>
                <p>
                    The model may struggle with sarcasm, neutral reviews, or context-based meaning
                    because TF-IDF treats text as independent words without semantic understanding.
                </p>
            </div>
        </div>

        <div class="footer">
            Developed using FastAPI, Machine Learning & TF-IDF
        </div>
    </body>
    </html>
    """

# ================= PREDICTION PAGE =================
@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...), model: str = Form(...)):

    if model == "manual":
        result = manual_predict(text)
        model_name = "Manual Squared Hinge SVM"
    else:
        pred = library_model.predict(vectorizer.transform([text]))[0]
        result = "Positive" if pred == 1 else "Negative"
        model_name = "Library Logistic Regression"

    emoji = "😃" if result == "Positive" else "😞"
    color = "#4caf50" if result == "Positive" else "#f44336"

    return f"""
    <html>
    <body style="font-family: Segoe UI; background:#e0f2f1; padding:50px;">
        <div style="background:white; width:650px; margin:auto; padding:30px;
                    border-radius:15px; box-shadow:0 10px 20px rgba(0,0,0,0.2);">
            <h2>Prediction Result</h2>
            <p><b>Model Used:</b> {model_name}</p>
            <h1 style="color:{color};">{emoji} {result}</h1>
            <a href="/">⬅ Try another review</a>
        </div>
    </body>
    </html>
    """
