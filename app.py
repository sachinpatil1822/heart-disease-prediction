from flask import Flask, render_template, request
import pickle

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form['age'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    thalach = float(request.form['thalach'])
    oldpeak = float(request.form['oldpeak'])

    # Create feature list
    features = [[
        age,
        trestbps,
        chol,
        thalach,
        oldpeak
    ]]

    # Predict
    prediction = model.predict(features)

    # High Risk
    if prediction[0] == 1:

        result = "High Risk of Heart Disease"

        tips = [
            "Avoid smoking and alcohol",
            "Reduce oily and junk food",
            "Exercise daily",
            "Drink more water",
            "Reduce stress",
            "Maintain proper sleep",
            "Avoid sugary drinks"
        ]

    # Low Risk
    else:

        result = "Low Risk of Heart Disease"

        tips = [
            "Maintain healthy lifestyle",
            "Continue regular exercise",
            "Eat healthy food",
            "Stay hydrated",
            "Do regular health checkups"
        ]

    return render_template(
        "index.html",
        prediction_text=result,
        health_tips=tips
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)