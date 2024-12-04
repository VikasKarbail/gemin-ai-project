from flask import Flask, request, render_template, jsonify
from prediction import predict_heart_disease, get_accuracy
from geminai import get_health_recommendations

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("ui.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_data = [
        data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
        data["fbs"], data["restecg"], data["thalach"], data["exang"],
        data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]
    prediction = predict_heart_disease(user_data)
    accuracy = get_accuracy()
    recommendations = get_health_recommendations(prediction, data["sex"], data["age"], accuracy)
    return jsonify({
        "prediction": prediction,
        "accuracy": accuracy,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
