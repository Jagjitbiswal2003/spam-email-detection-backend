from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from preprocessing import clean_email


app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL SAFELY ----------------
try:
    model = joblib.load("EmailSpam_Detection_model.pkl")
    print("Model loaded successfully ✅")
except Exception as e:
    print("Model loading failed ❌:", e)
    model = None

# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return "Spam Detection API is running"

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔴 Check model loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # 🔴 Check JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        # 🔴 Check key exists
        if "email" not in data:
            return jsonify({"error": "Missing 'email' field"}), 400

        email = data.get("email")

        # 🔴 Validate input type
        if not isinstance(email, str):
            return jsonify({"error": "Email must be a string"}), 400

        # 🔴 Check empty input
        if email.strip() == "":
            return jsonify({"error": "Email cannot be empty"}), 400

        # 🔴 CLEAN INPUT
        try:
            cleaned_email = clean_email(email)
        except Exception as e:
            return jsonify({"error": f"Cleaning failed: {str(e)}"}), 500

        # 🔴 MODEL PREDICTION
        try:
            prediction = model.predict([cleaned_email])[0]
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        print("Unexpected ERROR:", e)
        return jsonify({"error": "Internal Server Error"}), 500


# ---------------- GLOBAL ERROR HANDLER ----------------
@app.errorhandler(Exception)
def handle_exception(e):
    print("Global ERROR:", e)
    return jsonify({"error": "Something went wrong"}), 500


if __name__ == "__main__":
    app.run(debug=True)