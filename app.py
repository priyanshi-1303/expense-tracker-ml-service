from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)

# ---------- Root Test Route ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML Service is running ✅"})


# ---------- Categorize Expense ----------
def categorize_expense(title: str) -> str:
    title = title.lower()
    if any(word in title for word in ["pizza", "burger", "biryani", "food", "restaurant", "meal"]):
        return "Food"
    elif any(word in title for word in ["uber", "ola", "train", "bus", "flight", "taxi", "travel"]):
        return "Travel"
    elif any(word in title for word in ["jeans", "t-shirt", "shirt", "clothes", "dress", "mall", "shopping", "shoes"]):
        return "Shopping"
    elif any(word in title for word in ["electricity", "water", "gas", "wifi", "internet", "rent", "bill", "recharge"]):
        return "Bills"
    else:
        return "Other"


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
    title = (body.get("title") or "").strip()

    if not title:
        return jsonify({"error": "title is required"}), 400

    predicted_category = categorize_expense(title)
    return jsonify({"predicted_category": predicted_category})


# ---------- Simple Linear Regression for Future Prediction ----------
months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Jan–Jun
expenses = np.array([1200, 1500, 1700, 1600, 2000, 2100])  # past expenses

model_reg = LinearRegression()
model_reg.fit(months, expenses)


@app.route("/predict_future", methods=["GET"])
def predict_future():
    next_month = np.array([[7]])  # July
    prediction = model_reg.predict(next_month)[0]
    return jsonify({"next_month_prediction": round(float(prediction), 2)})


# ---------- Analyze Spending ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    category = data.get("category", "").lower()
    current_spend = float(data.get("current_spend", 0))

    averages = {
        "food": 1500,
        "travel": 1000,
        "shopping": 1200,
        "bills": 800
    }

    avg = averages.get(category, 1000)
    alert = None

    if current_spend > avg * 1.5:
        alert = f"⚠️ {category.capitalize()} spending is unusually high! ({current_spend} vs avg {avg})"

    return jsonify({
        "category": category,
        "current_spend": current_spend,
        "average": avg,
        "alert": alert or "✅ Spending is normal"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
