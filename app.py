"""
House Price Predictor - Flask app
Uses the model trained in the Colab notebook (house_price_model.ipynb).
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.joblib")
ZIPCODES_PATH = os.path.join(BASE_DIR, "zipcodes.json")
INFO_PATH = os.path.join(BASE_DIR, "model_info.json")

app = Flask(__name__)

# The data is from 2014-2015, so the model measures a house's age against 2015
DATA_YEAR = 2015

# Default columns (same order as in the notebook). Replaced by model_info.json if it exists.
DEFAULT_NUM = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
               "waterfront", "view", "condition", "grade", "sqft_basement",
               "house_age", "was_renovated", "has_basement"]
DEFAULT_CAT = ["zipcode"]

GRADE_LABELS = {
    3: "3 - Very low", 4: "4 - Low", 5: "5 - Fair", 6: "6 - Below average",
    7: "7 - Average (typical house)", 8: "8 - Good", 9: "9 - Better",
    10: "10 - Very good", 11: "11 - Excellent", 12: "12 - Luxury", 13: "13 - Mansion",
}
VIEW_LABELS = {0: "0 - No view", 1: "1 - Fair", 2: "2 - Average", 3: "3 - Good", 4: "4 - Excellent"}
CONDITION_LABELS = {1: "1 - Poor", 2: "2 - Fair", 3: "3 - Average", 4: "4 - Good", 5: "5 - Very good"}
FLOOR_OPTIONS = [1, 1.5, 2, 2.5, 3, 3.5]

# Default values shown in the form
DEFAULTS = {
    "bedrooms": "3", "bathrooms": "2", "sqft_living": "1800", "sqft_lot": "5000",
    "floors": "1", "waterfront": "0", "view": "0", "condition": "3", "grade": "7",
    "sqft_basement": "0", "yr_built": "1990", "yr_renovated": "0", "zipcode": "",
}


# ---------------------------------------------------------------- load the files
def load_everything():
    problems = []
    model, zipcodes, info = None, [], {}

    try:
        with open(INFO_PATH) as f:
            info = json.load(f)
    except Exception:
        info = {}

    try:
        with open(ZIPCODES_PATH) as f:
            zipcodes = [str(z) for z in json.load(f)]
    except Exception as e:
        problems.append(f"Could not read zipcodes.json ({e}).")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        problems.append(f"Could not load house_price_model.joblib ({e}).")

    if model is not None and info.get("sklearn_version"):
        try:
            import sklearn
            if sklearn.__version__ != info["sklearn_version"]:
                print("WARNING: model was trained with scikit-learn", info["sklearn_version"],
                      "but this server has", sklearn.__version__)
        except Exception:
            pass

    return model, zipcodes, info, problems


MODEL, ZIPCODES, INFO, LOAD_PROBLEMS = load_everything()
NUM_FEATURES = INFO.get("numeric_features", DEFAULT_NUM)
CAT_FEATURES = INFO.get("categorical_features", DEFAULT_CAT)
TEST_MAE = INFO.get("test_mae")
TEST_R2 = INFO.get("test_r2")


# ---------------------------------------------------------------- checking the input
def _number(raw, name, label, lo, hi, as_int, errors):
    text = str(raw if raw is not None else "").strip().replace(",", "")
    if text == "":
        errors[name] = f"{label}: please type a number."
        return None
    try:
        value = float(text)
    except ValueError:
        errors[name] = f"{label}: '{raw}' is not a number."
        return None
    if as_int and value != int(value):
        errors[name] = f"{label}: please use a whole number."
        return None
    if value < lo or value > hi:
        errors[name] = f"{label}: please use a value between {lo:,} and {hi:,}."
        return None
    return int(value) if as_int else value


def clean_input(data):
    """Check the form values. Returns (clean_values, errors)."""
    errors = {}
    v = {}

    v["bedrooms"] = _number(data.get("bedrooms"), "bedrooms", "Bedrooms", 1, 10, True, errors)
    v["bathrooms"] = _number(data.get("bathrooms"), "bathrooms", "Bathrooms", 0.5, 8, False, errors)
    v["sqft_living"] = _number(data.get("sqft_living"), "sqft_living", "Living space", 300, 13540, True, errors)
    v["sqft_lot"] = _number(data.get("sqft_lot"), "sqft_lot", "Land size", 500, 1700000, True, errors)
    v["floors"] = _number(data.get("floors"), "floors", "Floors", 1, 3.5, False, errors)
    v["waterfront"] = _number(data.get("waterfront"), "waterfront", "Waterfront", 0, 1, True, errors)
    v["view"] = _number(data.get("view"), "view", "View", 0, 4, True, errors)
    v["condition"] = _number(data.get("condition"), "condition", "Condition", 1, 5, True, errors)
    v["grade"] = _number(data.get("grade"), "grade", "Grade", 3, 13, True, errors)
    v["sqft_basement"] = _number(data.get("sqft_basement"), "sqft_basement", "Basement size", 0, 5000, True, errors)
    v["yr_built"] = _number(data.get("yr_built"), "yr_built", "Year built", 1900, DATA_YEAR, True, errors)

    renovated_raw = str(data.get("yr_renovated", "") or "").strip()
    if renovated_raw in ("", "0"):
        v["yr_renovated"] = 0
    else:
        v["yr_renovated"] = _number(renovated_raw, "yr_renovated", "Year renovated", 1900, DATA_YEAR, True, errors)

    zipcode = str(data.get("zipcode", "") or "").strip()
    if zipcode == "":
        errors["zipcode"] = "Zipcode: please choose one."
    elif ZIPCODES and zipcode not in ZIPCODES:
        errors["zipcode"] = "Zipcode: please choose one from the list."
    v["zipcode"] = zipcode

    # checks that need two values
    if not errors:
        if v["sqft_basement"] > v["sqft_living"]:
            errors["sqft_basement"] = "Basement size cannot be bigger than the living space."
        if v["yr_renovated"] and v["yr_renovated"] < v["yr_built"]:
            errors["yr_renovated"] = "A house cannot be renovated before it was built."

    return v, errors


# ---------------------------------------------------------------- predicting
def make_row(v):
    """Make the one-row table the model expects (same steps as in the notebook)."""
    row = {
        "bedrooms": v["bedrooms"],
        "bathrooms": v["bathrooms"],
        "sqft_living": v["sqft_living"],
        "sqft_lot": v["sqft_lot"],
        "floors": v["floors"],
        "waterfront": v["waterfront"],
        "view": v["view"],
        "condition": v["condition"],
        "grade": v["grade"],
        "sqft_basement": v["sqft_basement"],
        "house_age": DATA_YEAR - v["yr_built"],
        "was_renovated": 1 if v["yr_renovated"] > 0 else 0,
        "has_basement": 1 if v["sqft_basement"] > 0 else 0,
        "zipcode": str(v["zipcode"]),
    }
    return pd.DataFrame([row], columns=NUM_FEATURES + CAT_FEATURES)


def predict_price(v):
    price = float(MODEL.predict(make_row(v))[0])
    result = {"price": round(price, 2), "price_text": f"${round(price, -3):,.0f}"}
    if TEST_MAE:
        low = max(price - TEST_MAE, 0)
        high = price + TEST_MAE
        result["range_text"] = f"${round(low, -3):,.0f} to ${round(high, -3):,.0f}"
    return result


# ---------------------------------------------------------------- pages
def render_page(form, result=None, errors=None, status=200):
    return render_template(
        "index.html",
        form=form,
        result=result,
        errors=errors or {},
        zipcodes=ZIPCODES,
        grade_labels=GRADE_LABELS,
        view_labels=VIEW_LABELS,
        condition_labels=CONDITION_LABELS,
        floor_options=FLOOR_OPTIONS,
        load_problems=LOAD_PROBLEMS,
        test_mae=TEST_MAE,
        test_r2=TEST_R2,
        data_year=DATA_YEAR,
    ), status


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_page(dict(DEFAULTS))

    form = {k: request.form.get(k, "") for k in DEFAULTS}

    if MODEL is None or not ZIPCODES:
        return render_page(form, errors={"_general": "The model is not loaded on the server."}, status=500)

    values, errors = clean_input(form)
    if errors:
        return render_page(form, errors=errors, status=400)

    try:
        result = predict_price(values)
    except Exception as e:
        return render_page(form, errors={"_general": f"Something went wrong while predicting: {e}"}, status=500)

    return render_page(form, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON version. Send the same fields as the form."""
    if MODEL is None or not ZIPCODES:
        return jsonify({"error": "The model is not loaded on the server."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the house details."}), 400

    values, errors = clean_input(data)
    if errors:
        return jsonify({"error": "Some values are not valid.", "details": errors}), 400

    try:
        return jsonify(predict_price(values))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok" if MODEL is not None and ZIPCODES else "model not loaded"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
