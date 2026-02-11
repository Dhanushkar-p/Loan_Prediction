from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model safely ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")

def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # If the pickle stores a dict (common when saving extra metadata),
    # find the value that is a real ML model (has .predict)
    if isinstance(obj, dict):
        # Try common keys first
        for key in ("model", "clf", "classifier", "pipeline", "estimator"):
            if key in obj and hasattr(obj[key], "predict"):
                print(f"[DEBUG] Loaded dict. Using key: '{key}'")
                return obj[key]

        # Otherwise scan all values and pick the first with .predict
        for k, v in obj.items():
            if hasattr(v, "predict"):
                print(f"[DEBUG] Loaded dict. Using first predict-capable key: '{k}'")
                return v

        # If nothing has predict, fail clearly
        raise TypeError(
            f"Pickle loaded a dict but no value has .predict. Keys: {list(obj.keys())}"
        )

    # Normal case: object itself is a model
    if hasattr(obj, "predict"):
        print(f"[DEBUG] Loaded model object: {type(obj).__name__}")
        return obj

    raise TypeError(f"Loaded object type {type(obj).__name__} has no .predict")

model = load_model(MODEL_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────────
def encode_input(form):
    """
    IMPORTANT: These mappings MUST match your training preprocessing.
    Education mapping used here (common LabelEncoder order):
      Graduate -> 0
      Not Graduate -> 1
    """

    gender = 1 if form.get("Gender") == "Male" else 0
    married = 1 if form.get("Married") == "Yes" else 0

    dep_raw = form.get("Dependents", "0")
    dependents = 3 if dep_raw == "3+" else int(dep_raw)

    # ✅ Common LabelEncoder mapping
    education = 0 if form.get("Education") == "Graduate" else 1

    self_employed = 1 if form.get("Self_Employed") == "Yes" else 0

    applicant_inc = float(form.get("ApplicantIncome", 0))
    coapplicant_inc = float(form.get("CoapplicantIncome", 0))
    loan_amount = float(form.get("LoanAmount", 0))
    loan_term = float(form.get("Loan_Amount_Term", 360))
    credit_history = float(form.get("Credit_History", 1))

    area_raw = form.get("Property_Area", "Urban")
    area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area = area_map.get(area_raw, 2)

    features = np.array([[
        gender, married, dependents, education, self_employed,
        applicant_inc, coapplicant_inc, loan_amount,
        loan_term, credit_history, property_area
    ]], dtype=float)

    print(f"\n[DEBUG] Encoded features: {features[0].tolist()}")
    return features


def approved_value_from_model(m):
    """
    Decide which prediction value means 'Approved' using model.classes_ when present.
    """
    if not hasattr(m, "classes_"):
        return None

    classes = list(m.classes_)
    norm = [str(c).strip().lower() for c in classes]

    for positive in ("y", "approved", "yes", "true", "1"):
        if positive in norm:
            return classes[norm.index(positive)]

    if 1 in classes:
        return 1

    return classes[-1]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = encode_input(request.form)
        raw = model.predict(features)[0]

        print(f"[DEBUG] Raw model output: {repr(raw)} ({type(raw).__name__})")
        if hasattr(model, "classes_"):
            print(f"[DEBUG] classes_: {model.classes_}")
        if hasattr(model, "predict_proba"):
            try:
                print(f"[DEBUG] proba: {model.predict_proba(features)[0]}")
            except Exception as _:
                pass

        approved_val = approved_value_from_model(model)
        if approved_val is not None:
            result = "Approved" if raw == approved_val else "Rejected"
        else:
            # fallback (if classes_ not stored)
            approved_labels = {"y", "1", "yes", "approved", "true"}
            result = "Approved" if str(raw).strip().lower() in approved_labels else "Rejected"

        return jsonify({"result": result, "raw": str(raw)})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"result": "Error", "message": str(e)}), 500


@app.route("/debug")
def debug():
    info = {
        "model_type": type(model).__name__,
        "has_predict": hasattr(model, "predict"),
        "classes": list(model.classes_) if hasattr(model, "classes_") else "N/A",
        "n_features": int(model.n_features_in_) if hasattr(model, "n_features_in_") else "unknown",
    }

    sample = np.array([[1, 1, 0, 0, 0, 9000, 4000, 100, 360, 1, 1]], dtype=float)
    raw = model.predict(sample)[0]
    info["sample_prediction_raw"] = repr(raw)

    if hasattr(model, "predict_proba"):
        try:
            info["sample_probabilities"] = model.predict_proba(sample)[0].tolist()
        except Exception as _:
            info["sample_probabilities"] = "predict_proba failed"

    info["approved_value_used_by_app"] = repr(approved_value_from_model(model))
    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=True)
