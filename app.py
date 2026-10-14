from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("model.joblib")

# Train and save the model first by running: python train_model.py
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student Score Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 18px; color: #1f2937; }
    main { border: 1px solid #dbe3ee; border-radius: 16px; padding: 24px; box-shadow: 0 8px 28px #0f172a10; }
    h1 { color: #17365d; } label { display: block; margin-top: 14px; font-weight: 600; }
    input { width: 100%; box-sizing: border-box; padding: 11px; margin-top: 6px; border: 1px solid #cbd5e1; border-radius: 8px; }
    button { margin-top: 20px; padding: 12px 18px; background: #17365d; color: white; border: 0; border-radius: 8px; font-weight: bold; }
    #result { margin-top: 18px; padding: 12px; border-radius: 8px; background: #eff6ff; }
    small { color: #64748b; }
  </style>
</head>
<body>
<main>
  <h1>Student Score Predictor</h1>
  <p>Enter study and attendance details to estimate a practice score using a sample machine-learning model.</p>
  <form id="predict-form">
    <label>Study hours per day (0–16)</label><input name="study_hours" type="number" min="0" max="16" step="0.5" value="4" required>
    <label>Attendance percentage (0–100)</label><input name="attendance" type="number" min="0" max="100" value="80" required>
    <label>Previous score (0–100)</label><input name="previous_score" type="number" min="0" max="100" value="65" required>
    <button type="submit">Predict score</button>
  </form>
  <div id="result" hidden></div>
  <small>Educational demo only. This is a synthetic-data model, not an official academic assessment.</small>
</main>
<script>
document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  for (const key of Object.keys(payload)) payload[key] = Number(payload[key]);
  const result = document.getElementById("result");
  result.hidden = false;
  result.textContent = "Predicting...";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    result.textContent = `Estimated practice score: ${data.predicted_score} / 100`;
  } catch (error) {
    result.textContent = error.message;
  }
});
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Run python train_model.py first."}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    required = ["study_hours", "attendance", "previous_score"]
    try:
        values = [float(payload[key]) for key in required]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Required numeric fields: study_hours, attendance, previous_score."}), 400

    study_hours, attendance, previous_score = values
    if not (0 <= study_hours <= 16 and 0 <= attendance <= 100 and 0 <= previous_score <= 100):
        return jsonify({"error": "Values out of range: study_hours 0–16; attendance and previous_score 0–100."}), 400

    prediction = float(model.predict(np.array([values]))[0])
    prediction = max(0.0, min(100.0, prediction))
    return jsonify({"predicted_score": round(prediction, 2)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
