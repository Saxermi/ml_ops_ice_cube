#!/usr/bin/env python3
"""
Model Service Application.
- Provides REST API endpoints for model prediction (/predict) and health checks (/health).
- Loads the machine learning model from model.pkl.
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Dummy implementation for model prediction.
    return jsonify({"prediction": "dummy_result"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
