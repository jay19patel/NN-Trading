# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

app = Flask(__name__)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/strategies")
def list_strategies():
    """Returns a list of all strategies that have backtest results."""
    if not os.path.exists(RESULTS_DIR):
        return jsonify([])
    strategies = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return jsonify(strategies)

@app.route("/api/results/<strategy>")
def get_results(strategy):
    """Loads results.json for a specific strategy."""
    json_path = os.path.join(RESULTS_DIR, strategy.lower(), "results.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Results not found"}), 404
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
