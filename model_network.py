"""Anomaly Detection Model
This script provides:
    1) Synthetic data generation (network log features)
    2) Training of an IsolationForest anomaly detection model
    3) Model persistence (save/load)
    4) Scoring via CLI or HTTP endpoint

Features:
- Data generation/ingestion
- Model training
- Model inference
- Service integration

Usage:
    python model_network.py --train                      # train model and save "model.joblib"
    python model_network.py --score '{"bytes": 1000, "duration": 0.3}'
    python model_network.py --bytes 1000 --duration 0.3  # score a record without JSON quoting issues
    python model_network.py --serve                      # starts local HTTP server on http://localhost:8000/score

Note: Requires scikit-learn and pandas.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_PATH = "model.joblib"


def generate_synthetic_logs(n_normal: int = 200, n_anomalies: int = 10) -> pd.DataFrame:
    """Generate a small synthetic dataset resembling network log features."""

    rng = np.random.default_rng(42)

    # Normal traffic: bytes ~ N(500, 100), duration ~ N(0.4, 0.15)
    normal_bytes = rng.normal(loc=500, scale=100, size=n_normal).clip(min=50)
    normal_duration = rng.normal(loc=0.4, scale=0.15, size=n_normal).clip(min=0.01)

    # Anomalies: very large bytes or long duration
    anomaly_bytes = rng.normal(loc=2500, scale=300, size=n_anomalies).clip(min=500)
    anomaly_duration = rng.normal(loc=5.0, scale=1.0, size=n_anomalies).clip(min=0.5)

    df_normal = pd.DataFrame({"bytes": normal_bytes, "duration": normal_duration, "label": "normal"})
    df_anom = pd.DataFrame({"bytes": anomaly_bytes, "duration": anomaly_duration, "label": "anomaly"})

    return pd.concat([df_normal, df_anom], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


def train_model(df: pd.DataFrame) -> IsolationForest:
    """Train an IsolationForest model on numeric features."""
    model = IsolationForest(n_estimators=128, contamination=0.05, random_state=42)
    model.fit(df[["bytes", "duration"]])
    return model


def save_model(model: IsolationForest, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> IsolationForest:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run with --train first.")
    return joblib.load(path)


def score_record(model: IsolationForest, record: dict) -> dict:
    """Score a single record, returning anomaly score and label."""
    # Validate input keys
    if "bytes" not in record or "duration" not in record:
        raise ValueError("Record must include 'bytes' and 'duration' fields.")

    # Use DataFrame for stable feature names (matches training data)
    X = pd.DataFrame([{"bytes": float(record["bytes"]), "duration": float(record["duration"])}])
    score = float(model.decision_function(X)[0])
    is_anomaly = bool(model.predict(X)[0] == -1)

    return {"score": score, "is_anomaly": is_anomaly}


class ScoreHandler(BaseHTTPRequestHandler):
    """HTTP handler that exposes /score endpoint."""

    def _write_json(self, response: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def _not_found(self) -> None:
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")

    def do_GET(self):
        # Allow simple checks and browser-based usage.
        if not self.path.startswith("/score"):
            return self._not_found()

        # Support optional query params: /score?bytes=1000&duration=0.3
        try:
            q = self.path.split("?", 1)[1]
            params = dict(p.split("=", 1) for p in q.split("&") if "=" in p)
            record = {"bytes": float(params.get("bytes", "")), "duration": float(params.get("duration", ""))}
            response = score_record(self.server.model, record)
            return self._write_json(response)
        except Exception:
            # Friendly info using GET
            info = {
                "info": "Send POST to /score with JSON {\"bytes\": ..., \"duration\": ...}, or use query params like /score?bytes=1000&duration=0.3"
            }
            return self._write_json(info)

    def do_POST(self):
        if self.path != "/score":
            return self._not_found()

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
            response = score_record(self.server.model, payload)
            return self._write_json(response)
        except Exception as e:
            return self._write_json({"error": str(e)}, status=400)


def serve(model: IsolationForest, host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), ScoreHandler)
    server.model = model
    print(f"Serving /score at http://{host}:{port}/score (POST JSON with bytes/duration)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anomaly detection model for network log data")
    parser.add_argument("--train", action="store_true", help="Train the anomaly detection model and save it to disk")
    parser.add_argument("--score", type=str, help="Score a single JSON record, e.g. '{\"bytes\": 1000, \"duration\": 0.4}'")
    parser.add_argument("--bytes", type=float, help="Score using a bytes value (must be used with --duration)")
    parser.add_argument("--duration", type=float, help="Score using a duration value (must be used with --bytes)")
    parser.add_argument("--serve", action="store_true", help="Run a local HTTP server for scoring")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to save/load the model")

    args = parser.parse_args()

    if args.train:
        df = generate_synthetic_logs()
        model = train_model(df)
        save_model(model, args.model)
        print(f"Trained model and saved to {args.model} (data size={len(df)})")
        return

    if args.score or (args.bytes is not None and args.duration is not None):
        model = load_model(args.model)

        if args.score:
            try:
                record = json.loads(args.score)
            except json.JSONDecodeError:
                raise SystemExit("--score value must be valid JSON, e.g. '{\"bytes\": 1000, \"duration\": 0.4}'")
        else:
            record = {"bytes": args.bytes, "duration": args.duration}

        result = score_record(model, record)
        print(json.dumps(result, indent=2))
        return

    if args.serve:
        model = load_model(args.model)
        serve(model)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
