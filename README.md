# 🚀 Network Log Anomaly Detection System

This project is a lightweight, end-to-end Machine Learning pipeline designed to detect anomalies in network log data. It uses the **Isolation Forest** algorithm to identify unusual patterns in data transfer bytes and request durations.

---

## 📌 Overview
The system provides a full lifecycle for anomaly detection:
* **Data Generation:** Creates synthetic network logs (Normal vs. Anomalous).
* **Model Training:** Unsupervised outlier detection using `scikit-learn`.
* **Model Persistence:** Saves and loads trained models via `joblib`.
* **Multi-Interface:** Supports both a Command Line Interface (CLI) and a RESTful HTTP API.

---

## ⚙️ Technologies Used
* **Python 3.x**
* **Pandas & NumPy:** Data processing and feature engineering.
* **Scikit-learn:** Isolation Forest implementation.
* **Joblib:** Model serialization.
* **BaseHTTPRequestHandler:** Lightweight API serving.

---

## ▶️ Usage

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
2. Train the Model
Generate synthetic data and train the model.joblib file:

Bash
python model_network.py --train
3. Scoring via CLI
You can score a record using a JSON string or direct arguments:

Bash
# Using JSON
python model_network.py --score '{"bytes": 2500, "duration": 5.0}'

# Using arguments
python model_network.py --bytes 1000 --duration 0.3
4. Run the HTTP Server
Start the local scoring server (default port 8000):

Bash
python model_network.py --serve
Test the API:
Open your browser or use curl:
http://localhost:8000/score?bytes=2500&duration=5.0

📊 Example Output
JSON
{
  "score": -0.1542,
  "is_anomaly": true
}
Note: A negative score typically indicates an anomaly in Isolation Forest.