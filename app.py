from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import time
import logging

# ---------- Logging for basic monitoring ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Fraud Detection API")

# ---------- Load model and encoders at startup ----------
MODEL_PATH = "fraud_model_xgb.joblib"
ENCODERS_PATH = "id_encoders.joblib"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
cust_le = encoders["cust_le"]
term_le = encoders["term_le"]

# ---------- Pydantic model for request body ----------
class Transaction(BaseModel):
    transaction_id: int
    tx_datetime: str      # e.g. "2018-01-15 10:30:00"
    customer_id: int
    terminal_id: int
    tx_amount: float

# ---------- Helper: safe label encoding ----------
def safe_transform(le, values):
    """
    If an ID was not seen during training, map it to a fallback existing class.
    This avoids crashing on unseen IDs.
    """
    known_classes = set(le.classes_)
    fallback = le.classes_[0]

    safe_vals = [v if v in known_classes else fallback for v in values]
    return le.transform(safe_vals)

# ---------- Feature preparation (same structure as training) ----------
def build_features_df(tx: Transaction) -> pd.DataFrame:
    data = {
        "TRANSACTION_ID": [tx.transaction_id],
        "TX_DATETIME": [tx.tx_datetime],
        "CUSTOMER_ID": [tx.customer_id],
        "TERMINAL_ID": [tx.terminal_id],
        "TX_AMOUNT": [tx.tx_amount],
    }

    df = pd.DataFrame(data)

    # Parse datetime
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    # Basic time features
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DOW"] = df["TX_DATETIME"].dt.dayofweek

    # In real production these would come from a feature store / DB.
    # For now we set them to 0 to keep it simple.
    df["CUSTOMER_ID_1d_txn_count"] = 0
    df["CUSTOMER_ID_1d_txn_amount"] = 0.0
    df["TERMINAL_ID_1d_txn_count"] = 0
    df["TERMINAL_ID_1d_txn_amount"] = 0.0

    # Encode IDs using label encoders from training
    df["CUSTOMER_ID_ENC"] = safe_transform(cust_le, df["CUSTOMER_ID"])
    df["TERMINAL_ID_ENC"] = safe_transform(term_le, df["TERMINAL_ID"])

    feature_cols = [
        "CUSTOMER_ID_ENC", "TERMINAL_ID_ENC",
        "TX_AMOUNT", "TX_HOUR", "TX_DOW",
        "CUSTOMER_ID_1d_txn_count", "CUSTOMER_ID_1d_txn_amount",
        "TERMINAL_ID_1d_txn_count", "TERMINAL_ID_1d_txn_amount"
    ]

    return df[feature_cols]

# ---------- Health check endpoint ----------
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud Detection API is running"}

# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict(tx: Transaction):
    start_time = time.time()

    # Build features
    X = build_features_df(tx)

    # Predict probability
    proba = float(model.predict_proba(X)[:, 1][0])
    is_fraud = int(proba >= 0.5)

    latency_ms = (time.time() - start_time) * 1000
    logging.info(
        f"Prediction: tx_id={tx.transaction_id}, proba={proba:.4f}, "
        f"is_fraud={is_fraud}, latency_ms={latency_ms:.2f}"
    )

    return {
        "transaction_id": tx.transaction_id,
        "fraud_probability": proba,
        "is_fraud": is_fraud,
        "latency_ms": latency_ms
    }
