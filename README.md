
# Fraud Transaction Detection – XGBoost + FastAPI + Docker

This repository contains an end-to-end **Fraud Transaction Detection** system:

- A **Google Colab notebook** that trains an XGBoost model on a simulated transactions dataset.
- A **FastAPI service (developed & run from VS Code)** that exposes a `/predict` endpoint for real-time fraud prediction.
- A **Docker setup** to containerize and run the API.

---

## 1. Problem Overview

**Objective:**  
Build a system that can classify whether a transaction is fraudulent (`TX_FRAUD = 1`) or legitimate (`TX_FRAUD = 0`).

The dataset is a **simulated card transactions dataset** with core fields such as: :contentReference[oaicite:0]{index=0}  

- `TRANSACTION_ID` – unique transaction ID  
- `TX_DATETIME` – transaction date and time  
- `CUSTOMER_ID` – unique ID for each customer  
- `TERMINAL_ID` – unique ID for each terminal/merchant  
- `TX_AMOUNT` – transaction amount  
- `TX_FRAUD` – binary label (0 = genuine, 1 = fraud)

Fraud labels are generated using three scenarios (high-amount frauds, compromised terminals, and compromised customers over time), described in **`docs/Fraud Transaction Detection.pdf`**.

---

## 2. Tech Stack

- **Language:** Python 3.10+  
- **Modeling:** pandas, scikit-learn, XGBoost, imbalanced-learn, joblib :contentReference[oaicite:1]{index=1}  
- **API:** FastAPI, Uvicorn :contentReference[oaicite:2]{index=2}  
- **Environment:**
  - **Model training:** Google Colab
  - **API development & deployment:** VS Code (using integrated terminal)
- **Containerization:** Docker

---

## 3. Project Structure

Repository layout:

```text
fraud-transaction-detection/
├── app.py                        # FastAPI application (VS Code)
├── Fraud_Detection.ipynb         # Google Colab notebook (model training)
├── Dockerfile                    # Docker image definition for the API
├── requirements.txt              # Python dependencies
├── fraud_model_xgb.joblib        # Trained XGBoost model (saved from Colab)
├── id_encoders.joblib            # Label encoders for CUSTOMER_ID & TERMINAL_ID
├── data/
│   └── dataset.zip               # Transactions dataset (zipped)
├── docs/
│   └── Fraud Transaction Detection.pdf   # Problem & dataset description
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
````

### Key Components

* **`Fraud_Detection.ipynb` (Google Colab)**

  * Loads the dataset from `data/dataset.zip`
  * Performs preprocessing & feature engineering
  * Handles class imbalance
  * Trains an XGBoost model
  * Saves:

    * `fraud_model_xgb.joblib`
    * `id_encoders.joblib`

* **`app.py` (FastAPI in VS Code)** 

  * Loads the trained model and ID encoders at startup
  * Prepares features for a single transaction
  * Exposes:

    * `GET /` – health check
    * `POST /predict` – fraud prediction API

* **`Dockerfile`**

  * Builds a container image for the FastAPI app.

---

## 4. End-to-End Workflow

### High-Level Flow

1. **Model Training in Google Colab**

   * Open `Fraud_Detection.ipynb` in Colab.
   * Load and explore the dataset.
   * Engineer features (time-based + basic aggregates).
   * Handle class imbalance and train XGBoost.
   * Evaluate using metrics like precision, recall, F1-score.
   * Save `fraud_model_xgb.joblib` and `id_encoders.joblib` to download and place in the repo root.

2. **API Development & Deployment in VS Code (FastAPI)**

   * Open the repository in **VS Code**.
   * Create a Python virtual environment.
   * Install dependencies from `requirements.txt`.
   * Run FastAPI with Uvicorn from the **VS Code terminal**.
   * Test the `/predict` endpoint via **Swagger UI**.

3. **Optional Docker Deployment**

   * Build a Docker image.
   * Run the container.
   * Access the same Swagger UI from the containerized app.

---

## 5. Local Setup in VS Code (FastAPI)

### 5.1. Clone or open the repository

```bash
# If you haven't cloned yet:
git clone https://github.com/PardhivAryan/fraud-transaction-detection.git
cd fraud-transaction-detection
```

Then open this folder in **VS Code**:

* VS Code → **File → Open Folder** → select `fraud-transaction-detection`.

### 5.2. Create and activate a virtual environment (macOS / Linux)

Open the **VS Code terminal** (View → Terminal) and run:

```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

### 5.3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies include FastAPI, Uvicorn, pandas, scikit-learn, xgboost, imbalanced-learn, joblib. 

### 5.4. Make sure model files are present

Check that these files are in the **repository root**:

* `fraud_model_xgb.joblib`
* `id_encoders.joblib`

These are loaded in `app.py` at startup. 

---

## 6. Running the FastAPI Server (VS Code)

From the **VS Code terminal** (with venv activated):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

* `app:app`

  * First `app` = `app.py` (file name without `.py`)
  * Second `app` = FastAPI instance inside `app.py`
* `--reload` = auto-reload on code changes (for development)
* `--port 8000` = API runs on port 8000

If everything is correct, you should see log messages like:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.1. Health Check

Visit in your browser:

* `http://127.0.0.1:8000/`

Expected response:

```json
{
  "status": "ok",
  "message": "Fraud Detection API is running"
}
```

---

## 7. Swagger UI (API Documentation)

FastAPI automatically provides Swagger UI.

### 7.1. Open Swagger UI

Go to:

* `http://127.0.0.1:8000/docs`

You should see:

* `GET /` – health check endpoint
* `POST /predict` – prediction endpoint

### 7.2. /predict – Request Schema

The request body is defined by the `Transaction` Pydantic model in `app.py`: 

* `transaction_id`: integer
* `tx_datetime`: string (e.g. `"2018-01-15 10:30:00"`)
* `customer_id`: integer
* `terminal_id`: integer
* `tx_amount`: float

Example JSON:

```json
{
  "transaction_id": 1,
  "tx_datetime": "2018-01-15 10:30:00",
  "customer_id": 12345,
  "terminal_id": 6789,
  "tx_amount": 250.75
}
```

### 7.3. Example Response

Example response from `/predict`:

```json
{
  "transaction_id": 1,
  "fraud_probability": 0.87,
  "is_fraud": 1,
  "latency_ms": 3.21
}
```

* `fraud_probability`: probability that the transaction is fraudulent
* `is_fraud`: `1` if fraud, otherwise `0` (threshold 0.5)
* `latency_ms`: inference latency in milliseconds

You can test this right in Swagger UI by clicking **“Try it out”** on `/predict`.

---

## 8. Model Training in Google Colab

> **Note:** You can run the API with the already-trained model.
> Use this section if you want to **retrain the model** or **experiment**.

Steps:

1. Upload `Fraud_Detection.ipynb` and `data/dataset.zip` to your Google Drive
   or upload them directly in Colab.
2. Open **Google Colab** → **File → Upload notebook** → select `Fraud_Detection.ipynb`.
3. In Colab, set the runtime:

   * **Runtime → Change runtime type → Python 3 + GPU (optional)**.
4. Run the notebook cells sequentially:

   * Import libraries
   * Load `dataset.zip`
   * Data cleaning and feature engineering
   * Train-test split
   * Handle class imbalance (e.g., using imbalanced-learn)
   * Train XGBoost classifier
   * Evaluate metrics (confusion matrix, classification report, ROC-AUC)
5. At the end of the notebook, there will be code to **export the model & encoders**:

   * `fraud_model_xgb.joblib`
   * `id_encoders.joblib`
6. Download both `.joblib` files from Colab and place them in the project root (same folder as `app.py`).

Now the updated model will be used automatically by the FastAPI app in VS Code.

---

## 9. Docker Setup & Commands

You can also run this project in a Docker container.

### 9.1. Build Docker Image

From the project root (where `Dockerfile` is located):

```bash
docker build -t fraud-detection-api .
```

* `fraud-detection-api` is the image name (you can change it).

### 9.2. Run Docker Container

```bash
docker run -p 8000:8000 fraud-detection-api
```

* `-p 8000:8000` maps container port 8000 → host port 8000.

### 9.3. Access Swagger UI from Docker

Once the container is running, open:

* `http://127.0.0.1:8000/docs`

The behavior of `/` and `/predict` is exactly the same as when running via VS Code directly.

### 9.4. Stop Container

Press `Ctrl + C` in the terminal where you ran `docker run`,
or list and stop containers using:

```bash
# List running containers
docker ps

# Stop by container ID
docker stop <CONTAINER_ID>
```

---

## 10. Logging & Monitoring

Basic logging is configured in `app.py` using Python’s `logging` module: 

* Logs prediction details:

  * `transaction_id`
  * `fraud_probability`
  * `is_fraud`
  * `latency_ms`

These logs appear in the terminal (VS Code or Docker logs) and can be extended for production monitoring.

---


## 11. Credits

* Dataset and problem statement based on a simulated fraud transactions dataset
  (see `docs/Fraud Transaction Detection.pdf` for details). 
* Model training performed in **Google Colab**.
* API development and deployment implemented using **FastAPI in VS Code**.
