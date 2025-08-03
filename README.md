# 💳 Fraud Transaction Detection Using Transactions Dataset

This project aims to identify fraudulent transactions using machine learning techniques on a simulated dataset. It demonstrates how data preprocessing, feature engineering, and model evaluation can be combined to build a reliable fraud detection system.

---

## 📌 Objective

The primary goal of this project is to develop a model that can classify transactions as **fraudulent (1)** or **legitimate (0)** based on key transactional attributes.

---

## 🧾 Dataset Overview

The dataset is a **simulated representation** of real-world banking transactions and includes three types of fraud scenarios:

1. **Amount-based fraud**: Transactions above a fixed threshold (220).
2. **Terminal-based fraud**: Certain terminals are compromised over time.
3. **Customer-based fraud**: High-value transactions linked to leaked credentials.

### 🔑 Key Columns

| Column Name     | Description                                       |
|-----------------|---------------------------------------------------|
| `TRANSACTION_ID`| Unique identifier for each transaction            |
| `TX_DATETIME`   | Timestamp of the transaction                      |
| `CUSTOMER_ID`   | Unique ID of the customer                         |
| `TERMINAL_ID`   | Unique ID of the merchant terminal                |
| `TX_AMOUNT`     | Amount of the transaction                         |
| `TX_FRAUD`      | Label: 1 = Fraud, 0 = Legitimate transaction      |

> 🔎 **Note**: Fraud labels are simulated using logical rules described in the project report.

---

## 🛠️ Technologies Used

- **Python 3**
- **Google Colab**
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `seaborn`, `xgboost`

---

## 📂 Project Structure

- 📓 Fraud_Detection.ipynb: Main notebook with fraud detection code  
- 📄 [Fraud_Transaction_Detection.pdf](https://github.com/PardhivAryan/Fraud-transaction-detection/blob/main/Fraud_Transaction_Detection.pdf): Project report  
- 📘 README.md: Project overview


---

## 🚀 Getting Started

### ▶️ Run Online
You can run the notebook directly in Google Colab using this link:  
🔗 [Open in Colab](https://colab.research.google.com/drive/1SWdFF9pAzddAHH_ZcF66NHjhl5uoot2C)

### 💻 Run Locally
1. Clone the repo:
```bash
git clone https://github.com/PardhivAryan/Fraud-transaction-detection.git
cd Fraud-transaction-detection

