"""Download and preprocess IBM Telco Customer Churn dataset for TFX pipeline.

Downloads the dataset from IBM's GitHub repository, selects relevant
features, and cleans the data for ingestion by CsvExampleGen.
"""
import csv
import os
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_PATH = os.path.join("data", "churn_raw.csv")
OUTPUT_PATH = os.path.join("data", "churn.csv")

SELECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def download_dataset():
    """Download raw dataset from IBM GitHub if not exists."""
    if os.path.exists(RAW_PATH):
        print(f"Dataset already exists: {RAW_PATH}")
        return

    os.makedirs("data", exist_ok=True)
    print(f"Downloading dataset from IBM GitHub...")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    size = os.path.getsize(RAW_PATH)
    print(f"Downloaded: {RAW_PATH} ({size:,} bytes)")


def main():
    """Download, clean, and save dataset."""
    download_dataset()

    with open(RAW_PATH, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    cleaned = []
    skipped = 0
    for row in rows:
        if not row["TotalCharges"].strip():
            skipped += 1
            continue
        selected = {col: row[col] for col in SELECTED_COLUMNS}
        cleaned.append(selected)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=SELECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned)

    churn_count = sum(1 for r in cleaned if r["Churn"] == "Yes")
    total = len(cleaned)
    print(f"âœ… Dataset preprocessed: {OUTPUT_PATH}")
    print(f"   Total records : {total}")
    print(f"   Skipped       : {skipped} (empty TotalCharges)")
    print(f"   Churn rate    : {churn_count}/{total} ({churn_count/total*100:.1f}%)")


if __name__ == "__main__":
    main()