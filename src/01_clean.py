"""Load raw data, audit quality, fix types, engineer features."""
import json
import numpy as np, pandas as pd
from config import RAW, CLEAN, OUT

df = pd.read_csv(RAW)
log = {"rows_raw": len(df), "cols_raw": df.shape[1]}

# --- audit ---
log["duplicate_ids"] = int(df.customerID.duplicated().sum())
log["duplicate_rows"] = int(df.duplicated().sum())

# TotalCharges is text because 11 rows are blank (all tenure == 0 -> brand new customers)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blank = df["TotalCharges"].isna()
log["blank_total_charges"] = int(blank.sum())
log["blank_total_charges_all_tenure0"] = bool((df.loc[blank, "tenure"] == 0).all())
df.loc[blank, "TotalCharges"] = 0.0   # no billing yet -> 0, not a missing value

# --- consistency ---
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
# "No internet service" / "No phone service" are redundant with InternetService / PhoneService
svc = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for c in svc + ["MultipleLines"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# --- feature engineering ---
df["tenure_bucket"] = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72],
                             labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"])
df["n_addons"] = (df[["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"]] == "Yes").sum(axis=1)
df["n_streaming"] = (df[["StreamingTV","StreamingMovies"]] == "Yes").sum(axis=1)
df["auto_pay"] = df.PaymentMethod.str.contains("automatic").map({True: "Yes", False: "No"})
# Does the customer's bill look high/low vs what their tenure-implied average says? (price drift)
df["avg_monthly_hist"] = np.where(df.tenure > 0, df.TotalCharges / df.tenure, df.MonthlyCharges)
df["charge_drift"] = df.MonthlyCharges - df.avg_monthly_hist

log["rows_clean"] = len(df)
log["churn_rate"] = round(df.Churn.mean(), 4)
log["missing_after_clean"] = int(df.isna().sum().sum())
df.to_csv(CLEAN, index=False)
(OUT / "data_quality_log.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))
