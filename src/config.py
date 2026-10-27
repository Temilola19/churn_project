from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "telco_churn.csv"
CLEAN = ROOT / "data" / "telco_clean.csv"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports"
SEED = 42
# Business assumptions for the retention-campaign simulation (edit freely)
OFFER_COST = 100         # $ cost of a retention offer (discount + handling) per contacted customer
SAVE_RATE = 0.25         # share of true churners who accept the offer and stay
LIFETIME_MONTHS = 12     # months of revenue saved per retained customer
