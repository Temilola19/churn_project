"""EDA + hypothesis tests + revenue-at-risk. Saves figures and eda_results.json."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from config import CLEAN, FIG, OUT

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.titleweight": "bold"})
df = pd.read_csv(CLEAN)
df["tenure_bucket"] = pd.Categorical(df.tenure_bucket, ["0-6m","7-12m","13-24m","25-48m","49-72m"], ordered=True)
base = df.Churn.mean()
res = {"overall_churn": round(base, 4)}

def rate_bar(col, fname, title, order=None, rot=0):
    g = df.groupby(col, observed=True).Churn.agg(["mean", "size"]).reset_index()
    if order: g[col] = pd.Categorical(g[col], order, ordered=True); g = g.sort_values(col)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=g, x=col, y="mean", ax=ax, color=sns.color_palette()[0])
    ax.axhline(base, color="crimson", ls="--", lw=1.5, label=f"Overall {base:.1%}")
    for i, r in g.reset_index(drop=True).iterrows():
        ax.text(i, r["mean"] + .01, f'{r["mean"]:.0%}\n(n={r["size"]:,})', ha="center", fontsize=11)
    ax.set_ylim(0, g["mean"].max() * 1.25); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set(title=title, xlabel="", ylabel="Churn rate"); ax.legend(); plt.xticks(rotation=rot)
    fig.savefig(FIG / fname); plt.close(fig)
    return g

# 1. Headline segments
rate_bar("Contract", "01_churn_by_contract.png", "Churn rate by contract type")
rate_bar("tenure_bucket", "02_churn_by_tenure.png", "Churn is an early-life problem")
rate_bar("InternetService", "03_churn_by_internet.png", "Fiber customers churn far more")
rate_bar("PaymentMethod", "04_churn_by_payment.png", "Churn by payment method", rot=15)
rate_bar("n_addons", "05_churn_by_addons.png", "More protective add-ons, less churn")

# 2. Monthly charges / tenure distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, t in zip(axes, ["tenure", "MonthlyCharges"], ["Tenure (months)", "Monthly charges ($)"]):
    sns.kdeplot(data=df, x=col, hue=df.Churn.map({0: "Stayed", 1: "Churned"}), fill=True, common_norm=False, ax=ax)
    ax.set(title=t, ylabel="Density")
fig.savefig(FIG / "06_distributions.png"); plt.close(fig)

# 3. Contract x Internet interaction heatmap
pv = df.pivot_table(index="Contract", columns="InternetService", values="Churn", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pv, annot=True, fmt=".0%", cmap="Reds", cbar=False, ax=ax)
ax.set_title("Churn rate: contract x internet service"); ax.set(xlabel="", ylabel="")
fig.savefig(FIG / "07_contract_internet_heatmap.png"); plt.close(fig)

# 4. Statistical tests
def cramers_v(x, y):
    ct = pd.crosstab(x, y); chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return chi2, p, v
cats = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaperlessBilling","PaymentMethod","auto_pay"]
rows = []
for c in cats:
    chi2, p, v = cramers_v(df[c], df.Churn); rows.append((c, chi2, p, v))
assoc = pd.DataFrame(rows, columns=["feature","chi2","p_raw","cramers_v"])
m = len(assoc); assoc = assoc.sort_values("p_raw")
assoc["p_holm"] = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * assoc.p_raw.values))
assoc = assoc.sort_values("cramers_v", ascending=False)
assoc.to_csv(OUT / "association_tests.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(data=assoc, y="feature", x="cramers_v", ax=ax, color=sns.color_palette()[0])
ax.set(title="Association with churn (Cramer's V)", xlabel="Effect size", ylabel="")
fig.savefig(FIG / "08_association_strength.png"); plt.close(fig)
res["assoc_top5"] = assoc.head(5)[["feature","cramers_v"]].round(3).to_dict("records")
res["not_significant_holm"] = assoc[assoc.p_holm > .05].feature.tolist()

# continuous: Mann-Whitney + rank-biserial
for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    a, b = df.loc[df.Churn == 1, c], df.loc[df.Churn == 0, c]
    u, p = stats.mannwhitneyu(a, b); rb = 2 * u / (len(a) * len(b)) - 1
    res[f"mwu_{c}"] = {"median_churn": float(a.median()), "median_stay": float(b.median()),
                       "p": float(p), "rank_biserial": round(rb, 3)}

# 5. Effect with confidence interval: month-to-month vs contracted (risk ratio + bootstrap-free Wald CI)
mtm = df.Contract == "Month-to-month"
a, n1, c_, n2 = df[mtm].Churn.sum(), mtm.sum(), df[~mtm].Churn.sum(), (~mtm).sum()
rr = (a / n1) / (c_ / n2); se = np.sqrt(1/a - 1/n1 + 1/c_ - 1/n2)
res["risk_ratio_mtm"] = {"rr": round(rr, 2), "ci95": [round(np.exp(np.log(rr) - 1.96*se), 2), round(np.exp(np.log(rr) + 1.96*se), 2)]}

# 6. Revenue at risk
churn_rev = df[df.Churn == 1].MonthlyCharges.sum()
res["monthly_revenue_lost_to_churn"] = round(float(churn_rev), 0)
res["share_of_monthly_revenue_lost"] = round(float(churn_rev / df.MonthlyCharges.sum()), 3)
seg = df.groupby(["Contract","InternetService"]).agg(customers=("Churn","size"), churners=("Churn","sum"),
        churn_rate=("Churn","mean"), mrr_lost=("MonthlyCharges", lambda s: s[df.loc[s.index,"Churn"]==1].sum())).reset_index()
seg["share_of_mrr_lost"] = seg.mrr_lost / seg.mrr_lost.sum()
seg = seg.sort_values("mrr_lost", ascending=False); seg.to_csv(OUT / "segments_revenue_at_risk.csv", index=False)
res["top_segment_mrr"] = seg.head(3).round(3).to_dict("records")

# 7. Hidden-confounder check: tenure-controlled fiber effect
fiber = df.groupby(["tenure_bucket","InternetService"], observed=True).Churn.mean().unstack()
res["fiber_vs_dsl_by_tenure"] = fiber[["DSL","Fiber optic"]].round(3).to_dict("index")
(OUT / "eda_results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
