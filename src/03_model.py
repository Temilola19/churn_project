"""Modeling: baseline -> logistic -> gradient boosting; CV, calibration, explainability, profit-based threshold."""
import json, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
                             precision_recall_curve, precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from config import CLEAN, FIG, OUT, SEED, OFFER_COST, SAVE_RATE, LIFETIME_MONTHS
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk"); plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

df = pd.read_csv(CLEAN)
num = ["tenure", "MonthlyCharges", "TotalCharges", "n_addons", "n_streaming", "charge_drift"]
cat = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService",
       "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
       "Contract","PaperlessBilling","PaymentMethod"]
X, y = df[num + cat], df.Churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)  # test set untouched until the end
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def prep(scale):
    return ColumnTransformer([("n", StandardScaler() if scale else "passthrough", num),
                              ("c", OneHotEncoder(drop="if_binary" if scale else None, handle_unknown="ignore"), cat)])
models = {
    "Baseline (prior)": Pipeline([("p", prep(True)), ("m", DummyClassifier(strategy="prior"))]),
    "Logistic regression": Pipeline([("p", prep(True)), ("m", LogisticRegression(C=0.3, max_iter=2000))]),
    "Gradient boosting": Pipeline([("p", prep(False)), ("m", HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=250, max_depth=3, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED))]),
}

# ---- 1. cross-validation on TRAIN only
rows = []
for name, mdl in models.items():
    s = cross_validate(mdl, Xtr, ytr, cv=cv, scoring=["roc_auc", "average_precision"])
    rows.append({"model": name, "cv_auc": s["test_roc_auc"].mean(), "cv_auc_sd": s["test_roc_auc"].std(),
                 "cv_pr_auc": s["test_average_precision"].mean()})
cvtab = pd.DataFrame(rows).round(4); print(cvtab.to_string(index=False))

# ---- 2. fit on train, evaluate once on test
for m in models.values(): m.fit(Xtr, ytr)
pt = {n: m.predict_proba(Xte)[:, 1] for n, m in models.items()}
test_rows = [{"model": n, "auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
              "brier": brier_score_loss(yte, p)} for n, p in pt.items()]
testtab = pd.DataFrame(test_rows).round(4); print(testtab.to_string(index=False))
# bootstrap CI for the winner's AUC
best = cvtab.sort_values("cv_auc").iloc[-1].model  # chosen on CV only; test set stays untouched
rng = np.random.default_rng(SEED); yv = yte.values; pb = pt[best]
boots = [roc_auc_score(yv[i], pb[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000))]
auc_ci = np.percentile(boots, [2.5, 97.5]).round(3).tolist()

# ---- 3. ROC / PR / calibration
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for n in ["Logistic regression", "Gradient boosting"]:
    fpr, tpr, _ = roc_curve(yte, pt[n]); ax[0].plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(yte, pt[n]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, pt[n]); ax[1].plot(rc, pr, label=n)
    fp, mp = calibration_curve(yte, pt[n], n_bins=8, strategy="quantile"); ax[2].plot(mp, fp, "o-", label=n)
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend(fontsize=11)
ax[1].axhline(yte.mean(), color="k", ls="--", lw=1); ax[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
ax[2].plot([0, 1], [0, 1], "k--", lw=1); ax[2].set(title="Calibration", xlabel="Predicted probability", ylabel="Observed churn rate"); ax[2].legend(fontsize=11)
fig.savefig(FIG / "09_model_performance.png"); plt.close(fig)

# ---- 4. Explainability
lr = models["Logistic regression"]
names = lr.named_steps["p"].get_feature_names_out()
coef = pd.Series(lr.named_steps["m"].coef_[0], index=[n.split("__")[1] for n in names])
top = pd.concat([coef.nlargest(8), coef.nsmallest(8)]).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top.index, np.exp(top.values), color=["#2a9d8f" if v < 0 else "#e76f51" for v in top.values])
ax.axvline(1, color="k", lw=1); ax.set(title="Odds ratios (logistic regression)", xlabel="Odds ratio (>1 raises churn risk; numeric = per 1 SD)")
fig.savefig(FIG / "10_odds_ratios.png"); plt.close(fig)
pi = permutation_importance(models["Gradient boosting"], Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(9, 6)); imp.plot.barh(ax=ax, color=sns.color_palette()[0])
ax.set(title="Permutation importance (gradient boosting)", xlabel="Drop in test AUC when shuffled")
fig.savefig(FIG / "11_permutation_importance.png"); plt.close(fig)

# ---- 5. Business value: choose contact threshold by expected profit (picked on out-of-fold train preds, scored on test)
def profit(y_true, p, mrr, t):
    sel = p >= t
    gain = (y_true[sel] * SAVE_RATE * mrr[sel] * LIFETIME_MONTHS).sum()
    return gain - OFFER_COST * sel.sum()
oof = cross_val_predict(models[best], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0, .9, 91)
oof_profit = [profit(ytr.values, oof, Xtr.MonthlyCharges.values, t) for t in ths]
t_star = float(ths[int(np.argmax(oof_profit))])
p_best, mrr_te = pt[best], Xte.MonthlyCharges.values
prof_te = [profit(yv, p_best, mrr_te, t) for t in ths]
strategies = {
    "Do nothing": 0.0,
    "Contact everyone": profit(yv, np.ones(len(yv)), mrr_te, .5),
    f"Model-targeted (t={t_star:.2f})": profit(yv, p_best, mrr_te, t_star),
}
sel = p_best >= t_star
op = {"threshold": t_star, "contacted_pct": round(float(sel.mean()), 3),
      "precision": round(precision_score(yv, sel), 3), "recall": round(recall_score(yv, sel), 3),
      "f1": round(f1_score(yv, sel), 3)}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].plot(ths, prof_te, color="#2a9d8f", lw=2.5); ax[0].axvline(t_star, color="crimson", ls="--", label=f"chosen t={t_star:.2f}")
ax[0].set(title="Expected campaign profit vs. threshold (test set)", xlabel="Contact if P(churn) >=", ylabel="Profit ($)"); ax[0].legend()
ax[1].bar(list(strategies), list(strategies.values()), color=["#adb5bd", "#e9c46a", "#2a9d8f"])
for i, v in enumerate(strategies.values()): ax[1].text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=12)
ax[1].set(title="Retention strategy comparison (test set)", ylabel="Profit ($)"); plt.setp(ax[1].get_xticklabels(), rotation=12)
fig.savefig(FIG / "12_profit_threshold.png"); plt.close(fig)

# decile lift
d = pd.DataFrame({"p": p_best, "y": yv}); d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decile", observed=True).y.mean() / yv.mean()
fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(lift.index.astype(str), lift.values, color=sns.color_palette()[0]); ax.axhline(1, color="k", ls="--")
ax.set(title="Lift by risk decile (1 = highest risk)", xlabel="Decile", ylabel="Churn rate / baseline")
fig.savefig(FIG / "13_lift_by_decile.png"); plt.close(fig)

out = {"cv": cvtab.to_dict("records"), "test": testtab.to_dict("records"), "best_model": best, "best_auc_ci95": auc_ci,
       "operating_point": op, "profit_by_strategy": {k: round(v, 0) for k, v in strategies.items()},
       "top_decile_lift": round(float(lift.iloc[0]), 2), "top2_deciles_capture": round(float(d[d.decile.astype(int) <= 2].y.sum() / yv.sum()), 3),
       "top_risk_drivers": coef.nlargest(5).round(2).to_dict(), "top_protective": coef.nsmallest(5).round(2).to_dict(),
       "perm_importance_top5": imp.tail(5).round(4).to_dict(), "test_n": len(yv)}
(OUT / "model_results.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
