"""
src/evaluate.py — STEP 7: evaluation deep-dive + figures for the PPT/dashboard.
Run from project root:  python src/evaluate.py
Saves PNGs to docs/figures/ and a metrics table to docs/evaluation_report.md
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save PNGs without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             mean_absolute_error, precision_score, recall_score, f1_score)
import joblib
import xgboost as xgb
import lightgbm as lgb

FIG = "docs/figures"
TARGET_CLF = "DELAYED_15"
TARGET_REG = "ARRIVAL_DELAY"

def main():
    os.makedirs(FIG, exist_ok=True)
    test = pd.read_parquet("data/processed/test.parquet")
    feats = json.load(open("models/feature_list.json"))
    X, yc, yr = test[feats], test[TARGET_CLF], test[TARGET_REG]

    xgb_clf = xgb.XGBClassifier(); xgb_clf.load_model("models/xgb_classifier.json")
    lgb_booster = lgb.Booster(model_file="models/lgbm_classifier.txt")   # LightGBM loads via Booster, not load_model
    xgb_reg = xgb.XGBRegressor(); xgb_reg.load_model("models/xgb_regressor.json")
    px = xgb_clf.predict_proba(X)[:, 1]
    pl = lgb_booster.predict(X)   # binary model -> returns P(delayed) directly
    rx = xgb_reg.predict(X)
    metrics = json.load(open("models/metrics.json"))

    # 1) ROC curve (both classifiers)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in [("XGBoost", px), ("LightGBM", pl)]:
        fpr, tpr, _ = roc_curve(yc, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(yc, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — predicting delay > 15 min"); ax.legend()
    fig.savefig(f"{FIG}/roc_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 2) best threshold (Youden's J) on XGBoost
    fpr, tpr, thr = roc_curve(yc, px)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    pb = (px >= best_thr).astype(int)
    best = {"threshold": round(best_thr, 3), "precision": round(precision_score(yc, pb), 4),
            "recall": round(recall_score(yc, pb), 4), "f1": round(f1_score(yc, pb), 4)}
    print("Best threshold (Youden's J):", best)

    # 3) confusion matrix @0.5
    cm = confusion_matrix(yc, px >= 0.5)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["not delayed", "delayed >15m"], yticklabels=["not delayed", "delayed >15m"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion matrix @ 0.5 (XGBoost)")
    fig.savefig(f"{FIG}/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 4) feature importance
    imp = pd.read_csv("models/feature_importance.csv", index_col=0).iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.tail(15).plot.barh(ax=ax, color="#1f77b4")
    ax.set_title("Top 15 features — XGBoost classifier")
    fig.savefig(f"{FIG}/feature_importance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 5) actual vs predicted delay (regressor, 50k sample)
    idx = np.random.RandomState(42).choice(len(test), 50_000, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yr.iloc[idx], rx[idx], s=4, alpha=0.3)
    ax.plot([-100, 300], [-100, 300], "r--")
    ax.set_xlim(-100, 300); ax.set_ylim(-100, 300)
    ax.set_xlabel("actual arrival delay (min)"); ax.set_ylabel("predicted delay (min)")
    ax.set_title(f"Regression: actual vs predicted (MAE = {mean_absolute_error(yr, rx):.1f} min)")
    fig.savefig(f"{FIG}/actual_vs_predicted.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 6) insights: delay rate by hour / month / airline
    test = test.copy(); test["pred_prob"] = px
    fig, ax = plt.subplots(figsize=(9, 4))
    test.groupby("SCHED_HOUR")[TARGET_CLF].mean().plot.bar(ax=ax, color="#d62728")
    ax.set_title("Delay rate (>15 min) by scheduled departure hour"); ax.set_xlabel("hour of day"); ax.set_ylabel("delay rate")
    fig.savefig(f"{FIG}/delay_by_hour.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    test.groupby("MONTH")[TARGET_CLF].mean().plot.bar(ax=ax, color="#ff7f0e")
    ax.set_title("Delay rate by month (2015)"); ax.set_xlabel("month"); ax.set_ylabel("delay rate")
    fig.savefig(f"{FIG}/delay_by_month.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    enc = joblib.load("models/encoders.pkl")
    inv = {i: c for c, i in enc["AIRLINE"].items()}
    airlines = pd.read_parquet("data/processed/airlines.parquet")
    name = {r["IATA_CODE"]: r["AIRLINE"] for _, r in airlines.iterrows()}
    test["AIRLINE_CODE"] = test["AIRLINE_enc"].map(inv)
    top10 = test["AIRLINE_CODE"].value_counts().head(10).index
    by_air = test[test["AIRLINE_CODE"].isin(top10)].groupby("AIRLINE_CODE")[TARGET_CLF].mean().sort_values()
    by_air.index = [name.get(c, c)[:25] for c in by_air.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    by_air.plot.barh(ax=ax, color="#2ca02c")
    ax.set_title("Delay rate by airline (top 10 by volume)"); ax.set_xlabel("delay rate")
    fig.savefig(f"{FIG}/delay_by_airline.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 7) metrics report (Markdown) for the PPT / README
    lines = ["# Evaluation Report (test set = 1,142,802 flights, 20% holdout)", "",
             "## Classification — P(delay > 15 min)", "",
             "| Model | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |", "|---|---|---|---|---|",
             f"| XGBoost | {metrics['xgb_classifier']['roc_auc']} | {metrics['xgb_classifier']['precision@0.5']} | {metrics['xgb_classifier']['recall@0.5']} | {metrics['xgb_classifier']['f1@0.5']} |",
             f"| LightGBM | {metrics['lgbm_classifier']['roc_auc']} | {metrics['lgbm_classifier']['precision@0.5']} | {metrics['lgbm_classifier']['recall@0.5']} | {metrics['lgbm_classifier']['f1@0.5']} |", "",
             f"XGBoost best threshold (Youden's J): {best}", "",
             "## Regression — predicted delay minutes", "",
             "| Model | MAE (min) | RMSE (min) |", "|---|---|---|",
             f"| XGBoost | {metrics['xgb_regressor']['mae_min']} | {metrics['xgb_regressor']['rmse_min']} |",
             f"| LightGBM | {metrics['lgbm_regressor']['mae_min']} | {metrics['lgbm_regressor']['rmse_min']} |", "",
             "## Figures", "- roc_curve.png, confusion_matrix.png, feature_importance.png",
             "- actual_vs_predicted.png, delay_by_hour.png, delay_by_month.png, delay_by_airline.png", "",
             "## Notes",
             "- No target leakage: features use only pre-departure info (schedule, route, airline, weather, traffic).",
             "- Class imbalance 82/18 handled with scale_pos_weight; ROC-AUC is the primary metric.",
             "- Weather available for top-30 origin airports (62% of flights); rest use median-filled weather (has_weather flag)."]
    with open("docs/evaluation_report.md", "w") as f:
        f.write("\n".join(lines))
    print("Saved -> docs/evaluation_report.md")
    print("figures:", sorted(os.listdir(FIG)))

if __name__ == "__main__":
    main()