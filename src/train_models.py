"""
src/train_models.py — STEP 6: train XGBoost + LightGBM (classifier + regressor)
Run from project root:  python src/train_models.py
Saves into models/: xgb_classifier.json, lgbm_classifier.txt, xgb_regressor.json,
                    lgbm_regressor.txt, metrics.json, feature_list.json, feature_importance.csv
"""
import json, os
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, root_mean_squared_error)
import xgboost as xgb
import lightgbm as lgb

SAMPLE = None  # set to 1_000_000 for a fast test run on a slow laptop
TARGET_CLF = "DELAYED_15"
TARGET_REG = "ARRIVAL_DELAY"

def main():
    os.makedirs("models", exist_ok=True)
    train = pd.read_parquet("data/processed/train.parquet")
    test  = pd.read_parquet("data/processed/test.parquet")
    if SAMPLE:
        train = train.sample(n=SAMPLE, random_state=42)
        print(f"Quick run: training on {SAMPLE:,} sampled rows")
    feats = [c for c in train.columns if c not in (TARGET_CLF, TARGET_REG)]
    with open("models/feature_list.json", "w") as f:
        json.dump(feats, f)
    X_train, X_test = train[feats], test[feats]
    yc_train, yc_test = train[TARGET_CLF], test[TARGET_CLF]
    yr_train, yr_test = train[TARGET_REG], test[TARGET_REG]
    print(f"features: {len(feats)}   train: {len(train):,}   test: {len(test):,}")

    pos = int(yc_train.sum()); neg = len(yc_train) - pos
    spw = neg / pos
    print(f"class balance pos={pos:,} neg={neg:,} -> scale_pos_weight={spw:.2f}")
    metrics = {}
    common = dict(n_estimators=300, max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)

    # 1) XGBoost CLASSIFIER
    xgb_clf = xgb.XGBClassifier(**common, scale_pos_weight=spw, tree_method="hist", eval_metric="auc")
    print("\n[1/4] Training XGBoost classifier..."); xgb_clf.fit(X_train, yc_train)
    p = xgb_clf.predict_proba(X_test)[:, 1]
    metrics["xgb_classifier"] = {"roc_auc": round(roc_auc_score(yc_test, p), 4),
        "precision@0.5": round(precision_score(yc_test, p > 0.5), 4),
        "recall@0.5": round(recall_score(yc_test, p > 0.5), 4),
        "f1@0.5": round(f1_score(yc_test, p > 0.5), 4)}
    xgb_clf.save_model("models/xgb_classifier.json")

    # 2) LightGBM CLASSIFIER
    lgb_clf = lgb.LGBMClassifier(**common, scale_pos_weight=spw)
    print("[2/4] Training LightGBM classifier..."); lgb_clf.fit(X_train, yc_train)
    p = lgb_clf.predict_proba(X_test)[:, 1]
    metrics["lgbm_classifier"] = {"roc_auc": round(roc_auc_score(yc_test, p), 4),
        "precision@0.5": round(precision_score(yc_test, p > 0.5), 4),
        "recall@0.5": round(recall_score(yc_test, p > 0.5), 4),
        "f1@0.5": round(f1_score(yc_test, p > 0.5), 4)}
    lgb_clf.booster_.save_model("models/lgbm_classifier.txt")

    # 3) XGBoost REGRESSOR
    xgb_reg = xgb.XGBRegressor(**common, tree_method="hist", objective="reg:squarederror")
    print("[3/4] Training XGBoost regressor..."); xgb_reg.fit(X_train, yr_train)
    pr = xgb_reg.predict(X_test)
    metrics["xgb_regressor"] = {"mae_min": round(mean_absolute_error(yr_test, pr), 3),
                                "rmse_min": round(root_mean_squared_error(yr_test, pr), 3)}
    xgb_reg.save_model("models/xgb_regressor.json")

    # 4) LightGBM REGRESSOR
    lgb_reg = lgb.LGBMRegressor(**common)
    print("[4/4] Training LightGBM regressor..."); lgb_reg.fit(X_train, yr_train)
    pr = lgb_reg.predict(X_test)
    metrics["lgbm_regressor"] = {"mae_min": round(mean_absolute_error(yr_test, pr), 3),
                                 "rmse_min": round(root_mean_squared_error(yr_test, pr), 3)}
    lgb_reg.booster_.save_model("models/lgbm_regressor.txt")

    imp = pd.Series(xgb_clf.feature_importances_, index=feats).sort_values(ascending=False)
    imp.to_csv("models/feature_importance.csv")
    print("\nTop 10 most important features:\n", imp.head(10).round(4).to_string())

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved -> models/metrics.json")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()