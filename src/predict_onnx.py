"""
src/predict_onnx.py — STEP 11b: onnxruntime inference + equivalence check vs XGBoost.
Run:  python src/predict_onnx.py     Output: models/onnx_comparison.json
"""
import json
import numpy as np
import pandas as pd
import onnxruntime as rt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score

N = 200_000  # compare on first 200k test rows

def main():
    test = pd.read_parquet("data/processed/test.parquet")
    feats = json.load(open("models/feature_list.json"))
    X   = test[feats].head(N).astype(np.float32)    # ONNX input is float32
    X64 = test[feats].head(N).astype(np.float64)    # native xgboost
    yc = test["DELAYED_15"].head(N).to_numpy()
    yr = test["ARRIVAL_DELAY"].head(N).to_numpy()

    sess_c = rt.InferenceSession("models/xgb_classifier.onnx", providers=["CPUExecutionProvider"])
    sess_r = rt.InferenceSession("models/xgb_regressor.onnx",  providers=["CPUExecutionProvider"])
    out_c = sess_c.run(None, {"float_input": X.to_numpy()})
    probs = out_c[1]                                 # classifier outputs [label, probabilities Nx2]
    p_onnx = probs[:, 1] if isinstance(probs, np.ndarray) else np.array([d[1] for d in probs])
    r_onnx = sess_r.run(None, {"float_input": X.to_numpy()})[0].ravel()

    clf = xgb.XGBClassifier(); clf.load_model("models/xgb_classifier.json")
    reg = xgb.XGBRegressor();  reg.load_model("models/xgb_regressor.json")
    p_xgb = clf.predict_proba(X64)[:, 1]
    r_xgb = reg.predict(X64)

    report = {
        "rows_compared": N,
        "classifier_max_abs_prob_diff": float(np.max(np.abs(p_onnx - p_xgb))),
        "classifier_roc_auc_onnx": round(roc_auc_score(yc, p_onnx), 6),
        "classifier_roc_auc_xgboost": round(roc_auc_score(yc, p_xgb), 6),
        "regressor_max_abs_min_diff": float(np.max(np.abs(r_onnx - r_xgb))),
        "regressor_mae_onnx_min": round(mean_absolute_error(yr, r_onnx), 4),
        "regressor_mae_xgboost_min": round(mean_absolute_error(yr, r_xgb), 4),
    }
    print(json.dumps(report, indent=2))
    with open("models/onnx_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved -> models/onnx_comparison.json")

    from src.predict import build_features
    row = build_features("WN", "DAL", "PHX", "2015-12-20", 1915, 2030, 135).astype(np.float32)
    out1 = sess_c.run(None, {"float_input": row.to_numpy()})
    probs1 = out1[1]
    p1 = float(probs1[0, 1] if isinstance(probs1, np.ndarray) else probs1[0][1])
    m1 = float(sess_r.run(None, {"float_input": row.to_numpy()})[0].ravel()[0])
    print(f"ONNX single flight WN DAL->PHX 2015-12-20: P={p1:.4f}  delay={m1:.1f} min")

if __name__ == "__main__":
    main()