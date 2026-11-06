"""
src/export_onnx.py — STEP 11a: convert trained XGBoost models to ONNX.
Run:  python src/export_onnx.py     Outputs: models/xgb_classifier.onnx, models/xgb_regressor.onnx
"""
import json
import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    feats = json.load(open("models/feature_list.json"))
    n = len(feats)
    print(f"Exporting with {n} input features...")
    initial_types = [("float_input", FloatTensorType([None, n]))]
    jobs = [("classifier", xgb.XGBClassifier(), "models/xgb_classifier.json"),
            ("regressor", xgb.XGBRegressor(), "models/xgb_regressor.json")]
    for name, model, path in jobs:
        model.load_model(path)
        onnx_model = convert_sklearn(model, initial_types=initial_types)
        out = f"models/xgb_{name}.onnx"
        with open(out, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported -> {out}")

if __name__ == "__main__":
    main()