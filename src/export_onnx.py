"""
src/export_onnx.py — STEP 11a: convert trained XGBoost models to ONNX.
Gotchas handled (both verified): skl2onnx>=1.17 dropped its XGBoost converter ->
use onnxmltools.convert_xgboost on a Booster from file (opset 15, onnxmltools'
own FloatTensorType); booster feature names must be 'f0','f1',... -> renamed on
a SEPARATE booster so the sklearn wrapper is untouched.
Run:  python src/export_onnx.py
"""
import json
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OXFloatTensor

def export_one(json_path, onnx_path, n_features):
    booster = xgb.Booster()
    booster.load_model(json_path)                                # separate booster
    booster.feature_names = [f"f{i}" for i in range(n_features)] # f0, f1, ...
    initial_types = [("float_input", OXFloatTensor([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_types, target_opset=15)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported {json_path} -> {onnx_path}")

def main():
    feats = json.load(open("models/feature_list.json"))
    n = len(feats)
    print(f"Exporting ONNX with {n} input features...")
    export_one("models/xgb_classifier.json", "models/xgb_classifier.onnx", n)
    export_one("models/xgb_regressor.json",  "models/xgb_regressor.onnx", n)

if __name__ == "__main__":
    main()