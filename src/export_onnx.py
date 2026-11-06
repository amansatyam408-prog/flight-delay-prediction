"""
src/export_onnx.py — STEP 11a: convert trained XGBoost models to ONNX via onnxmltools
(skl2onnx >=1.17 removed its XGBoost converter, so we convert the Booster directly).
Run:  python src/export_onnx.py     Outputs: models/xgb_classifier.onnx, models/xgb_regressor.onnx
"""
import json
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OXFloatTensor

def main():
    feats = json.load(open("models/feature_list.json"))
    n = len(feats)
    print(f"Exporting ONNX with {n} input features...")
    initial_types = [("float_input", OXFloatTensor([None, n]))]   # onnxmltools' OWN tensor type!
    jobs = [("classifier", xgb.XGBClassifier(), "models/xgb_classifier.json"),
            ("regressor",  xgb.XGBRegressor(),  "models/xgb_regressor.json")]
    for name, model, path in jobs:
        model.load_model(path)
        booster = model.get_booster()                     # converter needs the Booster
        onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_types, target_opset=15)
        out = f"models/xgb_{name}.onnx"
        with open(out, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported -> {out}")

if __name__ == "__main__":
    main()