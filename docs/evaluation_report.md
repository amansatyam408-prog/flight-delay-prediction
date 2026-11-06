# Evaluation Report (test set = 1,142,802 flights, 20% holdout)

## Classification — P(delay > 15 min)

| Model | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|
| XGBoost | 0.7542 | 0.3336 | 0.6626 | 0.4437 |
| LightGBM | 0.7317 | 0.3102 | 0.6565 | 0.4213 |

XGBoost best threshold (Youden's J): {'threshold': 0.498, 'precision': 0.3319, 'recall': 0.6666, 'f1': 0.4432}

## Regression — predicted delay minutes

| Model | MAE (min) | RMSE (min) |
|---|---|---|
| XGBoost | 18.863 | 35.865 |
| LightGBM | 19.421 | 36.601 |

## Figures
- roc_curve.png, confusion_matrix.png, feature_importance.png
- actual_vs_predicted.png, delay_by_hour.png, delay_by_month.png, delay_by_airline.png

## Notes
- No target leakage: features use only pre-departure info (schedule, route, airline, weather, traffic).
- Class imbalance 82/18 handled with scale_pos_weight; ROC-AUC is the primary metric.
- Weather available for top-30 origin airports (62% of flights); rest use median-filled weather (has_weather flag).