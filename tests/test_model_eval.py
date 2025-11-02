import os
import math
import numpy as np
import pytest
import joblib

# Project paths
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PROJ_DIR))

CLF_PATH = os.path.join(ROOT_DIR, "rf_class.pkl")
REG_PATH = os.path.join(ROOT_DIR, "rf_reg.pkl")


def _heuristic_flag_and_value(ego_speed: float, rel_speed: float, distance: float):
    # Matches the rule used in training and fallback
    rel_now = max(0.0, rel_speed)
    ttc = (distance / rel_now) if rel_now > 0 else math.inf
    flag = int((distance < 10.0) or (ttc < 2.0))
    max_decel = 6.8
    req_decel = (rel_now ** 2) / (2.0 * max(distance, 1e-3)) if rel_now > 0 else 0.0
    value = float(min(1.0, req_decel / max_decel))
    return flag, value


@pytest.mark.skipif(not os.path.exists(CLF_PATH) or not os.path.exists(REG_PATH),
                    reason="Trained model files not found: rf_class.pkl and/or rf_reg.pkl")
def test_model_eval_classification_and_regression():
    print("\n=== Starting Model Evaluation ===")
    print(f"Classifier: {CLF_PATH}")
    print(f"Regressor: {REG_PATH}")
    
    # Load models directly
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print("\nModel metadata:")
    print(f"Classifier: {clf.__class__.__name__} (n_estimators={clf.n_estimators})")
    print(f"Regressor: {reg.__class__.__name__} (n_estimators={reg.n_estimators})")

    # Build synthetic evaluation grid
    ego_speeds = np.linspace(0, 40, 5)        # 0..40 m/s (reduced for readability)
    rel_speeds = np.linspace(-10, 20, 5)      # -10..20 m/s (reduced for readability)
    distances = np.linspace(5, 80, 5)         # 5..80 m (reduced for readability)

    X = []
    y_flag = []
    y_val = []
    for v in ego_speeds:
        for r in rel_speeds:
            for d in distances:
                f, val = _heuristic_flag_and_value(float(v), float(r), float(d))
                X.append([float(v), float(r), float(d)])
                y_flag.append(f)
                y_val.append(val)
    X = np.array(X)
    y_flag = np.array(y_flag)
    y_val = np.array(y_val)

    print("\nSample test cases:")
    print("Ego Speed | Rel Speed | Distance | True Flag | True Value")
    print("-" * 60)
    for i in range(min(5, len(X))):
        print(f"{X[i][0]:9.1f} | {X[i][1]:9.1f} | {X[i][2]:8.1f} | {y_flag[i]:9d} | {y_val[i]:.4f}")
    if len(X) > 5:
        print(f"... and {len(X)-5} more test cases")

    # Predictions
    print("\nMaking predictions...")
    y_flag_pred = clf.predict(X).astype(int)
    y_val_pred = reg.predict(X).astype(float)

    # Metrics
    acc = (y_flag_pred == y_flag).mean()
    mae = np.abs(y_val_pred - y_val).mean()
    mse = ((y_val_pred - y_val) ** 2).mean()
    
    # Classification report
    from sklearn.metrics import classification_report
    print("\n=== Classification Report ===")
    print(classification_report(y_flag, y_flag_pred, target_names=["No Brake", "Brake"]))
    
    # Regression metrics
    print("\n=== Regression Metrics ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    
    # Print some prediction examples
    print("\n=== Prediction Examples ===")
    print("Ego Speed | Rel Speed | Distance | True Flag | Pred Flag | True Value | Pred Value")
    print("-" * 85)
    for i in range(min(5, len(X))):
        print(f"{X[i][0]:9.1f} | {X[i][1]:9.1f} | {X[i][2]:8.1f} | "
              f"{y_flag[i]:9d} | {y_flag_pred[i]:9d} | {y_val[i]:10.4f} | {y_val_pred[i]:.4f}")

    # Assertions with thresholds
    print("\n=== Test Results ===")
    print(f"Classification accuracy: {acc:.3f} (threshold: >= 0.75)")
    print(f"Regression MAE: {mae:.4f} (threshold: <= 0.15)")
    print(f"Regression MSE: {mse:.4f} (threshold: <= 0.05)")
    
    assert acc >= 0.75, f"Classification accuracy too low: {acc:.3f}"
    assert mae <= 0.15, f"Regression MAE too high: {mae:.3f}"
    assert mse <= 0.05, f"Regression MSE too high: {mse:.3f}"
    
    print("\n✅ All tests passed!")