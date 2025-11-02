import os
import joblib
import numpy as np
from typing import Optional, Tuple


MODEL_CLASS_PATH = os.environ.get("AEB_MODEL_CLASS", "rf_class.pkl")
MODEL_REG_PATH = os.environ.get("AEB_MODEL_REG", "rf_reg.pkl")


class AEBModel:
    """A thin wrapper around trained models with a rule-based fallback.

    Exposes:
    - predict(ego_speed, rel_speed, distance) -> (flag:int, value:float)
    """

    def __init__(self):
        self.clf = None
        self.reg = None
        self._load()

    def _load(self):
        try:
            if os.path.exists(MODEL_CLASS_PATH):
                self.clf = joblib.load(MODEL_CLASS_PATH)
            if os.path.exists(MODEL_REG_PATH):
                self.reg = joblib.load(MODEL_REG_PATH)
        except Exception:
            # If load fails, keep as None to use fallback
            self.clf, self.reg = None, None

    @staticmethod
    def _fallback(ego_speed: float, rel_speed: float, distance: float) -> Tuple[int, float]:
        # Simple TTC/distance based heuristic fallback
        rel_now = max(0.0, rel_speed)
        flag = int((distance < 10) or ((rel_now > 0) and (distance / max(1e-6, rel_now) < 2)))
        val = min(1.0, (rel_now ** 2) / (2 * max(1e-3, distance)) / 6.8)
        return flag, float(val)

    def predict(self, ego_speed: float, rel_speed: float, distance: float) -> Tuple[int, float]:
        if self.clf is not None and self.reg is not None:
            try:
                X = np.array([[ego_speed, rel_speed, distance]])
                flag = int(self.clf.predict(X)[0])
                val = float(self.reg.predict(X)[0])
                return flag, val
            except Exception:
                pass
        return self._fallback(ego_speed, rel_speed, distance)