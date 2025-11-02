"""
Train ML models for AEB using data collected from MetaDrive.

- Collects dataset across multiple episodes.
- Trains RandomForest classifier (brake flag) and regressor (brake value).
- Saves to rf_class.pkl and rf_reg.pkl at project root.

This script requires MetaDrive and rendering dependencies. Run in an environment where
MetaDrive can be imported.
"""
import os
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Try import MetaDrive lazily and fail with a friendly message if unavailable
try:
    from metadrive import MetaDriveEnv
except Exception as e:
    MetaDriveEnv = None  # type: ignore
    _md_error = e
else:
    _md_error = None


@dataclass
class MDConfig:
    use_render: bool = False
    accident_prob: float = 0.5
    traffic_density: float = 0.25
    num_scenarios: int = 100
    start_seed: int = 0


def _ensure_metadrive() -> None:
    if MetaDriveEnv is None:
        raise RuntimeError(
            f"MetaDrive is not available. Please install it (pip install git+https://github.com/metadriverse/metadrive.git).\n"
            f"Underlying error: {_md_error}"
        )


def collect_data(env: "MetaDriveEnv", num_episodes: int, steps_per_ep: int) -> pd.DataFrame:
    rows = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        for step in range(steps_per_ep):
            action = [0.0, 0.8, 0.0]  # steer, throttle, brake
            obs, reward, terminated, truncated, info = env.step(action)

            # Ego
            ego = env.agent
            ego_speed = float(np.linalg.norm(ego.speed))
            ego_pos = ego.position
            ego_lane = ego.lane_index

            # Lead search (same lane, ahead)
            min_dist, lead_speed = float("inf"), 0.0
            for v in env.engine.traffic_manager.vehicles:
                if v.id == ego.id:
                    continue
                if v.lane_index != ego_lane:
                    continue
                dx, dy = v.position - ego_pos
                if dx > 0:
                    dist = float(np.linalg.norm([dx, dy]))
                    if dist < min_dist:
                        min_dist = dist
                        lead_speed = float(np.linalg.norm(v.speed))

            if np.isfinite(min_dist):
                rel_speed = ego_speed - lead_speed
                ttc = min_dist / max(1e-3, rel_speed) if rel_speed > 0 else float("inf")
                brake_flag = int((min_dist < 10) or (ttc < 2))

                max_decel = 6.8
                req_decel = (rel_speed ** 2) / (2 * min_dist) if rel_speed > 0 else 0.0
                brake_value = float(min(1.0, req_decel / max_decel))

                rows.append([ep, step, ego_speed, rel_speed, min_dist, ttc, brake_flag, brake_value])

            if terminated or truncated:
                break

    df = pd.DataFrame(rows, columns=[
        "episode", "step", "ego_speed", "rel_speed", "distance", "ttc", "brake_flag", "brake_value"
    ])
    return df


def train_models(df: pd.DataFrame) -> Tuple[RandomForestClassifier, RandomForestRegressor]:
    Xc = df[["ego_speed", "rel_speed", "distance"]]
    yc = df["brake_flag"]
    Xr = df[["ego_speed", "rel_speed", "distance"]]
    yr = df["brake_value"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42)

    clf.fit(Xc_train, yc_train)
    reg.fit(Xr_train, yr_train)

    # Simple reporting
    yc_pred = clf.predict(Xc_test)
    print("\nClassification Report (RF):\n", classification_report(yc_test, yc_pred))
    print("Confusion Matrix:\n", confusion_matrix(yc_test, yc_pred))

    yr_pred = reg.predict(Xr_test)
    print("\nRegression Metrics (RF):")
    print("MSE:", mean_squared_error(yr_test, yr_pred))
    print("R2:", r2_score(yr_test, yr_pred))

    return clf, reg


def main():
    parser = argparse.ArgumentParser(description="Train AEB models using MetaDrive data")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=150, help="Steps per episode")
    parser.add_argument("--out_clf", type=str, default="rf_class.pkl", help="Output path for classifier")
    parser.add_argument("--out_reg", type=str, default="rf_reg.pkl", help="Output path for regressor")
    args = parser.parse_args()

    _ensure_metadrive()

    # Headless-friendly
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    cfg = dict(
        use_render=False,
        accident_prob=0.5,
        traffic_density=0.25,
        num_scenarios=max(100, args.episodes),
        start_seed=0,
    )

    env = MetaDriveEnv(cfg)
    try:
        print(f"Collecting data: episodes={args.episodes}, steps={args.steps} ...")
        df = collect_data(env, num_episodes=args.episodes, steps_per_ep=args.steps)
        print("Dataset shape:", df.shape)

        print("Training models ...")
        clf, reg = train_models(df)

        print(f"Saving models -> {args.out_clf}, {args.out_reg}")
        joblib.dump(clf, args.out_clf)
        joblib.dump(reg, args.out_reg)
        print("✅ Done.")
    finally:
        try:
            from metadrive.engine.engine_utils import close_engine
            close_engine()
        except Exception:
            pass


if __name__ == "__main__":
    main()