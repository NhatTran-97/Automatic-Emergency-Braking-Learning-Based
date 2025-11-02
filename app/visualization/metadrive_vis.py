import os
from typing import Tuple
import multiprocessing as mp
import threading

# Guard MetaDrive import to allow environments without it
try:
    from metadrive import MetaDriveEnv
    import numpy as np
    import cv2
    import imageio
    METADRIVE_OK = True
except Exception:
    MetaDriveEnv = None  # type: ignore
    METADRIVE_OK = False

from ..app_config import METADRIVE_CONFIG
from ..model.aeb_model import AEBModel


def metadrive_available() -> bool:
    return METADRIVE_OK and MetaDriveEnv is not None


def _ensure_env(use_3d: bool = False):
    if not metadrive_available():
        raise RuntimeError("MetaDrive is not available.")
    # Headless-friendly defaults: only force dummy for 2D top-down
    if not use_3d:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    cfg = dict(METADRIVE_CONFIG)
    cfg["use_render"] = bool(use_3d)
    return MetaDriveEnv(cfg)


def run_episode_with_predictions(ep_idx: int = 11, steps: int = 150, use_3d: bool = False) -> Tuple[str, str]:
    """Run a MetaDrive episode with AEB model in closed-loop.
       Produces two GIFs: raw scene and prediction overlay.
       Returns (raw_gif_path, pred_gif_path).
    """
    # If called from a non-main thread (e.g., Gradio callback), run in a subprocess
    if threading.current_thread() is not threading.main_thread():
        q: mp.Queue = mp.Queue()
        p = mp.Process(target=_episode_worker, args=(int(ep_idx), int(steps), bool(use_3d), q))
        p.start()
        raw_gif, pred_gif, err = q.get()
        p.join()
        if err:
            raise RuntimeError(err)
        return raw_gif, pred_gif

    env = _ensure_env(use_3d=use_3d)
    obs, info = env.reset(seed=int(ep_idx))

    frames, annotated_frames = [], []
    model = AEBModel()

    for _ in range(int(steps)):
        # --- Ego vehicle state ---
        ego = env.agent
        ego_speed = float(np.linalg.norm(ego.speed))
        ego_pos = ego.position
        ego_lane = ego.lane_index

        # --- Find nearest lead vehicle in same lane ---
        min_dist, lead_speed = float("inf"), 0.0
        for v in env.engine.traffic_manager.vehicles:
            if v.id == ego.id:
                continue
            if v.lane_index != ego_lane:
                continue
            dx, dy = v.position - ego_pos
            if dx > 0:  # vehicle ahead only
                dist = float(np.linalg.norm([dx, dy]))
                if dist < min_dist:
                    min_dist = dist
                    lead_speed = float(np.linalg.norm(v.speed))

        # --- Decide action (CLOSED-LOOP) ---
        if np.isfinite(min_dist):
            rel_speed = ego_speed - lead_speed
            flag, val = model.predict(ego_speed, rel_speed, min_dist)

            if flag == 1:  # Brake
                action = [0.0, 0.0, float(np.clip(val, 0, 1))]
            else:          # Safe → throttle
                action = [0.0, 0.8, 0.0]
        else:
            # No lead vehicle
            action = [0.0, 0.8, 0.0]

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Render frame ---
        try:
            if use_3d:
                frame_bgr = env.render(mode="rgb_array")
            else:
                frame_bgr = env.render(mode="top_down", screen_size=(500, 500), screen_record=True)
        except Exception:
            frame_bgr = env.render(mode="top_down", screen_size=(500, 500), screen_record=True)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Annotated overlay ---
        overlay = frame_rgb.copy()
        if np.isfinite(min_dist):
            status = "BRAKE" if flag == 1 else "SAFE"
            color = (255, 0, 0) if flag == 1 else (0, 255, 0)
            cv2.putText(overlay, f"{status} | Brake={val:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"Dist={min_dist:.1f} | Ego={ego_speed:.1f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(overlay, "No lead vehicle", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        frames.append(frame_rgb)
        annotated_frames.append(overlay)

        if terminated or truncated:
            break

    # --- Save GIFs ---
    raw_gif = "scene.gif"
    try:
        env.top_down_renderer.generate_gif(raw_gif)
    except Exception:
        imageio.mimsave(raw_gif, frames, fps=10)

    pred_gif = "scene_pred.gif"
    imageio.mimsave(pred_gif, annotated_frames, fps=10)

    # --- Clean up ---
    try:
        from metadrive.engine.engine_utils import close_engine
        close_engine()
    except Exception:
        pass

    return raw_gif, pred_gif


def _episode_worker(ep_idx: int, steps: int, use_3d: bool, q: "mp.Queue") -> None:
    """Worker process to generate GIFs and send results via queue."""
    try:
        raw, pred = run_episode_with_predictions(ep_idx=ep_idx, steps=steps, use_3d=use_3d)
        q.put((raw, pred, None))
    except Exception as e:
        q.put((None, None, str(e)))
