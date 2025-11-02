import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from typing import List

from ..model.aeb_model import AEBModel

_model = AEBModel()


def aeb_demo(ego_speed: float, rel_speed: float, distance: float) -> str:
    """Generate a GIF simulating ego/lead along a straight road with closed-loop AEB braking.

    Returns a filepath to the saved GIF.
    """
    lead_speed = ego_speed - rel_speed
    dt, steps = 0.1, 150
    ego_x, lead_x = 10.0, 10.0 + distance
    frames: List[np.ndarray] = []

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for _ in range(steps):
        # Distance and relative speed
        dist = max(0.0, lead_x - ego_x)
        rel_now = ego_speed - lead_speed

        # --- Predict using AEB model ---
        flag, val = _model.predict(ego_speed, rel_now, dist)

        # --- Closed-loop ego update ---
        if flag == 1:  # Brake predicted
            decel = float(val) * 6.8  # max decel ≈ 6.8 m/s²
            ego_speed = max(0.0, ego_speed - decel * dt)

        # Update positions based on new speeds
        ego_x += ego_speed * dt
        lead_x += lead_speed * dt

        # --- Visualization ---
        W, H = 640, 160
        img = Image.new("RGB", (W, H), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, H//3, W, 2*H//3], fill=(50, 50, 50))
        for x in range(0, W, 40):
            draw.line([(x, H//2), (x+20, H//2)], fill=(200, 200, 200), width=2)

        world_to_img = lambda wx: int((W*0.2) + (wx - ego_x) * 5.0)
        ego_img_x, lead_img_x = world_to_img(ego_x), world_to_img(lead_x)

        # Draw ego (blue) and lead (green/red depending on brake status)
        draw.rectangle([ego_img_x-12, H//2-18, ego_img_x+12, H//2+18], fill=(0, 120, 255))
        color = (255, 0, 0) if flag == 1 else (0, 200, 0)
        draw.rectangle([lead_img_x-12, H//2-18, lead_img_x+12, H//2+18], fill=color)

        # Text overlays
        draw.text((10, 10), f"Dist: {dist:.1f} m", fill=(255, 255, 255), font=font)
        draw.text((10, 30), f"Ego speed: {ego_speed:.1f} m/s", fill=(255, 255, 255), font=font)
        draw.text((10, 50), f"Rel speed: {rel_now:.1f} m/s", fill=(255, 255, 255), font=font)
        draw.text((10, 70), f"Pred: {'BRAKE' if flag else 'SAFE'}", fill=color, font=font)
        draw.text((10, 90), f"Brake val: {val:.2f}", fill=(255, 255, 255), font=font)

        frames.append(np.array(img))

        # Stop if collision
        if dist <= 0:
            break

    gif_path = "aeb_demo.gif"
    imageio.mimsave(gif_path, frames, fps=10)
    return gif_path