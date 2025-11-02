import os

# App-level configuration flags
HEADLESS = os.environ.get("AEB_HEADLESS", "1") == "1"

# MetaDrive
METADRIVE_CONFIG = {
    "use_render": False,
    "accident_prob": 0.5,
    "traffic_density": 0.25,
    "num_scenarios": 50,
}