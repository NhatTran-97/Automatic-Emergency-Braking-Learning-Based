import os
import gradio as gr

from .visualization.slider_vis import aeb_demo

# Optional MetaDrive integration is guarded
try:
    from .visualization.metadrive_vis import metadrive_available, run_episode_with_predictions
except Exception:
    def metadrive_available() -> bool:
        return False
    def run_episode_with_predictions(*args, **kwargs):
        raise RuntimeError("MetaDrive not available in this environment.")


def launch():
    """Build and launch the Gradio app with two tabs: Slider and MetaDrive.

    - Slider tab: always available; produces an AEB GIF based on parameters.
    - MetaDrive tab: available only if MetaDrive can be imported and initialized.
    """
    with gr.Blocks(title="Machine Learning Based AEB") as demo:
        gr.Markdown("""
        # 🚗 Automatic Emergency Braking (AEB)
        Two visualization modes are available:
        - Slider: Lightweight, pure-Python visualization using synthetic kinematics.
        - MetaDrive: Generates a top-down simulation episode and (optionally) overlays predictions.
        """)

        with gr.Tab("Slider"):
            speed = gr.Slider(0, 50, value=15, label="Ego Speed (m/s)")
            rel = gr.Slider(-30, 30, value=-5, label="Relative Speed (ego - lead) (m/s)")
            dist = gr.Slider(1, 100, value=25, label="Distance to Lead Vehicle (m)")
            out_preview = gr.Image(type="filepath", label="AEB Simulation Preview (GIF)")
            out_download = gr.File(label="Download AEB GIF")
            run_btn = gr.Button("Run Slider Demo")

            def _run_slider(s, r, d):
                path = aeb_demo(s, r, d)
                return path, path

            run_btn.click(fn=_run_slider, inputs=[speed, rel, dist], outputs=[out_preview, out_download])

        with gr.Tab("MetaDrive"):
            info = gr.Markdown(visible=True)
            ep = gr.Slider(0, 200, value=11, step=1, label="Episode Seed")
            steps = gr.Slider(10, 400, value=150, step=10, label="Steps")
            use_3d = gr.Checkbox(value=False, label="Use 3D renderer if available (GPU)")
            gen_raw = gr.Checkbox(value=False, label="Return raw scene GIF as well")
            md_preview = gr.Image(type="filepath", label="Prediction Preview (GIF)")
            md_out = gr.File(label="Download Prediction GIF")
            md_raw = gr.File(label="Download Raw GIF (optional)", visible=True)
            run_md = gr.Button("Run MetaDrive Episode")

            def _run_md(ep_idx, steps_count, return_raw, want_3d):
                if not metadrive_available():
                    return None, None, None, ("MetaDrive is not available in this environment.\n" \
                                              "Install MetaDrive and necessary rendering deps (Panda3D/OpenGL).")
                try:
                    # Ensure headless-friendly rendering only for 2D mode
                    if not bool(want_3d):
                        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
                    raw_gif, pred_gif = run_episode_with_predictions(ep_idx=int(ep_idx), steps=int(steps_count), use_3d=bool(want_3d))
                    if return_raw:
                        return pred_gif, pred_gif, raw_gif, "✅ Episode generated."
                    else:
                        return pred_gif, pred_gif, None, "✅ Episode generated."
                except Exception as e:
                    return None, None, None, f"❌ Error running MetaDrive: {e}"

            run_md.click(fn=_run_md, inputs=[ep, steps, gen_raw, use_3d], outputs=[md_preview, md_out, md_raw, info])

    # Return interface to allow programmatic launch in run_app.py
    return demo


if __name__ == "__main__":
    launch().launch()