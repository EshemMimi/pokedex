"""Local Gradio UI for Gen 1 Pokédex MVP. Uses a dummy predictor until a trained checkpoint exists."""

from __future__ import annotations

import os

# Gradio launch() probes localhost via httpx. Corporate NO_PROXY often lists "127.0.0.*" which
# httpx does not reliably match for 127.0.0.1, so traffic still uses HTTP_PROXY and returns 404.
_np = os.environ.get("NO_PROXY", "")
_bypass = "127.0.0.1,localhost,::1"
os.environ["NO_PROXY"] = f"{_bypass},{_np}" if _np else _bypass

import json
import random
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
LABEL_MAP_PATH = ROOT / "artifacts" / "label_map.json"


def load_label_map() -> list[dict]:
    if not LABEL_MAP_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {LABEL_MAP_PATH}. Run: python scripts/generate_label_map.py"
        )
    data = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = data["classes"]
    if len(classes) != data.get("num_classes", 151):
        raise ValueError("label_map.json: classes length does not match num_classes")
    return classes


class DummyPredictor:
    """Returns a random Top-3 over Gen 1 classes (replace with torch model + weights)."""

    def __init__(self, classes: list[dict]) -> None:
        self._classes = classes

    def predict(self, image) -> list[tuple[int, str, float]]:
        _ = image  # real model would preprocess tensor(batch)
        k = min(3, len(self._classes))
        indices = random.sample(range(len(self._classes)), k)
        raw = [random.random() for _ in range(k)]
        total = sum(raw)
        probs = [x / total for x in raw]
        rows: list[tuple[int, str, float]] = []
        for idx, p in zip(indices, probs, strict=True):
            c = self._classes[idx]
            rows.append((int(c["dex"]), str(c["name"]), float(p)))
        rows.sort(key=lambda r: -r[2])
        return rows


def build_demo(classes: list[dict]) -> gr.Blocks:
    predictor = DummyPredictor(classes)

    def predict_ui(image):
        if image is None:
            return "Upload an image to see dummy Top-3 predictions."
        rows = predictor.predict(image)
        lines = [
            f"{i + 1}. **#{dex:03d}** {name} — {prob * 100:.1f}%"
            for i, (dex, name, prob) in enumerate(rows)
        ]
        return "\n\n".join(lines)

    with gr.Blocks(title="Pokédex (MVP)") as demo:
        gr.Markdown(
            "# Pokédex — Generation 1\n"
            "Dummy model: **random Top-3** with fake probabilities. "
            "Swap `DummyPredictor` for a trained classifier when ready."
        )
        img = gr.Image(type="pil", label="Photo or artwork")
        out = gr.Markdown(label="Predictions")
        img.change(fn=predict_ui, inputs=img, outputs=out)

    return demo


def main() -> None:
    classes = load_label_map()
    demo = build_demo(classes)
    # server_port=None: try 7860, then 7861… if the default port is already in use.
    demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True, ssr_mode=False)


if __name__ == "__main__":
    main()
