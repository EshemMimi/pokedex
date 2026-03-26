"""Local Gradio UI for Gen 1 Pokédex MVP. Uses trained weights when present, else a dummy predictor."""

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
    """Random Top-3 (used when no checkpoint or PyTorch unavailable)."""

    def __init__(self, classes: list[dict]) -> None:
        self._classes = classes

    def predict(self, image) -> list[tuple[int, str, float]]:
        _ = image
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


def _resolve_predictor(classes: list[dict]):
    """Returns (predictor, description_md_line)."""
    try:
        from pokedex.inference import try_load_torch_predictor
    except ImportError:
        return (
            DummyPredictor(classes),
            "PyTorch is not installed; using **dummy** Top-3. "
            "Install with `pip install torch torchvision` and add `artifacts/best_model.pt`.",
        )
    pred, status = try_load_torch_predictor(classes, ROOT)
    if pred is not None:
        return (
            pred,
            "**Trained model** (ResNet-18): predictions from `artifacts/best_model.pt`.",
        )
    if status == "no_checkpoint":
        return (
            DummyPredictor(classes),
            "No checkpoint at `artifacts/best_model.pt`; using **dummy** Top-3. "
            "Train with `python scripts/train.py` first.",
        )
    if status == "missing_train_config":
        return (
            DummyPredictor(classes),
            "Checkpoint is missing `num_classes` / `image_size` metadata; using **dummy** Top-3. "
            "Re-train with the current `scripts/train.py` or restore `artifacts/train_config.json`.",
        )
    return (
        DummyPredictor(classes),
        f"Could not load checkpoint ({status}); using **dummy** Top-3.",
    )


def build_demo(classes: list[dict]) -> gr.Blocks:
    predictor, banner_md = _resolve_predictor(classes)

    def predict_ui(image):
        if image is None:
            return "Upload an image to see Top-3 predictions."
        rows = predictor.predict(image)
        lines = [
            f"{i + 1}. **#{dex:03d}** {name} — {prob * 100:.1f}%"
            for i, (dex, name, prob) in enumerate(rows)
        ]
        return "\n\n".join(lines)

    with gr.Blocks(title="Pokédex (MVP)") as demo:
        gr.Markdown("# Pokédex — Generation 1\n" + banner_md)
        img = gr.Image(type="pil", label="Photo or artwork")
        out = gr.Markdown(label="Predictions")
        img.change(fn=predict_ui, inputs=img, outputs=out)

    return demo


def main() -> None:
    classes = load_label_map()
    demo = build_demo(classes)
    demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True, ssr_mode=False)


if __name__ == "__main__":
    main()
