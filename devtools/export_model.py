#!/usr/bin/env python3
"""Obtain a pretrained model and export it to ONNX. Developer machine only.

Run this once, on a machine with a network. It produces a `.onnx` file in
`models/`, which is the artifact the engine loads. Nothing here ships, and
nothing in the engine imports anything from this file — see `devtools/README.md`
for why that separation is load-bearing rather than tidy.

**Why ultralytics.** It is the maintained tool for these weights, and the rule
is to use a mature package rather than write one. It is also precisely what the
offline audit exists to keep out of shipped source: it fetches weights on first
use and carries analytics. Both facts are true, which is why it lives here and
why the model — not the library — is what crosses into the product.

The export is deliberately plain: no NMS baked in, no dynamic axes, no half
precision. The engine already does letterboxing, per-class NMS and coordinate
un-letterboxing against a known input size, all of it tested. A model that did
some of that internally would mean two implementations of the same arithmetic,
disagreeing at the edges.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

#: Exported at this size. 640 is what these weights were trained at, and
#: exporting at a different one silently costs accuracy that is then blamed on
#: the detector.
INPUT_SIZE = 640


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export-model",
        description="Fetch pretrained YOLO weights and export ONNX into models/.",
    )
    parser.add_argument("--task", choices=("segment", "detect"), default="segment",
                        help="segment gives per-instance masks; detect gives boxes only")
    parser.add_argument("--size", choices=("n", "s", "m", "l", "x"), default="n",
                        help="model size. n is the smallest and fastest")
    parser.add_argument("--imgsz", type=int, default=INPUT_SIZE)
    args = parser.parse_args(argv)

    # Off before ultralytics is imported: it reads this at import time, and a
    # developer machine is still not a thing that should report usage anywhere.
    os.environ.setdefault("YOLO_OFFLINE", "false")
    os.environ.setdefault("ULTRALYTICS_ANALYTICS", "false")

    try:
        from ultralytics import YOLO
        from ultralytics import settings as ultralytics_settings
    except ImportError:
        print(
            "ultralytics is not installed. It is a developer tool, not a "
            "dependency of the product:\n\n"
            "    pip install ultralytics onnxslim\n",
            file=sys.stderr,
        )
        return 1

    try:
        ultralytics_settings.update({"sync": False})
    except Exception:  # noqa: BLE001 - older versions have no such setting
        pass

    suffix = "-seg" if args.task == "segment" else ""
    name = f"yolov8{args.size}{suffix}"

    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"fetching {name}.pt (this reaches the network — the product never does)")
    model = YOLO(f"{name}.pt")

    print(f"exporting to ONNX at {args.imgsz}x{args.imgsz}")
    exported = Path(
        model.export(format="onnx", imgsz=args.imgsz, simplify=True,
                     dynamic=False, nms=False, opset=17)
    )

    target = MODELS / f"{name}.onnx"
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())
        exported.unlink(missing_ok=True)

    # The weights themselves are not the artifact and should not linger next to
    # it, where somebody will eventually ship one by accident.
    for stray in (ROOT / f"{name}.pt", MODELS / f"{name}.pt", Path(f"{name}.pt")):
        stray.unlink(missing_ok=True)

    digest = sha256_of(target)
    print()
    print(f"  {target.relative_to(ROOT)}")
    print(f"  {target.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256  {digest}")
    print()
    print("  That digest is what the engine records on every event this model")
    print("  produces, so a detection can be traced to the exact file months later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
