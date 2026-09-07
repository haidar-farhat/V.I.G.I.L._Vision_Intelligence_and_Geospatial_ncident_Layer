# devtools

Tools that run on a **developer's connected machine** and never ship.

This directory exists because of a genuine tension. The product must work with
the network cable unplugged, and `tools/offline_audit.py` enforces that over
everything under `tools/`, `engine/sentinel/`, `apps/console/sentinel_console/`
and `core/src/`. But *obtaining* a model is a connected act by definition, and
the mature tool for it — `ultralytics` — is exactly the sort of package the
audit exists to keep out of shipped source: it downloads weights on first use
and it has analytics of its own.

Both things are true and neither should be bent:

- **Nothing in here is imported by the engine, the console or the packaged
  build.** The offline audit does not scan this directory, and that is safe
  only because nothing here is reachable from anything that ships. A test
  asserts that.
- **What comes out of here is an artifact, not a dependency.** `export_model.py`
  writes a `.onnx` file into `models/`, which is gitignored and operator-owned.
  The engine loads that file through `onnxruntime` and has never heard of
  ultralytics.

That is the same shape as the wheelhouse in `docs/SECURITY.md`: the network is
available once, to obtain something, and never again at runtime.

## export_model.py

Downloads pretrained YOLO weights and exports them to ONNX.

```bash
python devtools/export_model.py --task segment --size n
```

Writes `models/yolov8n-seg.onnx` and prints its SHA-256, which is what the
engine records on every event the model produces.

Requires `pip install ultralytics onnxslim` — deliberately **not** in
`engine/pyproject.toml`, because adding it there would put a weight-downloader
in the shipped dependency set.
