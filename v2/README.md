# Sentinel Vision v2

Local-first multi-camera incident intelligence, rebuilt from what v1 taught.
Read [REVIEW_OF_V1.md](REVIEW_OF_V1.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md)
for the shape, [DECISIONS.md](DECISIONS.md) for what is and is not decided,
and [CAPABILITIES.md](CAPABILITIES.md) — generated, never hand-edited — for
what exists and how well.

```
pip install -e .[dev]
python tasks.py check          # offline audit + every suite
python -m vigil where          # paths, principal, alert sinks
python -m vigil cameras add gate device:0 --place 33.8938,35.5018,2,180,-15
python -m vigil zones add yard "33.8937,35.5018;33.8937,35.5019;33.8936,35.5019;33.8936,35.5018" --watch person
python -m vigil run --for 30   # motion-only unless a model is in the models folder
python -m vigil incidents
python -m vigil export inc-…   # report + clips + verifiable manifest
```

Accounts: `python -m vigil users add root --role ADMIN`; after the first account
every command runs as `--as NAME` with the password prompted (or on stdin with
`--password-stdin`). A store with no accounts is open and says so.

Packaging: `python tasks.py package` builds `dist/vigil/vigil.exe`;
`python tasks.py exetest --seconds 20` runs it on `device:0` and judges the run.

## Camera runs on this machine

| When (UTC) | Build | Command | Result |
|---|---|---|---|
| 2026-09-05 22:55 | `ef509ed+dirty` (560 MB bundle, `yolov8n-seg.onnx` shipped) | `python tasks.py exetest --seconds 20 --record` | **PASS** in 21 s: 191 frames at 10 fps through the segmentation model, one 191-frame clip written and indexed, exit 0, no traceback. The room was dark (the captured frame is near black), so 0 detections — the same model on v1's reference footage produced masked detections, which is how the ONNX path was validated tonight. A run with a lit room and a person in view is the next thing to record here. |

The first packaged build failed the same test in 0 s: a frozen `__main__`
cannot use relative imports (`packaging/entry.py` is the fix). The exetest
found it before anybody else could.
