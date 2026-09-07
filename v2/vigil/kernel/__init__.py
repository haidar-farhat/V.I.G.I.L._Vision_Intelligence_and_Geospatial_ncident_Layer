"""The numeric kernel: the arithmetic every layer above stands on.

Below the domain rather than beside it, and that placement is the point. The
domain must be able to say "predict this track" without knowing that a shared
library exists, and the layering test enforces that it cannot find out.

- `native`: the Rust core through ctypes, and the NumPy paths that stand in
  for the parts of it a checkout without a toolchain can still run.
- `filtering`: the box Kalman filter in NumPy, which is both that stand-in and
  the readable statement of the model `core/src/track.rs` implements.

Nothing here imports anything else from `vigil`, and nothing here touches
OpenCV, the database or the network.
"""
