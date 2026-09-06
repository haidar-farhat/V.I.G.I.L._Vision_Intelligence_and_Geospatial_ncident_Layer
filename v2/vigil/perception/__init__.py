"""Pixels to measurements: what a frame says before anything reasons about it.

The layer the product did not have. v1 and v2 both went straight from
"decode a frame" to "run the detector", which meant nothing in either version
could answer three questions that decide whether the detector's answer is
worth acting on:

- **Did the camera move?** A gust on a mast translates every box at once and a
  tracker with no warp reads it as everything accelerating together.
- **Is this frame worth looking at?** A frame that is out of focus, blown out
  or black produces detections whose confidence means nothing, and reporting
  them at face value is how an operator learns to ignore the screen.
- **What does this object look like?** Appearance is what stops one person
  becoming eleven objects, and extracting it is pixel work that does not
  belong in the domain.

Everything here takes a frame and returns a measurement with its own
confidence. Nothing here decides anything.
"""
