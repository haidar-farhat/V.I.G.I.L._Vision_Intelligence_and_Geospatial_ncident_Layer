"""Reading a colour descriptor off a crop.

The measurement half of `vigil.domain.appearance`, which owns the type and the
comparison. Split because the domain does not import OpenCV, and because a
tracker that does its own colour conversion is a tracker that cannot be tested
without an image.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..domain.appearance import (
    Appearance, HUE_BINS, MINIMUM_PIXELS, SATURATION_BINS, VALUE_BINS,
)

_LENGTH = HUE_BINS + SATURATION_BINS + VALUE_BINS


def _unit(histogram: np.ndarray) -> np.ndarray:
    total = float(np.linalg.norm(histogram))
    return histogram / total if total > 0 else histogram


def describe(image: np.ndarray, box: tuple[float, float, float, float],
             mask: np.ndarray | None = None) -> Appearance:
    """The descriptor of one detection.

    `box` is `(x, y, width, height)` in normalised image coordinates. `mask`,
    when the detector produced one, is a boolean or 0/255 array covering the
    box; the histogram is taken over the masked pixels only.

    Without a mask the sample is the **central band** of the box rather than
    all of it. A box around a person is mostly not the person: the corners are
    whatever they are standing in front of, and a histogram that includes them
    describes the background, which every box in that part of the frame
    shares. That is how a colour descriptor comes to link two different people
    who walked past the same wall.
    """
    height, width = image.shape[:2]
    x0 = int(round(max(0.0, box[0]) * width))
    y0 = int(round(max(0.0, box[1]) * height))
    x1 = int(round(min(1.0, box[0] + box[2]) * width))
    y1 = int(round(min(1.0, box[1] + box[3]) * height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return Appearance(np.zeros(HUE_BINS + SATURATION_BINS + VALUE_BINS, dtype=np.float32), 0)
    crop = image[y0:y1, x0:x1]

    if mask is not None and mask.size:
        resized = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        selector = resized > 0
    else:
        selector = np.zeros(crop.shape[:2], dtype=bool)
        h, w = crop.shape[:2]
        selector[int(h * 0.15):int(h * 0.85) or 1, int(w * 0.25):int(w * 0.75) or 1] = True

    pixels = int(np.count_nonzero(selector))
    if pixels < MINIMUM_PIXELS:
        return Appearance(np.zeros(HUE_BINS + SATURATION_BINS + VALUE_BINS, dtype=np.float32), pixels)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Histogrammed through OpenCV with a mask rather than by indexing the
    # selected pixels out and counting them in NumPy.
    #
    # The obvious form -- `hsv[selector]` then three `bincount`s -- copies
    # every selected pixel into a new array and allocates three more, and
    # measured at 0.215 ms a box against 0.061 ms here: 3.5x, and this runs
    # once per detection per frame. The results are **identical**, not close:
    # checked over 300 randomised crops and at the top of the hue range, which
    # is the boundary where an integer division and a float binning could
    # have disagreed.
    #
    # This is why there is no Rust appearance kernel, although the plan listed
    # one. The win was available from a library already in the process, and a
    # Rust version would have meant reimplementing OpenCV's BGR-to-HSV
    # conversion -- a second implementation of a colour space, to be held to
    # the first for ever, for a fraction of what this already recovers.
    mask = selector.astype(np.uint8)
    hue = cv2.calcHist([hsv], [0], mask, [HUE_BINS], [0, 180]).ravel()
    saturation = cv2.calcHist([hsv], [1], mask, [SATURATION_BINS], [0, 256]).ravel()
    value = cv2.calcHist([hsv], [2], mask, [VALUE_BINS], [0, 256]).ravel()
    # Hue is unreliable where saturation is low, so it is weighted by how much
    # colour there actually was. A grey coat then leans on value, which is the
    # channel that can still tell it from a white one.
    colourfulness = float(cv2.mean(hsv, mask=mask)[1]) / 255.0
    vector = np.concatenate([
        _unit(hue.astype(np.float32)) * colourfulness,
        _unit(saturation.astype(np.float32)) * 0.5,
        _unit(value.astype(np.float32)) * 0.5,
    ])
    return Appearance(_unit(vector).astype(np.float32), pixels)
