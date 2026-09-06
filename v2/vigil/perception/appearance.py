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
    chosen = hsv[selector]
    hue = np.bincount((chosen[:, 0].astype(np.int32) * HUE_BINS) // 180, minlength=HUE_BINS)[:HUE_BINS]
    saturation = np.bincount((chosen[:, 1].astype(np.int32) * SATURATION_BINS) // 256,
                             minlength=SATURATION_BINS)[:SATURATION_BINS]
    value = np.bincount((chosen[:, 2].astype(np.int32) * VALUE_BINS) // 256,
                        minlength=VALUE_BINS)[:VALUE_BINS]
    # Hue is unreliable where saturation is low, so it is weighted by how much
    # colour there actually was. A grey coat then leans on value, which is the
    # channel that can still tell it from a white one.
    colourfulness = float(np.mean(chosen[:, 1])) / 255.0
    vector = np.concatenate([
        _unit(hue.astype(np.float32)) * colourfulness,
        _unit(saturation.astype(np.float32)) * 0.5,
        _unit(value.astype(np.float32)) * 0.5,
    ])
    return Appearance(_unit(vector).astype(np.float32), pixels)
