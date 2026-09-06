"""What a detector says about one frame, and what a detector is."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geo import Vec2

#: The class id of a detection from a detector that does not classify.
UNCLASSIFIED = -1


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalised to the frame: 0..1 in both axes, (x, y) the top-left corner."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def bottom_center(self) -> Vec2:
        return Vec2(self.x + self.width / 2, self.bottom)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ix = max(0.0, min(self.right, other.right) - max(self.x, other.x))
        iy = max(0.0, min(self.bottom, other.bottom) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clamped(self) -> "BoundingBox":
        x = min(1.0, max(0.0, self.x))
        y = min(1.0, max(0.0, self.y))
        return BoundingBox(x, y, min(1.0 - x, max(0.0, self.width)), min(1.0 - y, max(0.0, self.height)))


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int = UNCLASSIFIED
    #: Where the object meets the ground, in normalised image coordinates.
    #: The bottom-centre of the box unless a mask measured better.
    contact: Vec2 | None = None
    #: The segmentation mask over the box, as a small 0/1 array, when the
    #: model produced one.
    #:
    #: Carried because an appearance descriptor taken over the whole box is
    #: mostly a descriptor of the *background* — the corners of a box around a
    #: person are whatever they are standing in front of, which every box in
    #: that part of the frame shares. That is how a colour descriptor comes to
    #: link two different people who walked past the same wall.
    #:
    #: Excluded from equality: an array does not compare to a bool, and two
    #: detections are the same detection when their box, class and confidence
    #: are the same.
    mask: Any = field(default=None, compare=False)

    @property
    def ground_contact(self) -> Vec2:
        return self.contact if self.contact is not None else self.bbox.bottom_center


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    """What drew the conclusions. Travels with every event into the evidence."""

    kind: str
    name: str
    model_path: str | None = None
    model_sha256: str | None = None
    input_size: tuple[int, int] | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    classifies: bool = False
    #: The onnxruntime execution provider the session actually got.
    #:
    #: Read back off the session rather than assumed from what was requested:
    #: onnxruntime falls back silently when a provider will not initialise, so
    #: a CUDA build with the wrong driver runs on the CPU and says nothing.
    #: "Why is this slow" is answered by this field more often than by
    #: anything in the model.
    provider: str | None = None

    def label_for(self, class_id: int) -> str | None:
        """The label, or ``None`` when this detector cannot say. A blob is not a person."""
        if not self.classifies or class_id == UNCLASSIFIED:
            return None
        return self.class_names.get(class_id, f"class_{class_id}")
