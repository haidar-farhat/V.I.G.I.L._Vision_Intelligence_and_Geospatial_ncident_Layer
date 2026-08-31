"""A real ONNX detection model, built locally.

`OnnxDetector` is the path a production model runs through: letterboxing,
normalisation, session execution, output-layout inference, coordinate
un-letterboxing, per-class NMS. Without weights, none of that had ever executed —
it was code that compiled and had never run, which is exactly the state this
project treats as unfinished.

**No model may be downloaded.** So one is built here instead, with real ONNX
operators, and it does real work: it reduces the image to brightness, pools it
into a grid, and emits one candidate box per cell whose confidence *is* that
cell's mean brightness. A crude detector, but not a fake one — feed it a white
square on black and it finds the square, in the right place, at the right size.

That distinction is the whole point. A stub returning constants would exercise
the plumbing while proving nothing about whether pixels reach the output. This
model makes the output a function of the input, so a test can assert that moving
the object moves the box — which is the property that catches a letterboxing or
transpose error, the two mistakes that actually happen here.

What it does *not* establish: anything about detection quality, classes, or
real-world behaviour. It is a test fixture for the machinery around a model, not
a substitute for one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Input the model declares. Square, so the reference scene (which is not) has to
#: be letterboxed — the case where coordinate mistakes actually show up.
INPUT_SIZE = 640

#: Cell size in input pixels. 640 / 32 = a 20x20 grid, 400 candidate boxes.
CELL = 32
GRID = INPUT_SIZE // CELL
CELLS = GRID * GRID


def build_model(path: Path) -> Path:
    """Write a working ONNX detection model and return its path.

    The graph, in order:

        images  [1,3,640,640]
          -> ReduceMean over channels          [1,1,640,640]   brightness
          -> AveragePool 32x32 stride 32       [1,1,20,20]     one value per cell
          -> Reshape                           [1,1,400]
          -> Concat with constant cx,cy,w,h    [1,5,400]

    The output is the YOLOv8-family layout — ``[1, 4 + classes, anchors]`` with
    box values in input-tensor pixels — because that is what the reader must
    handle, and a fixture in a different layout would test the wrong thing.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    centres = (np.arange(GRID, dtype=np.float32) + 0.5) * CELL
    grid_y, grid_x = np.meshgrid(centres, centres, indexing="ij")

    def constant(name: str, values: np.ndarray):
        return numpy_helper.from_array(
            values.reshape(1, 1, CELLS).astype(np.float32), name
        )

    initializers = [
        constant("cx", grid_x),
        constant("cy", grid_y),
        constant("bw", np.full(CELLS, float(CELL))),
        constant("bh", np.full(CELLS, float(CELL))),
        numpy_helper.from_array(np.array([1, 1, CELLS], dtype=np.int64), "flat_shape"),
    ]

    nodes = [
        # Mean over the colour channels. Opset 13 takes `axes` as an attribute;
        # from 18 it is an input, which is why the opset is pinned below.
        helper.make_node("ReduceMean", ["images"], ["gray"], axes=[1], keepdims=1),
        helper.make_node(
            "AveragePool",
            ["gray"],
            ["pooled"],
            kernel_shape=[CELL, CELL],
            strides=[CELL, CELL],
        ),
        helper.make_node("Reshape", ["pooled", "flat_shape"], ["scores"]),
        helper.make_node(
            "Concat", ["cx", "cy", "bw", "bh", "scores"], ["predictions"], axis=1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "brightness-detector",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, [1, 5, CELLS]
            )
        ],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="sentinel-test-fixture",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Class names travel inside the model, the way a real one carries them. The
    # detector must read them from here rather than assume a standard list.
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="names", value="{0: 'bright_region'}")
    )

    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def expected_cell(x_pixels: float, y_pixels: float) -> tuple[float, float]:
    """The centre of the grid cell containing a point in input-tensor pixels."""
    column = min(GRID - 1, max(0, int(x_pixels // CELL)))
    row = min(GRID - 1, max(0, int(y_pixels // CELL)))
    return (column + 0.5) * CELL, (row + 0.5) * CELL
