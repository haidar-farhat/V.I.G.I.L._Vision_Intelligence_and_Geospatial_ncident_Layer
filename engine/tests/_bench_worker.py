"""One camera's pipeline, runnable in its own process. Used by bench_scaling."""
import sys, time
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

def run(video: str, frames_wanted: int) -> float:
    import cv2
    cv2.setNumThreads(1)
    from sentinel.core import CameraPose, LatLon
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector
    from sentinel.pipeline import Pipeline

    pose = CameraPose(LatLon(33.8938, 35.5018), 6.0, 180.0, -22.0,
                      horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0)
    with Pipeline(VideoSource(Path(video)), MotionDetector(), pose=pose) as pipeline:
        start = time.perf_counter()
        count = 0
        for _ in pipeline.run():
            count += 1
            if count >= frames_wanted:
                break
        return count / (time.perf_counter() - start)
