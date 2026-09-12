"""CPU full-range detection, explicit crop, and MediaPipe Tasks landmarks."""

import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from preprocessing import (
    build_face_bbox_from_landmarks,
    crop_and_resize_frames,
    frames_bgr_to_pil_from_video,
    process_bbox,
)


def fill_short_gaps(landmarks, max_gap_frames):
    """Interpolate short gaps; reject long or entirely missing face tracks."""
    valid = np.array([points is not None for points in landmarks], dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise RuntimeError("MediaPipe detected no face in the video.")
    missing = np.flatnonzero(~valid)
    if len(missing):
        groups = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
        if max(map(len, groups)) > max_gap_frames:
            raise RuntimeError(f"MediaPipe face track has a gap longer than {max_gap_frames} frames.")
    points = np.stack([landmarks[index] for index in indices])
    result = np.empty((len(landmarks), *points.shape[1:]), dtype=np.float32)
    for point in range(points.shape[1]):
        for coordinate in range(points.shape[2]):
            result[:, point, coordinate] = np.interp(
                np.arange(len(landmarks)), indices, points[:, point, coordinate]
            )
    return result, missing.tolist()


def crop_detected_face(pixels, box):
    """Return a 256px RGB crop and its pixel-to-original affine transform."""
    side = max(box.width, box.height) * 1.5
    if side <= 0:
        raise ValueError("MediaPipe returned an empty detection box.")
    x = box.origin_x + box.width / 2 - side / 2
    y = box.origin_y + box.height / 2 - side / 2
    matrix = np.array([[side / 256, 0, x], [0, side / 256, y]], dtype=np.float32)
    crop = cv2.warpAffine(
        pixels, matrix, (256, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, matrix


def preprocess_video_with_mediapipe(video_path, model_path, report_path=None):
    """Isolate MediaPipe native libraries from the PyTorch generation process."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tbdub-mediapipe-") as temporary:
        worker_report = Path(temporary) / "detection.json"
        command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
                   "--worker", "--video", str(video_path), "--model", str(model_path),
                   "--report", str(worker_report)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(command, env=environment, check=False)
        if worker_report.exists():
            report = json.loads(worker_report.read_text())
        else:
            report = {"success": False, "worker_returncode": completed.returncode,
                      "error": "MediaPipe worker exited without a report; see its stderr."}
        if completed.returncode:
            if report_path:
                target = Path(report_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2))
            completed.check_returncode()
        if not report.get("success"):
            raise RuntimeError(report.get("error", "MediaPipe preprocessing failed."))
    report["worker_wall_seconds"] = time.perf_counter() - started
    report["execution"] = "isolated_cpu_subprocess"
    frames = frames_bgr_to_pil_from_video(video_path)
    if len(frames) != report["frames"]:
        raise RuntimeError("Video frame count changed between detection and cropping.")
    boxes = report["bboxes"]
    crops = crop_and_resize_frames(frames, boxes)
    report["preprocessing_seconds"] = time.perf_counter() - started
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))
    return frames, crops, boxes, report["crop_mode"]


def _preprocess_video_in_process(video_path, model_path, report_path=None):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    started = time.perf_counter()
    model_path = Path(model_path).expanduser().resolve()
    detector_model_path = Path(mp.__file__).parent / "modules/face_detection/face_detection_full_range_sparse.tflite"
    for path in (model_path, detector_model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing MediaPipe model: {path}")
    with imageio.get_reader(video_path) as reader:
        fps = float(reader.get_meta_data().get("fps", 25.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video frame rate: {fps}")
    frames = frames_bgr_to_pil_from_video(video_path)
    if not frames:
        raise ValueError("The input video contains no readable frames.")

    def cpu_options(path):
        return python.BaseOptions(model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU)

    report = {
        "backend": "mediapipe_full_range_sparse_v2", "version": mp.__version__, "delegate": "CPU",
        "model_path": str(model_path), "detector_model_path": str(detector_model_path),
        "source_fps": fps, "frames": len(frames), "frame_diagnostics": [],
        "confidence_thresholds": {"detection": 0.5, "presence": 0.5, "tracking": 0.5},
    }

    def save_report():
        if report_path:
            target = Path(report_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2))

    try:
        # 0.10.21's Tasks FaceDetector assumes short-range tensor dimensions.
        # Use its official full-range solution and bundled sparse model instead.
        # Close the detector before creating the landmark task.
        detection_started = time.perf_counter()
        face_inputs = []
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            for index, frame in enumerate(frames):
                pixels = np.asarray(frame)
                result = detector.process(pixels)
                detections = result.detections or []
                row = {"frame": index, "detected_faces": len(detections), "landmark_faces": 0}
                if detections:
                    detection = max(detections, key=lambda item: item.score[0])
                    relative = detection.location_data.relative_bounding_box
                    width, height = frame.size
                    box = SimpleNamespace(
                        origin_x=int(relative.xmin * width), origin_y=int(relative.ymin * height),
                        width=int(relative.width * width), height=int(relative.height * height),
                    )
                    crop, matrix = crop_detected_face(pixels, box)
                    face_inputs.append((crop, matrix))
                    row.update(detector_score=float(detection.score[0]),
                               detection_box=[box.origin_x, box.origin_y, box.width, box.height])
                else:
                    face_inputs.append(None)
                report["frame_diagnostics"].append(row)
        report["detection_seconds"] = time.perf_counter() - detection_started
        report["detected_frames"] = sum(item is not None for item in face_inputs)

        landmark_started = time.perf_counter()
        landmark_options = vision.FaceLandmarkerOptions(base_options=cpu_options(model_path), num_faces=1)
        landmarks = []
        with vision.FaceLandmarker.create_from_options(landmark_options) as landmarker:
            for index, face_input in enumerate(face_inputs):
                points = None
                if face_input is not None:
                    crop, matrix = face_input
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                    report["frame_diagnostics"][index]["landmark_faces"] = len(result.face_landmarks)
                    if result.face_landmarks:
                        local = np.array([(point.x * 256, point.y * 256, 1)
                                          for point in result.face_landmarks[0]], dtype=np.float32)
                        points = (local @ matrix.T) / np.array(frames[index].size, dtype=np.float32)
                landmarks.append(points)
        del face_inputs
        report["landmark_seconds"] = time.perf_counter() - landmark_started
        report["landmark_frames"] = sum(item is not None for item in landmarks)
        report["missing_frames"] = [i for i, item in enumerate(landmarks) if item is None]
        points, interpolated = fill_short_gaps(landmarks, max_gap_frames=max(1, round(fps * 0.4)))
        width, height = frames[0].size
        boxes = build_face_bbox_from_landmarks(points, width, height)
        boxes, case_flag = process_bbox(boxes, width, height, force_fix=False)
        report.update(success=True, interpolated_frames=interpolated, crop_mode=case_flag,
                      bboxes=boxes, landmark_count=points.shape[1],
                      preprocessing_seconds=time.perf_counter() - started)
    except Exception as exc:
        report.update(success=False, error=str(exc))
        save_report()
        raise
    save_report()
    print(f"[MediaPipe] detected={report['detected_frames']}/{len(frames)}, "
          f"landmarks={report['landmark_frames']}/{len(frames)}, "
          f"interpolated={len(interpolated)}, crop_mode={case_flag}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Internal isolated CPU face preprocessing worker.")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    options = parser.parse_args()
    _preprocess_video_in_process(options.video, options.model, options.report)
