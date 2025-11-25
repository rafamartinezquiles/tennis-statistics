from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import pandas as pd
import pickle
from ultralytics import YOLO  # YOLO11-compatible


# Single ball bounding box in [x1, y1, x2, y2] format
BBox = List[float]

# Per-frame mapping: frame_index -> {ball_id -> BBox}
BallDetections = List[Dict[int, BBox]]


@dataclass
class BallTrackerConfig:
    """
    Configuration for the YOLO-based ball tracker.

    Attributes
    ----------
    model_path : str
        Path to a YOLO model (e.g. fine-tuned for tennis balls).
    confidence_threshold : float
        Minimum detection confidence for keeping predictions.
    hit_detection_window : int
        Rolling window size for smoothing vertical motion.
    min_frames_for_hit : int
        Minimum number of consecutive frames in one direction required
        to declare a ball hit event.
    """
    model_path: str
    confidence_threshold: float = 0.15
    hit_detection_window: int = 5
    min_frames_for_hit: int = 25


class BallTracker:
    """
    Detects and tracks the tennis ball across frames, interpolates missing
    positions, and estimates likely ball-hit frames from motion patterns.
    """

    def __init__(self, cfg: BallTrackerConfig):
        """
        Load the YOLO model and store configuration.
        """
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    # ------------------------------------------------------------------ #
    # Core detection
    # ------------------------------------------------------------------ #

    def detect_frame(self, frame) -> Dict[int, BBox]:
        """
        Run ball detection on a single frame.

        Assumes the model is trained with a single ball class. The ball
        is always stored under key 1 to keep the downstream interface simple.
        """
        results = self.model.predict(
            source=frame,
            conf=self.cfg.confidence_threshold,
            verbose=False,
        )[0]

        ball_dict: Dict[int, BBox] = {}
        for box in results.boxes:
            # If trained with a single "ball" class, class ID is not needed.
            xyxy = box.xyxy.tolist()[0]
            ball_dict[1] = xyxy  # always key 1 for the ball; last prediction wins

        return ball_dict

    def detect_frames(
        self,
        frames,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> BallDetections:
        """
        Run ball detection over a sequence of frames.

        Optionally uses a pickle stub to cache and reuse previous results.
        """
        if read_from_stub and stub_path:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        all_detections: BallDetections = []
        for frame in frames:
            all_detections.append(self.detect_frame(frame))

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(all_detections, f)

        return all_detections

    # ------------------------------------------------------------------ #
    # Interpolation and ball-hit detection
    # ------------------------------------------------------------------ #

    def _detections_to_dataframe(self, detections: BallDetections) -> pd.DataFrame:
        """
        Convert raw detections into a DataFrame and interpolate missing values.

        Frames without a detected ball are filled using linear interpolation
        (and backward fill for leading gaps), which yields a smooth trajectory.
        """
        raw_positions = [frame_det.get(1, []) for frame_det in detections]
        df = pd.DataFrame(raw_positions, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return df

    def interpolate_ball_positions(self, detections: BallDetections) -> BallDetections:
        """
        Return a new detection list where missing frames are filled by
        interpolation, so every frame has a ball bounding box.
        """
        df = self._detections_to_dataframe(detections)
        as_list = df.to_numpy().tolist()
        return [{1: coords} for coords in as_list]

    def detect_ball_hit_frames(self, detections: BallDetections) -> List[int]:
        """
        Heuristically detect frames where the ball is likely hit by a player.

        The method:
        - Computes the vertical midpoint of the bounding box over time.
        - Smooths it with a rolling mean.
        - Looks for sign changes in the smoothed vertical velocity (delta_y)
          followed by sustained motion in the new direction.
        """
        df = self._detections_to_dataframe(detections)

        # Midpoint of the ball in the vertical direction.
        df["mid_y"] = 0.5 * (df["y1"] + df["y2"])

        # Smooth vertical trajectory to reduce noise.
        df["mid_y_smooth"] = df["mid_y"].rolling(
            window=self.cfg.hit_detection_window,
            min_periods=1,
        ).mean()

        # Approximate vertical velocity.
        df["delta_y"] = df["mid_y_smooth"].diff()
        df["ball_hit"] = 0

        # Look ahead slightly longer than the minimum hit window.
        horizon = int(self.cfg.min_frames_for_hit * 1.2)

        for idx in range(1, len(df) - horizon):
            current_sign = df["delta_y"].iloc[idx]
            next_sign = df["delta_y"].iloc[idx + 1]

            sign_change_negative = current_sign > 0 and next_sign < 0
            sign_change_positive = current_sign < 0 and next_sign > 0

            if not (sign_change_negative or sign_change_positive):
                continue

            same_direction_count = 0
            for future_idx in range(idx + 1, idx + horizon + 1):
                future_value = df["delta_y"].iloc[future_idx]
                if sign_change_negative and current_sign > 0 and future_value < 0:
                    same_direction_count += 1
                elif sign_change_positive and current_sign < 0 and future_value > 0:
                    same_direction_count += 1

            if same_direction_count >= self.cfg.min_frames_for_hit:
                df.loc[idx, "ball_hit"] = 1

        hit_frames = df.index[df["ball_hit"] == 1].tolist()
        return hit_frames

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    @staticmethod
    def draw_bounding_boxes(frames, detections: BallDetections):
        """
        Draw labeled ball bounding boxes on each frame.
        """
        output_frames = []
        for frame, frame_balls in zip(frames, detections):
            for _, (x1, y1, x2, y2) in frame_balls.items():
                cv2.putText(
                    frame,
                    "Ball",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2,
                )
            output_frames.append(frame)
        return output_frames
