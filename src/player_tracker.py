from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import pickle
from ultralytics import YOLO  # YOLO11-compatible

from src.geometry_utils import bbox_center, euclidean_distance

# Bounding box in [x1, y1, x2, y2] format
BBox = List[float]

# Per-frame mapping: frame_index -> {track_id -> BBox}
PlayerDetections = List[Dict[int, BBox]]


@dataclass
class PlayerTrackerConfig:
    """
    Configuration for the YOLO-based player tracker.

    Attributes
    ----------
    model_path : str
        Path to a YOLO model, e.g. "yolo11x.pt" fine-tuned for tennis.
    person_class_name : str
        Name of the person class in the model's label set.
    """
    model_path: str
    person_class_name: str = "person"


class PlayerTracker:
    """
    Wrapper around a YOLO detector to track tennis players over a sequence
    of frames and keep consistent track IDs across time.
    """

    def __init__(self, cfg: PlayerTrackerConfig):
        """
        Load the YOLO model and store basic configuration.
        """
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #

    def detect_frame(self, frame) -> Dict[int, BBox]:
        """
        Run tracking on a single frame and return player detections.

        YOLO's internal tracker is used with `persist=True` so that track
        IDs remain consistent across frames.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            verbose=False,
        )[0]

        id_to_name = results.names
        player_dict: Dict[int, BBox] = {}

        for box in results.boxes:
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_to_name[cls_id]
            if cls_name != self.cfg.person_class_name:
                continue

            track_id = int(box.id.tolist()[0])
            xyxy = box.xyxy.tolist()[0]
            player_dict[track_id] = xyxy

        return player_dict

    def detect_frames(
        self,
        frames,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> PlayerDetections:
        """
        Run tracking over a sequence of frames.

        Optionally reads/writes a pickle stub to avoid recomputing detections
        for the same video.
        """
        if read_from_stub and stub_path:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        all_detections: PlayerDetections = []
        for frame in frames:
            all_detections.append(self.detect_frame(frame))

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(all_detections, f)

        return all_detections

    # ------------------------------------------------------------------ #
    # Choosing the two main players
    # ------------------------------------------------------------------ #

    @staticmethod
    def _choose_two_closest_to_court(
        court_keypoints_flat: List[float],
        first_frame_detections: Dict[int, BBox],
    ) -> List[int]:
        """
        Select the two track IDs whose centers are closest to the known court keypoints, using only the first frame.

        This assumes that the two main players are typically nearest to the court at the beginning of the sequence.
        """
        distances = []
        for track_id, bbox in first_frame_detections.items():
            center = bbox_center(bbox)

            min_dist = float("inf")
            for i in range(0, len(court_keypoints_flat), 2):
                kp = (court_keypoints_flat[i], court_keypoints_flat[i + 1])
                d = euclidean_distance(center, kp)
                if d < min_dist:
                    min_dist = d

            distances.append((track_id, min_dist))

        distances.sort(key=lambda x: x[1])
        # Assume at least two players are present; caller should ensure this.
        return [distances[0][0], distances[1][0]]

    def choose_and_filter_players(
        self,
        court_keypoints_flat: List[float],
        all_detections: PlayerDetections,
    ) -> PlayerDetections:
        """
        From all tracked persons, keep only the two that are closest to the court on the first frame, and filter all subsequent frames to 
        those IDs. This provides a consistent pair of player tracks across the rally.
        """
        if not all_detections:
            return all_detections

        chosen_ids = self._choose_two_closest_to_court(
            court_keypoints_flat,
            all_detections[0],
        )

        filtered: PlayerDetections = []
        for frame_dets in all_detections:
            filtered.append(
                {tid: box for tid, box in frame_dets.items() if tid in chosen_ids}
            )
        return filtered

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _label_for_track_id(track_id: int) -> str:
        """
        Map a track ID to a human-readable label.

        Adjust this mapping to reflect the actual players in the footage.
        """
        if track_id == 1:
            return "Medvedev"
        if track_id == 2:
            return "Djokovic"
        return f"Player {track_id}"

    @classmethod
    def draw_bounding_boxes(cls, frames, detections: PlayerDetections):
        """
        Draw labeled bounding boxes for each tracked player on each frame.
        """
        output_frames = []
        for frame, frame_players in zip(frames, detections):
            for track_id, (x1, y1, x2, y2) in frame_players.items():
                label = cls._label_for_track_id(track_id)

                # Draw label above the bounding box.
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                # Draw bounding box.
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
            output_frames.append(frame)
        return output_frames
