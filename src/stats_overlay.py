from __future__ import annotations

from typing import List

import cv2
import numpy as np
import pandas as pd


def draw_player_stats_overlay(
    frames: List[np.ndarray],
    stats_df: pd.DataFrame,
) -> List[np.ndarray]:
    """
    Draw a per-frame statistics panel with shot and movement speeds
    for both players.

    The function assumes that `stats_df` has one row per frame and contains:
        - player_1_last_shot_speed
        - player_2_last_shot_speed
        - player_1_last_player_speed
        - player_2_last_player_speed
        - player_1_average_shot_speed
        - player_2_average_shot_speed
        - player_1_average_player_speed
        - player_2_average_player_speed
    """
    if len(frames) != len(stats_df):
        raise ValueError("frames and stats_df must have the same length")

    # Work in-place but keep the type annotation explicit
    output_frames: List[np.ndarray] = frames

    for idx, row in stats_df.iterrows():
        frame = output_frames[idx]

        # Latest values
        p1_shot = row["player_1_last_shot_speed"]
        p2_shot = row["player_2_last_shot_speed"]
        p1_speed = row["player_1_last_player_speed"]
        p2_speed = row["player_2_last_player_speed"]

        # Running averages
        p1_shot_avg = row["player_1_average_shot_speed"]
        p2_shot_avg = row["player_2_average_shot_speed"]
        p1_speed_avg = row["player_1_average_player_speed"]
        p2_speed_avg = row["player_2_average_player_speed"]

        height, width = frame.shape[:2]

        # Overlay box geometry in bottom-right corner
        box_width = 360
        box_height = 220
        margin_x = 30
        margin_y = 40

        box_x1 = width - box_width - margin_x
        box_y1 = height - box_height - margin_y
        box_x2 = width - margin_x
        box_y2 = height - margin_y

        # Semi-transparent background for readability
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0, dst=frame)

        # Header row
        header = "     Player 1      Player 2"
        cv2.putText(
            frame,
            header,
            (box_x1 + 65, box_y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Row 1: shot speed (instant)
        cv2.putText(
            frame,
            "Shot Speed",
            (box_x1 + 10, box_y1 + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"{p1_shot:.1f} km/h    {p2_shot:.1f} km/h",
            (box_x1 + 130, box_y1 + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Row 2: player movement speed (instant)
        cv2.putText(
            frame,
            "Player Speed",
            (box_x1 + 10, box_y1 + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"{p1_speed:.1f} km/h    {p2_speed:.1f} km/h",
            (box_x1 + 130, box_y1 + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Row 3: average shot speed
        cv2.putText(
            frame,
            "avg. S. Speed",
            (box_x1 + 10, box_y1 + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"{p1_shot_avg:.1f} km/h    {p2_shot_avg:.1f} km/h",
            (box_x1 + 130, box_y1 + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Row 4: average player speed
        cv2.putText(
            frame,
            "avg. P. Speed",
            (box_x1 + 10, box_y1 + 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"{p1_speed_avg:.1f} km/h    {p2_speed_avg:.1f} km/h",
            (box_x1 + 130, box_y1 + 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        output_frames[idx] = frame

    return output_frames
