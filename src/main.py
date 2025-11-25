from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import pandas as pd

from src.ball_tracker import BallTracker, BallTrackerConfig
from src.config import CourtDimensions
from src.court_line_detector import CourtLineDetector, CourtLineDetectorConfig
from src.geometry_utils import bbox_center, convert_pixels_to_meters, euclidean_distance
from src.mini_court import MiniCourt
from src.player_tracker import PlayerTracker, PlayerTrackerConfig
from src.stats_overlay import draw_player_stats_overlay
from src.video_io import read_video_frames, write_video_frames


MiniCourtPoint = Tuple[float, float]


def compute_player_stats(
    frames: List,
    court_width_px: int,
    ball_mini_tracks: List[Dict[int, MiniCourtPoint]],
    player_mini_tracks: List[Dict[int, MiniCourtPoint]],
    ball_hit_frames: List[int],
    court_dims: CourtDimensions,
    fps: float = 24.0,
) -> pd.DataFrame:
    """
    Compute per-frame shot and movement statistics for both players.

    For each consecutive pair of detected ball-hit frames:
        - Estimate ball speed between hits, assign it to the hitter.
        - Estimate opponent movement speed between hits, assign it to the opponent.
    Accumulate totals and derive running averages over time.
    """
    # Initial stats row (frame 0, all counters and aggregates at zero)
    stats_history = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0.0,
            "player_1_last_shot_speed": 0.0,
            "player_1_total_player_speed": 0.0,
            "player_1_last_player_speed": 0.0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0.0,
            "player_2_last_shot_speed": 0.0,
            "player_2_total_player_speed": 0.0,
            "player_2_last_player_speed": 0.0,
        }
    ]

    for i in range(len(ball_hit_frames) - 1):
        start_f = ball_hit_frames[i]
        end_f = ball_hit_frames[i + 1]

        duration_s = (end_f - start_f) / fps
        if duration_s <= 0:
            continue

        # Ball travel distance between hits, measured on the mini court
        ball_start = ball_mini_tracks[start_f].get(1)
        ball_end = ball_mini_tracks[end_f].get(1)
        if ball_start is None or ball_end is None:
            continue

        ball_dist_px = euclidean_distance(ball_start, ball_end)
        ball_dist_m = convert_pixels_to_meters(
            ball_dist_px,
            court_dims.DOUBLE_LINE_WIDTH_M,
            court_width_px,
        )
        ball_speed_kmh = ball_dist_m / duration_s * 3.6

        # Determine which player hit the ball at start_f:
        # the hitter is the player closest to the ball position at that frame.
        players_start = player_mini_tracks[start_f]
        if not players_start:
            continue

        hitter_id = min(
            players_start.keys(),
            key=lambda pid: euclidean_distance(players_start[pid], ball_start),
        )
        opponent_id = 1 if hitter_id == 2 else 2

        # Opponent movement between the two hits
        opp_start = player_mini_tracks[start_f].get(opponent_id)
        opp_end = player_mini_tracks[end_f].get(opponent_id)
        if opp_start is None or opp_end is None:
            continue

        opp_dist_px = euclidean_distance(opp_start, opp_end)
        opp_dist_m = convert_pixels_to_meters(
            opp_dist_px,
            court_dims.DOUBLE_LINE_WIDTH_M,
            court_width_px,
        )
        opp_speed_kmh = opp_dist_m / duration_s * 3.6

        # Update cumulative statistics based on the previous state
        prev = deepcopy(stats_history[-1])
        prev["frame_num"] = start_f

        # Hitter: shot-related stats
        prev[f"player_{hitter_id}_number_of_shots"] += 1
        prev[f"player_{hitter_id}_total_shot_speed"] += ball_speed_kmh
        prev[f"player_{hitter_id}_last_shot_speed"] = ball_speed_kmh

        # Opponent: movement-related stats
        prev[f"player_{opponent_id}_total_player_speed"] += opp_speed_kmh
        prev[f"player_{opponent_id}_last_player_speed"] = opp_speed_kmh

        stats_history.append(prev)

    stats_df = pd.DataFrame(stats_history)

    # Ensure a dense, frame-wise time series and forward-fill stats
    frames_df = pd.DataFrame({"frame_num": list(range(len(frames)))})
    stats_df = frames_df.merge(stats_df, on="frame_num", how="left")
    stats_df = stats_df.ffill().fillna(0.0)

    # Compute running averages for shot and movement speeds
    for pid in (1, 2):
        shots = stats_df[f"player_{pid}_number_of_shots"].replace(0, pd.NA)
        stats_df[f"player_{pid}_average_shot_speed"] = (
            stats_df[f"player_{pid}_total_shot_speed"] / shots
        ).fillna(0.0)

        moves = stats_df[f"player_{pid}_number_of_shots"].replace(0, pd.NA)
        stats_df[f"player_{pid}_average_player_speed"] = (
            stats_df[f"player_{pid}_total_player_speed"] / moves
        ).fillna(0.0)

    return stats_df


def main():
    """
    End-to-end pipeline:

        1. Read input video frames.
        2. Run YOLO-based player and ball tracking.
        3. Predict court keypoints.
        4. Project trajectories onto a mini court.
        5. Compute player and ball statistics between hits.
        6. Render visual overlays and write annotated output video.
    """
    # ------------------------------------------------------------------ #
    # 1. Read video
    # ------------------------------------------------------------------ #
    input_video_path = "input_videos/input_video1.mp4"
    frames = read_video_frames(input_video_path)
    fps = 24.0  # Update if the source video has a different frame rate

    # ------------------------------------------------------------------ #
    # 2. Track players and ball (YOLO11 models)
    # ------------------------------------------------------------------ #
    player_tracker = PlayerTracker(
        PlayerTrackerConfig(model_path="yolo11x.pt")
    )  # Replace with fine-tuned player weights if available

    ball_tracker = BallTracker(
        BallTrackerConfig(model_path="models/tennis_ball_yolo11.pt")
    )

    player_dets = player_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path="tracker_stubs/player_detections.pkl",
    )

    ball_dets = ball_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path="tracker_stubs/ball_detections.pkl",
    )
    # Fill in missing ball detections for a smooth trajectory
    ball_dets = ball_tracker.interpolate_ball_positions(ball_dets)

    # ------------------------------------------------------------------ #
    # 3. Court keypoints
    # ------------------------------------------------------------------ #
    court_detector = CourtLineDetector(
        CourtLineDetectorConfig(model_path="models/keypoints_model(resnet50).pth")
    )
    court_kps_flat = court_detector.predict_keypoints(frames[0])

    # Keep only the two main players closest to the court
    player_dets = player_tracker.choose_and_filter_players(court_kps_flat, player_dets)

    # ------------------------------------------------------------------ #
    # 4. Mini court and coordinate conversion
    # ------------------------------------------------------------------ #
    mini_court = MiniCourt(frames[0])

    # Detect frames where the ball is likely hit
    ball_hit_frames = ball_tracker.detect_ball_hit_frames(ball_dets)

    # Convert player and ball positions from full-frame to mini-court coordinates
    player_mini_tracks, ball_mini_tracks = mini_court.convert_to_mini_court_tracks(
        player_dets,
        ball_dets,
        court_kps_flat,
    )

    # ------------------------------------------------------------------ #
    # 5. Player statistics
    # ------------------------------------------------------------------ #
    court_dims = CourtDimensions()
    stats_df = compute_player_stats(
        frames=frames,
        court_width_px=mini_court.mini_court_width(),
        ball_mini_tracks=ball_mini_tracks,
        player_mini_tracks=player_mini_tracks,
        ball_hit_frames=ball_hit_frames,
        court_dims=court_dims,
        fps=fps,
    )

    # ------------------------------------------------------------------ #
    # 6. Drawing overlays
    # ------------------------------------------------------------------ #
    output_frames = player_tracker.draw_bounding_boxes(frames, player_dets)
    output_frames = ball_tracker.draw_bounding_boxes(output_frames, ball_dets)
    output_frames = court_detector.draw_keypoints_on_video(output_frames, court_kps_flat)

    # Mini court visualization
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points(output_frames, player_mini_tracks, color=(255, 0, 0))
    output_frames = mini_court.draw_points(output_frames, ball_mini_tracks, color=(0, 255, 255))

    # Stats box overlay
    output_frames = draw_player_stats_overlay(output_frames, stats_df)

    # Frame index in top-left corner for debugging
    for idx, frame in enumerate(output_frames):
        cv2.putText(
            frame,
            f"Frame: {idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    # ------------------------------------------------------------------ #
    # 7. Save video
    # ------------------------------------------------------------------ #
    write_video_frames(output_frames, "output_videos/output_video.avi", fps=fps)


if __name__ == "__main__":
    main()
