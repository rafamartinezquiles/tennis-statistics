from __future__ import annotations

import pickle
import matplotlib.pyplot as plt
import pandas as pd


def load_ball_detections_stub(path: str) -> pd.DataFrame:
    """
    Load a pickled list of ball detections and return them as a DataFrame.

    The pickled object is expected to be a list of per-frame dicts:
        [{1: [x1, y1, x2, y2]}, {...}, ...]

    Missing detections are filled by interpolation and backward fill to ensure
    a continuous trajectory for analysis.
    """
    with open(path, "rb") as f:
        detections = pickle.load(f)

    positions = [d.get(1, []) for d in detections]
    df = pd.DataFrame(positions, columns=["x1", "y1", "x2", "y2"])
    df = df.interpolate().bfill()
    return df


def compute_vertical_dynamics(df: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """
    Add vertical motion metrics to the DataFrame.

    Columns added:
        - mid_y: midpoint of the bounding box vertically.
        - mid_y_smooth: rolling mean of mid_y.
        - delta_y: frame-to-frame change in smoothed mid_y.

    These form the basis for detecting direction changes linked to ball hits.
    """
    df = df.copy()
    df["mid_y"] = 0.5 * (df["y1"] + df["y2"])
    df["mid_y_smooth"] = df["mid_y"].rolling(
        window=smooth_window,
        min_periods=1,
    ).mean()
    df["delta_y"] = df["mid_y_smooth"].diff()
    return df


def mark_hits(df: pd.DataFrame, min_frames_for_hit: int = 25) -> pd.DataFrame:
    """
    Identify frames where the ball likely changes direction sharply,
    which is used as a proxy for ball-hit events.

    The algorithm detects sign changes in delta_y and confirms them by
    checking that the succeeding frames maintain the new sign for a
    sufficient duration.
    """
    df = df.copy()
    df["ball_hit"] = 0
    horizon = int(min_frames_for_hit * 1.2)

    for idx in range(1, len(df) - horizon):
        curr = df["delta_y"].iloc[idx]
        nxt = df["delta_y"].iloc[idx + 1]

        neg_change = curr > 0 and nxt < 0
        pos_change = curr < 0 and nxt > 0

        if not (neg_change or pos_change):
            continue

        count = 0
        for future_idx in range(idx + 1, idx + horizon + 1):
            future_val = df["delta_y"].iloc[future_idx]

            if neg_change and curr > 0 and future_val < 0:
                count += 1
            elif pos_change and curr < 0 and future_val > 0:
                count += 1

        if count >= min_frames_for_hit:
            df.loc[idx, "ball_hit"] = 1

    return df


def analyze_ball_stub(stub_path: str):
    """
    Convenience function for offline analysis of ball detections.

    Loads detections, computes vertical dynamics, marks hits,
    prints hit frames, and plots two diagnostic curves:
        - mid_y_smooth over time
        - delta_y over time
    """
    df = load_ball_detections_stub(stub_path)
    df = compute_vertical_dynamics(df)
    df = mark_hits(df)

    plt.figure()
    plt.title("Ball mid_y smoothed")
    plt.plot(df["mid_y_smooth"])

    plt.figure()
    plt.title("Delta mid_y smoothed")
    plt.plot(df["delta_y"])

    hit_indices = df.index[df["ball_hit"] == 1].tolist()
    print("Frames with ball hits:", hit_indices)
    print(df[df["ball_hit"] == 1])

    plt.show()


if __name__ == "__main__":
    analyze_ball_stub("../tracker_stubs/ball_detections.pkl")
