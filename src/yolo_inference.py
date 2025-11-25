from ultralytics import YOLO  # YOLO11 family

if __name__ == "__main__":
    """
    Minimal YOLO11 inference script for tracking on a video file.

    This script:
        - Loads a YOLO model (official or custom fine-tuned weights).
        - Runs object tracking on the input video.
        - Saves the annotated output automatically under runs/track/.

    Adjust the model path, confidence threshold, or input video path
    as needed for your workflow.
    """
    model = YOLO("yolo11x.pt")  # Replace with custom weights if available

    # Run tracking on a video and save results
    results = model.track(
        source="input/input_video.mp4",
        conf=0.2,       # minimum confidence for detections
        save=True,      # save output video to runs/
        verbose=True,   # print detailed inference logs
    )
