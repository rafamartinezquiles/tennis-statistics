from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np
import torch
from torchvision import models, transforms


@dataclass
class CourtLineDetectorConfig:
    """
    Configuration for the ResNet-based court line keypoint regressor.

    Attributes
    ----------
    model_path : str
        Path to a trained checkpoint containing weights for the regression head.
    input_size : int
        Spatial size to which input frames are resized before prediction.
    num_keypoints : int
        Number of court keypoints predicted (each producing x, y).
    """
    model_path: str
    input_size: int = 224
    num_keypoints: int = 14


class CourtLineDetector:
    """
    Predicts 2D tennis-court keypoints from a single RGB frame using a
    ResNet50 backbone followed by a regression layer.

    The model outputs a flat vector of length 2 * num_keypoints containing
    pixel coordinates normalized to the 224×224 training resolution; these
    are rescaled back to the original frame size.
    """

    def __init__(self, cfg: CourtLineDetectorConfig) -> None:
        """
        Load the trained model and preprocessing pipeline.
        """
        self.cfg = cfg
        self.model = self._load_model(cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((cfg.input_size, cfg.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self, cfg: CourtLineDetectorConfig) -> torch.nn.Module:
        """
        Load a ResNet50 backbone with a regression head for keypoint prediction.

        The final fully connected layer is replaced with a Linear layer that
        outputs 2 × num_keypoints values.
        """
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        base_model.fc = torch.nn.Linear(
            base_model.fc.in_features,
            cfg.num_keypoints * 2,
        )

        state = torch.load(cfg.model_path, map_location="cpu")
        base_model.load_state_dict(state)
        base_model.eval()
        return base_model

    def predict_keypoints(self, bgr_image: np.ndarray) -> List[float]:
        """
        Predict a flattened keypoint array [x0, y0, x1, y1, ...] in original
        image coordinates.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)

        kps = out.squeeze().cpu().numpy()
        h, w = bgr_image.shape[:2]

        # Rescale: model outputs relative to the 224×224 input grid
        kps[0::2] *= w / self.cfg.input_size
        kps[1::2] *= h / self.cfg.input_size

        return kps.tolist()

    @staticmethod
    def draw_keypoints(
        frame: np.ndarray,
        flat_keypoints: Iterable[float],
        radius: int = 5,
    ) -> np.ndarray:
        """
        Render predicted keypoints and their indices onto a frame.
        """
        output = frame.copy()
        flat = list(flat_keypoints)

        for idx in range(0, len(flat), 2):
            x = int(flat[idx])
            y = int(flat[idx + 1])
            cv2.circle(output, (x, y), radius, (0, 0, 255), thickness=-1)
            cv2.putText(
                output,
                str(idx // 2),
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        return output

    def draw_keypoints_on_video(
        self,
        frames: List[np.ndarray],
        flat_keypoints: Iterable[float],
    ) -> List[np.ndarray]:
        """
        Apply the keypoint overlay to each frame in a sequence.

        Returns a new list of frames with the same ordering.
        """
        return [self.draw_keypoints(f, flat_keypoints) for f in frames]
