"""Minimal DWPose wrapper used by TBDub preprocessing."""

import torch

from .wholebody import Wholebody


class DWposeDetector:
    """Detect whole-body landmarks and return the fields needed for face crops."""

    def __init__(
        self,
        det_config,
        det_ckpt,
        pose_config,
        pose_ckpt,
        device="cuda",
        type="pth",
        cuda_stream=None,
    ):
        if cuda_stream is None and torch.cuda.is_available():
            cuda_stream = torch.cuda.current_stream()
        self.pose_estimation = Wholebody(
            det_config=det_config,
            det_ckpt=det_ckpt,
            pose_config=pose_config,
            pose_ckpt=pose_ckpt,
            device=device,
            type=type,
            cuda_stream=cuda_stream,
        )

    def __call__(self, image_np_hwc, box_ext=None):
        _, _, candidate, subset, _, bbox = self.pose_estimation(
            image_np_hwc.copy(),
            box_ext=box_ext,
        )
        return candidate, subset, bbox


__all__ = ["DWposeDetector"]
