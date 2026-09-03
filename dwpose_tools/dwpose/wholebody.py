import numpy as np
import torch
from contextlib import contextmanager

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector


@contextmanager
def patch_torch_load_weights_only_false(): # to make mmengine support PyTorch 2.6+
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat
    try:
        yield
    finally:
        torch.load = original_torch_load


class Wholebody:
    def __init__(self, det_config=None, det_ckpt=None, 
                 pose_config=None, pose_ckpt=None,
                det_onnx=None, pose_onnx=None, \
                det_trt=None, pose_trt=None, \
                device="cpu", type='pt', cuda_stream=None):
        self.is_rtmw = True        
        self.type = type
        self.cuda_stream = cuda_stream
        if self.type not in ("pt", "pth"):
            raise NotImplementedError("Current local lip-sync preprocessing only supports the PyTorch DWpose path.")

        with patch_torch_load_weights_only_false():
            self.detector = init_detector(det_config, det_ckpt, device=device)
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
            self.pose_estimator = init_pose_estimator(
                pose_config,
                pose_ckpt,
                device=device,
            )
    
    def __call__(self, oriImg, box_ext=None):
        if box_ext is None:
            det_result = inference_detector(self.detector, oriImg)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)]
            bboxes = bboxes[nms(bboxes, 0.7)]

            if len(bboxes) == 1:
                bboxes = bboxes[:, :4]
            elif len(bboxes) >= 2:
                final_scores = bboxes[:, 4:]
                bboxes = bboxes[:, :4]
                best_idx = np.argmax(final_scores)
                bboxes = bboxes[best_idx][np.newaxis, :]
            else:
                bboxes = np.empty((0, 4), dtype=np.float32)
        else:
            bboxes = box_ext

        if len(bboxes) == 0:
            pose_results = inference_topdown(self.pose_estimator, oriImg)
        else:
            pose_results = inference_topdown(self.pose_estimator, oriImg, bboxes)
        preds = merge_data_samples(pose_results).pred_instances

        keypoints = preds.get('transformed_keypoints', preds.keypoints)
        if 'keypoint_scores' in preds:
            scores = preds.keypoint_scores
        else:
            scores = np.ones(keypoints.shape[:-1])

        if 'keypoints_visible' in preds:
            visible = preds.keypoints_visible
        else:
            visible = np.ones(keypoints.shape[:-1])
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None], visible[..., None]),
            axis=-1,
        )
        det_result = bboxes

        keypoints_info_133 = keypoints_info.copy()
        scores_133 = scores.copy()

        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.5,
            keypoints_info[:, 6, 2:4] > 0.5).astype(int)
        
        if self.is_rtmw:
            neck[:, 2:3] = neck[:, 2:3] * 10
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores, visible = keypoints_info[
            ..., :2], keypoints_info[..., 2], keypoints_info[..., 3]
   
        return keypoints_info_133, scores_133, keypoints, scores, visible,  det_result
