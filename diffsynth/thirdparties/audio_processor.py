"""HuBERT feature extraction used by TBDub inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .utils import load_audio, load_audio_chunk, pad_with_fadeout_silence


class HubertProcessor(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_frames: int = 77,
        audio_feat_window_size: int = 0,
        vid_fps: int = 25,
        aud_feature_fps: int = 50,
        embedding_num_layers: int = 4,
        sample_rate: int = 16000,
        layer_indices: tuple[int, ...] = (9, 10, 11, 12),
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.audio_feat_window_size = audio_feat_window_size
        self.embedding_num_layers = embedding_num_layers
        self.vid_fps = vid_fps
        self.aud_feature_fps = aud_feature_fps
        self.token_frames = (num_frames - 1) // 4 + 1
        self.sample_rate = sample_rate
        self.layer_indices = list(layer_indices)
        if len(self.layer_indices) != embedding_num_layers:
            raise ValueError("layer_indices must match embedding_num_layers.")

        model_path = str(Path(model_path).expanduser().resolve())
        self.model = HubertModel.from_pretrained(model_path, local_files_only=True).to(device=device)
        self.model.eval()
        self.model.feature_extractor._freeze_parameters()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self.model_type = Path(model_path).name

        configured_cache = cache_dir or os.environ.get("TBDUB_CACHE_DIR")
        self.cache_dir = Path(configured_cache).expanduser() if configured_cache else Path.home() / ".cache" / "tbdub" / "audio"

    @property
    def device(self):
        return next(self.parameters()).device

    def get_sliced_feature(self, features, video_index: int, is_first_token: bool = False):
        features = features[:, -self.embedding_num_layers :, :]
        features = rearrange(features, "f n c -> f (c n)")
        audio_length = features.shape[0]
        audio_to_video_ratio = int(self.aud_feature_fps / self.vid_fps)
        if audio_to_video_ratio != 2:
            raise ValueError("TBDub currently expects 50-fps audio features and 25-fps video.")

        center = video_index * audio_to_video_ratio
        if is_first_token:
            left = center - (3 + self.audio_feat_window_size) * audio_to_video_ratio
            right = center + (1 + self.audio_feat_window_size) * audio_to_video_ratio
        else:
            left = center - self.audio_feat_window_size * audio_to_video_ratio
            right = center + (4 + self.audio_feat_window_size) * audio_to_video_ratio

        indices = [min(audio_length - 1, max(0, index)) for index in range(left, right)]
        return torch.stack([features[index] for index in indices])

    def crop_overlap_audio_window(self, features, start_index: int):
        windows = []
        for token_index in range(self.token_frames):
            if token_index == 0:
                video_index = start_index
                window = self.get_sliced_feature(features, video_index, is_first_token=True)
            else:
                video_index = start_index + (token_index - 1) * 4 + 1
                window = self.get_sliced_feature(features, video_index)
            windows.append(window)
        return torch.stack(windows)

    def _extract_features(self, audio):
        values = np.squeeze(self.feature_extractor(audio, sampling_rate=self.sample_rate).input_values)
        values = torch.from_numpy(values).unsqueeze(0).float().to(self.device)
        outputs = self.model(values, output_hidden_states=True)
        selected = [outputs.hidden_states[index] for index in self.layer_indices]
        features = torch.stack(selected, dim=1).squeeze(0)
        return rearrange(features, "l f c -> f l c").detach().cpu().float()

    @torch.no_grad()
    def _audio2feat(self, audio_path_or_array):
        if isinstance(audio_path_or_array, np.ndarray):
            audio = audio_path_or_array
        else:
            audio = load_audio(audio_path_or_array, sr=self.sample_rate)
            pad_samples = int(round(self.num_frames / self.vid_fps * self.sample_rate))
            audio = pad_with_fadeout_silence(audio, len(audio) + pad_samples, sr=self.sample_rate)
        return self._extract_features(audio)

    def _cache_path(self, audio_path: str) -> Path:
        source = Path(audio_path).expanduser().resolve()
        stat = source.stat()
        identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{self.model_type}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{source.stem}-{digest}.npy"

    def audio2feat(self, audio_path: str, use_cache: bool = False):
        if not use_cache:
            return self._audio2feat(audio_path)

        cache_path = self._cache_path(audio_path)
        if cache_path.exists():
            try:
                return torch.from_numpy(np.load(cache_path).astype(np.float32))
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)

        features = self._audio2feat(audio_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features.numpy())
        return features

    def audio2feat_chunk(self, audio_path: str, start_index: int):
        audio_to_video_ratio = int(self.aud_feature_fps / self.vid_fps)
        first_audio_index = start_index * audio_to_video_ratio - (
            3 + self.audio_feat_window_size
        ) * audio_to_video_ratio
        last_video_center = start_index + (self.token_frames - 2) * 4 + 1
        last_audio_index = last_video_center * audio_to_video_ratio + (
            4 + self.audio_feat_window_size
        ) * audio_to_video_ratio

        padding = 10
        first_feature = max(0, first_audio_index - padding)
        last_feature = last_audio_index + padding
        audio = load_audio_chunk(
            audio_path,
            first_feature / self.aud_feature_fps,
            last_feature / self.aud_feature_fps,
            sr=self.sample_rate,
        )
        features = self._audio2feat(audio)
        adjusted_start = start_index - first_feature // audio_to_video_ratio
        return self.crop_overlap_audio_window(features, int(adjusted_start))


__all__ = ["HubertProcessor"]
