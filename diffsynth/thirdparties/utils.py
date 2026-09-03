import ffmpeg
import numpy as np
import random
import torch

SAMPLE_RATE = 16000

# Default fade-out length (in milliseconds) applied at the boundary between
# real audio and the trailing silence padding. Avoids a click/discontinuity
# (which would otherwise produce broadband noise in hubert features near the
# boundary) by linearly ramping the last few samples down to zero before
# appending silence.
DEFAULT_FADE_OUT_MS = 10.0


def pad_with_fadeout_silence(
    audio: np.ndarray,
    target_length: int,
    sr: int = SAMPLE_RATE,
    fade_ms: float = DEFAULT_FADE_OUT_MS,
) -> np.ndarray:
    """Pad `audio` to `target_length` samples by appending silence.

    The original audio is left completely untouched. Only the appended segment
    is shaped: its first `fade_samples` ramp linearly from the last real-audio
    sample value down to 0, then the remainder stays at 0. This keeps the
    speech -> silence boundary continuous (click-free) without modifying the
    real audio.

    No-op if `audio` is already at or beyond `target_length`.
    """
    audio = np.asarray(audio)
    if len(audio) >= target_length:
        return audio
    out = audio.astype(np.float32, copy=True)
    pad_len = target_length - len(out)
    padding_value = np.zeros(pad_len, dtype=out.dtype)
    fade_samples = max(0, min(pad_len, int(round(fade_ms / 1000.0 * sr))))
    if fade_samples > 0 and len(out) > 0:
        last_sample = float(out[-1])
        # ramp from the last real sample value down to (but not including) 0
        ramp = np.linspace(last_sample, 0.0, fade_samples + 1, dtype=np.float32)[1:]
        padding_value[:fade_samples] = ramp
    return np.concatenate([out, padding_value])


def load_audio(file: str, sr: int = SAMPLE_RATE):
    if file.endswith(".npy"):
        return np.load(file).astype(np.float32)

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_chunk(file: str, start_sec: float, end_sec: float, sr: int = SAMPLE_RATE):
    """Load a specific time range [start_sec, end_sec] from an audio file.

    If the requested range extends past the end of the real audio, the returned
    array is padded with trailing silence (zeros) up to the full requested
    length, with a short linear fade-out applied to the real audio tail so the
    speech-to-silence boundary is click-free. This gives the final chunk a
    real "speech -> silence" context so the model closes the mouth naturally,
    instead of the audio-feature window being clamped to (i.e. repeating) the
    last speech frame.
    """
    expected_samples = max(0, int(round((end_sec - start_sec) * sr)))

    if file.endswith(".npy"):
        audio = np.load(file).astype(np.float32)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        chunk = audio[start_sample:end_sample]
        if len(chunk) < expected_samples:
            chunk = pad_with_fadeout_silence(chunk, expected_samples, sr=sr)
        return chunk

    try:
        input_kwargs = {"threads": 0, "ss": start_sec}
        out, _ = (
            ffmpeg.input(file, **input_kwargs)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr, t=end_sec - start_sec)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio chunk: {e.stderr.decode()}") from e

    chunk = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    if len(chunk) < expected_samples:
        chunk = pad_with_fadeout_silence(chunk, expected_samples, sr=sr)
    return chunk

def post_process_audio_feat(audio_feat, begin_process_rate=0.1, end_process_rate=0.1):
    f, p, c = audio_feat.shape
    left_repeat_length = p // 2 + 2

    if random.random() < begin_process_rate:
        audio_feat_new = audio_feat.clone()
        audio_feat_new[0, 0:left_repeat_length] = audio_feat[0, left_repeat_length].unsqueeze(0).repeat(left_repeat_length, 1)
        audio_feat = audio_feat_new 

    return audio_feat
