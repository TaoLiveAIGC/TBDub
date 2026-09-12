<div align="center">
  <h1>TBDub: Production-Oriented Visual Dubbing</h1>
  <p><strong>Bihan Li<sup>*</sup>, Xinyang Li<sup>*</sup>, Zeran Xu, Meiguang Jin<sup>†</sup>, Junfeng Ma</strong></p>
  <p>Taobao &amp; Tmall Group, Alibaba Group</p>
  <p>
    <a href="https://arxiv.org/abs/2609.06144"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv Paper"></a>
    <a href="https://taoliveaigc.github.io/TBDub/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://github.com/TaoLiveAIGC/TBDub"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="GitHub"></a>
    <a href="https://huggingface.co/TaoLiveAIGC/TBDub"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="Hugging Face"></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow" alt="License: Apache-2.0"></a>
  </p>
</div>

**High-quality lip sync. Fast, multilingual, and robust.**

Give an existing video a new voice. TBDub synchronizes the speaker's lips with new speech while preserving their appearance, motion, and background.

**Code · Teacher · Student — all open source under Apache-2.0.**

## Why TBDub?

- **🏆 Leading perceptual quality.** Highest mean opinion scores in all three dimensions among the open-source methods in our study: **3.85 lip sync**, **3.80 visual quality**, and **3.78 identity preservation**. Student leads the first two; Teacher leads identity. [See results](https://taoliveaigc.github.io/TBDub/#subjective-results).
- **⚡ Fast, two-step generation.** **7.13 FPS** on a single **NVIDIA H20 at 512 × 512**, or **13.93×** the Teacher's core generation speed. [See benchmark](https://taoliveaigc.github.io/TBDub/#efficiency).
- **🌍 Multilingual lip sync.** Lip sync across languages, with examples including **English, Chinese, Japanese, Korean, and Russian**. See lip sync and identity preservation in multilingual reconstruction. [Watch examples](https://taoliveaigc.github.io/TBDub/#subgroup-self-multilingual).
- **🛡️ Robust in challenging scenes.** Stable lip sync and appearance through **large head turns, hands and microphones over the mouth, and rapid motion**. [Head turns](https://taoliveaigc.github.io/TBDub/#cross-large-pose) · [Occlusions](https://taoliveaigc.github.io/TBDub/#cross-occlusion-02).

MOS: 38 TalkVid clips, 114 ratings per method, on a 0–5 scale. Speed: VAE encode to decode, excluding preprocessing, audio encoding, and file output. [Full evaluation protocol](https://arxiv.org/html/2609.06144v1).

**[Watch comparisons](https://taoliveaigc.github.io/TBDub/#demos) · [Get the code](https://github.com/TaoLiveAIGC/TBDub#quick-start) · [Download models](https://huggingface.co/TaoLiveAIGC/TBDub)**

## Inference pipeline

1. Detect and track facial landmarks with DWPose.
2. Crop and align the face region to `512 x 512`.
3. Encode the reference video into latent tokens.
4. Generate audio-aligned target latents with the TBDub DiT and cross-clip motion conditioning.
5. Decode the generated latents, correct color statistics, and paste the face region back into the source video.

For an already aligned `512 x 512` face video, pass `--cropped-input` to skip steps 1 and 5.

## Requirements

- Linux with an NVIDIA GPU
- Python 3.10+
- CUDA-compatible PyTorch
- `ffmpeg` available on `PATH`

Create an environment and install PyTorch for your CUDA version first. Then install the remaining packages:

```bash
pip install -r requirements.txt
pip install -U openmim
mim install "mmcv==2.1.0"
```

The DWPose preprocessing path also uses `mmengine`, `mmdet`, and `mmpose`, which are listed in `requirements.txt`. Follow the official OpenMMLab compatibility matrix if your CUDA or PyTorch version requires different package versions.

## Checkpoints

Place the following files under `checkpoints/`, or provide their paths through the corresponding command-line options:

```text
checkpoints/
├── tbdub_base.safetensors
├── tbdub_finetune.safetensors
├── tbdub_student.safetensors       # standalone BF16 Student; no Base needed
├── Wan2.2_VAE.safetensors
├── null_prompt_emb.pt
└── hubert-large-ll60k/

dwpose_tools/models/
├── yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
└── rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth
```



Download the configuration manifest together with the model variant you plan to use.

Teacher (30 steps):

```bash
hf download TaoLiveAIGC/TBDub \
  config.json null_prompt_emb.pt \
  tbdub_base.safetensors tbdub_finetune.safetensors \
  --local-dir checkpoints
```

Student (2 steps):

```bash
hf download TaoLiveAIGC/TBDub \
  config.json null_prompt_emb.pt \
  tbdub_student.safetensors \
  --local-dir checkpoints
```

The versioned `config.json` records the runtime layout and is the model-level query file Hugging Face uses for download statistics. The remaining auxiliary checkpoints can be downloaded from [KlingTeam/X-Dub on Hugging Face](https://huggingface.co/KlingTeam/X-Dub).

Both Teacher and Student use the precomputed context in `null_prompt_emb.pt`.
This file is required even though the prompt is empty; its contents are not an
all-zero tensor. Inference does not load a T5 text encoder or tokenizer, so do not
download `models_t5_umt5-xxl-enc-bf16.safetensors` or `umt5-xxl/` for this code.
The former `--text-encoder-checkpoint` and `--tokenizer-path` options have been
removed. Direct pipeline callers must supply `prompt_emb`; `from_pretrained`
no longer accepts `tokenizer_config`. The DiT's context projection and attention
weights remain required. Transformers is still used by HuBERT.

## Quick start

```bash
bash infer.sh path/to/source.mp4 path/to/driving.wav results
```

Equivalent Python command:

```bash
python inference.py \
  --video path/to/source.mp4 \
  --audio path/to/driving.wav \
  --dit-checkpoint \
    checkpoints/tbdub_base.safetensors \
    checkpoints/tbdub_finetune.safetensors \
  --ref-cfg-scale 2.0 \
  --audio-cfg-scale 6.0 \
  --num-inference-steps 30 \
  --seed 42 \
  --output-dir results
```

For Teacher, the two DiT files are loaded in order: the fine-tuned checkpoint overlays the base checkpoint. The defaults above reproduce the parameter configuration used by the project inference script. Student uses a single complete checkpoint.

### Distilled Student inference

A compatible DMD2 Student checkpoint can run with two denoising steps and no classifier-free guidance:

```bash
python inference.py \
  --video path/to/source.mp4 \
  --audio path/to/driving.wav \
  --dit-checkpoint checkpoints/tbdub_student.safetensors \
  --inference-mode student \
  --num-student-steps 2 \
  --sigma-shift 1.0 \
  --motion-from-latents \
  --seed 42 \
  --output-dir results
```

The Student checkpoint contains the entire DiT; it is not an overlay and does
not require `tbdub_base.safetensors`. The BF16 inference file is about 12.59 GB,
compared with 37.77 GB for the former Base + mostly-FP32 Student download.
VAE, HuBERT, `null_prompt_emb.pt`, and the selected preprocessing models are
still required. Specify individual files in `hf download` as shown above to
avoid downloading Teacher weights.

With `--inference-mode student`, omitting `--dit-checkpoint` also defaults to
the single `checkpoints/tbdub_student.safetensors` file. Existing FP32 Student
files still work alone; inference casts them to BF16. Student mode uses
single-branch denoising and first-clip temporal padding by default. The checkpoint
must be distilled for the same two-step schedule, sigma shift, HuBERT conditioning,
and 77-frame temporal layout.

To export an existing original Student file without overwriting it:

```bash
python scripts/convert_student_bf16.py \
  checkpoints/tbdub_student.safetensors \
  checkpoints/tbdub_student_bf16.safetensors \
  --report student_conversion.json
```

The CPU exporter checks the complete model signature, retains all parameter
names and shapes, verifies every saved tensor against the source cast to BF16,
and records a SHA-256 checksum. It needs enough host memory for the complete
BF16 model and serialization buffers. This reduces download and storage size;
the inference model was already BF16, so its resident GPU memory does not halve.

Useful options:

- `--cropped-input`: the source is already an aligned face video.
- `--motion-from-latents`: pass the previous segment's tail latents to the next segment.
- `--per-chunk-audio`: extract HuBERT features separately for each segment.
- `--preprocess-cache cache/sample.pkl`: reuse face crops and bounding boxes.
- `--save-comparison`: additionally save side-by-side diagnostic videos.
- `--preprocess-only`: run face preprocessing without loading TBDub checkpoints.
- `--no-student-first-clip-padding`: disable the default five-frame Student pre-roll.

Run `python inference.py --help` for checkpoint-path and sampling options.

Only load preprocessing cache files that you created or trust, because Python pickle files can execute code while loading.

MediaPipe preprocessing is available as an alternative to the OpenMMLab path.
Install `requirements-mediapipe.txt` instead of `requirements.txt`, after installing
a CUDA-compatible PyTorch/torchvision build. This environment uses MediaPipe 1.0.1
Tasks (not the legacy `mp.solutions` API). Full-range face detection is followed
by an explicit crop and the official Face Landmarker task on CPU. Preprocessing
runs in a separate CPU process to isolate MediaPipe native libraries from PyTorch.
It does not require MMCV, MMEngine, MMDetection, or MMPose. Use only the pinned
`opencv-contrib-python` package in this environment; do not also install
`opencv-python`, since both provide `cv2`.

Download the [Face Landmarker model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
(3,758,596 bytes):

```bash
mkdir -p checkpoints
curl -fL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -o checkpoints/face_landmarker.task
echo '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  checkpoints/face_landmarker.task' | sha256sum -c -
curl -fL https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite \
  -o checkpoints/blaze_face_full_range.tflite
echo '3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b  checkpoints/blaze_face_full_range.tflite' | sha256sum -c -
```

Add these arguments to the Student inference command above:

```bash
--preprocess-backend mediapipe \
--mediapipe-model checkpoints/face_landmarker.task \
--mediapipe-detector-model checkpoints/blaze_face_full_range.tflite \
--preprocess-report results/face_detection.json
```

The additional full-range detector is 1,083,786 bytes. The backend selects the
highest-scoring face in each frame and is intended for single-person footage.
It expands each detection by 1.5 for the landmark task, then constructs the final
Student face crop with the existing 1.45 padding
factor and smoothing policy, and preserves the crop/paste-back interface. Brief
missing detections (up to 0.4 seconds) are interpolated and listed in the report;
long gaps or a completely missing face raise an error. Use separate preprocessing
caches for each backend. Model landmarks differ between backends, so compare the
resulting crops and generated videos before adopting it for new footage.

## Citation

```bibtex
@misc{li2026tbdubproductionorientedvisualdubbing,
  title={TBDub: Production-Oriented Visual Dubbing},
  author={Bihan Li and Xinyang Li and Zeran Xu and Meiguang Jin and Junfeng Ma},
  year={2026},
  eprint={2609.06144},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.06144},
}
```

## License

The code and released model weights are available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Third-party dependencies and checkpoints remain subject to their respective licenses.
