<div align="center">
  <h1>TBDub: Production-Oriented Visual Dubbing</h1>
  <p><strong>Bihan Li<sup>*</sup>, Xinyang Li<sup>*</sup>, Zeran Xu, Meiguang Jin<sup>†</sup>, Junfeng Ma</strong></p>
  <p>Taobao &amp; Tmall Group, Alibaba Group</p>
  <p>
    <a href="https://arxiv.org/abs/2609.06144"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv Paper"></a>
    <a href="https://taoliveaigc.github.io/TBDub/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://github.com/TaoLiveAIGC/TBDub"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="GitHub"></a>
    <a href="https://huggingface.co/TaoLiveAIGC/TBDub"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="Hugging Face"></a>
    <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Release-V1.1-green" alt="Release V1.1"></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow" alt="License: Apache-2.0"></a>
  </p>
</div>

**High-quality lip sync. Fast, multilingual, and robust.**

Give an existing video a new voice. TBDub synchronizes the speaker's lips with new speech while preserving their appearance, motion, and background.

**Code · Teacher · Student — all open source under Apache-2.0.**

## What's new in V1.1

V1.1 makes the existing Teacher and Student easier to deploy:

- **One installation:** `requirements.txt` includes the validated MediaPipe 0.10.21 face preprocessing setup.
- **Compact models:** download one complete BF16 DiT checkpoint, about **12.59 GB**, for the variant you want. Including shared auxiliary model files, a single variant needs about **15.27 GB** in total.
- **Optional lower GPU memory:** add `--cpu-offload` when GPU memory is limited. Measured process peaks with this option were **15.01 GiB for Student** and **19.49 GiB for Teacher** on the workload described below.

[Full V1.1 changelog](CHANGELOG.md) · [Project update](https://taoliveaigc.github.io/TBDub/#v1-1)

## Why TBDub?

- **🏆 Leading perceptual quality.** Highest mean opinion scores in all three dimensions among the open-source methods in our study: **3.85 lip sync**, **3.80 visual quality**, and **3.78 identity preservation**. Student leads the first two; Teacher leads identity. [See results](https://taoliveaigc.github.io/TBDub/#subjective-results).
- **⚡ Fast, two-step generation.** **7.13 FPS** on a single **NVIDIA H20 at 512 × 512**, or **13.93×** the Teacher's core generation speed. [See benchmark](https://taoliveaigc.github.io/TBDub/#efficiency).
- **🌍 Multilingual lip sync.** Lip sync across languages, with examples including **English, Chinese, Japanese, Korean, and Russian**. See lip sync and identity preservation in multilingual reconstruction. [Watch examples](https://taoliveaigc.github.io/TBDub/#subgroup-self-multilingual).
- **🛡️ Robust in challenging scenes.** Stable lip sync and appearance through **large head turns, hands and microphones over the mouth, and rapid motion**. [Head turns](https://taoliveaigc.github.io/TBDub/#cross-large-pose) · [Occlusions](https://taoliveaigc.github.io/TBDub/#cross-occlusion-02).

MOS: 38 TalkVid clips, 114 ratings per method, on a 0–5 scale. Speed: VAE encode to decode, excluding preprocessing, audio encoding, and file output. [Full evaluation protocol](https://arxiv.org/html/2609.06144v1).

**[Watch comparisons](https://taoliveaigc.github.io/TBDub/#demos) · [Get the code](https://github.com/TaoLiveAIGC/TBDub#quick-start) · [Download models](https://huggingface.co/TaoLiveAIGC/TBDub)**

## Inference pipeline

1. Detect faces with MediaPipe, crop each detected region, and extract facial landmarks.
2. Crop and align the face region to `512 x 512`.
3. Encode the reference video into latent tokens.
4. Generate audio-aligned target latents with the TBDub DiT and cross-clip motion conditioning.
5. Decode the generated latents, correct color statistics, and paste the face region back into the source video.

For an already aligned `512 x 512` face video, pass `--cropped-input` to skip steps 1 and 5.

## Requirements

- Linux with an NVIDIA GPU
- Python 3.10 (validated environment)
- CUDA-compatible PyTorch and torchvision
- `ffmpeg` available on `PATH`

Clone the repository and create a fresh environment:

```bash
git clone https://github.com/TaoLiveAIGC/TBDub.git
cd TBDub
python3.10 -m venv .venv
source .venv/bin/activate
```

Install PyTorch for your CUDA environment, then the single dependency file.
The validated stack uses PyTorch 2.9.0 and torchvision 0.24.0 with CUDA 12.8:

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Keep the pinned MediaPipe, Protobuf, CPU JAX, and `opencv-contrib-python`
versions from this file. Install only that OpenCV package in this environment.

## Checkpoints

Choose **one** model variant. Each uses a single complete BF16 DiT file; you do
not need the other variant or any additional DiT checkpoint.

Student (two steps):

```bash
hf download TaoLiveAIGC/TBDub \
  config.json null_prompt_emb.pt tbdub_student.safetensors \
  --local-dir checkpoints
```

Or Teacher (30 steps):

```bash
hf download TaoLiveAIGC/TBDub \
  config.json null_prompt_emb.pt tbdub_teacher.safetensors \
  --local-dir checkpoints
```

Both variants also need the shared VAE and HuBERT files:

```bash
hf download KlingTeam/X-Dub Wan2.2_VAE.safetensors --local-dir checkpoints
hf download facebook/hubert-large-ll60k \
  config.json preprocessor_config.json pytorch_model.bin \
  --local-dir checkpoints/hubert-large-ll60k
```

For face detection and landmarks, download the Face Landmarker bundle
(3,758,596 bytes). The face detector is already included in MediaPipe:

```bash
mkdir -p checkpoints
curl -fL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -o checkpoints/face_landmarker.task
echo '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  checkpoints/face_landmarker.task' | sha256sum -c -
```

For example, a complete Student setup contains:

```text
checkpoints/
├── config.json
├── tbdub_student.safetensors
├── Wan2.2_VAE.safetensors
├── null_prompt_emb.pt
├── face_landmarker.task
└── hubert-large-ll60k/
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```

For Teacher, replace the Student DiT file with `tbdub_teacher.safetensors`.
The selected DiT plus all shared model files totals about **15.27 GB** (decimal
GB), excluding Python packages. Use the individual download lists above to
avoid downloading both variants. Keep `null_prompt_emb.pt`: it supplies the
fixed context used by both models. Already aligned inputs with `--cropped-input`
can skip the Face Landmarker download.

## Quick start

Student, with two-step generation and latent motion:

```bash
python inference.py \
  --video path/to/source.mp4 \
  --audio path/to/driving.wav \
  --inference-mode student \
  --num-student-steps 2 \
  --sigma-shift 1.0 \
  --motion-from-latents \
  --seed 42 \
  --output-dir results
```

Teacher:

```bash
bash infer.sh path/to/source.mp4 path/to/driving.wav results
```

Equivalent Teacher Python command:

```bash
python inference.py \
  --video path/to/source.mp4 \
  --audio path/to/driving.wav \
  --inference-mode teacher \
  --ref-cfg-scale 2.0 \
  --audio-cfg-scale 6.0 \
  --num-inference-steps 30 \
  --seed 42 \
  --output-dir results
```

These commands select the matching DiT checkpoint automatically and use MediaPipe
for full-frame videos. Models remain on the GPU by default for faster inference.
If a run fails because GPU memory is insufficient, add `--cpu-offload` to the
Python command or `infer.sh` command and run again. Memory mode is selected manually; the program
does not automatically detect available memory or switch modes after an error.

### Lower GPU memory usage

With `--cpu-offload`, DiT and VAE weights move between CPU memory and the GPU at stage
boundaries. The active DiT stays on the GPU throughout denoising. HuBERT runs
on the GPU in FP32 for feature extraction, then returns to CPU memory.
DiT/VAE computation stays BF16.

This saves GPU memory while using more system RAM and CPU/GPU transfer time.
Omit the option to use the default GPU-resident mode.

For example, rerun Teacher with:

```bash
bash infer.sh path/to/source.mp4 path/to/driving.wav results --cpu-offload
```

Measured on one RTX PRO 5000 72GB: 512×512, 15.04-second audio, 376 output
frames; Teacher 30 steps with reference/audio CFG 2.5/10, Student two steps
with latent motion. The table reports sampled `nvidia-smi` process peaks,
including CUDA overhead. Both columns use the same V1.1 models.

| Model | Default GPU-resident mode | With `--cpu-offload` |
|---|---:|---:|
| Teacher | 27.95 GiB | 19.49 GiB |
| Student | 27.93 GiB | 15.01 GiB |

Both offload runs completed under a 22 GiB PyTorch allocator limit. Their
376 decoded video frames and audio matched the respective resident-mode
outputs exactly. These are workstation measurements, separate from the
paper's H20 speed benchmark; they are not an RTX 4090 hardware test.

Host-process RSS peaked at 21.9 GiB for Teacher and 21.3 GiB for Student;
the system and other applications need additional RAM. In single cold runs,
inference plus output time changed from 563.6 to 595.3 seconds for Teacher
and 63.1 to 82.1 seconds for Student. These timings exclude preprocessing,
model loading, and process startup.

Teacher and Student have the same parameter count and BF16 weight size.
Teacher evaluates three CFG branches together, sharing weights but producing
more intermediate tensors and workspaces; Student evaluates one branch.
In this test, final VAE decoding dominated both resident-mode peaks, whereas
DiT denoising dominated with CPU offload. Longer inputs can still require
more memory because final VAE decoding spans the assembled sequence.

### Face preprocessing

MediaPipe detects the face, crops the detected region, and extracts landmarks
with Face Landmarker in a separate CPU process. It selects the highest-scoring
face per frame and is intended for single-person footage. Short detection gaps
(up to 0.4 seconds) are interpolated; longer gaps raise an error. To save
per-frame diagnostics, add `--preprocess-report results/face_detection.json`.

The pinned 0.10.21 version has been validated with the included full-range
face detector and CPU landmark task. Keep the versions in `requirements.txt`
when reproducing these results.

### Useful options

- `--cpu-offload`: reduce GPU memory usage by moving inactive models to system RAM.
- `--cropped-input`: use an already aligned face video and skip preprocessing and paste-back.
- `--preprocess-cache cache/sample.pkl`: reuse face crops and bounding boxes.
- `--preprocess-only`: preview face preprocessing without loading the generative models.
- `--mediapipe-model path/to/face_landmarker.task`: override the landmark model path.
- `--save-comparison`: save an additional comparison video.
- `--per-chunk-audio`: extract HuBERT features separately for each segment.

Run `python inference.py --help` for checkpoint and sampling options. Only load
preprocessing caches you created or trust, because they use Python pickle.
When a cache format is incompatible, regenerate it with the current version.

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
