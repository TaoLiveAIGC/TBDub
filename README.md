# TBDub

TBDub is an audio-driven video dubbing system designed for high-fidelity identity preservation, temporal consistency, and robust lip synchronization under challenging motion, pose, occlusion, and cross-domain conditions.

This repository contains the inference code only. Model checkpoints are not included yet.

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
├── models_t5_umt5-xxl-enc-bf16.safetensors
├── Wan2.2_VAE.safetensors
├── null_prompt_emb.pt
├── umt5-xxl/
└── hubert-large-ll60k/

dwpose_tools/models/
├── yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
└── rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth
```

Checkpoint download links and model terms will be added before the public release.

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

The two DiT files are loaded in order: the fine-tuned checkpoint overlays the base checkpoint. The defaults above reproduce the parameter configuration used by the project inference script. A single consolidated checkpoint can also be supplied with one `--dit-checkpoint` path.

Useful options:

- `--cropped-input`: the source is already an aligned face video.
- `--motion-from-latents`: pass the previous segment's tail latents to the next segment.
- `--per-chunk-audio`: extract HuBERT features separately for each segment.
- `--preprocess-cache cache/sample.pkl`: reuse face crops and bounding boxes.
- `--save-comparison`: additionally save side-by-side diagnostic videos.
- `--preprocess-only`: run face preprocessing without loading TBDub checkpoints.

Run `python inference.py --help` for checkpoint-path and sampling options.

Only load preprocessing cache files that you created or trust, because Python pickle files can execute code while loading.

## Repository structure

```text
TBDub/
├── inference.py                  # public inference entry point
├── preprocessing.py              # DWPose-based face crop and tracking
├── video_utils.py                # video blending, color, and audio muxing
├── infer.sh                      # minimal shell wrapper
├── diffsynth/
│   ├── pipelines/tbdub.py        # TBDub sampling pipeline
│   ├── models/tbdub_dit.py       # TBDub DiT architecture
│   └── ...                       # required DiffSynth runtime modules
└── dwpose_tools/                 # minimal DWPose inference wrapper/configs
```

Internal batch jobs, data-generation scripts, evaluation utilities, experiment launchers, editor settings, and hard-coded cluster paths from the development workspace have been intentionally excluded.

## Acknowledgements and license

TBDub builds on open-source components including DiffSynth-Studio, Wan, HuBERT, and DWPose. This cleaned release was derived from and substantially modified relative to the X-Dub codebase; provenance and modification notes are retained in [NOTICE.md](NOTICE.md).

The code in this repository is released under the Apache License 2.0 unless a subcomponent states otherwise. See [LICENSE](LICENSE).
