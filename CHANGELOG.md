# Changelog

## V1.1 — 2026-09-12

This release collects the inference and deployment improvements since V1.0.
It does not introduce retrained Teacher or Student models.

- **MediaPipe is the default face preprocessing backend.** The pinned 0.10.21
  environment uses face detection, an explicit crop, and Face Landmarker on
  CPU. The default installation needs no MMCV, MMEngine, MMDetection, or MMPose.
  DWPose remains an [optional compatibility path](docs/dwpose.md).
- **Remove unused T5 loading and downloads.** Inference uses the existing
  `null_prompt_emb.pt` context directly. That embedding and the DiT context
  layers remain required; HuBERT still uses Transformers.
- **Publish one complete BF16 checkpoint per model.** Each is about 12.59 GB.
  Student no longer needs a Base download, and its storage changes from mostly
  FP32 to BF16. Teacher combines the previous Base and Fine-tune files using
  the existing runtime merge rule. No inference parameters are removed.
- **Add optional `--cpu-offload`.** Inactive DiT/VAE weights remain in system
  memory, and HuBERT uses the GPU only while extracting audio features. This
  reduces GPU memory at the cost of host memory and transfer time.
- **Document auxiliary downloads and measured memory.** The README provides
  individual commands for VAE, HuBERT, and Face Landmarker plus the GPU-memory
  protocol and Teacher CFG explanation.

### Memory validation

| Model | GPU-resident process peak | With CPU offload |
|---|---:|---:|
| Student | 27.93 GiB | 15.01 GiB |
| Teacher | 27.95 GiB | 19.49 GiB |

Both columns use the same V1.1 configuration; this comparison isolates CPU
offload and is not a complete V1.0-versus-V1.1 benchmark. Measurements used an
RTX PRO 5000 72GB, 512×512 crops, 15.04-second audio and 376 output frames. CPU
offload ran with a 22 GiB PyTorch allocator limit. Teacher used 30 steps and
reference/audio CFG 2.5/10; Student used two steps and latent motion.

The complete decoded videos and audio were identical to the corresponding
GPU-resident outputs. Single-pass inference plus output time increased from
63.1 to 82.1 seconds for Student and 563.6 to 595.3 seconds for Teacher. Host
process RSS peaked at 21.3 and 21.9 GiB respectively; additional RAM is needed
for the operating system. This was not an RTX 4090 hardware benchmark.

Teacher and Student share the same architecture, parameter count, and BF16 weight size.
Teacher's three CFG branches share weights but need more intermediate tensors
than Student's single branch. The overall peak also depends on the stage:
in this test, with weights kept on the GPU, final VAE decoding dominated both peaks. Longer
videos can require more memory even with CPU offload.

### Migrating from V1.0

Use a fresh environment with `requirements.txt` and download
`checkpoints/face_landmarker.task` as shown in the README. Existing
`requirements-mediapipe.txt` commands remain valid. The default preprocessing
backend changes to MediaPipe; use `--preprocess-backend dwpose` with its optional
environment to retain DWPose. Regenerate MediaPipe caches from older backend
revisions and keep caches separate between backends.

Download only the Teacher or Student DiT file you need, plus the shared VAE,
HuBERT, and `null_prompt_emb.pt`. T5 and its tokenizer are no longer used.
Original Student files still work alone, and explicit legacy Teacher
Base + Fine-tune lists remain supported. CPU offload is opt-in.

## V1.0 — Initial public release

The original public code and Teacher/Student release, before the V1.1
deployment changes. It used DWPose/OpenMMLab preprocessing, loaded the T5 text
encoder, and documented the earlier checkpoint layout.
