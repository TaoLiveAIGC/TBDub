# Changelog

## Unreleased

- Read and validate the downloaded TBDub manifest; explicit CLI options override its defaults.
- Apply custom checkpoint directories consistently to all model files.

## V1.1 — 2026-09-12

An inference and deployment update using the existing trained Teacher and Student.

- **Smaller downloads:** one complete BF16 DiT file per variant, about 12.59 GB; about 15.27 GB including the shared auxiliary model files for either variant. No manual model combination is needed.
- **Optional lower GPU memory:** add `--cpu-offload` to move inactive models to system RAM. Models remain on the GPU by default for faster inference; users select the memory mode manually.

### Memory measurements

| Model | Default GPU-resident mode | With `--cpu-offload` |
|---|---:|---:|
| Student | 27.93 GiB | 15.01 GiB |
| Teacher | 27.95 GiB | 19.49 GiB |

Both columns use the same V1.1 configuration and compare memory modes. The
measurement device was an RTX PRO 5000 72GB, with 512×512 crops, 15.04-second
audio and 376 output frames. Offload runs completed under a 22 GiB PyTorch
allocator limit. Teacher used 30 steps and reference/audio CFG 2.5/10; Student
used two steps and latent motion. These data are separate from the paper's
H20 speed benchmark and do not constitute an RTX 4090 hardware test.

The complete 376-frame Teacher and Student outputs match their corresponding
GPU-resident runs exactly when using CPU offload.

CPU offload trades host memory and transfer time for GPU memory. Teacher's
three CFG branches share weights but require more intermediate tensors than
Student's single branch. Peak usage also depends on input length and the
active stage. See the [full measurement scope and usage](README.md#lower-gpu-memory-usage).

See the [README](README.md) for installation, downloads, and inference commands.
Regenerate incompatible preprocessing caches with this version.

## V1.0 — Initial public release

The original release of the inference code, Teacher and Student models, and
research results.
