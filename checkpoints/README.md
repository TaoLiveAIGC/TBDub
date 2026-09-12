# Checkpoints

Model weights are intentionally excluded from source control. The default inference configuration expects one of the two DiT files and the shared auxiliary files below:

- `tbdub_teacher.safetensors` (standalone BF16 Teacher)
- `tbdub_student.safetensors` (standalone BF16 DMD2 Student; no Base required)
- `Wan2.2_VAE.safetensors`
- `null_prompt_emb.pt`
- `hubert-large-ll60k/`
- `face_landmarker.task` (default MediaPipe preprocessing)

`null_prompt_emb.pt` supplies the precomputed empty-prompt context for both
Teacher and Student. It must be kept; the T5 encoder checkpoint and `umt5-xxl/`
tokenizer are no longer loaded or required. The repository README gives individual
download commands for the VAE, HuBERT, and Face Landmarker. The pinned
MediaPipe 0.10.21 package includes its full-range detector; DWPose weights are
only needed for the DWPose backend.

Teacher and Student each load one complete checkpoint (about 12.59 GB in BF16).
The Teacher already combines the former Base + Fine-tune parameters. Explicit
legacy checkpoint lists are still applied from left to right. Original
FP32 Student files also work alone and are converted to BF16 during loading.
See the repository README and `python inference.py --help` for path overrides,
sampling options, and the verified BF16 export script.
