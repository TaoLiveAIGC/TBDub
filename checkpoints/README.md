# Checkpoints

Model weights are intentionally excluded from source control. The default inference configuration expects:

- `tbdub_base.safetensors`
- `tbdub_finetune.safetensors`
- `tbdub_student.safetensors` (optional DMD2 Student)
- `Wan2.2_VAE.safetensors`
- `null_prompt_emb.pt`
- `hubert-large-ll60k/`

`null_prompt_emb.pt` supplies the precomputed empty-prompt context for both
Teacher and Student. It must be kept; the T5 encoder checkpoint and `umt5-xxl/`
tokenizer are no longer loaded or required. For MediaPipe preprocessing, also
download `face_landmarker.task` and `blaze_face_full_range.tflite` as described in
the repository README; DWPose weights are only needed for the DWPose backend.

Checkpoint state dictionaries are applied from left to right. Teacher inference normally loads the fine-tuned checkpoint after the base checkpoint. Distilled inference instead loads the compatible Student checkpoint after the base checkpoint so its parameters take precedence. See the repository README and `python inference.py --help` for path overrides and sampling options.
