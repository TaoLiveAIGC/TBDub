# Checkpoints

Download one complete BF16 DiT checkpoint for the model variant you want:

- `tbdub_teacher.safetensors` for Teacher, or
- `tbdub_student.safetensors` for Student.

Download `config.json` with either variant: inference reads and validates this
manifest for runtime defaults.

Both variants share these auxiliary files:

- `Wan2.2_VAE.safetensors`
- `null_prompt_emb.pt`
- `hubert-large-ll60k/config.json`
- `hubert-large-ll60k/preprocessor_config.json`
- `hubert-large-ll60k/pytorch_model.bin`
- `face_landmarker.task` for full-frame face preprocessing.

Each DiT is about 12.59 GB, and a complete single-variant setup totals about
15.27 GB in model files. The face detector is bundled with MediaPipe.
Use the [individual download commands](../README.md#checkpoints) to obtain
only the required files. Already aligned inputs with `--cropped-input` can
skip the Face Landmarker download. Model files are excluded from source control.
