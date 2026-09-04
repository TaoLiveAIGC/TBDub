# Checkpoints

Model weights are intentionally excluded from source control. The default inference configuration expects:

- `tbdub_base.safetensors`
- `tbdub_finetune.safetensors`
- `tbdub_student.safetensors` (optional DMD2 Student)
- `models_t5_umt5-xxl-enc-bf16.safetensors`
- `Wan2.2_VAE.safetensors`
- `null_prompt_emb.pt`
- `umt5-xxl/`
- `hubert-large-ll60k/`

Checkpoint state dictionaries are applied from left to right. Teacher inference normally loads the fine-tuned checkpoint after the base checkpoint. Distilled inference instead loads the compatible Student checkpoint after the base checkpoint so its parameters take precedence. See the repository README and `python inference.py --help` for path overrides and sampling options.
