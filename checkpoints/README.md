# Checkpoints

Model weights are intentionally excluded from source control. The default inference configuration expects:

- `tbdub_base.safetensors`
- `tbdub_finetune.safetensors`
- `models_t5_umt5-xxl-enc-bf16.safetensors`
- `Wan2.2_VAE.safetensors`
- `null_prompt_emb.pt`
- `umt5-xxl/`
- `hubert-large-ll60k/`

The TBDub fine-tuned state dictionary is applied after the base state dictionary. See the repository README and `python inference.py --help` for path overrides.
