"""Model signatures required by TBDub inference."""

MODEL_CONFIGS = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE38",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        "model_hash": "da12f645b7edaedb855f6bd8cee7b24c",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.tbdub_dit.TBDubDiT",
        "extra_kwargs": {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "num_layers": 30,
            "eps": 1e-6,
            # Keep the original parameter spelling for checkpoint compatibility.
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
        },
        "state_dict_converter": "diffsynth.utils.state_dict_converters.tbdub_dit.TBDubDiTStateDictConverter",
    },
]
