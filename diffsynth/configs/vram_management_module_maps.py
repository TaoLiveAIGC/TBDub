"""Module wrapping rules for the models used by TBDub."""

VRAM_MANAGEMENT_MODULE_MAPS = {
    "diffsynth.models.tbdub_dit.TBDubDiT": {
        "diffsynth.models.audio_modules.AudioProjModule": "diffsynth.core.vram.layers.AutoWrappedNonRecurseModule",
        "diffsynth.models.tbdub_dit.MLP": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.tbdub_dit.DiTBlock": "diffsynth.core.vram.layers.AutoWrappedNonRecurseModule",
        "diffsynth.models.tbdub_dit.Head": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.tbdub_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.wan_video_text_encoder.WanTextEncoder": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_text_encoder.T5RelativeEmbedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_text_encoder.T5LayerNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.wan_video_vae.WanVideoVAE": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.GroupNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.RMS_norm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.CausalConv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.Upsample": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.wan_video_vae.WanVideoVAE38": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.GroupNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.RMS_norm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.CausalConv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.wan_video_vae.Upsample": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
}
