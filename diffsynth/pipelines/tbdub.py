import math
import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from einops import rearrange
from tqdm import tqdm
from typing_extensions import Literal

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.flow_match import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.tbdub_dit import TBDubDiT, sinusoidal_embedding_1d 
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE

from ..models.model_loader import ModelPool
import torch.nn.functional as F

# audio
from ..thirdparties.audio_processor import HubertProcessor
from ..thirdparties.utils import post_process_audio_feat


def cosine_blend(t, t_start, t_end):
    normalized = (t - t_start) / (t_end - t_start)
    return (1 - torch.cos(normalized * math.pi)) / 2


def get_ref_guidance_scale_schedule(t, init_scale, trans_p1=0.8, trans_p2=0.4, trans_width=0.1): # 0.8 0.4
    trans1_start = trans_p1 + trans_width / 2
    trans1_end = trans_p1 - trans_width / 2
    trans2_start = trans_p2 + trans_width / 2
    trans2_end = trans_p2 - trans_width / 2

    c1 = init_scale * 1.0
    c2 = init_scale * 0.6
    c3 = init_scale * 0.3

    result = torch.zeros_like(t)
    result = torch.where(t >= trans1_start, c1, result)

    mask = (t >= trans1_end) & (t < trans1_start)
    blend = cosine_blend(t, trans1_start, trans1_end)
    result = torch.where(mask, c1 * (1 - blend) + c2 * blend, result)

    result = torch.where((t >= trans2_start) & (t < trans1_end), c2, result)

    mask = (t >= trans2_end) & (t < trans2_start)
    blend = cosine_blend(t, trans2_start, trans2_end)
    result = torch.where(mask, c2 * (1 - blend) + c3 * blend, result)

    result = torch.where(t < trans2_end, c3, result)
    return result


def get_xt_from_x0(ref_latents, noise, sigma_shift):
    sigma_shift = sigma_shift.view(-1, 1, 1, 1, 1)
    return (1.0 - sigma_shift) * ref_latents + sigma_shift * noise

# start_frame: motion frame 
def replace_latent_borders(latents, border_latents, start_frame=0, border_width=2):
    if border_width <= 0 or start_frame >= latents.shape[2]:
        return latents
    updated_latents = latents.clone()
    for offset in range(border_width):
        updated_latents[:, :, start_frame:, offset, :] = border_latents[:, :, start_frame:, offset, :]
        updated_latents[:, :, start_frame:, :, offset] = border_latents[:, :, start_frame:, :, offset]
        updated_latents[:, :, start_frame:, -(offset + 1), :] = border_latents[:, :, start_frame:, -(offset + 1), :]
        updated_latents[:, :, start_frame:, :, -(offset + 1)] = border_latents[:, :, start_frame:, :, -(offset + 1)]
    return updated_latents


class TBDubPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: TBDubDiT = None
        self.vae: WanVideoVAE = None
        self.hubert_processor: HubertProcessor = None
        self.in_iteration_models = ("dit",) 
        self.units = [
            WanVideoUnit_ShapeChecker(),   
            WanVideoUnit_NoiseInitializer(),  
            # Empty-prompt embeddings are loaded by the public entry point.
            WanVideoUnit_TargetVideoEmbedder(),
            WanVideoUnit_ReferenceVideoEmbedder(), 
            WanVideoUnit_MotionVideoEmbedder(), 
            WanVideoUnit_MaskEmbedder(),
            WanVideoUnit_LipSync(), 
            WanVideoUnit_TeaCache(), # TODO: teacache accelerate
        ]
        self.model_fn = model_fn_wan_video


    def download_and_load_models(self, model_configs: list[ModelConfig] = [], vram_limit: float = None):
        model_pool = ModelPool()
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or self.torch_dtype
            vram_config["computation_device"] = vram_config["computation_device"] or self.device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
                state_dict=model_config.state_dict,
            )
        return model_pool
    

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        vram_limit: float = None,
        args = None,
        hubert_ckpt_path: str = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = { 
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
       # Initialize pipeline
        pipe = TBDubPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit) 
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit", index=2) 
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2 # 16 * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')

        if hubert_ckpt_path is None:
            raise ValueError("`hubert_ckpt_path` is required.")

        num_frames = 77 if args is None or getattr(args, "num_frames", None) is None else args.num_frames
        audio_feat_window_size = (
            0
            if args is None or getattr(args, "audio_feat_window_size", None) is None
            else args.audio_feat_window_size
        )
        pipe.hubert_processor = HubertProcessor(
            model_path=hubert_ckpt_path,
            num_frames=num_frames,
            audio_feat_window_size=audio_feat_window_size,
            embedding_num_layers=4,
            device=device,
            layer_indices=(9, 10, 11, 12),
            # layer_indices=(13, 14, 15, 16),
        )
        pipe.hubert_processor.eval()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        prompt_emb: Optional[torch.Tensor] = None,
        # Video-to-video
        ref_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0, # maybe useless
        # Lip Sync
        audio_wav_path: Optional[str] = None,
        audio_npy_path: Optional[str] = None,
        hubert_feat: Optional[torch.Tensor] = None,
        per_chunk_audio: Optional[bool] = False,
        start_idx: Optional[int] = 0,
        motion_video: Optional[list[Image.Image]] = None, 
        motion_latents: Optional[torch.Tensor] = None, 
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_frames: Optional[int] = 77,
        motion_latents_num_frames: Optional[int] = 2,
        # Classifier-free guidance
        ref_cfg_scale: Optional[float] = 2.0,
        audio_cfg_scale: Optional[float] = 7.0,
        use_dynamic_cfg: Optional[bool] = False,
        replace_border_latents: Optional[bool] = False,
        replace_border_latents_width: Optional[int] = 1,
        replace_border_each_step: Optional[bool] = True,
        cfg_merge: Optional[bool] = True,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = False,
        tile_size: Optional[tuple[int, int]] = (32, 32),
        tile_stride: Optional[tuple[int, int]] = (16, 16),
        decode_output: Optional[bool] = True,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "context": prompt_emb
            # "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "context": prompt_emb
            # "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "ref_video": ref_video, # [PIL.Image]
            "motion_video": motion_video, # [PIL.Image] or None
            "motion_latents": motion_latents, # pre-encoded motion latents or None
            "audio_wav_path": audio_wav_path,
            "audio_npy_path": audio_npy_path,   # none
            "hubert_feat": hubert_feat,   # none
            "per_chunk_audio": per_chunk_audio,
            "start_idx": start_idx, # audio_start_idx
            "denoising_strength": denoising_strength, # 1.0
            "seed": seed, "rand_device": rand_device, 
            "height": height, "width": width, "num_frames": num_frames, 
            "motion_latents_num_frames": motion_latents_num_frames, # 2
            # cfg
            "ref_cfg_scale": ref_cfg_scale,     # 2.5
            "audio_cfg_scale": audio_cfg_scale, # 10
            "cfg_merge": cfg_merge, # True
            "cfg_scale": 1.0, # should not be changed
            # others
            "sigma_shift": sigma_shift,  # 5.0
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }

        for unit in self.units:
            # print(f"Running unit: {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # merge inputs
        inputs = {**inputs_shared, **inputs_posi, **inputs_nega}

        # keep ref_latents & noise for border replace
        ref_latents = inputs["ref_latents"].clone()     # torch.Size([1, 48, 20, 32, 32])
        noise = inputs["noise"].clone()     # torch.Size([1, 48, 20, 32, 32])
        
        # prepare for cfg merge 
        if cfg_merge:
            # cfg
            inputs["ref_latents"] = torch.cat([torch.zeros_like(inputs["ref_latents"]), inputs["ref_latents"], inputs["ref_latents"]], dim=0) 
            inputs["audio_feats"] = torch.cat([torch.zeros_like(inputs["audio_feats"]), torch.zeros_like(inputs["audio_feats"]), inputs["audio_feats"]], dim=0)
            # no cfg
            inputs["context"] = torch.cat([inputs["context"]] * 3, dim=0) 

        # Prepare model for denoising
        # onload dit; offload others 
        self.load_models_to_device(self.in_iteration_models)

        # Denoise 
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device) 
            # border replace
            if replace_border_latents and replace_border_each_step:
                sigma = timestep / self.scheduler.num_train_timesteps
                sigma_shift = torch.clamp(sigma, min=0.0, max=0.95).to(dtype=ref_latents.dtype, device=ref_latents.device)
                ref_latents_noised = get_xt_from_x0(ref_latents, noise, sigma_shift)
                inputs["latents"] = replace_latent_borders(
                    inputs["latents"],
                    ref_latents_noised,
                    start_frame=motion_latents_num_frames if inputs.get("motion_latents") is not None else 0,
                    border_width=replace_border_latents_width,
                )

            # Inference
            noise_pred = self.model_fn(**models, **inputs, timestep=timestep)
            if cfg_merge:
                noise_pred_nega, noise_pred_posi_r, noise_pred_posi_ra = noise_pred.chunk(3, dim=0)
                if use_dynamic_cfg:
                    t = timestep / self.scheduler.num_train_timesteps
                    _audio_guidance_scale = audio_cfg_scale * max((t ** 1.5), 0.5)
                    _ref_guidance_scale = get_ref_guidance_scale_schedule(t, ref_cfg_scale, trans_p1=0.8, trans_p2=0.4, trans_width=0.1)
                else:
                    _audio_guidance_scale = audio_cfg_scale
                    _ref_guidance_scale = ref_cfg_scale

                noise_pred = noise_pred_nega + \
                    _ref_guidance_scale * (noise_pred_posi_r - noise_pred_nega) + \
                    _audio_guidance_scale * (noise_pred_posi_ra - noise_pred_posi_r)

            # Scheduler
            inputs["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs["latents"])
            
            # Motion conditiioning
            if inputs.get("motion_latents") is not None: 
                inputs["latents"][:, :, 0:motion_latents_num_frames] = inputs["motion_latents"]

        if replace_border_latents:  # true
            inputs["latents"] = replace_latent_borders(
                inputs["latents"],
                ref_latents,
                start_frame=motion_latents_num_frames if inputs.get("motion_latents") is not None else 0,
                border_width=replace_border_latents_width,
            )
            if inputs.get("motion_latents") is not None:
                inputs["latents"][:, :, 0:motion_latents_num_frames] = inputs["motion_latents"]

        if not decode_output:
            self.load_models_to_device([])
            return None, inputs

        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video, inputs # reuse audio feats



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: TBDubPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "batch_size", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: TBDubPipeline, height, width, num_frames, batch_size, seed, rand_device):
        length = (num_frames - 1) // 4 + 1  # 77->20
        batch_size = 1 if batch_size is None else batch_size
        shape = (batch_size, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)   # (1, 48, 20, 32, 32)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    
class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "prompt_embed": "prompt_embed", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "prompt_embed": "prompt_embed", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: TBDubPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device) # [bs, 512] 
        mask = mask.to(pipe.device) # [bs, 512] 
        seq_lens = mask.gt(0).sum(dim=1).long() 
        with torch.no_grad():
            prompt_emb = pipe.text_encoder(ids, mask) # [1,512,4096]
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0 
        return prompt_emb

    def process(self, pipe: TBDubPipeline, prompt, positive, prompt_embed):
        prompt = prompt if isinstance(prompt, list) else [prompt]
        prompt_embed = prompt_embed if isinstance(prompt_embed, list) else [prompt_embed]

        if all(prompt_embed_item is not None for prompt_embed_item in prompt_embed):
            context = []
            for prompt_embed_item in prompt_embed:
                if prompt_embed_item.ndim == 2:
                    prompt_embed_item = prompt_embed_item.unsqueeze(0)
                prompt_embed_item = prompt_embed_item.to(dtype=pipe.torch_dtype, device=pipe.device)
                context_item = torch.zeros(
                    (prompt_embed_item.shape[0], 512, prompt_embed_item.shape[-1]),
                    dtype=prompt_embed_item.dtype,
                    device=prompt_embed_item.device,
                )
                valid_length = min(prompt_embed_item.shape[1], 512)
                context_item[:, :valid_length, :] = prompt_embed_item[:, :valid_length, :]
                context.append(context_item)
            return {"context": torch.cat(context, dim=0)}

        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}
    
class WanVideoUnit_TargetVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("tgt_video", "tgt_latents", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "tgt_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: TBDubPipeline, tgt_video, tgt_latents, noise, tiled, tile_size, tile_stride):
        if isinstance(tgt_video, list):
            if len(tgt_video) > 0 and isinstance(tgt_video[0], Image.Image):
                tgt_video = [tgt_video]
        else:
            tgt_video = [tgt_video]
        tgt_latents = tgt_latents if isinstance(tgt_latents, list) else [tgt_latents]

        # inference
        if all(tgt_video_item is None for tgt_video_item in tgt_video) and all(tgt_latents_item is None for tgt_latents_item in tgt_latents):
            return {"latents": noise.clone()}

        if all(tgt_latents_item is not None for tgt_latents_item in tgt_latents): # training
            tgt_latents = torch.stack([
                tgt_latents_item.to(dtype=pipe.torch_dtype, device=pipe.device)
                for tgt_latents_item in tgt_latents
            ], dim=0)
            return {"latents": noise.clone(), "tgt_latents": tgt_latents}
        pipe.load_models_to_device(self.onload_model_names) 
        tgt_video = torch.cat([
            pipe.preprocess_video(tgt_video_item, torch_dtype=pipe.torch_dtype, device=pipe.device)
            for tgt_video_item in tgt_video
        ], dim=0) 
        with torch.no_grad():
            tgt_latents = pipe.vae.encode(tgt_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) # torch.Size([B, 48, 20, 32, 32])
        return {"latents": noise.clone(), "tgt_latents": tgt_latents} 

class WanVideoUnit_ReferenceVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("ref_video", "ref_latents", "tiled", "tile_size", "tile_stride"),
            output_params=("ref_latents",),
            onload_model_names=("vae",)
        )
    def process(self, pipe: TBDubPipeline, ref_video, ref_latents, tiled, tile_size, tile_stride):
        if isinstance(ref_video, list):     # [PIL.Image]
            if len(ref_video) > 0 and isinstance(ref_video[0], Image.Image):
                ref_video = [ref_video]     # [[PIL.Image]]
        else:
            ref_video = [ref_video]
        ref_latents = ref_latents if isinstance(ref_latents, list) else [ref_latents]   # [None]

        if all(ref_latents_item is not None for ref_latents_item in ref_latents):
            ref_latents = torch.stack([
                ref_latents_item.to(dtype=pipe.torch_dtype, device=pipe.device)
                for ref_latents_item in ref_latents
            ], dim=0)
            return {"ref_latents": ref_latents}
        pipe.load_models_to_device(self.onload_model_names)
        ref_video = torch.cat([
            pipe.preprocess_video(ref_video_item, torch_dtype=pipe.torch_dtype, device=pipe.device)
            for ref_video_item in ref_video
        ], dim=0)
        with torch.no_grad():
            ref_latents = pipe.vae.encode(ref_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) # torch.Size([B, 48, 20, 32, 32])
        return {"ref_latents": ref_latents}


class WanVideoUnit_MotionVideoEmbedder(PipelineUnit): 
    def __init__(self):
        super().__init__(
            input_params=("motion_video", "motion_latents", "latents", "motion_latents_num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "use_motion_latents", "motion_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: TBDubPipeline, motion_video, motion_latents, latents, motion_latents_num_frames, tiled, tile_size, tile_stride):
        if motion_latents is None and motion_video is None:    # none
            return {}

        if motion_latents is not None:
            # caller passed the previous clip's tail latents: no decode/encode round trip
            motion_latents = motion_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        else:
            pipe.load_models_to_device(self.onload_model_names)
            if isinstance(motion_video, torch.Tensor): 
                motion_video = motion_video.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device) 
            else: 
                motion_video = pipe.preprocess_video(motion_video, torch_dtype=pipe.torch_dtype, device=pipe.device)    # torch.Size([1, 3, 5, 512, 512])
            with torch.no_grad():
                motion_latents = pipe.vae.encode(motion_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) # 1 c=48 f=2 h=32 w=32 torch.Size([1, 48, 2, 32, 32])
        f = motion_latents.shape[2]
        assert f == motion_latents_num_frames, f"motion_latents_num_frames mismatch: expected={motion_latents_num_frames}, got={f}"
        latents[:, :, 0: f] = motion_latents # replace motion latents
        return {"latents": latents, "use_motion_latents": True, "motion_latents": motion_latents}


class WanVideoUnit_MaskEmbedder(PipelineUnit): 
    """
    downsample lip masks
    """
    def __init__(self):
        super().__init__(
            input_params=("lip_mask", "height", "width", "num_frames"),
            output_params=("lip_mask_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: TBDubPipeline, lip_mask, height, width, num_frames):
        # inference
        if lip_mask is None:
            return {}
        # training
        if isinstance(lip_mask, list) and len(lip_mask) > 0 and isinstance(lip_mask[0], Image.Image):
            lip_mask = [lip_mask]
        lip_mask = torch.stack([
            torch.stack([self.preprocess_mask(mask) for mask in lip_mask_item], dim=0) 
            for lip_mask_item in lip_mask 
        ], dim=0) 
        lip_mask = lip_mask.unsqueeze(1) 
        # causal downsample (union by max-pooling)
        spatial_down_factor = pipe.vae.upsampling_factor
        temporal_down_factor = 4
        first_lip_mask_latents = lip_mask[:, :, 0:1, :, :]   
        rest_lip_mask_latents = lip_mask[:, :, 1:, :, :]     

        first_lip_mask_latents = F.max_pool3d(
            first_lip_mask_latents,
            kernel_size=(1, spatial_down_factor, spatial_down_factor),
            stride=(1, spatial_down_factor, spatial_down_factor),)

        rest_lip_mask_latents = F.max_pool3d(
            rest_lip_mask_latents,
            kernel_size=(temporal_down_factor, spatial_down_factor, spatial_down_factor),
            stride=(temporal_down_factor, spatial_down_factor, spatial_down_factor),)
        lip_mask_latents = torch.cat([first_lip_mask_latents, rest_lip_mask_latents], dim=2).to(
            dtype=pipe.torch_dtype, device=pipe.device) 
        return {"lip_mask_latents": lip_mask_latents}

    def preprocess_mask(self, image):
        image = torch.tensor(np.array(image, dtype=np.float32)) / 255.0
        return image


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: TBDubPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: TBDubPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_LipSync(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process(self, pipe: TBDubPipeline, inputs_shared, inputs_posi, inputs_nega):
        # audio path
        audio_wav_path = inputs_shared.get("audio_wav_path", None)
        audio_npy_path = inputs_shared.get("audio_npy_path", None)
        audio_wav_path = audio_wav_path if isinstance(audio_wav_path, list) else [audio_wav_path]   # [path]
        audio_npy_path = audio_npy_path if isinstance(audio_npy_path, list) else [audio_npy_path]
        # npy > wav
        audio_path = [
            audio_npy_path_item if audio_npy_path_item is not None else audio_wav_path_item
            for audio_npy_path_item, audio_wav_path_item in zip(audio_npy_path, audio_wav_path)
        ]
        
        # sample 
        start_idx = inputs_shared.get("start_idx", None)
        assert start_idx is not None, "start_idx is required for sampling audio features in lip sync."
        start_idx = start_idx if isinstance(start_idx, list) else [start_idx]

        per_chunk_audio = inputs_shared.get("per_chunk_audio", False)

        if per_chunk_audio:
            assert all(audio_path_item is not None for audio_path_item in audio_path), "No audio input provided for lip sync."
            hubert_feat = torch.stack([
                pipe.hubert_processor.audio2feat_chunk(audio_path_item, int(start_idx_item))
                for audio_path_item, start_idx_item in zip(audio_path, start_idx)
            ], dim=0)  # b f p_audio c torch.Size([1, 20, 8, 4096])
        else:
            hubert_feat = inputs_shared.get("hubert_feat", None)
            hubert_feat = hubert_feat if isinstance(hubert_feat, list) else [hubert_feat]

            if any(feat_item is None for feat_item in hubert_feat):
                assert all(audio_path_item is not None for audio_path_item in audio_path), "No audio input provided for lip sync."
                hubert_feat = [
                    pipe.hubert_processor.audio2feat(audio_path_item, use_cache=True)
                    for audio_path_item in audio_path
                ]   # [torch.Size([T, 4, 1024])]
                inputs_shared.update({"hubert_feat": hubert_feat if len(hubert_feat) > 1 else hubert_feat[0]})

            hubert_feat = torch.stack([
                pipe.hubert_processor.crop_overlap_audio_window(feat_item, int(start_idx_item))
                for feat_item, start_idx_item in zip(hubert_feat, start_idx)
            ], dim=0)  # b f p_audio c torch.Size([1, 20, 8, 4096])

        audio_feat = hubert_feat.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # training augmentation
        if inputs_shared.get("is_training", False):
            audio_feat = torch.stack([
                post_process_audio_feat(audio_feat_item, begin_process_rate=0.3, end_process_rate=0.1)
                for audio_feat_item in audio_feat
            ], dim=0)

        audio_feats = audio_feat    # torch.Size([1, 20, 8, 4096])

        inputs_posi.update({"audio_feats": audio_feats})
        
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: TBDubDiT, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video( 
    dit: TBDubDiT, 
    latents: torch.Tensor = None,   # b, c, f, h, w
    timestep: torch.Tensor = None,
    ref_latents: torch.Tensor = None,   # merge cfg torch.Size([3, 48, 20, 32, 32]) b=3
    context: torch.Tensor = None,       # torch.Size([3, 512, 4096]) b=3
    audio_feats: Optional[torch.Tensor] = None, # torch.Size([3, 20, 8, 2048])
    motion_latents: Optional[torch.Tensor] = None, 
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    cfg_merge: bool = False, 
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    use_motion_latents: Union[bool, torch.Tensor] = False, 
    motion_latents_num_frames: int = 2, 
    **kwargs,
):

    # Timestep
    batch_size = latents.shape[0]   # latents: torch.Size([1, 48, 20, 32, 32])
    timestep = timestep.to(dtype=latents.dtype, device=latents.device) 
    use_motion_latents = (
        torch.tensor([use_motion_latents], dtype=torch.bool, device=latents.device)
        if not isinstance(use_motion_latents, torch.Tensor)
        else use_motion_latents.to(dtype=torch.bool, device=latents.device)
    ) 

    timestep = timestep[:, None, None].expand(
        batch_size, latents.shape[2], latents.shape[3] * latents.shape[4] // 4
    ).clone() # bs, f, h'w'
    # Motion latents and fixed reference latents use a zero timestep.
    timestep[use_motion_latents, :motion_latents_num_frames, :] = 0
    timestep = torch.cat([torch.zeros_like(timestep), timestep], dim=1)
    timestep = timestep.flatten(1) # bs, 2fh'w' torch.Size([1, 10240])
    t = sinusoidal_embedding_1d(dit.freq_dim, timestep.flatten()).reshape(batch_size, -1, dit.freq_dim)     # torch.Size([1, 10240, 256])
    t = dit.time_embedding(t) # bs, 2fh'w', c   torch.Size([1, 10240, 3072])
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim)) # bs, 2fh'w', 6, c    torch.Size([1, 10240, 6, 3072])
    
    # text emb
    context = dit.text_embedding(context) # bs, 512, 3072

    # audio 
    audio_embeds = dit.audio_embedding(audio_feats) # torch.Size([3, 20, 8, 2048]) --> b, f, p, c torch.Size([3, 20, 8, 3072])

    x = latents
    # Merge the conditional branches into one model call.
    if x.shape[0] != context.shape[0]:
        repeat_factor = context.shape[0] // x.shape[0]
        x = torch.concat([x] * repeat_factor, dim=0)
    if timestep.shape[0] != context.shape[0]:
        repeat_factor = context.shape[0] // timestep.shape[0]
        timestep = torch.concat([timestep] * repeat_factor, dim=0)
    if t.shape[0] != context.shape[0]:
        repeat_factor = context.shape[0] // t.shape[0]
        t = torch.concat([t] * repeat_factor, dim=0)
        t_mod = torch.concat([t_mod] * repeat_factor, dim=0)
        
    # Concatenate reference tokens before target tokens for joint self-attention.
    x = torch.cat([ref_latents, x], dim=2) # bs, c, 2f, h, w    torch.Size([3, 48, 40, 32, 32])

    # Pathify
    x = dit.patchify(x) # bs, c, 2f, h, w   torch.Size([3, 3072, 40, 16, 16])

    f_total, h, w = x.shape[2:]
    f = f_total // 2
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    ref_freqs = torch.cat([ # ref rope
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),         # dit.freqs[0]: torch.Size([1024, 22])
        dit.freqs[1][h:2 * h].view(1, h, 1, -1).expand(f, h, w, -1),    # dit.freqs[1]: torch.Size([1024, 21])
        dit.freqs[2][w:2 * w].view(1, 1, w, -1).expand(f, h, w, -1)     # dit.freqs[1]: torch.Size([1024, 21])
    ], dim=-1)  # torch.Size([20, 16, 16, 64]) 
    tgt_freqs = torch.cat([ # tgt rope
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1)  # torch.Size([20, 16, 16, 64])
    freqs = torch.cat([ref_freqs, tgt_freqs], dim=0).reshape(2 * f * h * w, 1, -1).to(x.device) # 2fhw,1,c_rope=64 torch.Size([10240, 1, 64])

    
    # TeaCache
    if tea_cache is not None: # pass
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
          
    # blocks
    if tea_cache_update: # pass
        x = tea_cache.update(x)
    else: # here
        for block_id, block in enumerate(dit.blocks):
            # Block
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, audio_embeds, t_mod, freqs
            )
              
            
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = x[:, f * h * w:, :]
    t = t[:, f * h * w:, :]
    x = dit.head(x, t) 

    x = dit.unpatchify(x, (f, h, w)) 
    return x
