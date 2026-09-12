"""Resolve the published TBDub manifest without importing GPU dependencies."""
import json
import math
from pathlib import Path
import warnings


def configure_runtime(parser, args, argv):
    explicit = {
        action.dest for action in parser._actions
        if any(token.split('=', 1)[0] in action.option_strings for token in argv)
    }
    root = Path(args.checkpoint_dir).expanduser().resolve()
    args.checkpoint_dir = str(root)
    manifest_path = Path(args.config).expanduser().resolve() if args.config else root / 'config.json'
    data = None
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text())
            validate_manifest(data, args.inference_mode)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            parser.error(f'Invalid TBDub manifest {manifest_path}: {error}')
        args.config = str(manifest_path)
    elif args.config:
        parser.error(f'TBDub manifest does not exist: {manifest_path}')
    elif not args.preprocess_only:
        warnings.warn(f'No TBDub manifest at {manifest_path}; using legacy CLI defaults. '
                      'Download config.json with the model to enable configuration validation.')

    def default(name, value):
        if name not in explicit:
            setattr(args, name, value)

    paths = {
        'vae_checkpoint': 'Wan2.2_VAE.safetensors',
        'hubert_checkpoint': 'hubert-large-ll60k',
        'prompt_embedding': 'null_prompt_emb.pt',
        'mediapipe_model': 'face_landmarker.task',
    }
    if data:
        paths['prompt_embedding'] = data['shared_files']['prompt_embedding']
        # Optional named paths extend format v2 while retaining existing manifests.
        paths.update({key: value for key, value in data.get('runtime_paths', {}).items() if key in paths})
        variant = data['variants'][args.inference_mode]
        default('dit_checkpoint', [str(root / p) for p in variant['checkpoints']])
        default('cpu_offload', data['generation']['cpu_offload'])
        if args.inference_mode == 'student':
            default('num_student_steps', variant['num_inference_steps'])
            default('sigma_shift', variant['sigma_shift'])
            default('motion_from_latents', variant['motion_from_latents'])
        else:
            for key in ('num_inference_steps', 'ref_cfg_scale', 'audio_cfg_scale'):
                default(key, variant[key])
            default('sigma_shift', variant.get('sigma_shift', 5.0))
    elif args.dit_checkpoint is None:
        args.dit_checkpoint = [str(root / f'tbdub_{args.inference_mode}.safetensors')]
    for name, value in paths.items():
        default(name, str(root / value))
    return args


def validate_manifest(data, mode):
    if not isinstance(data, dict):
        raise ValueError('expected a JSON object')
    expected = {
        'format_version': 2, 'model_id': 'TaoLiveAIGC/TBDub',
        'model_type': 'tbdub', 'pipeline_class': 'TBDubPipeline',
        'task': 'audio-driven-video-dubbing',
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f'{key} must be {value!r}')
    for section, values in {
        'input': {'video_fps': 25, 'face_resolution': [512, 512]},
        'generation': {'clip_num_frames': 77, 'motion_num_frames': 5},
    }.items():
        for key, value in values.items():
            if data[section][key] != value:
                raise ValueError(f'{section}.{key} must be {value!r} for this runtime')
    if type(data['generation']['cpu_offload']) is not bool:
        raise ValueError('generation.cpu_offload must be a boolean')
    pre = data['preprocessing']
    if pre['default_backend'] != 'mediapipe' or pre['skip_for_cropped_input'] is not True:
        raise ValueError('unsupported preprocessing configuration')
    backend = pre['backends']['mediapipe']
    if backend['version'] != '0.10.21' or backend['detector'] != 'bundled_face_detection_full_range_sparse':
        raise ValueError('unsupported MediaPipe version or detector')
    if backend['external_dependencies'] != ['face_landmarker.task']:
        raise ValueError('unsupported preprocessing dependencies')
    if data['external_dependencies'] != ['Wan2.2_VAE.safetensors', 'hubert-large-ll60k']:
        raise ValueError('unsupported shared model dependencies')
    variant = data['variants'][mode]
    if variant['storage_dtype'] != 'bfloat16' or variant['standalone'] is not True:
        raise ValueError('expected a complete BF16 checkpoint')
    checkpoints = variant['checkpoints']
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise ValueError('expected one standalone DiT checkpoint')
    def relative_file(value):
        if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('manifest model paths must be relative to checkpoint-dir')
    for value in checkpoints + [data['shared_files']['prompt_embedding']]:
        relative_file(value)
    allowed = {'vae_checkpoint', 'hubert_checkpoint', 'prompt_embedding', 'mediapipe_model'}
    for key, value in data.get('runtime_paths', {}).items():
        if key not in allowed:
            raise ValueError(f'unknown runtime_paths key: {key}')
        relative_file(value)
    steps = variant['num_inference_steps']
    if type(steps) is not int or steps <= 0:
        raise ValueError('num_inference_steps must be a positive integer')
    for key in (('sigma_shift',) if mode == 'student' else ('ref_cfg_scale', 'audio_cfg_scale')):
        value = variant[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'sigma_shift' and value == 0):
            raise ValueError(f'invalid {key}')
    if 'sigma_shift' in variant:
        shift = variant['sigma_shift']
        if type(shift) not in (int, float) or not math.isfinite(shift) or shift <= 0:
            raise ValueError('sigma_shift must be finite and positive')
    if mode == 'student' and type(variant['motion_from_latents']) is not bool:
        raise ValueError('motion_from_latents must be a boolean')
