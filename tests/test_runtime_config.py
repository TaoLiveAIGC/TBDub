"""CPU-only regression tests for the real CLI parser and manifest resolver."""
import argparse
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_config import configure_runtime

# Execute the unchanged parser function without importing torch/model modules.
tree = ast.parse((ROOT / 'inference.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
namespace = dict(argparse=argparse, os=os, sys=sys, Path=Path,
                 DEFAULT_CHECKPOINT_DIR=ROOT/'checkpoints', configure_runtime=configure_runtime)
exec(compile(ast.Module(body=[function], type_ignores=[]), 'inference.py', 'exec'), namespace)
parse = namespace['parse_args']
MANIFEST = {
    'format_version': 2, 'model_id': 'TaoLiveAIGC/TBDub', 'model_type': 'tbdub',
    'task': 'audio-driven-video-dubbing', 'pipeline_class': 'TBDubPipeline',
    'input': {'video_fps': 25, 'face_resolution': [512, 512]},
    'generation': {'clip_num_frames': 77, 'motion_num_frames': 5, 'cpu_offload': False},
    'shared_files': {'prompt_embedding': 'null_prompt_emb.pt'},
    'external_dependencies': ['Wan2.2_VAE.safetensors', 'hubert-large-ll60k'],
    'preprocessing': {'default_backend': 'mediapipe', 'skip_for_cropped_input': True,
        'backends': {'mediapipe': {'version': '0.10.21',
            'detector': 'bundled_face_detection_full_range_sparse',
            'external_dependencies': ['face_landmarker.task']}}},
    'variants': {
        'teacher': {'num_inference_steps': 30, 'ref_cfg_scale': 2., 'audio_cfg_scale': 6.,
            'checkpoints': ['tbdub_teacher.safetensors'], 'storage_dtype': 'bfloat16', 'standalone': True},
        'student': {'num_inference_steps': 2, 'sigma_shift': 1., 'motion_from_latents': True,
            'checkpoints': ['tbdub_student.safetensors'], 'storage_dtype': 'bfloat16', 'standalone': True},
    },
}

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = copy.deepcopy(MANIFEST)
        self.write()

    def write(self):
        (self.root/'config.json').write_text(json.dumps(self.data))

    def run_args(self, *args):
        return parse(['--video', 'input.mp4', '--checkpoint-dir', str(self.root), *args])

    def test_both_variants_and_all_paths(self):
        for variant in ['teacher', 'student']:
            a = self.run_args('--inference-mode', variant)
            self.assertEqual(a.dit_checkpoint, [str(self.root/f'tbdub_{variant}.safetensors')])
            for key in ['vae_checkpoint', 'hubert_checkpoint', 'prompt_embedding', 'mediapipe_model']:
                self.assertEqual(Path(getattr(a, key)).parent, self.root)
        self.assertTrue(a.motion_from_latents)
        self.assertFalse(a.cpu_offload)

    def test_manifest_defaults_and_cli_overrides(self):
        self.data['variants']['student']['num_inference_steps'] = 4
        self.data['generation']['cpu_offload'] = True
        self.data['runtime_paths'] = {'vae_checkpoint': 'shared/vae.safetensors'}
        self.write()
        a = self.run_args('--inference-mode', 'student')
        self.assertEqual(a.num_student_steps, 4)
        self.assertTrue(a.cpu_offload)
        self.assertEqual(a.vae_checkpoint, str(self.root/'shared/vae.safetensors'))
        a = self.run_args('--inference_mode=student', '--num_student_steps=3',
                          '--no-cpu-offload', '--no-motion-from-latents', '--vae-checkpoint', 'custom.vae')
        self.assertEqual(a.num_student_steps, 3)
        self.assertFalse(a.cpu_offload)
        self.assertFalse(a.motion_from_latents)
        self.assertEqual(a.vae_checkpoint, 'custom.vae')

    def test_incompatible_manifest_fails(self):
        for section, key, value in [('input','video_fps',30), ('generation','clip_num_frames',81),
                                    ('preprocessing','default_backend','old_backend')]:
            self.data = copy.deepcopy(MANIFEST); self.data[section][key] = value; self.write()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_args()

    def test_missing_and_malformed_manifest(self):
        (self.root/'config.json').unlink()
        with warnings.catch_warnings(record=True) as seen:
            a = self.run_args('--inference-mode','student')
        self.assertEqual(len(seen), 1)
        self.assertFalse(a.cpu_offload)
        self.assertEqual(a.dit_checkpoint, [str(self.root/'tbdub_student.safetensors')])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_args('--config',str(self.root/'missing.json'))
        (self.root/'config.json').write_text('{broken')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_args()

    def test_teacher_sampling_override(self):
        self.data['variants']['teacher']['ref_cfg_scale'] = 2.5; self.write()
        self.assertEqual(self.run_args().ref_cfg_scale, 2.5)
        self.assertEqual(self.run_args('--ref_cfg_scale=2.0').ref_cfg_scale, 2.0)

    def test_shell_custom_directory(self):
        # Stub only the Python executable, capture the shell's real arguments.
        executable = self.root/'python'
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o755)
        env = dict(os.environ, PATH=str(self.root)+os.pathsep+os.environ['PATH'],
                   TBDUB_CHECKPOINT_DIR=str(self.root))
        out = subprocess.check_output(['bash',str(ROOT/'infer.sh'),'source.mp4','audio.wav','--cpu-offload'],env=env,text=True).splitlines()
        args = parse(out[1:])
        self.assertTrue(args.cpu_offload)
        for key in ['vae_checkpoint','hubert_checkpoint','prompt_embedding','mediapipe_model']:
            self.assertEqual(Path(getattr(args,key)).parent,self.root)

if __name__ == '__main__':
    unittest.main()
