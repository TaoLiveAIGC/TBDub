# Optional DWPose preprocessing

V1.1 uses MediaPipe by default. This compatibility path keeps the original
DWPose detector and landmark model available for existing workflows. It requires
MMCV's compiled operators plus MMEngine, MMDetection, and MMPose; none of these
packages is required by the default MediaPipe installation.

Create a separate Python environment and install a CUDA-compatible
PyTorch/torchvision build first. Install `requirements-dwpose.txt` **instead of**
`requirements.txt`; the two environments use different OpenCV and protobuf
packages.

```bash
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation "chumpy==0.70"
python -m pip install -r requirements-dwpose.txt
python -m pip install openmim
mim install "mmcv==2.1.0"
```

MMCV must include its compiled operators (`mmcv-lite` is insufficient). Check
the [official MMCV installation guidance](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
for a wheel compatible with your PyTorch and CUDA versions. If no compatible
wheel exists, this optional path requires a source build. In our original
PyTorch 2.9/CUDA 12.8 environment, MMCV needed compilation; the default V1.1
MediaPipe path avoids this step.

Download the detector and landmark weights into the existing configuration
directory from the repository root:

```bash
hf download KlingTeam/X-Dub \
  dwpose_tools/models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
  dwpose_tools/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
  --local-dir .
```

Select this backend explicitly with the Python entry point:

```bash
python inference.py \
  --video path/to/source.mp4 --audio path/to/driving.wav \
  --preprocess-backend dwpose \
  --output-dir results
```

The same argument works with Student sampling. `--dwpose-model-dir` overrides
the directory containing both `.pth` weights and their `.py` configuration
files. The `infer.sh` convenience command uses MediaPipe.

Keep preprocessing caches separate for DWPose and MediaPipe. Their landmarks
and crop regions can differ; review outputs when switching backends. A backend
mismatch is rejected when loading a cache.
