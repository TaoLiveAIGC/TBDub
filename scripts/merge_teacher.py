"""Export one complete BF16 Teacher from Base and the fine-tuned overlay."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsynth.core.loader.file import hash_model_file
from convert_student_bf16 import sha256_file


MODEL_SIGNATURE = "da12f645b7edaedb855f6bd8cee7b24c"


def merge(base: Path, finetune: Path, output: Path):
    if output.exists() or output.resolve() in (base.resolve(), finetune.resolve()):
        raise ValueError("Choose a new output path; existing files are never overwritten.")
    if hash_model_file([str(base), str(finetune)]) != MODEL_SIGNATURE:
        raise ValueError("The combined keys/shapes do not describe a complete TBDub DiT.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        sources = [stack.enter_context(safe_open(str(p), framework="pt", device="cpu"))
                   for p in (base, finetune)]
        # Exactly the same last-file-wins rule as inference's DiskMap.
        origins = {key: index for index, source in enumerate(sources) for key in source.keys()}
        counts = Counter(origins.values())
        tensors = {}
        for key, index in origins.items():
            value = sources[index].get_tensor(key)
            if value.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"Unexpected dtype for {key}: {value.dtype}")
            value = value.to(torch.bfloat16).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite BF16 values in {key}")
            tensors[key] = value
        numel = sum(value.numel() for value in tensors.values())
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".safetensors", delete=False) as f:
            temporary = Path(f.name)
        try:
            save_file(tensors, str(temporary), metadata={
                "format": "pt", "variant": "teacher", "dtype": "bfloat16",
                "standalone": "true", "merge_order": "base_then_finetune",
            })
            del tensors
            with safe_open(str(temporary), framework="pt", device="cpu") as saved:
                if set(saved.keys()) != set(origins):
                    raise RuntimeError("Exported keys differ from the combined inputs.")
                for key, index in origins.items():
                    expected = sources[index].get_tensor(key).to(torch.bfloat16)
                    actual = saved.get_tensor(key)
                    if actual.dtype != torch.bfloat16 or not torch.equal(actual, expected):
                        raise RuntimeError(f"Exported tensor differs from runtime merge: {key}")
            if hash_model_file(str(temporary)) != MODEL_SIGNATURE:
                raise RuntimeError("Exported model signature changed.")
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "base": str(base), "finetune": str(finetune), "output": str(output),
        "source_bytes": base.stat().st_size + finetune.stat().st_size,
        "output_bytes": output.stat().st_size, "output_sha256": sha256_file(output),
        "tensor_count": len(origins), "numel": numel, "dtype": "BF16",
        "tensors_from_base": counts[0], "tensors_from_finetune": counts[1],
        "model_signature": MODEL_SIGNATURE, "all_tensors_equal_to_runtime_merge": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("finetune", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = merge(args.base, args.finetune, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
