"""Export a standalone Student checkpoint in the dtype used by TBDub inference."""

import argparse
from collections import Counter
import hashlib
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


STUDENT_SIGNATURE = "da12f645b7edaedb855f6bd8cee7b24c"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, output: Path):
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Choose a new output path; existing files are never overwritten.")
    if hash_model_file(str(source)) != STUDENT_SIGNATURE:
        raise ValueError("Expected a complete TBDub Student checkpoint with 4096-channel audio input.")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    tensors = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            counts[str(tensor.dtype)] += 1
            if tensor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"Unexpected dtype for {name}: {tensor.dtype}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values in {name}")
            tensor = tensor.to(torch.bfloat16).contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"BF16 conversion overflow in {name}")
            tensors[name] = tensor
    numel = sum(t.numel() for t in tensors.values())
    # Keep an incomplete export out of the final path if serialization fails.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".safetensors", delete=False) as file:
        temporary = Path(file.name)
    try:
        save_file(tensors, str(temporary), metadata={
            "format": "pt", "variant": "student", "dtype": "bfloat16",
            "standalone": "true", "source_file": source.name,
        })
        del tensors
        with safe_open(str(source), framework="pt", device="cpu") as before, safe_open(
            str(temporary), framework="pt", device="cpu"
        ) as after:
            if before.keys() != after.keys():
                raise RuntimeError("Exported parameter names differ from the source.")
            for name in before.keys():
                expected = before.get_tensor(name).to(torch.bfloat16)
                actual = after.get_tensor(name)
                if actual.dtype != torch.bfloat16 or not torch.equal(expected, actual):
                    raise RuntimeError(f"BF16 tensor verification failed: {name}")
        if hash_model_file(str(temporary)) != STUDENT_SIGNATURE:
            raise RuntimeError("Exported model signature changed.")
        # Publish atomically without clobbering a file created during export.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source": str(source), "output": str(output),
        "source_bytes": source.stat().st_size, "output_bytes": output.stat().st_size,
        "source_dtype_tensor_counts": dict(counts), "output_dtype": "BF16",
        "tensor_count": sum(counts.values()), "numel": numel,
        "model_signature": STUDENT_SIGNATURE, "all_tensors_equal_to_source_bf16": True,
        "output_sha256": sha256_file(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = convert(args.source, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
