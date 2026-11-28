"""Compare a complete ComfyUI checkpoint workflow in isolated Torch environments."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
import sys
from time import perf_counter


COMFY_ROOT = Path(__file__).resolve().parents[1] / "comfyui"


def parse_args():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a checkpoint under the current Python environment")
    run.add_argument("--backend", choices=("reference", "candle"), required=True)
    run.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    run.add_argument("--comfy-root", type=Path, default=COMFY_ROOT)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--checkpoint-sha256")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--width", type=int, default=256)
    run.add_argument("--height", type=int, default=256)
    run.add_argument("--seed", type=int, default=20260428)
    run.add_argument("--steps", type=int, default=4)
    run.add_argument("--cfg", type=float, default=7.0)
    run.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    run.add_argument("--positive", default="A white ceramic mug on a wooden table, morning light")
    run.add_argument("--negative", default="")
    run.add_argument("--sampler", default="euler")
    run.add_argument("--scheduler", default="normal")
    compare = commands.add_parser("compare", help="compare artifacts recorded by two run commands")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--noise-atol", type=float, default=1e-6)
    compare.add_argument("--clip-atol", type=float, default=2e-3)
    compare.add_argument("--latent-atol", type=float, default=5e-3)
    compare.add_argument("--rgb-atol", type=float, default=5e-3)
    compare.add_argument("--rtol", type=float, default=5e-3)
    return root.parse_args()


def sha256_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def capture(name, operation):
    print(f"Starting {name}", flush=True)
    start = perf_counter()
    result = operation()
    print(f"Finished {name} in {perf_counter() - start:.2f}s", flush=True)
    return result


def run_checkpoint(args):
    if args.width < 64 or args.height < 64 or args.width % 64 or args.height % 64:
        raise ValueError("width and height must be positive multiples of 64")
    if args.steps < 1 or not 0 <= args.seed < 2**64:
        raise ValueError("steps must be positive and seed must fit uint64")
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("a complete single-file safetensors checkpoint is required")
    digest = capture("checkpoint SHA256", lambda: sha256_file(checkpoint))
    if args.checkpoint_sha256 and digest != args.checkpoint_sha256.lower():
        raise ValueError(f"checkpoint hash differs: {digest}")

    if args.backend == "candle":
        try:
            distribution("torch")
        except PackageNotFoundError:
            pass
        else:
            raise RuntimeError("Candle gate requires a Python environment without original PyTorch")

    import numpy as np
    from PIL import Image
    import torch

    if bool(getattr(torch, "__candle_torch__", False)) != (args.backend == "candle"):
        raise RuntimeError(f"unexpected torch module for {args.backend}: {torch.__file__}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"the selected interpreter cannot access the physical Metal device: {torch.__file__}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"the selected interpreter cannot access a CUDA device: {torch.__file__}")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("arrays.npz", "rgb.png", "manifest.json")):
        raise FileExistsError(f"existing workflow artifacts in {out}; use a fresh output directory")
    comfy_root = args.comfy_root.resolve(strict=True)
    sys.path.insert(0, str(comfy_root))
    sys.argv[:] = [
        str(comfy_root / "main.py"), "--base-directory", str(out),
        "--disable-assets", "--disable-dynamic-vram",
    ]
    if args.precision == "fp32":
        sys.argv.extend(("--fp32-unet", "--fp32-vae", "--fp32-text-enc"))
    if args.device == "cpu":
        sys.argv.append("--cpu")
    from comfy import options
    options.enable_args_parsing()

    import comfy.model_management as models
    import comfy.sample
    import comfy.utils
    import folder_paths
    import nodes

    device = models.get_torch_device()
    if device.type != args.device:
        raise RuntimeError(f"ComfyUI selected {device} with {args.device} requested")
    folder_paths.add_model_folder_path("checkpoints", str(checkpoint.parent), is_default=True)
    selected = folder_paths.get_full_path_or_raise("checkpoints", checkpoint.name)
    if Path(selected).resolve() != checkpoint:
        raise RuntimeError("ComfyUI resolved a different checkpoint")
    comfy.utils.PROGRESS_BAR_ENABLED = False
    with torch.inference_mode():
        model, clip, vae = capture(
            "CheckpointLoaderSimple", lambda: nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
        )
        if clip is None or vae is None:
            raise RuntimeError("checkpoint must include its CLIP encoder and VAE")
        component_devices = {
            "diffusion": str(model.load_device), "clip": str(clip.patcher.load_device), "vae": str(vae.device),
        }
        component_dtypes = {
            "diffusion_weights": str(model.model.get_dtype()),
            "diffusion_compute": str(model.model.get_dtype_inference()), "vae": str(vae.vae_dtype),
        }
        print(f"ComfyUI component devices: {component_devices}", flush=True)
        print(f"ComfyUI component dtypes: {component_dtypes}", flush=True)
        encode = nodes.CLIPTextEncode().encode
        (positive,) = capture("CLIPTextEncode positive", lambda: encode(clip, args.positive))
        (negative,) = capture("CLIPTextEncode negative", lambda: encode(clip, args.negative))
        (empty,) = capture("EmptyLatentImage", lambda: nodes.EmptyLatentImage().generate(args.width, args.height))
        noise = capture("seeded initial noise", lambda: comfy.sample.prepare_noise(empty["samples"], args.seed))
        values = {"noise": noise, "clip_positive": positive[0][0], "clip_negative": negative[0][0]}
        arrays = {name: value.detach().float().cpu().contiguous().numpy() for name, value in values.items()}
        np.savez(out / "inputs.npz", **arrays)
        (latents,) = capture("KSampler", lambda: nodes.KSampler().sample(
            model, args.seed, args.steps, args.cfg, args.sampler, args.scheduler, positive, negative, empty
        ))
        (rgb,) = capture("VAEDecode", lambda: nodes.VAEDecode().decode(vae, latents))
        values = {"latent": latents["samples"], "rgb": rgb}
        arrays.update({name: value.detach().float().cpu().contiguous().numpy() for name, value in values.items()})
    for name, array in arrays.items():
        if not array.size or not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} is empty or contains NaN or infinity")
    if arrays["rgb"].shape != (1, args.height, args.width, 3):
        raise RuntimeError(f"VAEDecode produced an unexpected image shape: {arrays['rgb'].shape}")
    np.savez(out / "arrays.npz", **arrays)
    Image.fromarray((np.clip(arrays["rgb"][0], 0, 1) * 255).astype(np.uint8)).save(out / "rgb.png")
    workflow = {name: getattr(args, name) for name in (
        "width", "height", "seed", "steps", "cfg", "positive", "negative", "sampler", "scheduler", "precision"
    )}
    result = {
        "backend": args.backend, "torch_version": torch.__version__, "torch_file": torch.__file__,
        "device": str(device), "component_devices": component_devices, "component_dtypes": component_dtypes,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "workflow": workflow,
        "arrays": {
            name: {
                "shape": array.shape, "dtype": str(array.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest(),
            } for name, array in arrays.items()
        },
        "png_sha256": sha256_file(out / "rgb.png"),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def compare_runs(args):
    import numpy as np

    expected = json.loads((args.reference / "manifest.json").read_text())
    actual = json.loads((args.candidate / "manifest.json").read_text())
    if expected["backend"] != "reference" or actual["backend"] != "candle":
        raise ValueError("compare requires a reference run and a Candle run")
    if expected["checkpoint_sha256"] != actual["checkpoint_sha256"]:
        raise ValueError("runs used different checkpoint files")
    if {"precision": "fp32", **expected["workflow"]} != {"precision": "fp32", **actual["workflow"]}:
        raise ValueError("runs used different workflow parameters")
    limits = {
        "noise": (args.noise_atol, 0),
        "clip_positive": (args.clip_atol, args.rtol),
        "clip_negative": (args.clip_atol, args.rtol),
        "latent": (args.latent_atol, args.rtol),
        "rgb": (args.rgb_atol, args.rtol),
    }
    if not all(np.isfinite(atol) and np.isfinite(rtol) and min(atol, rtol) >= 0 for atol, rtol in limits.values()):
        raise ValueError("tolerances must be finite and nonnegative")
    metrics = {}
    with np.load(args.reference / "arrays.npz", allow_pickle=False) as reference, np.load(
        args.candidate / "arrays.npz", allow_pickle=False
    ) as candidate:
        for name, (atol, rtol) in limits.items():
            ref, output = reference[name], candidate[name]
            if ref.shape != output.shape or ref.dtype != output.dtype or not ref.size:
                raise RuntimeError(f"{name} output shape/dtype differs: {ref.shape}/{ref.dtype} vs {output.shape}/{output.dtype}")
            if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(output))):
                raise RuntimeError(f"{name} contains NaN or infinity")
            delta = np.abs(ref.astype(np.float64) - output)
            metrics[name] = {
                "atol": atol, "rtol": rtol, "max_absolute": float(delta.max()),
                "rmse": float(np.sqrt(np.mean(delta**2))), "different": int((delta > atol + rtol * abs(ref)).sum()),
            }
    result = {
        "reference_device": expected["device"], "candle_device": actual["device"],
        "reference_dtypes": expected.get("component_dtypes"), "candle_dtypes": actual.get("component_dtypes"),
        "same_png_sha256": expected["png_sha256"] == actual["png_sha256"], "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    if any(metric["different"] for metric in metrics.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parameters = parse_args()
    if parameters.command == "run":
        run_checkpoint(parameters)
    else:
        compare_runs(parameters)
