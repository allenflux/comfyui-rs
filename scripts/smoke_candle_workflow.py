"""Run ComfyUI CPU workflow and sampler probes in an installed Candle environment."""

import asyncio
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comfyui"


class LocalEvents:
    """Provide a local event sink for the graph executor."""

    client_id = None
    last_node_id = None

    def __init__(self):
        self.sockets_metadata = {}

    def send_sync(self, *message):
        pass


async def run_image_workflow(output):
    from execution import CacheType, PromptExecutor, validate_prompt

    color = (16, 32, 48)
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 4, "batch_size": 1, "color": 0x102030}},
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0], "upscale_method": "nearest-exact", "width": 12, "height": 10, "crop": "disabled",
            },
        },
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "candle-smoke"}},
    }
    prompt_id = "candle-image-smoke"
    valid, errors, outputs, node_errors = await validate_prompt(prompt_id, prompt, None)
    if not valid or node_errors:
        raise RuntimeError(f"image prompt failed validation: {errors} {node_errors}")
    executor = PromptExecutor(LocalEvents(), cache_type=CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, prompt_id, execute_outputs=outputs)
    if not executor.success:
        raise RuntimeError(f"image prompt failed execution: {executor.status_messages}")
    (artifact,) = executor.history_result["outputs"]["3"]["images"]
    with Image.open(output / artifact["subfolder"] / artifact["filename"]) as image:
        if image.mode != "RGB" or image.size != (12, 10) or image.getextrema() != tuple((c, c) for c in color):
            raise RuntimeError("image workflow returned unexpected size, color space or pixels")


def run_euler_sampler():
    import torch
    from comfy.k_diffusion.sampling import sample_euler

    initial = torch.tensor([[[[-1, 0.25], [1, 3]]]], dtype=torch.float32)
    sigmas = torch.tensor([1, 0.5, 0.125, 0], dtype=torch.float32)
    steps = []

    def denoise(latent, sigma):
        return latent * 0.25 - sigma * 0.1

    result = sample_euler(denoise, initial, sigmas, callback=lambda event: steps.append(event["i"]), disable=True)
    expected = torch.tensor([[[[-0.095703125, -0.01025390625], [0.041015625, 0.177734375]]]], dtype=torch.float32)
    if steps != [0, 1, 2] or not torch.allclose(result, expected, atol=1e-7, rtol=0):
        raise RuntimeError(f"Euler sampler returned unexpected steps or output: {steps}, {result}")


def run_deis_gradient():
    import torch
    from comfy.k_diffusion.deis import cal_intergrand

    tensor = torch.tensor
    with torch.inference_mode():
        times = tensor([0.05, 0.1, 0.2, 0.4, 0.6, 0.9, 0.98], dtype=torch.float64)
        result = cal_intergrand(tensor(0.2, dtype=torch.float64), tensor(1.7, dtype=torch.float64), times)
        expected = tensor([1.273070774731, 1.077264089955, 0.9957426664295, 1.038310806977, 1.176216023307, 1.556321311253, 1.704439245552], dtype=torch.float64)
    if not torch.allclose(result, expected, atol=1e-11, rtol=1e-11):
        raise RuntimeError(f"DEIS integrand differs from the PyTorch reference {result}")


def main():
    try:
        distribution("torch")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("the Candle workflow smoke test requires an environment without the PyTorch distribution")

    import torch

    if not getattr(torch, "__candle_torch__", False):
        raise RuntimeError("the selected interpreter did not import candle-torch")

    sys.path.insert(0, str(SOURCE))
    from comfy import options

    with TemporaryDirectory(prefix="comfyui-candle-smoke-") as temp:
        sys.argv[:] = [str(SOURCE / "main.py"), "--cpu", "--disable-assets", "--base-directory", temp]
        options.enable_args_parsing()
        asyncio.run(run_image_workflow(Path(temp) / "output"))
        run_euler_sampler()
        run_deis_gradient()
    print("Candle workflow tests passed: built-in image prompt, Euler sampler and DEIS gradient")


if __name__ == "__main__":
    main()
