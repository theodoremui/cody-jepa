"""Device, mixed-precision, and compilation runtime support."""

import contextlib
import importlib.util
import sys

import torch


def amp_dtype(config, device):
    name = config.get("amp_dtype")
    if name in (None, "none", "float32"):
        return None
    if name == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured bfloat16 but the CUDA device lacks BF16 support")
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported amp_dtype {name!r}")


def autocast_context(config, device):
    dtype = amp_dtype(config, device)
    if dtype is not None and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def make_scaler(config, device):
    return torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype(config, device) is torch.float16
    )


def _validate_cuda_runtime(device):
    try:
        probe = torch.zeros(1, device=device)
        probe.add_(1)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        architectures = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        torch_cuda = getattr(torch.version, "cuda", None)
        original_error = str(error)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        has_required_arch = required_arch in architectures
        incompatible_build = any(
            signature in original_error.lower()
            for signature in (
                "no kernel image is available",
                "invalid device function",
                "not compatible with the current pytorch installation",
            )
        )
        remediation = (
            "This usually means the active environment loaded a PyTorch build that "
            f"does not include a runnable CUDA kernel for {required_arch}. If "
            "HAIC allocated an H100/Hopper GPU, reinstall the locked project "
            "environment with `uv sync --frozen --reinstall-package torch`, then "
            "rerun the training command through `uv run --frozen --no-sync`. If HAIC "
            "allocated a newer Blackwell GPU, use a PyTorch CUDA 12.8+ build "
            "or request a Hopper/H100 node."
            if incompatible_build
            else "The CUDA runtime failed before training; inspect the original "
            "error below and the active Python interpreter."
        )
        raise RuntimeError(
            "CUDA is visible, but this PyTorch build cannot execute a kernel on "
            f"{device_name} (cuda_compute_capability=sm_{capability[0]}{capability[1]}). "
            f"torch_version={torch.__version__}, torch_cuda_version={torch_cuda}, "
            f"torch_cuda_arch_list={architectures or 'unknown'}, "
            f"torch_has_required_cuda_arch={has_required_arch}, "
            f"python_executable={sys.executable}. {remediation} "
            f"original_cuda_error={original_error}"
        ) from error


def resolve_device(required_device="auto"):
    if required_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("this full run requires CUDA, but CUDA is unavailable")
        device = torch.device("cuda")
    elif required_device == "cpu":
        device = torch.device("cpu")
    elif required_device != "auto":
        device = torch.device(required_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable")
        _validate_cuda_runtime(device)
    return device


def validate_compile_runtime(device):
    if device.type != "cuda":
        return
    try:
        from torch.utils._triton import has_triton

        triton_available = bool(has_triton())
    except Exception:
        triton_available = False
    if triton_available:
        return
    triton_spec = importlib.util.find_spec("triton")
    triton_version = None
    if triton_spec is not None:
        try:
            import triton

            triton_version = getattr(triton, "__version__", "unknown")
        except Exception as error:
            triton_version = f"import_failed:{error!r}"
    raise RuntimeError(
        "compile=True requested CUDA torch.compile, but TorchInductor cannot "
        "use Triton in this environment. Run this HAIC training job with "
        "CONFIG['compile'] = False, or repair the environment with "
        "`uv sync --frozen --reinstall-package torch --reinstall-package triton` "
        "and rerun a small torch.compile CUDA smoke test before training. "
        f"torch_version={torch.__version__}, "
        f"triton_module={triton_spec.origin if triton_spec else None}, "
        f"triton_version={triton_version}, python_executable={sys.executable}"
    )


def maybe_compile(module, config, device):
    if not config.get("compile", False):
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile was requested but is unavailable")
    validate_compile_runtime(device)
    return torch.compile(module, dynamic=True)


__all__ = [
    "amp_dtype",
    "autocast_context",
    "make_scaler",
    "maybe_compile",
    "resolve_device",
    "validate_compile_runtime",
]
