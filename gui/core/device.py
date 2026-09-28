import os
import re


def resolve_device_env(choice: str) -> str | None:
    """Resolve a device choice string to its env value.

    Returns None for "Auto", "cpu" for "CPU", "cuda:n" for "CUDA:n".
    Raises ValueError for invalid choices.
    """
    m = re.fullmatch(r"^(?:Auto|CPU|CUDA:(\d+))$", choice)
    if not m:
        raise ValueError(f"Invalid device choice: {choice!r}")
    if choice == "Auto":
        return None
    if choice == "CPU":
        return "cpu"
    # choice matches CUDA:n
    return "cuda:" + m.group(1)


def apply_device(choice: str) -> None:
    """Apply a device choice, setting/clearing os.environ['DEVICE'].

    Raises ValueError on invalid choice.
    "Auto" clears the DEVICE env var.
    Valid choices set DEVICE to the resolved env value.
    """
    resolved = resolve_device_env(choice)
    if resolved is None:
        os.environ.pop("DEVICE", None)
    else:
        os.environ["DEVICE"] = resolved


def available_devices() -> list[str]:
    """Return list of available device strings.

    First entry is always "Auto". Lazily imports torch;
    if torch is available and CUDA is available, appends
    "CUDA:0" through "CUDA:<count-1>".
    Otherwise appends "CPU".
    """
    result = ["Auto"]
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                result.append(f"CUDA:{i}")
        else:
            result.append("CPU")
    except ImportError:
        result.append("CPU")
    return result
