import json
import torch


def get_device() -> str:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _runtime_config_path():
    from django.conf import settings
    return settings.PRISMADB_HOME / "runtime_config.json"


def load_runtime_config():
    """Load persisted runtime config (returns dict, never fails)."""
    path = _runtime_config_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_runtime_config(config: dict):
    """Merge config keys into the persisted runtime config file."""
    current = load_runtime_config()
    current.update(config)
    _runtime_config_path().write_text(json.dumps(current, indent=2))
