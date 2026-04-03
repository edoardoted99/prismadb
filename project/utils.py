import logging
import os

import torch

logger = logging.getLogger(__name__)

# Default values for all app settings
SETTING_DEFAULTS = {
    'ollama_base_url': 'http://localhost:11434',
}

# Map setting keys to environment variable names
SETTING_ENV_VARS = {
    'ollama_base_url': 'OLLAMA_HOST',
}


def get_device() -> str:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_setting(key: str) -> str:
    """Load a setting: DB > env var > default."""
    # 1. Database (highest priority — user explicitly saved via UI)
    try:
        from explorer.models import AppSetting
        obj = AppSetting.objects.filter(key=key).first()
        if obj is not None:
            return obj.value
    except Exception:
        pass  # DB not ready (e.g. during migrations)

    # 2. Environment variable (initial/default override)
    env_var = SETTING_ENV_VARS.get(key)
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val

    # 3. Default
    return SETTING_DEFAULTS.get(key, '')


def set_setting(key: str, value: str):
    """Persist a setting to the database."""
    from explorer.models import AppSetting
    AppSetting.objects.update_or_create(key=key, defaults={'value': value})
