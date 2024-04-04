import os
from pathlib import Path


def _load_env_var_as_path(env_var: str) -> Path:
    """Load an environment variable as a Path object."""
    path_str = os.getenv(env_var)

    if path_str is None:
        msg = f"Please set the environment variable {env_var}."
        raise ValueError(msg)

    path = Path(path_str)
    if not path.is_dir():
        msg = f"{path} is not a valid directory path."
        raise ValueError(msg)
    return path
