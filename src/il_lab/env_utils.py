from __future__ import annotations

from typing import Any

import numpy as np


def flatten_observation(observation: Any) -> np.ndarray:
    """Convert a ManiSkill/Gymnasium observation into one flat float32 vector."""
    if isinstance(observation, dict):
        arrays = [flatten_observation(observation[key]) for key in sorted(observation.keys())]
        return np.concatenate(arrays, axis=0).astype(np.float32)

    if isinstance(observation, (list, tuple)):
        arrays = [flatten_observation(item) for item in observation]
        return np.concatenate(arrays, axis=0).astype(np.float32)

    if hasattr(observation, "detach"):
        observation = observation.detach().cpu().numpy()

    array = np.asarray(observation, dtype=np.float32)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1).astype(np.float32)


def scalar_from_info(info: dict[str, Any], key: str) -> float | None:
    if key not in info:
        return None
    value = info[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return None
    return float(array.reshape(-1)[0])
