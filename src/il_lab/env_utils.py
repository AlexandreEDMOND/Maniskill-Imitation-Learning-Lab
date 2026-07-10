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


def extract_pick_cube_state(env: Any, info: dict[str, Any]) -> dict[str, float | bool]:
    extra = env.unwrapped._get_obs_extra(info)
    tcp_to_obj = np.asarray(extra["tcp_to_obj_pos"], dtype=np.float32).reshape(-1, 3)[0]
    obj_pose = np.asarray(extra["obj_pose"], dtype=np.float32).reshape(-1, 7)[0]
    return {
        "tcp_obj_distance": float(np.linalg.norm(tcp_to_obj)),
        "is_grasped": bool(np.asarray(extra["is_grasped"]).reshape(-1)[0]),
        "cube_z": float(obj_pose[2]),
    }
