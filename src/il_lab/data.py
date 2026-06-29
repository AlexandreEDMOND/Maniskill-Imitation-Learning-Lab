from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class DemoDataset:
    observations: np.ndarray
    actions: np.ndarray
    episodes: int
    source: str

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])


def load_bc_dataset(
    path: str | Path,
    max_episodes: int | None = None,
    observation_source: str = "auto",
) -> DemoDataset:
    """Load state observations and actions from a ManiSkill trajectory HDF5 file."""
    if observation_source not in {"auto", "obs", "env_states"}:
        raise ValueError("observation_source must be one of: auto, obs, env_states.")

    demo_path = Path(path).expanduser()
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    resolved_source: str | None = None

    with h5py.File(demo_path, "r") as file:
        episode_names = sorted(name for name in file.keys() if name.startswith("traj_"))
        if not episode_names:
            raise ValueError(
                f"No trajectory groups named 'traj_*' were found in {demo_path}. "
                "This does not look like a ManiSkill trajectory HDF5 file."
            )

        for name in episode_names[:max_episodes]:
            group = file[name]
            if "actions" not in group:
                raise ValueError(f"Trajectory {name} does not contain an 'actions' dataset.")

            episode_actions = _as_2d(np.asarray(group["actions"], dtype=np.float32), "actions", name)
            episode_obs, episode_source = _load_observation_array(group, name, observation_source)
            if resolved_source is None:
                resolved_source = episode_source
            elif resolved_source != episode_source:
                raise ValueError(
                    f"Mixed observation sources in one dataset: {resolved_source} and {episode_source}."
                )
            episode_obs = _align_observations(episode_obs, len(episode_actions), name)

            observations.append(episode_obs)
            actions.append(episode_actions)

    if not observations:
        raise ValueError(f"No usable episodes were loaded from {demo_path}.")

    obs_array = np.concatenate(observations, axis=0)
    action_array = np.concatenate(actions, axis=0)
    return DemoDataset(obs_array, action_array, episodes=len(observations), source=resolved_source or "unknown")


def describe_demo_file(path: str | Path, max_episodes: int = 3) -> str:
    """Return a readable summary of a ManiSkill HDF5 trajectory file."""
    demo_path = Path(path).expanduser()
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    lines = [f"File: {demo_path}"]
    with h5py.File(demo_path, "r") as file:
        root_keys = list(file.keys())
        lines.append("Root keys: " + _format_key_list(root_keys))
        episode_names = sorted(name for name in file.keys() if name.startswith("traj_"))
        lines.append(f"Trajectory groups: {len(episode_names)}")

        for name in episode_names[:max_episodes]:
            group = file[name]
            lines.append(f"\n{name}:")
            _describe_node(group, lines, indent="  ")

        try:
            dataset = load_bc_dataset(demo_path, max_episodes=max_episodes)
        except Exception as exc:
            lines.append("\nBC readiness: not ready")
            lines.append(f"Reason: {exc}")
        else:
            lines.append("\nBC readiness: ready")
            lines.append(f"Observation source: {dataset.source}")
            lines.append(f"Loaded sample count: {len(dataset.observations)}")
            lines.append(f"Observation dim: {dataset.obs_dim}")
            lines.append(f"Action dim: {dataset.action_dim}")

    return "\n".join(lines)


def _format_key_list(keys: list[str], limit: int = 20) -> str:
    if len(keys) <= limit:
        return ", ".join(keys)
    shown = ", ".join(keys[:limit])
    return f"{shown}, ... ({len(keys) - limit} more)"


def _flatten_h5_node(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return _as_2d(np.asarray(node, dtype=np.float32), node.name, "trajectory")

    arrays: list[np.ndarray] = []
    for key in sorted(node.keys()):
        child = node[key]
        if isinstance(child, h5py.Group):
            arrays.append(_flatten_h5_node(child))
        elif isinstance(child, h5py.Dataset) and np.issubdtype(child.dtype, np.number):
            arrays.append(_as_2d(np.asarray(child, dtype=np.float32), child.name, "trajectory"))

    if not arrays:
        raise ValueError(f"No numeric observation datasets were found under {node.name}.")

    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"Observation datasets under {node.name} have inconsistent lengths: {sorted(lengths)}"
        )

    return np.concatenate(arrays, axis=1)


def _load_observation_array(
    group: h5py.Group,
    trajectory_name: str,
    observation_source: str,
) -> tuple[np.ndarray, str]:
    errors: list[str] = []

    if observation_source in {"auto", "obs"}:
        if "obs" in group:
            try:
                return _flatten_h5_node(group["obs"]).astype(np.float32), "obs"
            except ValueError as exc:
                errors.append(str(exc))
        elif observation_source == "obs":
            errors.append(f"Trajectory {trajectory_name} does not contain 'obs'.")

    if observation_source in {"auto", "env_states"}:
        if "env_states" in group:
            return _flatten_h5_node(group["env_states"]).astype(np.float32), "env_states"
        errors.append(f"Trajectory {trajectory_name} does not contain 'env_states'.")

    detail = " ".join(errors)
    raise ValueError(
        f"Could not load observations for {trajectory_name}. {detail} "
        "Use --observation-source env_states to train directly from downloaded demos, "
        "or replay trajectories with scripts/prepare_state_demos.py to create obs_mode=state data."
    )


def _align_observations(observations: np.ndarray, action_count: int, trajectory_name: str) -> np.ndarray:
    if len(observations) == action_count:
        return observations
    if len(observations) == action_count + 1:
        return observations[:-1]
    raise ValueError(
        f"{trajectory_name} has {len(observations)} observations for {action_count} actions. "
        "Expected either the same count or one extra final observation."
    )


def _as_2d(array: np.ndarray, field_name: str, trajectory_name: str) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError(
            f"{trajectory_name}/{field_name} must have at least 2 dimensions, got shape {array.shape}."
        )
    return array.reshape(array.shape[0], -1)


def _describe_node(node: h5py.Dataset | h5py.Group, lines: list[str], indent: str) -> None:
    if isinstance(node, h5py.Dataset):
        lines.append(f"{indent}{Path(node.name).name}: shape={node.shape}, dtype={node.dtype}")
        return

    for key in node.keys():
        child: Any = node[key]
        if isinstance(child, h5py.Dataset):
            lines.append(f"{indent}{key}: shape={child.shape}, dtype={child.dtype}")
        else:
            lines.append(f"{indent}{key}/")
            _describe_node(child, lines, indent + "  ")
