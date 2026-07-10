from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from il_lab.data import _align_observations, _load_observation_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize timing and phase markers in ManiSkill demos.")
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--observation-source", choices=["auto", "obs", "env_states"], default="obs")
    parser.add_argument("--lift-threshold", type=float, default=0.03)
    parser.add_argument("--results-path", default="results/demo_phase_timing.json")
    args = parser.parse_args()

    demo_path = Path(args.demo_path).expanduser()
    summaries = []
    with h5py.File(demo_path, "r") as file:
        names = sorted(
            (name for name in file.keys() if name.startswith("traj_")),
            key=lambda name: int(name.removeprefix("traj_")),
        )[: args.max_episodes]
        for name in names:
            group = file[name]
            actions = np.asarray(group["actions"], dtype=np.float32).reshape(len(group["actions"]), -1)
            observations, source = _load_observation_array(group, name, args.observation_source)
            observations = _align_observations(observations, len(actions), name)
            summaries.append(summarize_episode(name, actions, observations, source, args.lift_threshold))

    result = {
        "demo_path": str(demo_path),
        "episodes": len(summaries),
        "lift_threshold": args.lift_threshold,
        "summaries": summaries,
        "aggregate": aggregate(summaries),
    }
    print_summary(result)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved phase timing: {results_path}")


def summarize_episode(
    name: str,
    actions: np.ndarray,
    observations: np.ndarray,
    source: str,
    lift_threshold: float,
) -> dict:
    gripper = actions[:, -1]
    changes = np.flatnonzero(np.abs(np.diff(gripper)) > 1e-4) + 1
    first_gripper_change = first_or_none(changes)
    qpos0 = observations[0, :9].astype(float).tolist()
    cube_xyz = observations[:, 29:32]
    tcp_to_cube = observations[:, 36:39]
    is_grasped = observations[:, 18] > 0.5
    grasp_search_start = first_gripper_change or 0
    grasped_after_close = np.flatnonzero(is_grasped[grasp_search_start:]) + grasp_search_start
    initial_cube_pose = observations[0, 29:36].astype(float).tolist()
    initial_cube_z = float(cube_xyz[0, 2])
    first_lift = first_or_none(np.flatnonzero(cube_xyz[:, 2] > initial_cube_z + lift_threshold))
    length = int(len(actions))

    return {
        "trajectory": name,
        "source": source,
        "length": length,
        "first_gripper_change_t": first_gripper_change,
        "first_gripper_change_t_norm": normalized_time(first_gripper_change, length),
        "first_negative_gripper_t": first_or_none(np.flatnonzero(gripper < 0.0)),
        "first_positive_gripper_t": first_or_none(np.flatnonzero(gripper > 0.0)),
        "first_is_grasped_t": first_or_none(grasped_after_close),
        "first_lift_t": first_lift,
        "first_lift_t_norm": normalized_time(first_lift, length),
        "min_tcp_cube_distance": float(np.min(np.linalg.norm(tcp_to_cube, axis=1))),
        "initial_cube_pose": initial_cube_pose,
        "initial_qpos": qpos0,
        "gripper_change_times": [int(index) for index in changes.tolist()],
    }


def first_or_none(indices: np.ndarray) -> int | None:
    return int(indices[0]) if len(indices) else None


def normalized_time(step: int | None, length: int) -> float | None:
    return step / max(length - 1, 1) if step is not None else None


def aggregate(summaries: list[dict]) -> dict:
    keys = [
        "length",
        "first_gripper_change_t",
        "first_gripper_change_t_norm",
        "first_negative_gripper_t",
        "first_is_grasped_t",
        "first_lift_t",
        "first_lift_t_norm",
        "min_tcp_cube_distance",
    ]
    output = {}
    for key in keys:
        values = [summary[key] for summary in summaries if summary[key] is not None]
        if values:
            output[key] = {
                "min": float(np.min(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
    return output


def print_summary(result: dict) -> None:
    print("trajectory length close_t neg_t grasp_t lift_t min_tcp_cube")
    for summary in result["summaries"]:
        print(
            f"{summary['trajectory']:>10} "
            f"{summary['length']:>6} "
            f"{str(summary['first_gripper_change_t']):>7} "
            f"{str(summary['first_negative_gripper_t']):>5} "
            f"{str(summary['first_is_grasped_t']):>7} "
            f"{str(summary['first_lift_t']):>6} "
            f"{summary['min_tcp_cube_distance']:.4f}"
        )
    print("Aggregate:", json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
