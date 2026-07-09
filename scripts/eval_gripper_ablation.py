from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
from evaluate_policy import clip_action
from il_lab.env_utils import scalar_from_info
from log_policy_divergence import (
    extract_policy_metrics,
    get_policy_observation,
    load_normalization,
    load_policy,
    predict_action,
)


MODES = (
    "expert_arm_mlp_gripper",
    "mlp_arm_expert_gripper",
    "mlp_arm_scripted_gripper",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate arm/gripper ablations for a BC policy.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--script-close-distance", type=float, default=0.003)
    parser.add_argument("--script-min-close-steps", type=int, default=8)
    parser.add_argument("--results-path", default="results/gripper_ablation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    policy = load_policy(checkpoint, args.device)
    normalization = load_normalization(checkpoint, args.device)
    observation_source = checkpoint.get("observation_source", "obs")
    gripper_binary = bool(checkpoint.get("gripper_binary", False))

    demo_path = Path(args.demo_path).expanduser()
    trajectory_name = f"traj_{args.trajectory_index}"
    with h5py.File(demo_path, "r") as demo_file:
        if trajectory_name not in demo_file:
            raise ValueError(f"{trajectory_name} not found in {demo_path}.")
        expert_actions = np.asarray(
            demo_file[trajectory_name]["actions"],
            dtype=np.float32,
        ).reshape(len(demo_file[trajectory_name]["actions"]), -1)

    results = []
    for mode in MODES:
        result = evaluate_mode(
            args=args,
            mode=mode,
            expert_actions=expert_actions,
            checkpoint=checkpoint,
            policy=policy,
            normalization=normalization,
            observation_source=observation_source,
            gripper_binary=gripper_binary,
        )
        results.append(result)
        print(
            f"{mode}: success={result['success']} return={result['return']:.3f} "
            f"length={result['length']} first_grasp={result['first_grasp_step']} "
            f"min_tcp_cube={result['min_distance_tcp_cube']:.6f}"
        )

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "demo_path": str(demo_path),
        "trajectory": trajectory_name,
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "observation_source": observation_source,
        "gripper_binary": gripper_binary,
        "script_close_distance": args.script_close_distance,
        "script_min_close_steps": args.script_min_close_steps,
        "results": results,
    }

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ablation results: {results_path}")


def evaluate_mode(
    args: argparse.Namespace,
    mode: str,
    expert_actions: np.ndarray,
    checkpoint: dict[str, Any],
    policy: torch.nn.Module,
    normalization: dict[str, torch.Tensor] | None,
    observation_source: str,
    gripper_binary: bool,
) -> dict[str, Any]:
    env = gym.make(
        args.env_id,
        obs_mode="state_dict",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,
    )
    total_reward = 0.0
    success = None
    first_grasp_step = None
    min_distance_tcp_cube = float("inf")
    close_hold_remaining = 0
    scripted_closed = False
    step_records: list[dict[str, Any]] = []

    try:
        observation, _ = env.reset(seed=args.trajectory_index)
        horizon = min(len(expert_actions), args.max_episode_steps)
        for step in range(horizon):
            expert_action = expert_actions[step]
            policy_obs = get_policy_observation(env, observation, observation_source)
            mlp_action = predict_action(
                policy,
                policy_obs,
                normalization,
                gripper_binary,
                args.device,
            )
            action = compose_action(
                mode=mode,
                expert_action=expert_action,
                mlp_action=mlp_action,
                observation=observation,
                script_close_distance=args.script_close_distance,
                script_min_close_steps=args.script_min_close_steps,
                close_hold_remaining=close_hold_remaining,
                scripted_closed=scripted_closed,
            )
            close_hold_remaining = action["close_hold_remaining"]
            scripted_closed = action["scripted_closed"]
            applied_action = clip_action(env.action_space, action["action"])

            metrics = extract_policy_metrics(observation)
            min_distance_tcp_cube = min(min_distance_tcp_cube, metrics["distance_tcp_cube"])
            if metrics["is_grasped"] and first_grasp_step is None:
                first_grasp_step = step

            observation, reward, terminated, truncated, info = env.step(applied_action)
            if args.render_mode != "none":
                env.render()
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            success_value = scalar_from_info(info, "success")
            if success_value is not None:
                success = bool(success_value)

            step_records.append(
                {
                    "t": step,
                    "expert_gripper": float(expert_action[-1]),
                    "mlp_gripper": float(mlp_action[-1]),
                    "applied_gripper": float(applied_action[-1]),
                    "distance_tcp_cube": metrics["distance_tcp_cube"],
                    "is_grasped": metrics["is_grasped"],
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "success": success,
                }
            )

            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                break
    finally:
        env.close()

    return {
        "mode": mode,
        "success": success,
        "return": total_reward,
        "length": len(step_records),
        "first_grasp_step": first_grasp_step,
        "min_distance_tcp_cube": min_distance_tcp_cube,
        "steps": step_records,
    }


def compose_action(
    mode: str,
    expert_action: np.ndarray,
    mlp_action: np.ndarray,
    observation: Any,
    script_close_distance: float,
    script_min_close_steps: int,
    close_hold_remaining: int,
    scripted_closed: bool,
) -> dict[str, Any]:
    action = np.empty_like(expert_action, dtype=np.float32)
    if mode == "expert_arm_mlp_gripper":
        action[:-1] = expert_action[:-1]
        action[-1] = mlp_action[-1]
    elif mode == "mlp_arm_expert_gripper":
        action[:-1] = mlp_action[:-1]
        action[-1] = expert_action[-1]
    elif mode == "mlp_arm_scripted_gripper":
        action[:-1] = mlp_action[:-1]
        gripper, close_hold_remaining, scripted_closed = scripted_gripper(
            observation=observation,
            close_distance=script_close_distance,
            min_close_steps=script_min_close_steps,
            close_hold_remaining=close_hold_remaining,
            scripted_closed=scripted_closed,
        )
        action[-1] = gripper
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")
    return {
        "action": action,
        "close_hold_remaining": close_hold_remaining,
        "scripted_closed": scripted_closed,
    }


def scripted_gripper(
    observation: Any,
    close_distance: float,
    min_close_steps: int,
    close_hold_remaining: int,
    scripted_closed: bool,
) -> tuple[float, int, bool]:
    close_value = -1.0
    open_value = 1.0
    if scripted_closed:
        return close_value, max(0, close_hold_remaining - 1), True
    if close_hold_remaining > 0:
        return close_value, close_hold_remaining - 1, True

    distance_tcp_cube = extract_policy_metrics(observation)["distance_tcp_cube"]
    if distance_tcp_cube <= close_distance:
        return close_value, max(0, min_close_steps - 1), True
    return open_value, 0, False


if __name__ == "__main__":
    main()
