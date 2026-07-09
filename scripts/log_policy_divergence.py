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
from evaluate_policy import build_eval_feature, clip_action, flatten_state_dict_sorted
from il_lab.data import _align_observations, _load_observation_array
from il_lab.env_utils import flatten_observation, scalar_from_info
from il_lab.model import MLPPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log step-by-step divergence between expert actions and policy actions."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--action-error-threshold", type=float, default=0.01)
    parser.add_argument("--qpos-error-threshold", type=float, default=0.02)
    parser.add_argument("--tcp-error-threshold", type=float, default=0.02)
    parser.add_argument("--cube-error-threshold", type=float, default=0.02)
    parser.add_argument("--results-path", default="results/policy_divergence.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    observation_source = checkpoint.get("observation_source", "obs")
    policy = load_policy(checkpoint, args.device)
    normalization = load_normalization(checkpoint, args.device)
    gripper_binary = bool(checkpoint.get("gripper_binary", False))
    frame_stack = int(checkpoint.get("frame_stack", 1))
    use_timestep = bool(checkpoint.get("use_timestep", False))
    use_previous_action = bool(checkpoint.get("use_previous_action", False))
    residual_arm = bool(checkpoint.get("residual_arm", False))
    timestep_horizon = int(checkpoint.get("timestep_horizon", args.max_episode_steps - 1))

    demo_path = Path(args.demo_path).expanduser()
    trajectory_name = f"traj_{args.trajectory_index}"
    with h5py.File(demo_path, "r") as demo_file:
        if trajectory_name not in demo_file:
            raise ValueError(f"{trajectory_name} not found in {demo_path}.")
        demo_group = demo_file[trajectory_name]
        expert_actions = np.asarray(demo_group["actions"], dtype=np.float32).reshape(
            len(demo_group["actions"]), -1
        )
        expert_observations, source = _load_observation_array(
            demo_group,
            trajectory_name,
            observation_source,
        )
        expert_observations = _align_observations(
            expert_observations,
            len(expert_actions),
            trajectory_name,
        )
        recorded_state_obs = np.asarray(demo_group["obs"], dtype=np.float32).reshape(
            len(demo_group["obs"]),
            -1,
        )

    env = gym.make(
        args.env_id,
        obs_mode="state_dict",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,
    )

    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    success = None
    try:
        observation, _ = env.reset(seed=args.trajectory_index)
        horizon = min(len(expert_actions), args.max_episode_steps)
        if args.max_steps is not None:
            horizon = min(horizon, args.max_steps)
        policy_obs0 = get_policy_observation(env, observation, observation_source)
        policy_history = [policy_obs0.copy() for _ in range(frame_stack)]
        previous_expert_action = np.zeros(expert_actions.shape[1], dtype=np.float32)
        previous_policy_action = np.zeros(expert_actions.shape[1], dtype=np.float32)

        for step in range(horizon):
            expert_action = expert_actions[step]
            policy_obs = get_policy_observation(env, observation, observation_source)
            expert_history = build_expert_history(expert_observations, step, frame_stack)
            policy_history.append(policy_obs.copy())
            policy_history = policy_history[-frame_stack:]
            expert_feature = build_eval_feature(
                observation_history=expert_history,
                previous_action=previous_expert_action,
                step=step,
                timestep_horizon=timestep_horizon,
                use_timestep=use_timestep,
                use_previous_action=use_previous_action,
            )
            policy_feature = build_eval_feature(
                observation_history=policy_history,
                previous_action=previous_policy_action,
                step=step,
                timestep_horizon=timestep_horizon,
                use_timestep=use_timestep,
                use_previous_action=use_previous_action,
            )

            mlp_on_expert_obs = predict_action(
                policy,
                expert_feature,
                normalization,
                gripper_binary,
                args.device,
            )
            mlp_on_policy_obs = predict_action(
                policy,
                policy_feature,
                normalization,
                gripper_binary,
                args.device,
            )
            if residual_arm:
                arm_dim = expert_actions.shape[1] - 1
                mlp_on_expert_obs[:arm_dim] = expert_observations[step, :arm_dim] + mlp_on_expert_obs[:arm_dim]
                mlp_on_policy_obs[:arm_dim] = policy_obs[:arm_dim] + mlp_on_policy_obs[:arm_dim]
            expert_action_applied = clip_action(env.action_space, expert_action.copy())
            mlp_on_expert_obs_applied = clip_action(env.action_space, mlp_on_expert_obs.copy())
            mlp_on_policy_obs_applied = clip_action(env.action_space, mlp_on_policy_obs.copy())

            expert_metrics = extract_expert_metrics(recorded_state_obs[step])
            policy_metrics = extract_policy_metrics(observation)
            action_error_on_expert = np.abs(mlp_on_expert_obs_applied - expert_action_applied)
            action_error_on_policy = np.abs(mlp_on_policy_obs_applied - expert_action_applied)
            metric_errors = compute_metric_errors(expert_metrics, policy_metrics)

            observation, reward, terminated, truncated, info = env.step(mlp_on_policy_obs_applied)
            if args.render_mode != "none":
                env.render()
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            success_value = scalar_from_info(info, "success")
            if success_value is not None:
                success = bool(success_value) if success is None else success or bool(success_value)

            steps.append(
                {
                    "t": step,
                    "expert_action": expert_action_applied.tolist(),
                    "mlp_action_on_expert_obs": mlp_on_expert_obs_applied.tolist(),
                    "mlp_action_on_policy_obs": mlp_on_policy_obs_applied.tolist(),
                    "error_on_expert_obs": summarize_action_error(action_error_on_expert),
                    "error_on_policy_obs": summarize_action_error(action_error_on_policy),
                    "qpos_error": metric_errors["qpos_error"],
                    "tcp_pose_error": metric_errors["tcp_pose_error"],
                    "cube_pose_error": metric_errors["cube_pose_error"],
                    "gripper_error": metric_errors["gripper_error"],
                    "distance_tcp_cube": policy_metrics["distance_tcp_cube"],
                    "expert_distance_tcp_cube": expert_metrics["distance_tcp_cube"],
                    "is_grasped": policy_metrics["is_grasped"],
                    "expert_is_grasped": expert_metrics["is_grasped"],
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "success": success,
                }
            )

            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                break
            previous_expert_action = expert_action_applied.copy()
            previous_policy_action = mlp_on_policy_obs_applied.copy()
    finally:
        env.close()

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "demo_path": str(demo_path),
        "trajectory": trajectory_name,
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "observation_source": source,
        "gripper_binary": gripper_binary,
        "frame_stack": frame_stack,
        "use_timestep": use_timestep,
        "use_previous_action": use_previous_action,
        "residual_arm": residual_arm,
        "return": total_reward,
        "length": len(steps),
        "success": success,
        "first_large_error": find_first_large_error(args, steps),
        "steps": steps,
    }

    print_summary(summary)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved divergence log: {results_path}")


def load_policy(checkpoint: dict[str, Any], device: str) -> MLPPolicy:
    policy = MLPPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def load_normalization(checkpoint: dict[str, Any], device: str) -> dict[str, torch.Tensor] | None:
    keys = ("obs_mean", "obs_std", "action_mean", "action_std")
    if not all(key in checkpoint for key in keys):
        return None
    return {key: checkpoint[key].to(device).float() for key in keys}


def get_policy_observation(env: gym.Env, observation: Any, observation_source: str) -> np.ndarray:
    if observation_source == "env_states":
        return flatten_state_dict_sorted(env.unwrapped.get_state_dict())
    if isinstance(observation, dict) and "agent" in observation and "extra" in observation:
        return state_vector_from_state_dict(observation)
    return flatten_observation(observation)


def state_vector_from_state_dict(observation: dict[str, Any]) -> np.ndarray:
    agent = observation["agent"]
    extra = observation["extra"]
    arrays = [
        np.asarray(agent["qpos"], dtype=np.float32).reshape(-1),
        np.asarray(agent["qvel"], dtype=np.float32).reshape(-1),
        np.asarray(extra["is_grasped"], dtype=np.float32).reshape(-1),
        np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1),
        np.asarray(extra["goal_pos"], dtype=np.float32).reshape(-1),
        np.asarray(extra["obj_pose"], dtype=np.float32).reshape(-1),
        np.asarray(extra["tcp_to_obj_pos"], dtype=np.float32).reshape(-1),
        np.asarray(extra["obj_to_goal_pos"], dtype=np.float32).reshape(-1),
    ]
    return np.concatenate(arrays).astype(np.float32)


def build_expert_history(
    expert_observations: np.ndarray,
    step: int,
    frame_stack: int,
) -> list[np.ndarray]:
    return [
        expert_observations[max(0, step - offset)].astype(np.float32)
        for offset in range(frame_stack - 1, -1, -1)
    ]


@torch.no_grad()
def predict_action(
    policy: MLPPolicy,
    obs_vector: np.ndarray,
    normalization: dict[str, torch.Tensor] | None,
    gripper_binary: bool,
    device: str,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(obs_vector).unsqueeze(0).to(device)
    if normalization is not None:
        obs_tensor = (obs_tensor - normalization["obs_mean"]) / normalization["obs_std"]
    action_tensor = policy(obs_tensor)
    if normalization is not None:
        action_tensor = action_tensor * normalization["action_std"] + normalization["action_mean"]
    action = action_tensor.cpu().numpy()[0].astype(np.float32)
    if gripper_binary:
        action[-1] = 1.0 if action[-1] > 0.0 else -1.0
    return action


def extract_expert_metrics(state_obs: np.ndarray) -> dict[str, Any]:
    qpos = state_obs[0:9]
    is_grasped = bool(state_obs[18] > 0.5)
    tcp_pose = state_obs[19:26]
    cube_pose = state_obs[29:36]
    tcp_to_obj = state_obs[36:39]
    return {
        "qpos": qpos,
        "tcp_pose": tcp_pose,
        "cube_pose": cube_pose,
        "gripper": qpos[-2:],
        "distance_tcp_cube": float(np.linalg.norm(tcp_to_obj)),
        "is_grasped": is_grasped,
    }


def extract_policy_metrics(observation: Any) -> dict[str, Any]:
    agent = observation["agent"]
    extra = observation["extra"]
    qpos = np.asarray(agent["qpos"], dtype=np.float32).reshape(-1)
    tcp_pose = np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1)
    cube_pose = np.asarray(extra["obj_pose"], dtype=np.float32).reshape(-1)
    tcp_to_obj = np.asarray(extra["tcp_to_obj_pos"], dtype=np.float32).reshape(-1)
    is_grasped = bool(np.asarray(extra["is_grasped"]).reshape(-1)[0])
    return {
        "qpos": qpos,
        "tcp_pose": tcp_pose,
        "cube_pose": cube_pose,
        "gripper": qpos[-2:],
        "distance_tcp_cube": float(np.linalg.norm(tcp_to_obj)),
        "is_grasped": is_grasped,
    }


def compute_metric_errors(expert: dict[str, Any], policy: dict[str, Any]) -> dict[str, float]:
    return {
        "qpos_error": max_abs(policy["qpos"] - expert["qpos"]),
        "tcp_pose_error": max_abs(policy["tcp_pose"] - expert["tcp_pose"]),
        "cube_pose_error": max_abs(policy["cube_pose"] - expert["cube_pose"]),
        "gripper_error": max_abs(policy["gripper"] - expert["gripper"]),
    }


def summarize_action_error(error: np.ndarray) -> dict[str, Any]:
    return {
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
        "arm_max_abs": float(np.max(error[:-1])),
        "gripper_abs": float(error[-1]),
        "per_dim_abs": [float(value) for value in error],
    }


def max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def find_first_large_error(args: argparse.Namespace, steps: list[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "expert_obs_action": None,
        "policy_obs_action": None,
        "qpos": None,
        "tcp": None,
        "cube": None,
        "gripper_state": None,
        "grasp_mismatch": None,
    }
    for step in steps:
        t = int(step["t"])
        if result["expert_obs_action"] is None and step["error_on_expert_obs"]["max_abs"] > args.action_error_threshold:
            result["expert_obs_action"] = t
        if result["policy_obs_action"] is None and step["error_on_policy_obs"]["max_abs"] > args.action_error_threshold:
            result["policy_obs_action"] = t
        if result["qpos"] is None and step["qpos_error"] > args.qpos_error_threshold:
            result["qpos"] = t
        if result["tcp"] is None and step["tcp_pose_error"] > args.tcp_error_threshold:
            result["tcp"] = t
        if result["cube"] is None and step["cube_pose_error"] > args.cube_error_threshold:
            result["cube"] = t
        if result["gripper_state"] is None and step["gripper_error"] > args.qpos_error_threshold:
            result["gripper_state"] = t
        if result["grasp_mismatch"] is None and step["is_grasped"] != step["expert_is_grasped"]:
            result["grasp_mismatch"] = t
    return result


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"trajectory={summary['trajectory']} success={summary['success']} "
        f"return={summary['return']:.3f} length={summary['length']}"
    )
    print("first_large_error=" + json.dumps(summary["first_large_error"], sort_keys=True))
    for step in summary["steps"][:25]:
        print(
            f"t={step['t']:03d} "
            f"err_expert={step['error_on_expert_obs']['max_abs']:.6f} "
            f"err_policy={step['error_on_policy_obs']['max_abs']:.6f} "
            f"qpos={step['qpos_error']:.6f} "
            f"tcp={step['tcp_pose_error']:.6f} "
            f"cube={step['cube_pose_error']:.6f} "
            f"gripper={step['gripper_error']:.6f} "
            f"dist={step['distance_tcp_cube']:.6f} "
            f"grasp={step['is_grasped']}"
        )


if __name__ == "__main__":
    main()
