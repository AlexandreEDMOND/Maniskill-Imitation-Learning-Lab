from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
from il_lab.env_utils import flatten_observation, scalar_from_info
from il_lab.model import MLPPolicy


def flatten_state_dict_sorted(state_dict: dict) -> np.ndarray:
    """Flatten a nested state dict with sorted keys, matching il_lab.data._flatten_h5_node."""
    arrays: list[np.ndarray] = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if isinstance(value, dict):
            arrays.append(flatten_state_dict_sorted(value))
        else:
            arrays.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(arrays).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained BC policy in ManiSkill.")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--replay-expert", action="store_true")
    parser.add_argument("--demo-path", help="Path to a ManiSkill trajectory HDF5 file for expert replay.")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--render-mode", choices=["human", "none"], default="human")
    parser.add_argument("--arm-action-smoothing", type=float, default=0.0)
    parser.add_argument("--results-path", default="results/pickcube_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.arm_action_smoothing < 1.0:
        raise ValueError("--arm-action-smoothing must be in [0.0, 1.0).")
    if args.replay_expert and not args.demo_path:
        raise ValueError("--demo-path is required when using --replay-expert.")
    if not args.replay_expert and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required unless --replay-expert is used.")

    checkpoint = None
    policy = None
    observation_source = "expert_actions"
    obs_mean = obs_std = action_mean = action_std = None
    has_normalization = False
    gripper_binary = False
    base_obs_dim = None
    frame_stack = 1
    use_timestep = False
    use_previous_action = False
    residual_arm = False
    timestep_horizon = args.max_episode_steps - 1
    if not args.replay_expert:
        checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
        observation_source = checkpoint.get("observation_source", "obs")
        base_obs_dim = int(checkpoint.get("base_obs_dim", checkpoint["obs_dim"]))
        frame_stack = int(checkpoint.get("frame_stack", 1))
        use_timestep = bool(checkpoint.get("use_timestep", False))
        use_previous_action = bool(checkpoint.get("use_previous_action", False))
        residual_arm = bool(checkpoint.get("residual_arm", False))
        timestep_horizon = int(checkpoint.get("timestep_horizon", timestep_horizon))
        policy = MLPPolicy(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            hidden_layers=int(checkpoint["hidden_layers"]),
        ).to(args.device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        obs_mean = checkpoint.get("obs_mean")
        obs_std = checkpoint.get("obs_std")
        action_mean = checkpoint.get("action_mean")
        action_std = checkpoint.get("action_std")
        has_normalization = all(
            value is not None for value in (obs_mean, obs_std, action_mean, action_std)
        )
        if has_normalization:
            obs_mean = obs_mean.to(args.device).float()
            obs_std = obs_std.to(args.device).float()
            action_mean = action_mean.to(args.device).float()
            action_std = action_std.to(args.device).float()
        gripper_binary = bool(checkpoint.get("gripper_binary", False))

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,
    )

    print(f"Mode: {'expert replay' if args.replay_expert else 'policy'}")
    print(f"Observation source: {observation_source}")
    print(f"Checkpoint normalization: {'enabled' if has_normalization else 'disabled'}")
    print(f"Binary gripper: {'enabled' if gripper_binary else 'disabled'}")
    print(f"Arm action smoothing: {args.arm_action_smoothing}")
    print(
        f"Features: frame_stack={frame_stack}, timestep={use_timestep}, "
        f"previous_action={use_previous_action}, residual_arm={residual_arm}"
    )

    episode_results = []
    try:
        demo_file = h5py.File(Path(args.demo_path).expanduser(), "r") if args.replay_expert else None
        for episode_offset in range(args.episodes):
            episode = args.start_seed + episode_offset
            trajectory_name = f"traj_{episode}"
            expert_actions = None
            if demo_file is not None:
                if trajectory_name not in demo_file:
                    print(f"episode={episode:03d} skipped: {trajectory_name} not found")
                    continue
                expert_actions = np.asarray(demo_file[trajectory_name]["actions"], dtype=np.float32)
            observation, _ = env.reset(seed=episode)
            total_reward = 0.0
            success = None
            previous_arm_action = None
            previous_action = None
            observation_history = []
            if expert_actions is None:
                initial_obs_vector = (
                    flatten_state_dict_sorted(env.unwrapped.get_state_dict())
                    if observation_source == "env_states"
                    else flatten_observation(observation)
                )
                observation_history = [initial_obs_vector.copy() for _ in range(frame_stack)]
                previous_action = np.zeros(int(checkpoint["action_dim"]), dtype=np.float32)

            for step in range(args.max_episode_steps):
                if expert_actions is not None:
                    if step >= len(expert_actions):
                        break
                    action = expert_actions[step].astype(np.float32)
                else:
                    if observation_source == "env_states":
                        obs_vector = flatten_state_dict_sorted(
                            env.unwrapped.get_state_dict()
                        )
                    else:
                        obs_vector = flatten_observation(observation)
                    if obs_vector.shape[0] != base_obs_dim:
                        raise ValueError(
                            f"Environment base observation dim is {obs_vector.shape[0]}, but checkpoint "
                            f"expects {base_obs_dim}. Check obs_mode and control_mode."
                        )
                    observation_history.append(obs_vector.copy())
                    observation_history = observation_history[-frame_stack:]
                    feature_vector = build_eval_feature(
                        observation_history=observation_history,
                        previous_action=previous_action,
                        step=step,
                        timestep_horizon=timestep_horizon,
                        use_timestep=use_timestep,
                        use_previous_action=use_previous_action,
                    )
                    if feature_vector.shape[0] != checkpoint["obs_dim"]:
                        raise ValueError(
                            f"Feature dim is {feature_vector.shape[0]}, but checkpoint "
                            f"expects {checkpoint['obs_dim']}."
                        )

                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(feature_vector).unsqueeze(0).to(args.device)
                        if has_normalization:
                            obs_tensor = (obs_tensor - obs_mean) / obs_std
                        action_tensor = policy(obs_tensor)
                        if has_normalization:
                            action_tensor = action_tensor * action_std + action_mean
                        action = action_tensor.cpu().numpy()[0].astype(np.float32)
                        if residual_arm:
                            action[:-1] = obs_vector[: action.shape[0] - 1] + action[:-1]
                        if gripper_binary:
                            action[-1] = 1.0 if action[-1] > 0.0 else -1.0
                        if previous_arm_action is not None and args.arm_action_smoothing > 0.0:
                            alpha = args.arm_action_smoothing
                            action[:-1] = alpha * previous_arm_action + (1.0 - alpha) * action[:-1]

                action = clip_action(env.action_space, action)
                previous_arm_action = action[:-1].copy()
                if previous_action is not None:
                    previous_action = action.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                if args.render_mode != "none":
                    env.render()
                total_reward += float(np.asarray(reward).reshape(-1)[0])

                success_value = scalar_from_info(info, "success")
                if success_value is not None:
                    success = bool(success_value) if success is None else success or bool(success_value)

                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break

            episode_results.append(
                {
                    "episode": episode,
                    "trajectory": trajectory_name if args.replay_expert else None,
                    "return": total_reward,
                    "length": step + 1,
                    "success": success,
                }
            )
            print(
                f"episode={episode:03d} return={total_reward:.3f} "
                f"length={step + 1} success={success}"
            )
    finally:
        if "demo_file" in locals() and demo_file is not None:
            demo_file.close()
        env.close()

    successes = [result["success"] for result in episode_results if result["success"] is not None]
    summary = {
        "mode": "expert_replay" if args.replay_expert else "policy",
        "demo_path": str(Path(args.demo_path).expanduser()) if args.demo_path else None,
        "checkpoint_path": args.checkpoint_path,
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "mean_return": float(np.mean([result["return"] for result in episode_results])),
        "mean_length": float(np.mean([result["length"] for result in episode_results])),
        "success_rate": float(np.mean(successes)) if successes else None,
        "episodes_detail": episode_results,
    }

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results: {results_path}")


def clip_action(action_space: gym.Space, action: np.ndarray) -> np.ndarray:
    if isinstance(action_space, gym.spaces.Box):
        return np.clip(action, action_space.low, action_space.high).astype(np.float32)
    return action


def build_eval_feature(
    observation_history: list[np.ndarray],
    previous_action: np.ndarray | None,
    step: int,
    timestep_horizon: int,
    use_timestep: bool,
    use_previous_action: bool,
) -> np.ndarray:
    parts = [observation.astype(np.float32) for observation in observation_history]
    if use_timestep:
        parts.append(np.asarray([step / max(1, timestep_horizon)], dtype=np.float32))
    if use_previous_action:
        if previous_action is None:
            raise ValueError("previous_action is required for this checkpoint.")
        parts.append(previous_action.astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


if __name__ == "__main__":
    main()
