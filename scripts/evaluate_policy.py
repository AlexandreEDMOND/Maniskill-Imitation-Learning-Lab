from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
from il_lab.env_utils import flatten_observation, scalar_from_info
from il_lab.model import MLPPolicy


def flatten_state_dict_sorted(state_dict: dict) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for key in sorted(state_dict):
        value = state_dict[key]
        if isinstance(value, dict):
            arrays.append(flatten_state_dict_sorted(value))
        else:
            arrays.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(arrays).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Behavior Cloning policy in ManiSkill.")
    parser.add_argument("--checkpoint-path", default="checkpoints/pushcube_bc.pt")
    parser.add_argument("--env-id", default="PushCube-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--results-path", default="results/pushcube_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    action_dim = int(checkpoint["action_dim"])
    action_horizon = int(checkpoint.get("action_horizon", 1))
    observation_history_size = int(checkpoint.get("observation_history", 1))
    policy = MLPPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=action_dim * action_horizon,
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(args.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    observation_source = checkpoint["observation_source"]
    obs_mean = checkpoint["obs_mean"].to(args.device).float()
    obs_std = checkpoint["obs_std"].to(args.device).float()
    action_mean = checkpoint["action_mean"].to(args.device).float()
    action_std = checkpoint["action_std"].to(args.device).float()
    if action_mean.numel() == action_dim:
        action_mean = action_mean.repeat(action_horizon)
        action_std = action_std.repeat(action_horizon)
    base_obs_dim = int(checkpoint.get("base_obs_dim", checkpoint["obs_dim"]))

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,
    )

    episode_results = []
    try:
        for episode in range(args.start_seed, args.start_seed + args.episodes):
            observation, _ = env.reset(seed=episode)
            total_reward = 0.0
            success = False
            predicted_chunks: list[np.ndarray] = []
            observation_history: list[np.ndarray] = []

            for step in range(args.max_episode_steps):
                observation_vector = (
                    flatten_state_dict_sorted(env.unwrapped.get_state_dict())
                    if observation_source == "env_states"
                    else flatten_observation(observation)
                )
                if observation_vector.shape[0] != base_obs_dim:
                    raise ValueError(
                        f"Environment observation dim is {observation_vector.shape[0]}, but "
                        f"the checkpoint expects {base_obs_dim}."
                    )
                observation_history.append(observation_vector)
                observation_history = observation_history[-observation_history_size:]
                feature_vector = build_observation_history(
                    observation_history,
                    observation_history_size,
                )

                with torch.no_grad():
                    observation_tensor = torch.from_numpy(feature_vector).unsqueeze(0).to(args.device)
                    normalized_observation = (observation_tensor - obs_mean) / obs_std
                    normalized_action = policy(normalized_observation)
                    action_chunk = (
                        (normalized_action * action_std + action_mean)
                        .cpu()
                        .numpy()[0]
                        .reshape(action_horizon, action_dim)
                    )
                predicted_chunks.append(action_chunk)
                predicted_chunks = predicted_chunks[-action_horizon:]
                action = temporal_ensemble(predicted_chunks)
                action = clip_action(env.action_space, action)

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(np.asarray(reward).reshape(-1)[0])
                success = success or bool(scalar_from_info(info, "success"))
                if args.render_mode != "none":
                    env.render()
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break

            result = {
                "episode": episode,
                "return": total_reward,
                "length": step + 1,
                "success": success,
            }
            episode_results.append(result)
            print(
                f"episode={episode:03d} return={total_reward:.3f} "
                f"length={step + 1} success={success}"
            )
    finally:
        env.close()

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "action_horizon": action_horizon,
        "observation_history": observation_history_size,
        "mean_return": float(np.mean([result["return"] for result in episode_results])),
        "mean_length": float(np.mean([result["length"] for result in episode_results])),
        "success_rate": float(np.mean([result["success"] for result in episode_results])),
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


def temporal_ensemble(predicted_chunks: list[np.ndarray]) -> np.ndarray:
    predictions = [
        chunk[len(predicted_chunks) - 1 - index]
        for index, chunk in enumerate(predicted_chunks)
    ]
    return np.mean(predictions, axis=0).astype(np.float32)


def build_observation_history(history: list[np.ndarray], size: int) -> np.ndarray:
    if not history:
        raise ValueError("Observation history cannot be empty.")
    frames = [history[0]] * (size - len(history)) + history
    return np.concatenate(frames).astype(np.float32)


if __name__ == "__main__":
    main()
