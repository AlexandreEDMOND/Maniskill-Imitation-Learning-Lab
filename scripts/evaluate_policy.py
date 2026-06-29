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
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--results-path", default="results/pickcube_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    observation_source = checkpoint.get("observation_source", "obs")
    policy = MLPPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(args.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode="human",
    )

    print(f"Observation source from checkpoint: {observation_source}")

    episode_results = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=episode)
            total_reward = 0.0
            success = None

            for step in range(args.max_episode_steps):
                if observation_source == "env_states":
                    obs_vector = flatten_state_dict_sorted(
                        env.unwrapped.get_state_dict()
                    )
                else:
                    obs_vector = flatten_observation(observation)
                if obs_vector.shape[0] != checkpoint["obs_dim"]:
                    raise ValueError(
                        f"Environment observation dim is {obs_vector.shape[0]}, but checkpoint "
                        f"expects {checkpoint['obs_dim']}. Check obs_mode and control_mode."
                    )

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_vector).unsqueeze(0).to(args.device)
                    action = policy(obs_tensor).cpu().numpy()[0].astype(np.float32)

                action = clip_action(env.action_space, action)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()
                total_reward += float(np.asarray(reward).reshape(-1)[0])

                success_value = scalar_from_info(info, "success")
                if success_value is not None:
                    success = bool(success_value)

                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break

            episode_results.append(
                {
                    "episode": episode,
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
        env.close()

    successes = [result["success"] for result in episode_results if result["success"] is not None]
    summary = {
        "env_id": args.env_id,
        "episodes": args.episodes,
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


if __name__ == "__main__":
    main()
