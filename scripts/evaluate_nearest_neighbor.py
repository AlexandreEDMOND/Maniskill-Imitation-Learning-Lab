from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
from evaluate_policy import build_eval_feature, clip_action
from il_lab.data import load_bc_dataset
from il_lab.env_utils import extract_pick_cube_state, flatten_observation, scalar_from_info
from train_bc import build_training_arrays, parse_episode_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate nearest-neighbor BC in closed loop.")
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--observation-source", choices=["auto", "obs", "env_states"], default="obs")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--use-timestep", action="store_true")
    parser.add_argument("--use-previous-action", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--reach-threshold", type=float, default=0.04)
    parser.add_argument("--lift-threshold", type=float, default=0.03)
    parser.add_argument("--results-path", default="results/nearest_neighbor_eval.json")
    args = parser.parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.frame_stack < 1:
        raise ValueError("--frame-stack must be at least 1.")

    dataset = load_bc_dataset(
        args.demo_path,
        max_episodes=args.max_episodes,
        observation_source=args.observation_source,
        episode_indices=parse_episode_indices(args.episode_indices),
    )
    train_features, _, timestep_horizon = build_training_arrays(
        dataset.observations,
        dataset.actions,
        dataset.episode_lengths,
        use_timestep=args.use_timestep,
        use_previous_action=args.use_previous_action,
        residual_arm=False,
        frame_stack=args.frame_stack,
    )
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    train_features = ((train_features - feature_mean) / feature_std).astype(np.float32)
    train_actions = dataset.actions.astype(np.float32)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode=None,
    )

    episode_results = []
    try:
        for episode_offset in range(args.episodes):
            episode = args.start_seed + episode_offset
            observation, reset_info = env.reset(seed=episode)
            initial_task_state = extract_pick_cube_state(env, reset_info)
            obs_vector = flatten_observation(observation)
            observation_history = [obs_vector.copy() for _ in range(args.frame_stack)]
            previous_action = np.zeros(dataset.action_dim, dtype=np.float32)
            total_reward = 0.0
            success = None
            reached = False
            grasped = False
            lifted = False
            min_tcp_obj_distance = initial_task_state["tcp_obj_distance"]

            for step in range(args.max_episode_steps):
                obs_vector = flatten_observation(observation)
                observation_history.append(obs_vector.copy())
                observation_history = observation_history[-args.frame_stack :]
                feature = build_eval_feature(
                    observation_history=observation_history,
                    previous_action=previous_action,
                    step=step,
                    timestep_horizon=timestep_horizon,
                    use_timestep=args.use_timestep,
                    use_previous_action=args.use_previous_action,
                )
                query = ((feature[None, :] - feature_mean) / feature_std).astype(np.float32)
                action = nearest_action(query, train_features, train_actions, args.k, args.weighted)
                action = clip_action(env.action_space, action)
                previous_action = action.copy()

                observation, reward, terminated, truncated, info = env.step(action)
                task_state = extract_pick_cube_state(env, info)
                min_tcp_obj_distance = min(
                    min_tcp_obj_distance,
                    task_state["tcp_obj_distance"],
                )
                reached = reached or task_state["tcp_obj_distance"] <= args.reach_threshold
                grasped = grasped or task_state["is_grasped"]
                lifted = lifted or (
                    task_state["cube_z"]
                    >= initial_task_state["cube_z"] + args.lift_threshold
                )
                total_reward += float(np.asarray(reward).reshape(-1)[0])
                success_value = scalar_from_info(info, "success")
                if success_value is not None:
                    success = bool(success_value) if success is None else success or bool(success_value)
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break

            episode_results.append(
                {
                    "episode": episode,
                    "return": total_reward,
                    "length": step + 1,
                    "success": success,
                    "reached": reached,
                    "grasped": grasped,
                    "lifted": lifted,
                    "min_tcp_obj_distance": min_tcp_obj_distance,
                }
            )
            print(f"episode={episode:03d} return={total_reward:.3f} length={step + 1} success={success}")
    finally:
        env.close()

    successes = [result["success"] for result in episode_results if result["success"] is not None]
    nontrivial_successes = [
        bool(result["success"]) and result["grasped"]
        for result in episode_results
        if result["success"] is not None
    ]
    summary = {
        "mode": "nearest_neighbor",
        "demo_path": str(Path(args.demo_path).expanduser()),
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "train_episodes": dataset.episodes,
        "train_samples": int(len(dataset.actions)),
        "k": args.k,
        "weighted": args.weighted,
        "use_timestep": args.use_timestep,
        "use_previous_action": args.use_previous_action,
        "frame_stack": args.frame_stack,
        "reach_threshold": args.reach_threshold,
        "lift_threshold": args.lift_threshold,
        "mean_return": float(np.mean([result["return"] for result in episode_results])),
        "mean_length": float(np.mean([result["length"] for result in episode_results])),
        "success_rate": float(np.mean(successes)) if successes else None,
        "nontrivial_success_rate": (
            float(np.mean(nontrivial_successes)) if nontrivial_successes else None
        ),
        "reach_rate": float(np.mean([result["reached"] for result in episode_results])),
        "grasp_rate": float(np.mean([result["grasped"] for result in episode_results])),
        "lift_rate": float(np.mean([result["lifted"] for result in episode_results])),
        "episodes_detail": episode_results,
    }
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results: {results_path}")


def nearest_action(
    query: np.ndarray,
    train_features: np.ndarray,
    train_actions: np.ndarray,
    k: int,
    weighted: bool,
) -> np.ndarray:
    distances = np.sum((train_features - query) ** 2, axis=1)
    if k == 1:
        return train_actions[int(np.argmin(distances))].copy()
    nearest = np.argpartition(distances, k - 1)[:k]
    if not weighted:
        return train_actions[nearest].mean(axis=0).astype(np.float32)
    weights = 1.0 / np.maximum(distances[nearest], 1e-8)
    weights = weights / weights.sum()
    return np.sum(train_actions[nearest] * weights[:, None], axis=0).astype(np.float32)


if __name__ == "__main__":
    main()
