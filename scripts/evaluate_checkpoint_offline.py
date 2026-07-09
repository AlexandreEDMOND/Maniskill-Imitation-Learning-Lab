from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from il_lab.data import _align_observations, _load_observation_array
from il_lab.model import MLPPolicy
from train_bc import build_training_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BC checkpoint offline against recorded demos.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--demo-path", default=None)
    parser.add_argument(
        "--observation-source",
        choices=["auto", "obs", "env_states"],
        default=None,
        help="Defaults to the observation source stored in the checkpoint.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gripper-window", type=int, default=2)
    parser.add_argument("--results-path", default="results/checkpoint_offline_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.gripper_window < 0:
        raise ValueError("--gripper-window must be non-negative.")

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    demo_path = Path(args.demo_path or checkpoint["demo_path"]).expanduser()
    observation_source = args.observation_source or checkpoint.get("observation_source", "auto")

    episodes = load_episodes(demo_path, observation_source, args.max_episodes)
    observations = np.concatenate([episode["observations"] for episode in episodes], axis=0)
    actions = np.concatenate([episode["actions"] for episode in episodes], axis=0)
    episode_lengths = tuple(len(episode["actions"]) for episode in episodes)
    model_observations = build_model_observations(checkpoint, observations, actions, episode_lengths)

    policy = MLPPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(args.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    predictions, gripper_scores = predict_actions(
        policy,
        checkpoint,
        model_observations,
        observations,
        args.batch_size,
        args.device,
    )
    metrics = compute_metrics(
        predictions=predictions,
        gripper_scores=gripper_scores,
        actions=actions,
        episodes=episodes,
        gripper_binary=bool(checkpoint.get("gripper_binary", False)),
        gripper_window=args.gripper_window,
    )

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "demo_path": str(demo_path),
        "observation_source": observation_source,
        "episodes": len(episodes),
        "samples": int(len(actions)),
        "obs_dim": int(observations.shape[1]),
        "model_obs_dim": int(model_observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "gripper_binary": bool(checkpoint.get("gripper_binary", False)),
        "gripper_window": args.gripper_window,
        **metrics,
    }

    print_summary(summary)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results: {results_path}")


def load_episodes(
    demo_path: Path,
    observation_source: str,
    max_episodes: int | None,
) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    with h5py.File(demo_path, "r") as file:
        episode_names = sorted(
            (name for name in file.keys() if name.startswith("traj_")),
            key=trajectory_index,
        )
        if max_episodes is not None:
            episode_names = episode_names[:max_episodes]
        for name in episode_names:
            group = file[name]
            if "actions" not in group:
                raise ValueError(f"{name} does not contain an 'actions' dataset.")
            actions = np.asarray(group["actions"], dtype=np.float32).reshape(len(group["actions"]), -1)
            observations, source = _load_observation_array(group, name, observation_source)
            observations = _align_observations(observations, len(actions), name)
            episodes.append(
                {
                    "name": name,
                    "source": source,
                    "observations": observations.astype(np.float32),
                    "actions": actions.astype(np.float32),
                }
            )
    if not episodes:
        raise ValueError(f"No episodes were loaded from {demo_path}.")
    sources = {str(episode["source"]) for episode in episodes}
    if len(sources) != 1:
        raise ValueError(f"Mixed observation sources found: {sorted(sources)}")
    return episodes


def trajectory_index(name: str) -> int:
    return int(name.removeprefix("traj_"))


def build_model_observations(
    checkpoint: dict,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_lengths: tuple[int, ...],
) -> np.ndarray:
    frame_stack = int(checkpoint.get("frame_stack", 1))
    use_timestep = bool(checkpoint.get("use_timestep", False))
    use_previous_action = bool(checkpoint.get("use_previous_action", False))
    residual_arm = bool(checkpoint.get("residual_arm", False))
    if frame_stack == 1 and not use_timestep and not use_previous_action:
        return observations
    model_observations, _, _ = build_training_arrays(
        observations,
        actions,
        episode_lengths,
        use_timestep=use_timestep,
        use_previous_action=use_previous_action,
        residual_arm=residual_arm,
        frame_stack=frame_stack,
    )
    return model_observations


@torch.no_grad()
def predict_actions(
    policy: MLPPolicy,
    checkpoint: dict,
    observations: np.ndarray,
    base_observations: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs_mean = checkpoint.get("obs_mean")
    obs_std = checkpoint.get("obs_std")
    action_mean = checkpoint.get("action_mean")
    action_std = checkpoint.get("action_std")
    has_normalization = all(value is not None for value in (obs_mean, obs_std, action_mean, action_std))
    if has_normalization:
        obs_mean = obs_mean.to(device).float()
        obs_std = obs_std.to(device).float()
        action_mean = action_mean.to(device).float()
        action_std = action_std.to(device).float()

    predictions: list[np.ndarray] = []
    gripper_scores: list[np.ndarray] = []
    for start in tqdm(range(0, len(observations), batch_size), desc="offline eval", unit="batch"):
        obs_tensor = torch.from_numpy(observations[start : start + batch_size]).to(device)
        if has_normalization:
            obs_tensor = (obs_tensor - obs_mean) / obs_std
        output = policy(obs_tensor)
        gripper_scores.append(output[:, -1].detach().cpu().numpy())
        if has_normalization:
            output = output * action_std + action_mean
        action = output.detach().cpu().numpy().astype(np.float32)
        if bool(checkpoint.get("residual_arm", False)):
            base_batch = base_observations[start : start + batch_size]
            action[:, :-1] = base_batch[:, : action.shape[1] - 1] + action[:, :-1]
        if bool(checkpoint.get("gripper_binary", False)):
            action[:, -1] = np.where(action[:, -1] > 0.0, 1.0, -1.0)
        predictions.append(action)

    return np.concatenate(predictions, axis=0), np.concatenate(gripper_scores, axis=0)


def compute_metrics(
    predictions: np.ndarray,
    gripper_scores: np.ndarray,
    actions: np.ndarray,
    episodes: list[dict[str, np.ndarray]],
    gripper_binary: bool,
    gripper_window: int,
) -> dict:
    arm_error = predictions[:, :-1] - actions[:, :-1]
    action_error = predictions - actions
    mean_action = actions.mean(axis=0, keepdims=True)
    baseline_error = np.repeat(mean_action, len(actions), axis=0) - actions

    predicted_gripper_closed = predictions[:, -1] > 0.0
    expert_gripper_closed = actions[:, -1] > 0.0
    gripper_accuracy = float(np.mean(predicted_gripper_closed == expert_gripper_closed))

    transition_mask = build_gripper_transition_mask(episodes, gripper_window)
    transition_metrics = compute_transition_metrics(
        predictions,
        actions,
        gripper_scores,
        transition_mask,
        gripper_binary,
    )

    return {
        "action_mse": float(np.mean(action_error**2)),
        "arm_mse": float(np.mean(arm_error**2)),
        "arm_mae": float(np.mean(np.abs(arm_error))),
        "arm_max_abs_error": float(np.max(np.abs(arm_error))),
        "per_dim_mse": [float(value) for value in np.mean(action_error**2, axis=0)],
        "mean_action_baseline_mse": float(np.mean(baseline_error**2)),
        "mean_action_baseline_arm_mse": float(np.mean(baseline_error[:, :-1] ** 2)),
        "gripper_accuracy": gripper_accuracy,
        "gripper_false_positive_rate": float(
            np.mean((predicted_gripper_closed == 1) & (expert_gripper_closed == 0))
        ),
        "gripper_false_negative_rate": float(
            np.mean((predicted_gripper_closed == 0) & (expert_gripper_closed == 1))
        ),
        "gripper_positive_rate_predicted": float(np.mean(predicted_gripper_closed)),
        "gripper_positive_rate_expert": float(np.mean(expert_gripper_closed)),
        **transition_metrics,
    }


def build_gripper_transition_mask(episodes: list[dict[str, np.ndarray]], window: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    for episode in episodes:
        actions = episode["actions"]
        closed = actions[:, -1] > 0.0
        mask = np.zeros(len(actions), dtype=bool)
        transition_indices = np.flatnonzero(closed[1:] != closed[:-1]) + 1
        for index in transition_indices:
            start = max(0, index - window)
            end = min(len(mask), index + window + 1)
            mask[start:end] = True
        masks.append(mask)
    return np.concatenate(masks, axis=0)


def compute_transition_metrics(
    predictions: np.ndarray,
    actions: np.ndarray,
    gripper_scores: np.ndarray,
    transition_mask: np.ndarray,
    gripper_binary: bool,
) -> dict:
    count = int(np.count_nonzero(transition_mask))
    if count == 0:
        return {
            "gripper_transition_sample_count": 0,
            "gripper_transition_accuracy": None,
            "gripper_transition_arm_mse": None,
            "gripper_transition_score_margin_mean": None,
        }

    predicted_closed = predictions[transition_mask, -1] > 0.0
    expert_closed = actions[transition_mask, -1] > 0.0
    arm_error = predictions[transition_mask, :-1] - actions[transition_mask, :-1]
    score_margin = np.abs(gripper_scores[transition_mask]) if gripper_binary else None
    return {
        "gripper_transition_sample_count": count,
        "gripper_transition_accuracy": float(np.mean(predicted_closed == expert_closed)),
        "gripper_transition_arm_mse": float(np.mean(arm_error**2)),
        "gripper_transition_score_margin_mean": float(np.mean(score_margin)) if score_margin is not None else None,
    }


def print_summary(summary: dict) -> None:
    print(
        f"Loaded {summary['samples']} samples from {summary['episodes']} episodes "
        f"(source={summary['observation_source']}, obs_dim={summary['obs_dim']}, "
        f"action_dim={summary['action_dim']})."
    )
    print(f"action_mse={summary['action_mse']:.8f}")
    print(f"arm_mse={summary['arm_mse']:.8f} arm_mae={summary['arm_mae']:.8f}")
    print(f"mean_action_baseline_mse={summary['mean_action_baseline_mse']:.8f}")
    print(
        f"gripper_accuracy={summary['gripper_accuracy']:.6f} "
        f"false_pos={summary['gripper_false_positive_rate']:.6f} "
        f"false_neg={summary['gripper_false_negative_rate']:.6f}"
    )
    if summary["gripper_transition_sample_count"]:
        print(
            "gripper_transition "
            f"samples={summary['gripper_transition_sample_count']} "
            f"accuracy={summary['gripper_transition_accuracy']:.6f} "
            f"arm_mse={summary['gripper_transition_arm_mse']:.8f}"
        )


if __name__ == "__main__":
    main()
