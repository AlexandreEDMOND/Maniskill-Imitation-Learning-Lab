from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from il_lab.data import load_bc_dataset
from il_lab.model import MLPPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple Behavior Cloning policy.")
    parser.add_argument("--demo-path", required=True, help="Path to a ManiSkill .h5 trajectory file.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated trajectory indices to load, for example '0' or '0,3,7'.",
    )
    parser.add_argument(
        "--observation-source",
        choices=["auto", "obs", "env_states"],
        default="auto",
        help="Use recorded obs, simulator env_states, or auto fallback.",
    )
    parser.add_argument("--checkpoint-path", default="checkpoints/pushcube_bc.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of future actions predicted from each observation.",
    )
    parser.add_argument(
        "--observation-history",
        type=int,
        default=1,
        help="Number of consecutive observations used to predict an action.",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation applied to normalized training observations.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be at least 1.")
    if args.observation_history < 1:
        raise ValueError("--observation-history must be at least 1.")
    if args.obs_noise_std < 0.0:
        raise ValueError("--obs-noise-std must be non-negative.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_bc_dataset(
        args.demo_path,
        max_episodes=args.max_episodes,
        observation_source=args.observation_source,
        episode_indices=parse_episode_indices(args.episode_indices),
    )
    observation_array, action_array = build_training_examples(
        dataset.observations,
        dataset.actions,
        dataset.episode_lengths,
        args.action_horizon,
        args.observation_history,
    )
    raw_observations = torch.from_numpy(observation_array)
    raw_actions = torch.from_numpy(action_array)
    obs_mean = raw_observations.mean(dim=0)
    obs_std = raw_observations.std(dim=0).clamp_min(1e-6)
    base_action_mean = torch.from_numpy(dataset.actions).mean(dim=0)
    base_action_std = torch.from_numpy(dataset.actions).std(dim=0).clamp_min(1e-6)
    action_mean = base_action_mean.repeat(args.action_horizon)
    action_std = base_action_std.repeat(args.action_horizon)
    observations = (raw_observations - obs_mean) / obs_std
    actions = (raw_actions - action_mean) / action_std
    tensor_dataset = TensorDataset(observations, actions)

    val_size = int(len(tensor_dataset) * args.val_fraction)
    train_size = len(tensor_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples to create a training split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_size else None

    policy = MLPPolicy(
        obs_dim=raw_observations.shape[1],
        action_dim=dataset.action_dim * args.action_horizon,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    print(
        f"Loaded {len(tensor_dataset)} samples from {dataset.episodes} episodes "
        f"(source={dataset.source}, input_dim={raw_observations.shape[1]}, "
        f"base_obs_dim={dataset.obs_dim}, action_dim={dataset.action_dim}, "
        f"observation_history={args.observation_history}, action_horizon={args.action_horizon})."
    )
    print("Training with observation and action normalization.")
    if args.obs_noise_std > 0.0:
        print(f"Adding Gaussian observation noise (std={args.obs_noise_std}).")

    last_val_loss = None
    progress = trange(1, args.epochs + 1, desc="training", unit="epoch")
    for epoch in progress:
        policy.train()
        train_losses: list[float] = []
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(args.device)
            batch_actions = batch_actions.to(args.device)
            if args.obs_noise_std > 0.0:
                batch_obs = batch_obs + torch.randn_like(batch_obs) * args.obs_noise_std
            predicted_actions = policy(batch_obs)
            loss = loss_fn(predicted_actions, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            last_val_loss = (
                evaluate_loss(
                    policy,
                    val_loader,
                    loss_fn,
                    args.device,
                )
                if val_loader
                else None
            )
        postfix = {"train_loss": f"{mean_train_loss:.6f}"}
        if last_val_loss is not None:
            postfix["val_loss"] = f"{last_val_loss:.6f}"
        progress.set_postfix(postfix)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": int(raw_observations.shape[1]),
            "base_obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "action_horizon": args.action_horizon,
            "observation_history": args.observation_history,
            "obs_noise_std": args.obs_noise_std,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "observation_source": dataset.source,
            "demo_path": str(Path(args.demo_path).expanduser()),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "action_mean": action_mean,
            "action_std": action_std,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


@torch.no_grad()
def evaluate_loss(
    policy: MLPPolicy,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: str,
) -> float:
    policy.eval()
    losses: list[float] = []
    for observations, actions in loader:
        observations = observations.to(device)
        actions = actions.to(device)
        predicted_actions = policy(observations)
        loss = loss_fn(predicted_actions, actions)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def parse_episode_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise ValueError("--episode-indices must contain at least one integer.")
    return indices


def build_training_examples(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_lengths: tuple[int, ...],
    action_horizon: int,
    observation_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_observations: list[np.ndarray] = []
    chunk_actions: list[np.ndarray] = []
    start = 0

    for length in episode_lengths:
        if length < action_horizon:
            start += length
            continue
        episode_observations = observations[start : start + length]
        episode_actions = actions[start : start + length]
        for step in range(length - action_horizon + 1):
            frames = [
                episode_observations[max(0, step - offset)]
                for offset in range(observation_history - 1, -1, -1)
            ]
            feature_observations.append(np.concatenate(frames))
            chunk_actions.append(episode_actions[step : step + action_horizon].reshape(-1))
        start += length

    if not feature_observations:
        raise ValueError("No episode is long enough for --action-horizon.")
    return np.stack(feature_observations), np.stack(chunk_actions)


if __name__ == "__main__":
    main()
