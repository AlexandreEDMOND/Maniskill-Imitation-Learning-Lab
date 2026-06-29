from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a ManiSkill trajectory file and save state observations for BC."
    )
    parser.add_argument("--traj-path", required=True, help="Path to the downloaded ManiSkill .h5 file.")
    parser.add_argument(
        "--output-name",
        default="trajectory.state.pd_joint_pos.physx_cpu.h5",
        help="Output HDF5 filename written next to --traj-path.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Replay environment count.")
    parser.add_argument("--count", type=int, default=None, help="Optional number of episodes to replay.")
    args = parser.parse_args()

    traj_path = Path(args.traj_path).expanduser()
    if not traj_path.exists():
        candidates = sorted(traj_path.parent.glob("**/*.h5"))
        if not candidates and traj_path.parent.parent.exists():
            candidates = sorted(traj_path.parent.parent.glob("**/*.h5"))
        candidate_text = "\n".join(f"  - {path}" for path in candidates) or "  none"
        raise FileNotFoundError(
            f"Trajectory file not found: {traj_path}\n"
            "Available HDF5 files near that path:\n"
            f"{candidate_text}\n"
            "Use one of these paths with --traj-path."
        )
    if not traj_path.with_suffix(".json").exists():
        raise FileNotFoundError(
            f"Expected metadata file next to the trajectory: {traj_path.with_suffix('.json')}"
        )

    output_path = traj_path.with_name(args.output_name)
    before = {path.resolve(): path.stat().st_mtime_ns for path in traj_path.parent.glob("*.h5")}
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(traj_path),
        "--use-env-states",
        "--obs-mode",
        "state",
        "--save-traj",
        "--num-envs",
        str(args.num_envs),
    ]
    if args.count is not None:
        command.extend(["--count", str(args.count)])

    print("Running:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Failed to replay the trajectory. If the traceback mentions Vulkan, svulkan2, "
            "or vk::createInstanceUnique, SAPIEN cannot create the ManiSkill environment on "
            "this machine. Otherwise, ManiSkill's replay CLI may have changed; run "
            "`uv run python -m mani_skill.trajectory.replay_trajectory --help` and adjust "
            "scripts/prepare_state_demos.py if your installed version uses different flags."
        ) from exc

    generated_path = _find_generated_h5(traj_path.parent, before)
    if generated_path != output_path:
        generated_json_path = generated_path.with_suffix(".json")
        output_json_path = output_path.with_suffix(".json")
        generated_path.replace(output_path)
        if generated_json_path.exists():
            generated_json_path.replace(output_json_path)

    print(f"Saved state observations to: {output_path}")


def _find_generated_h5(directory: Path, before: dict[Path, int]) -> Path:
    changed = []
    for path in directory.glob("*.h5"):
        resolved = path.resolve()
        if resolved not in before or path.stat().st_mtime_ns > before[resolved]:
            changed.append(path)

    if len(changed) != 1:
        candidates = "\n".join(str(path) for path in changed) or "none"
        raise RuntimeError(
            "Could not identify the replay output file unambiguously. "
            f"Changed HDF5 files:\n{candidates}"
        )
    return changed[0]


if __name__ == "__main__":
    main()
