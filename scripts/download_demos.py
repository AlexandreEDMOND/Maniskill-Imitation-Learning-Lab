from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ManiSkill demonstration trajectories.")
    parser.add_argument("--env-id", default="PushCube-v1", help="ManiSkill environment id.")
    args = parser.parse_args()

    command = [sys.executable, "-m", "mani_skill.utils.download_demo", args.env_id]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
