from __future__ import annotations

import argparse

from il_lab.data import describe_demo_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a ManiSkill HDF5 demonstration file.")
    parser.add_argument("--demo-path", required=True, help="Path to a ManiSkill .h5 trajectory file.")
    parser.add_argument("--max-episodes", type=int, default=3, help="Number of episodes to summarize.")
    args = parser.parse_args()

    print(describe_demo_file(args.demo_path, max_episodes=args.max_episodes))


if __name__ == "__main__":
    main()
