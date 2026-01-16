#!/usr/bin/env python3
"""
Utility to view and reconstruct conversations from rollout data.

Usage:
    python view_rollouts.py runs/self_play_20240106/rollouts/rollout_0.json
    python view_rollouts.py runs/self_play_20240106/rollouts/ --all
    python view_rollouts.py runs/self_play_20240106/rollouts/rollout_0.json --game 0 --full
"""

import argparse
import json
from pathlib import Path


def load_rollout(path: Path) -> dict:
    """Load rollout data from JSON file."""
    with open(path) as f:
        return json.load(f)


def group_samples_by_game(samples: list) -> list[list]:
    """Group samples by game (using group_index or consecutive order)."""
    if not samples:
        return []

    games = []
    current_game = []
    current_group = None

    for sample in samples:
        group_id = sample.get("group_index", sample.get("index", 0))

        if current_group is None:
            current_group = group_id

        # New game starts when group_index changes
        if group_id != current_group:
            if current_game:
                games.append(current_game)
            current_game = []
            current_group = group_id

        current_game.append(sample)

    if current_game:
        games.append(current_game)

    return games


def format_sample(sample: dict, show_tokens: bool = False) -> str:
    """Format a single sample for display."""
    lines = []

    player = sample.get("player", "unknown")
    response = sample.get("response", "")
    reward = sample.get("reward")
    status = sample.get("status")

    lines.append(f"[{player}]")
    lines.append("-" * 40)
    lines.append(response if response else "(empty response)")

    if reward is not None:
        lines.append(f"\n  Reward: {reward}")
    if status:
        lines.append(f"  Status: {status}")
    if show_tokens:
        tokens = sample.get("tokens", [])
        lines.append(f"  Tokens: {len(tokens)}")
        if sample.get("rollout_log_probs"):
            lines.append(f"  Logprobs: {len(sample['rollout_log_probs'])}")

    return "\n".join(lines)


def format_game(game: list, game_idx: int, show_tokens: bool = False) -> str:
    """Format a full game conversation."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"GAME {game_idx + 1}")
    lines.append("=" * 60)

    if game:
        reward = game[0].get("reward")
        lines.append(f"Final Reward: {reward}")
        lines.append(f"Turns: {len(game)}")
        total_tokens = sum(len(s.get("tokens", [])) for s in game)
        lines.append(f"Total Tokens: {total_tokens}")
    lines.append("")

    for sample in game:
        lines.append(format_sample(sample, show_tokens))
        lines.append("")

    return "\n".join(lines)


def summarize_rollout(data: dict) -> str:
    """Generate summary statistics for a rollout."""
    samples = data.get("samples", [])
    games = group_samples_by_game(samples)

    rewards = []
    turns_per_game = []
    tokens_per_game = []

    for game in games:
        if game:
            r = game[0].get("reward")
            if r is not None:
                rewards.append(r)
            turns_per_game.append(len(game))
            tokens_per_game.append(sum(len(s.get("tokens", [])) for s in game))

    lines = []
    lines.append("ROLLOUT SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total samples: {len(samples)}")
    lines.append(f"Total games: {len(games)}")

    if rewards:
        lines.append(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
        lines.append(f"Min reward: {min(rewards):.4f}")
        lines.append(f"Max reward: {max(rewards):.4f}")

    if turns_per_game:
        lines.append(f"Mean turns/game: {sum(turns_per_game) / len(turns_per_game):.1f}")

    if tokens_per_game:
        lines.append(f"Mean tokens/game: {sum(tokens_per_game) / len(tokens_per_game):.0f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="View rollout data")
    parser.add_argument("path", type=str, help="Path to rollout JSON file or directory")
    parser.add_argument("--all", action="store_true", help="Process all rollout files in directory")
    parser.add_argument("--game", type=int, default=None, help="Show specific game index")
    parser.add_argument("--games", type=int, default=3, help="Number of games to show")
    parser.add_argument("--full", action="store_true", help="Show full token info")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary")

    args = parser.parse_args()
    path = Path(args.path)

    if path.is_dir():
        files = sorted(path.glob("rollout_*.json"))
        if args.all:
            for f in files:
                print(f"\n{'#' * 60}")
                print(f"FILE: {f}")
                print(f"{'#' * 60}\n")
                data = load_rollout(f)
                print(summarize_rollout(data))
        else:
            print(f"Found {len(files)} rollout files. Use --all to process all.")
            if files:
                print(f"Processing latest: {files[-1]}")
                path = files[-1]
            else:
                return

    if path.is_file():
        data = load_rollout(path)
        print(summarize_rollout(data))

        if args.summary_only:
            return

        print("\n")

        samples = data.get("samples", [])
        games = group_samples_by_game(samples)

        if args.game is not None:
            if 0 <= args.game < len(games):
                print(format_game(games[args.game], args.game, args.full))
            else:
                print(f"Game {args.game} not found. Available: 0-{len(games)-1}")
        else:
            for i, game in enumerate(games[:args.games]):
                print(format_game(game, i, args.full))
                print("\n")

            if len(games) > args.games:
                print(f"... and {len(games) - args.games} more games. Use --games N to see more.")


if __name__ == "__main__":
    main()
