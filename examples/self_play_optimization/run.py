#!/usr/bin/env python3
"""
Python wrapper for self-play optimization training with SLIME.

Features:
- FSDP training backend (simpler than Megatron for 8B models)
- Dual TP2 SGLang servers
- Automatic checkpoint cleanup (keep only N latest)
- Rollout sample logging with conversation reconstruction
- WandB integration
- Full configuration management

Usage:
    python run.py --model Qwen/Qwen2.5-7B-Instruct --num-rollouts 100
    python run.py --resume runs/self_play_20240106_123456
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Training configuration."""

    # Model - Docker path (mount /home/nickatomlin/georgiazhou -> /workspace)
    model: str = "/workspace/self_play/full_sft_retrain_gpt5/global_step_61/huggingface"

    # Cluster
    num_gpus: int = 4
    gpus_per_sglang_engine: int = 2  # TP size; 4 GPUs / 2 = 2 engines

    # Training
    train_backend: str = "fsdp"  # "fsdp" or "megatron"
    num_rollouts: int = 100
    rollout_batch_size: int = 16  # Unique game configs per rollout
    n_samples_per_prompt: int = 8  # Games per config (GRPO group size)
    # Total samples per rollout = rollout_batch_size * n_samples_per_prompt * 2 players
    # Example: 16 * 8 * 2 = 256 samples
    global_batch_size: int = 256  # Train on all samples from one rollout
    learning_rate: float = 1e-6

    # Game settings
    max_turns: int = 30
    max_retries_per_turn: int = 8
    rollout_temperature: float = 0.7
    rollout_max_context_len: int = 8192
    rollout_max_response_len: int = 2048

    # GRPO/PPO
    use_kl_loss: bool = False  # Whether to use KL penalty against reference model
    kl_coef: float = 0.001
    entropy_coef: float = 0.01
    clip_eps: float = 0.2

    # Checkpointing
    save_interval: int = 10
    keep_n_checkpoints: int = 3  # Only keep N most recent checkpoints

    # Logging
    save_rollout_samples: bool = True
    max_rollout_samples_per_iter: int = 10  # Only save N samples per rollout

    # WandB
    use_wandb: bool = False
    wandb_project: str = "self-play-optimization"
    wandb_group: Optional[str] = None

    # Paths (set automatically)
    output_dir: Optional[str] = None
    resume_from: Optional[str] = None

    # Internal
    pythonpath: str = field(default_factory=lambda: os.environ.get("PYTHONPATH", ""))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Self-play optimization training")

    # Model
    parser.add_argument("--model", type=str, default=Config.model)

    # Cluster
    parser.add_argument("--num-gpus", type=int, default=Config.num_gpus)
    parser.add_argument("--gpus-per-engine", type=int, default=Config.gpus_per_sglang_engine)

    # Training
    parser.add_argument("--train-backend", choices=["fsdp", "megatron"], default=Config.train_backend)
    parser.add_argument("--num-rollouts", type=int, default=Config.num_rollouts)
    parser.add_argument("--rollout-batch-size", type=int, default=Config.rollout_batch_size)
    parser.add_argument("--n-samples-per-prompt", type=int, default=Config.n_samples_per_prompt,
                        help="GRPO group size: games per unique game config")
    parser.add_argument("--global-batch-size", type=int, default=Config.global_batch_size)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)

    # Game
    parser.add_argument("--max-turns", type=int, default=Config.max_turns)
    parser.add_argument("--temperature", type=float, default=Config.rollout_temperature)

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=Config.save_interval)
    parser.add_argument("--keep-n-checkpoints", type=int, default=Config.keep_n_checkpoints)

    # Logging
    parser.add_argument("--max-samples-per-iter", type=int, default=Config.max_rollout_samples_per_iter)

    # KL loss
    parser.add_argument("--use-kl-loss", action="store_true", help="Enable KL penalty against reference model")

    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-group", type=str, default=None)

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from run directory")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    config = Config(
        model=args.model,
        num_gpus=args.num_gpus,
        gpus_per_sglang_engine=args.gpus_per_engine,
        train_backend=args.train_backend,
        num_rollouts=args.num_rollouts,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        global_batch_size=args.global_batch_size,
        learning_rate=args.lr,
        max_turns=args.max_turns,
        rollout_temperature=args.temperature,
        save_interval=args.save_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        max_rollout_samples_per_iter=args.max_samples_per_iter,
        use_kl_loss=args.use_kl_loss,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        resume_from=args.resume,
        output_dir=args.output_dir,
    )

    return config


def setup_paths(config: Config) -> Config:
    """Set up output directory and paths."""

    if config.resume_from:
        config.output_dir = config.resume_from
        print(f"Resuming from: {config.output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"self_play_{timestamp}"
        config.output_dir = f"./runs/{run_name}"

    if config.wandb_group is None:
        config.wandb_group = Path(config.output_dir).name

    # Create directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{config.output_dir}/checkpoints").mkdir(exist_ok=True)
    Path(f"{config.output_dir}/rollouts").mkdir(exist_ok=True)
    Path(f"{config.output_dir}/conversations").mkdir(exist_ok=True)

    # Add SLIME root to PYTHONPATH (for examples.self_play_optimization imports)
    # In Docker: /workspace/slime
    slime_root = str(Path(__file__).parent.parent.parent)
    if slime_root not in config.pythonpath:
        config.pythonpath = f"{slime_root}:{config.pythonpath}" if config.pythonpath else slime_root

    # Add dialop to PYTHONPATH
    # In Docker: /workspace/self_play/scripts
    dialop_path = str(Path(__file__).parent.parent.parent.parent / "self_play" / "scripts")
    if dialop_path not in config.pythonpath:
        config.pythonpath = f"{config.pythonpath}:{dialop_path}"

    return config


def save_config(config: Config):
    """Save configuration to output directory."""
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    print(f"Config saved to: {config_path}")


def build_slime_command(config: Config) -> list[str]:
    """Build the SLIME training command."""

    # Get path to SLIME's train.py (in the repo root)
    slime_root = Path(__file__).parent.parent.parent
    train_py = slime_root / "train.py"

    cmd = ["python3", str(train_py)]

    # Model & checkpoints
    cmd.extend(["--hf-checkpoint", config.model])
    cmd.extend(["--save", f"{config.output_dir}/checkpoints/"])
    cmd.extend(["--save-interval", str(config.save_interval)])

    # Training backend
    cmd.extend(["--train-backend", config.train_backend])

    # Cluster configuration (non-colocate for separate rollout GPUs)
    cmd.extend(["--actor-num-nodes", "1"])
    cmd.extend(["--actor-num-gpus-per-node", str(config.num_gpus)])

    # Dual SGLang servers: 4 GPUs / 2 per engine = 2 engines
    cmd.extend(["--rollout-num-gpus", str(config.num_gpus)])
    cmd.extend(["--rollout-num-gpus-per-engine", str(config.gpus_per_sglang_engine)])
    cmd.extend(["--colocate"])  # Share GPUs, swap memory

    # Rollout configuration
    cmd.extend(["--custom-generate-function-path",
                "examples.self_play_optimization.rollout.generate_with_self_play"])
    cmd.extend(["--custom-config-path",
                str(Path(__file__).parent / "config.yaml")])

    # Each game produces 2 samples (one per player), not variable
    # Total samples per rollout = rollout_batch_size * n_samples_per_prompt * 2
    # Example: 16 configs * 8 replays * 2 players = 256 samples

    cmd.extend(["--apply-chat-template"])

    # Provide a dummy dataset - our custom generate function ignores the sample content
    # and generates random games based on group_index as seed
    dummy_data_path = str(Path(__file__).parent / "dummy_prompts.jsonl")
    cmd.extend(["--prompt-data", dummy_data_path])
    cmd.extend(["--input-key", "prompt"])

    # GRPO grouping: n_samples_per_prompt games share the same seed (group_index)
    # Each game produces ~20 samples (10 turns × 2 players)
    cmd.extend(["--num-rollout", str(config.num_rollouts)])
    cmd.extend(["--rollout-batch-size", str(config.rollout_batch_size)])
    cmd.extend(["--n-samples-per-prompt", str(config.n_samples_per_prompt)])

    cmd.extend(["--rollout-max-context-len", str(config.rollout_max_context_len)])
    cmd.extend(["--rollout-max-response-len", str(config.rollout_max_response_len)])
    cmd.extend(["--rollout-temperature", str(config.rollout_temperature)])

    # Training
    cmd.extend(["--global-batch-size", str(config.global_batch_size)])
    cmd.extend(["--use-dynamic-batch-size"])
    cmd.extend(["--max-tokens-per-gpu", "16384"])

    # GRPO/PPO
    cmd.extend(["--advantage-estimator", "grpo"])
    if config.use_kl_loss:
        cmd.extend(["--use-kl-loss"])
        cmd.extend(["--kl-loss-coef", str(config.kl_coef)])
        cmd.extend(["--ref-load", config.model])  # Reference model for KL
    cmd.extend(["--entropy-coef", str(config.entropy_coef)])
    cmd.extend(["--eps-clip", str(config.clip_eps)])

    # Optimizer
    cmd.extend(["--optimizer", "adam"])
    cmd.extend(["--lr", str(config.learning_rate)])
    cmd.extend(["--lr-decay-style", "constant"])
    cmd.extend(["--weight-decay", "0.1"])

    # SGLang settings
    cmd.extend(["--sglang-mem-fraction-static", "0.8"])

    # Debug/logging
    if config.save_rollout_samples:
        cmd.extend(["--save-debug-rollout-data",
                    f"{config.output_dir}/rollouts/rollout_{{rollout_id}}.json"])

    # WandB
    if config.use_wandb:
        cmd.extend(["--use-wandb"])
        cmd.extend(["--wandb-project", config.wandb_project])
        cmd.extend(["--wandb-group", config.wandb_group])
        if os.environ.get("WANDB_KEY"):
            cmd.extend(["--wandb-key", os.environ["WANDB_KEY"]])

    return cmd


def cleanup_old_checkpoints(config: Config):
    """Keep only the N most recent checkpoints."""
    ckpt_dir = Path(config.output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return

    # Find all checkpoint directories (iter_XXXXX format for Megatron, step_X for FSDP)
    checkpoints = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and (p.name.startswith("iter_") or p.name.startswith("step_")):
            # Extract step number
            try:
                step = int(p.name.split("_")[1])
                checkpoints.append((step, p))
            except (IndexError, ValueError):
                continue

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    # Keep only the N most recent
    to_delete = checkpoints[config.keep_n_checkpoints:]
    for step, path in to_delete:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path)


def process_rollout_samples(config: Config, rollout_id: int):
    """Process rollout samples: subset and reconstruct conversations."""
    rollout_path = Path(config.output_dir) / "rollouts" / f"rollout_{rollout_id}.json"
    if not rollout_path.exists():
        return

    try:
        with open(rollout_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {rollout_path}")
        return

    samples = data.get("samples", [])
    if not samples:
        return

    # Group samples by game (assuming consecutive samples are from same game)
    # Each game produces multiple samples (one per player turn)
    conversations = []
    current_game = []
    current_game_id = None

    for sample in samples:
        game_id = sample.get("group_index", sample.get("index", 0))
        if current_game_id is None:
            current_game_id = game_id

        if game_id != current_game_id:
            if current_game:
                conversations.append(current_game)
            current_game = []
            current_game_id = game_id

        current_game.append(sample)

    if current_game:
        conversations.append(current_game)

    # Save subset of conversations in readable format
    output_path = Path(config.output_dir) / "conversations" / f"rollout_{rollout_id}.txt"
    with open(output_path, "w") as f:
        for i, game_samples in enumerate(conversations[:config.max_rollout_samples_per_iter]):
            f.write(f"{'='*60}\n")
            f.write(f"GAME {i+1}\n")
            f.write(f"{'='*60}\n\n")

            reward = game_samples[0].get("reward", "N/A") if game_samples else "N/A"
            f.write(f"Reward: {reward}\n\n")

            for sample in game_samples:
                player = sample.get("player", "unknown")
                response = sample.get("response", "")
                f.write(f"[{player}]:\n{response}\n\n")

            f.write("\n")

    # Also save a compact version with just rewards and lengths
    stats_path = Path(config.output_dir) / "conversations" / f"rollout_{rollout_id}_stats.json"
    stats = {
        "rollout_id": rollout_id,
        "num_games": len(conversations),
        "games": [
            {
                "reward": game[0].get("reward") if game else None,
                "num_turns": len(game),
                "total_tokens": sum(len(s.get("tokens", [])) for s in game),
            }
            for game in conversations[:config.max_rollout_samples_per_iter]
        ]
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Optionally delete the full rollout file to save space
    # Uncomment if disk space is critical:
    # rollout_path.unlink()

    print(f"Processed rollout {rollout_id}: {len(conversations)} games, saved {min(len(conversations), config.max_rollout_samples_per_iter)} examples")


def run_training(config: Config):
    """Run the SLIME training process."""

    # Build command
    cmd = build_slime_command(config)

    # Save command for reproducibility
    cmd_path = Path(config.output_dir) / "command.txt"
    with open(cmd_path, "w") as f:
        f.write(" \\\n    ".join(cmd))
    print(f"Command saved to: {cmd_path}")

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = config.pythonpath

    # Kill any existing processes
    subprocess.run(["pkill", "-9", "sglang"], capture_output=True)
    time.sleep(2)
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    time.sleep(2)

    # Start Ray
    print("Starting Ray...")
    subprocess.run(
        ["ray", "start", "--head", f"--num-gpus={config.num_gpus}", "--disable-usage-stats"],
        check=True
    )
    time.sleep(3)

    # Log file
    log_path = Path(config.output_dir) / "train.log"

    try:
        print(f"Starting training... (logging to {log_path})")
        print(f"Command: {' '.join(cmd[:10])}...")

        # Run with ray job submit for proper environment handling
        runtime_env = json.dumps({
            "env_vars": {"PYTHONPATH": config.pythonpath},
        })

        full_cmd = [
            "ray", "job", "submit",
            "--address=http://127.0.0.1:8265",
            f"--runtime-env-json={runtime_env}",
            "--"
        ] + cmd

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1
            )

            last_rollout_id = -1

            for line in process.stdout:
                # Write to log
                log_file.write(line)
                log_file.flush()

                # Also print to console
                print(line, end="")

                # Check for rollout completion to trigger post-processing
                if "rollout" in line.lower() and "completed" in line.lower():
                    # Try to extract rollout ID
                    try:
                        # Pattern: "rollout X completed" or similar
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                rollout_id = int(part)
                                if rollout_id > last_rollout_id:
                                    last_rollout_id = rollout_id
                                    process_rollout_samples(config, rollout_id)
                                    cleanup_old_checkpoints(config)
                                break
                    except (ValueError, IndexError):
                        pass

            process.wait()

            if process.returncode != 0:
                print(f"Training failed with return code {process.returncode}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        process.terminate()
        process.wait(timeout=10)
    finally:
        # Cleanup
        subprocess.run(["ray", "stop", "--force"], capture_output=True)


def main():
    config = parse_args()
    config = setup_paths(config)
    save_config(config)

    print(f"\n{'='*60}")
    print("Self-Play Optimization Training")
    print(f"{'='*60}")
    print(f"Model: {config.model}")
    print(f"Output: {config.output_dir}")
    print(f"GPUs: {config.num_gpus} ({config.num_gpus // config.gpus_per_sglang_engine} SGLang engines)")
    print(f"Backend: {config.train_backend}")
    print(f"Rollouts: {config.num_rollouts}")
    print(f"Batch: {config.rollout_batch_size} prompts × {config.n_samples_per_prompt} games/prompt")
    print(f"WandB: {'enabled' if config.use_wandb else 'disabled'}")
    print(f"{'='*60}\n")

    run_training(config)


if __name__ == "__main__":
    main()
