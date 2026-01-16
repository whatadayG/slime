"""
SLIME-compatible rollout function for self-play optimization game.

Usage:
    python train.py \
        --custom-generate-function-path examples.self_play_optimization.rollout.generate_with_self_play \
        --custom-config-path examples/self_play_optimization/config.yaml \
        ...
"""

import random
from copy import deepcopy

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

# Import the game environment (requires PYTHONPATH setup)
# export PYTHONPATH="${PYTHONPATH}:/path/to/self_play/scripts"
from dialop.envs.optimization import OptimizationEnv

from .agent_system import run_self_play_game


# Default config - can be overridden via --custom-config-path
SELF_PLAY_DEFAULTS = {
    "max_turns": 30,
    "max_retries_per_turn": 8,
    "error_penalty": 0.0,  # Penalty per error (subtracted from reward)
}


async def generate_with_self_play(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[Sample]:
    """
    SLIME-compatible custom generate function for two-player self-play.

    This function is called by SLIME's rollout system for each prompt (Sample).
    It runs a full game and returns multiple Samples (one per player turn).

    Args:
        args: SLIME args object with model/rollout config
        sample: Input sample (contains prompt with game_state if using custom data source)
        sampling_params: Dict with temperature, max_new_tokens, etc.
        evaluation: Whether this is an eval run (affects context length)

    Returns:
        List of Sample objects from both players for training
    """
    # Set up tokenizer and context
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    # Attach to args for use by generate_response
    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer
    args.apply_chat_template = True  # Required for dialogue

    # Load custom config if provided
    config = SELF_PLAY_DEFAULTS.copy()
    for key in config:
        if hasattr(args, key):
            config[key] = getattr(args, key)

    # Create environment
    env = OptimizationEnv(
        max_turns=config["max_turns"],
        max_retries_per_turn=config["max_retries_per_turn"],
        error_penalty=config["error_penalty"],
    )

    # Run the self-play game
    samples = await run_self_play_game(
        args=args,
        sample=sample,
        env=env,
        max_turns=config["max_turns"],
        max_retries_per_turn=config["max_retries_per_turn"],
    )

    # Shuffle samples (as done in multi_agent example)
    random.shuffle(samples)

    return samples


# Alternative: Full rollout function replacement (more control)
async def generate_rollout(
    args,
    rollout_id: int,
    *,
    evaluation: bool = False,
) -> list[list[Sample]]:
    """
    Full rollout function replacement.

    Use this with --rollout-function-path if you need complete control
    over the rollout process (e.g., custom data source, batching).

    Returns:
        list[list[Sample]]: Grouped samples (outer list = groups, inner = samples per group)
    """
    # This is a more advanced pattern - usually you just need generate_with_self_play
    raise NotImplementedError(
        "Use --custom-generate-function-path instead for simpler integration. "
        "Implement this only if you need custom data source or batching logic."
    )
