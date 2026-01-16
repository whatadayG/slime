"""
Custom reward post-processing for self-play games.

In self-play, all samples from the same game should be grouped together
for GRPO normalization, regardless of --n-samples-per-prompt.

The grouping is based on `sample.group_index` which is set in agent_system.py.
"""

import torch
from slime.utils.types import Sample


def post_process_rewards_by_group_index(args, samples: list[Sample], **kwargs):
    """
    GRPO-style reward normalization using group_index instead of n_samples_per_prompt.

    This is needed for self-play because:
    - We use --n-samples-per-prompt 1 (one game per prompt)
    - But each game produces multiple samples (one per player turn)
    - All samples from the same game share the same group_index and should be normalized together

    Args:
        args: SLIME args
        samples: List of samples from rollout

    Returns:
        Tuple of (normalized_rewards, raw_rewards) - same format as default
    """
    # Group samples by group_index
    groups = {}
    for i, sample in enumerate(samples):
        gidx = sample.group_index
        if gidx is None:
            # Fallback: treat each sample as its own group (no normalization)
            gidx = i
        if gidx not in groups:
            groups[gidx] = []
        groups[gidx].append((i, sample))

    # Get raw rewards
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    # Normalize within groups
    normalized_rewards = [0.0] * len(samples)

    for gidx, group_samples in groups.items():
        indices = [idx for idx, _ in group_samples]
        group_rewards = torch.tensor([raw_rewards[idx] for idx in indices], dtype=torch.float32)

        # GRPO normalization: subtract mean
        if len(group_rewards) > 1:
            mean = group_rewards.mean()
            std = group_rewards.std()
            # Optionally normalize by std (SLIME's default doesn't, but some implementations do)
            # if std > 1e-8:
            #     group_rewards = (group_rewards - mean) / std
            # else:
            group_rewards = group_rewards - mean
        else:
            # Single sample in group: reward = 0 (no signal)
            group_rewards = torch.zeros_like(group_rewards)

        # Assign back
        for i, idx in enumerate(indices):
            normalized_rewards[idx] = group_rewards[i].item()
            samples[idx].reward = normalized_rewards[idx]

    return normalized_rewards, raw_rewards


def post_process_rewards_preserve_raw(args, samples: list[Sample], **kwargs):
    """
    Alternative: Skip GRPO normalization entirely, use raw rewards.

    Use this if you want to:
    - Use a different advantage estimator
    - Have already normalized rewards in the environment
    - Debug reward values
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    return raw_rewards, raw_rewards
