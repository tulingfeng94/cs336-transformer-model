from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id")

    sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        prompt_lens.append(len(prompt_ids))
        sequences.append(prompt_ids + output_ids)

    max_len = max(len(seq) for seq in sequences)
    seq_len = max_len - 1
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, (seq, prompt_len) in enumerate(zip(sequences, prompt_lens)):
        input_seq = seq[:-1]
        label_seq = seq[1:]
        n = len(input_seq)
        input_ids[i, :n] = torch.tensor(input_seq, dtype=torch.long)
        labels[i, :n] = torch.tensor(label_seq, dtype=torch.long)
        for j in range(n):
            if j >= prompt_len - 1:
                response_mask[i, j] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    batch_size, seq_len = input_ids.shape
    log_probs = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    token_entropy = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    logits = model(input_ids=input_ids).logits
    logits = logits.log_softmax(dim=-1)
    for i in range(seq_len):
        log_probs[:, i] = logits[torch.arange(batch_size), i, labels[:, i]]
        if return_token_entropy:
            token_entropy[:, i] = -torch.sum(logits[:, i].exp() * logits[:, i], dim=-1)
    return {
        "log_probs": log_probs,
        "token_entropy": token_entropy,
    }

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    raw_rewards = torch.zeros((rollout_batch_size,))
    metadata = {}
    reward_sum = 0
    format_reward_sum = 0
    for i in range(rollout_batch_size):
        reward = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        raw_rewards[i] = reward["reward"]
        reward_sum += reward["reward"]
        format_reward_sum += reward["format_reward"]
    metadata["mean_reward"] = reward_sum / rollout_batch_size
    metadata["mean_format_reward"] = format_reward_sum / rollout_batch_size

    return raw_rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    group_rewards = raw_rewards.reshape(-1, group_size)
    metadata: dict[str, float] = {}

    if baseline == "mean":
        group_rewards_baseline = group_rewards.mean(dim=-1, keepdim=True)
    elif baseline == "none":
        group_rewards_baseline = 0
    else:
        raise ValueError(f"Invalid baseline: {baseline}")

    advantage = group_rewards - group_rewards_baseline
    if advantage_normalizer == "std":
        advantage = advantage / (group_rewards.std(dim=-1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    elif advantage_normalizer == "mean":
        advantage = advantage / (group_rewards.mean(dim=-1, keepdim=True) + advantage_eps)
    else:
        raise ValueError(f"Invalid advantage normalizer: {advantage_normalizer}")

    metadata["mean_advantage"] = advantage.mean().item()
    metadata["std_advantage"] = advantage.std().item()
    metadata["min_advantage"] = advantage.min().item()
    metadata["max_advantage"] = advantage.max().item()
    return advantage.reshape(-1), metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantage = raw_rewards_or_advantages.reshape(-1, 1)
    batch_size, seq_len = policy_log_probs.shape
    loss = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    metadata = {}

    if importance_reweighting_method == "none":
        importance_weights = -advantage * policy_log_probs
    elif importance_reweighting_method == "noclip":
        ratios = torch.exp(policy_log_probs - old_log_probs)
        importance_weights = -ratios * advantage
    elif importance_reweighting_method == "grpo":
        ratios = torch.exp(policy_log_probs - old_log_probs)
        clipped = torch.clamp(ratios, 1 - cliprange, 1 + cliprange) * advantage
        unclipped = ratios * advantage
        importance_weights = -torch.minimum(clipped, unclipped)
    elif importance_reweighting_method == "gspo":
        t = policy_log_probs - old_log_probs
        mask = response_mask.to(dtype=t.dtype)
        delta = torch.sum(t * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(min=1e-8)
        ratios = torch.exp(delta.unsqueeze(-1))
        clipped = torch.clamp(ratios, 1 - cliprange, 1 + cliprange) * advantage
        unclipped = ratios * advantage
        importance_weights = -torch.minimum(clipped, unclipped)
        importance_weights = importance_weights.expand_as(policy_log_probs)
    else:
        raise ValueError(f"Invalid importance reweighting method: {importance_reweighting_method}")

    metadata["mean_importance_weights"] = importance_weights.mean().item()
    metadata["std_importance_weights"] = importance_weights.std().item()
    metadata["min_importance_weights"] = importance_weights.min().item()
    metadata["max_importance_weights"] = importance_weights.max().item()

    return importance_weights, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    if loss_normalization == "sequence":
        per_seq = torch.sum(per_token_policy_gradient_loss * mask, dim=-1) / torch.sum(mask, dim=-1)
        return per_seq.mean()
    elif loss_normalization == "constant":
        return torch.sum(per_token_policy_gradient_loss * mask) / normalization_constant
    else:
        raise ValueError(f"Invalid loss normalization: {loss_normalization}")

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    device = next(model.parameters()).device

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device=device, dtype=torch.float32)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    batch_size = input_ids.shape[0]
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by "
            f"gradient_accumulation_steps ({gradient_accumulation_steps})"
        )
    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    entropy_num = torch.tensor(0.0, device=device)
    entropy_den = torch.tensor(0.0, device=device)
    loss_metadata: dict[str, torch.Tensor | float] = {}

    for microbatch_idx in range(gradient_accumulation_steps):
        start = microbatch_idx * microbatch_size
        end = start + microbatch_size

        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_mask = response_mask[start:end]
        mb_advantages = advantages[start:end]
        mb_old_log_probs = (
            old_log_probs[start:end] if old_log_probs is not None else None
        )

        log_prob_out = get_response_log_probs(
            model=model,
            input_ids=mb_input_ids,
            labels=mb_labels,
            return_token_entropy=True,
        )
        policy_log_probs = log_prob_out["log_probs"]
        token_entropy = log_prob_out["token_entropy"]

        entropy_num = entropy_num + (token_entropy * mb_mask).sum()
        entropy_den = entropy_den + mb_mask.sum()

        per_token_loss, mb_loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_advantages,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_log_probs,
            cliprange=cliprange,
            response_mask=mb_mask,
        )
        loss_metadata.update(mb_loss_metadata)

        loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mb_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        # Sequence mean must be reweighted so accumulated grads match full-batch mean.
        # Constant normalization already uses a global divisor, so do not reweight.
        if loss_normalization == "sequence":
            scaled_loss = loss * (microbatch_size / batch_size)
        else:
            scaled_loss = loss

        scaled_loss.backward()
        total_loss = total_loss + scaled_loss.detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    optimizer.step()
    optimizer.zero_grad()

    metadata: dict[str, torch.Tensor | float] = {}
    metadata.update(reward_metadata)
    metadata.update(advantage_metadata)
    metadata.update(loss_metadata)
    metadata["loss"] = total_loss.detach()
    metadata["grad_norm"] = (
        float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
    )
    metadata["token_entropy"] = float(
        (entropy_num / entropy_den.clamp(min=1.0)).detach().cpu()
    )
    return total_loss, metadata