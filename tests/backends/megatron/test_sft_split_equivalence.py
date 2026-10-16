# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU coverage of packed SFT splitting, using the real slicer and CE."""

import ast
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from relax.engine.sft.runtime import is_sft_mode


@pytest.fixture(scope="module")
def helpers():
    # Load only the actual definitions: no Ray, Megatron, or CUDA initialization.
    root = Path(__file__).resolve().parents[3]
    namespace = dict(
        torch=torch,
        dist=dist,
        F=F,
        Namespace=Namespace,
        Iterator=Iterator,
        is_sft_mode=is_sft_mode,
        mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1),
    )
    for relative_path, names in (
        ("relax/backends/megatron/loss.py", {"get_responses"}),
        ("relax/backends/megatron/cp_utils.py", {"get_logits_and_tokens_offset_with_cp"}),
        (
            "relax/utils/training/ppo_utils.py",
            {
                "_maybe_all_reduce",
                "_get_vocab_parallel_rank_size",
                "_VocabParallelLogProbEntropy",
                "_calculate_log_probs_and_entropy_chunk",
                "calculate_log_probs_and_entropy",
            },
        ),
    ):
        path = root / relative_path
        definitions = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_responses"], namespace["calculate_log_probs_and_entropy"]


def _args(**overrides):
    return Namespace(
        **{
            "loss_type": "sft",
            "qkv_format": "thd",
            "sft_chunked_logits": False,
            "rollout_temperature": 1.0,
            "allgather_cp": False,
            **overrides,
        }
    )


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_split_loss_and_gradient_match_independent_slices(helpers, temperature):
    get_responses, log_probs = helpers
    tokens = [torch.tensor([1, 2, 3, 4]), torch.tensor([5]), torch.tensor([2, 1, 3])]
    masks = [torch.tensor([0, 1, 1, 0]), torch.tensor([0]), torch.tensor([1, 0, 0])]
    source = torch.randn(1, 10, 7, generator=torch.Generator().manual_seed(17))  # 8 tokens + 2 padding
    results = []
    for use_split in (True, False):
        logits = source.clone().requires_grad_()
        if use_split:
            pairs = get_responses(
                logits,
                args=_args(rollout_temperature=temperature),
                unconcat_tokens=tokens,
                total_lengths=[4, 1, 3],
                response_lengths=[4, 1, 3],
            )
        else:
            # Explicit old slices and hand-written next-token labels form the reference.
            pairs = [
                (logits[0, start:end] / temperature, torch.tensor(labels))
                for start, end, labels in [(0, 4, [2, 3, 4, 0]), (4, 5, [0]), (5, 8, [1, 3, 0])]
            ]
        loss = sum(
            -(log_probs(chunk, labels, None, with_entropy=False)[0].squeeze(-1) * mask).sum()
            for (chunk, labels), mask in zip(pairs, masks, strict=True)
        )
        loss.backward()
        assert torch.equal(logits.detach(), source)
        results.append((loss.detach(), logits.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_one_split_node_keeps_empty_samples_and_unscaled_labels_aligned(helpers):
    get_responses, _ = helpers
    logits = torch.arange(42, dtype=torch.float32).reshape(1, 6, 7).requires_grad_()
    pairs = list(
        get_responses(
            logits,
            args=_args(rollout_temperature=0.7),
            unconcat_tokens=[torch.tensor([], dtype=torch.long), torch.tensor([0, 1, 2]), torch.tensor([3])],
            total_lengths=[0, 3, 1],
            response_lengths=[0, 3, 1],
            apply_temperature=False,
        )
    )
    node = pairs[0][0].grad_fn
    assert type(node).__name__.startswith("SplitWithSizesBackward")
    assert all(chunk.grad_fn is node for chunk, _ in pairs)
    assert [labels.tolist() for _, labels in pairs] == [[], [1, 2, 0], [0]]
    assert torch.equal(pairs[1][0], logits[0, :3])
    sum(chunk.sum() for chunk, _ in pairs).backward()
    expected_gradient = torch.ones_like(logits)
    expected_gradient[:, 4:] = 0
    assert torch.equal(logits.grad, expected_gradient)


@pytest.mark.parametrize("mode", ["rl", "bshd", "cp2", "chunked", "seq_cls"])
def test_supported_fallbacks_keep_their_token_coordinates(helpers, mode):
    get_responses, _ = helpers
    args = _args(rollout_temperature=0.7)
    lengths, responses = [4, 4], [4, 4]
    rows, shape = [[0, 1, 2, 3], [4, 5, 6, 7]], (1, 8, 8)
    labels = [[1, 2, 3, 0], [1, 2, 3, 0]]
    kwargs = {}
    if mode == "rl":
        args.loss_type = "policy_loss"
        responses, rows, labels = [2, 2], [[1, 2], [5, 6]], [[2, 3], [2, 3]]
    elif mode == "bshd":
        args.qkv_format = "bshd"
        shape, rows, kwargs = (2, 6, 8), [[0, 1, 2, 3], [6, 7, 8, 9]], {"max_seq_lens": [6, 6]}
    elif mode == "cp2":
        lengths, responses, rows, labels, shape = [8], [8], [[0, 1, 2]], [[1, 2, 7]], (1, 4, 8)
        kwargs = {"dynamic_cp_size": 2, "dynamic_cp_rank": 0}
    elif mode == "chunked":
        args.sft_chunked_logits = True
    elif mode == "seq_cls":
        args.task_type = "seq_cls"
        responses, rows, labels = [1, 1], [[2], [6]], [[3], [3]]
        kwargs = {"apply_temperature": False}
    dtype = torch.bfloat16 if mode == "chunked" else torch.float32
    source = torch.arange(shape[0] * shape[1] * shape[2], dtype=dtype).reshape(shape)
    logits = source.clone().requires_grad_()
    pairs = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=[torch.arange(length) for length in lengths],
            total_lengths=lengths,
            response_lengths=responses,
            **kwargs,
        )
    )
    scale = 1.0 if mode in ("chunked", "seq_cls") else 0.7
    for (chunk, target), indices, expected_labels in zip(pairs, rows, labels, strict=True):
        assert torch.equal(chunk, source.reshape(-1, shape[-1])[indices] / scale)
        assert target.tolist() == expected_labels
    sum(chunk.float().sum() for chunk, _ in pairs).backward()
    expected_gradient = torch.zeros_like(logits).reshape(-1, shape[-1])
    expected_gradient[[index for indices in rows for index in indices]] = 1 / scale
    assert torch.equal(logits.grad.reshape(-1, shape[-1]), expected_gradient)
