# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real post-train guard without loading Ray or Megatron."""

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.engine.sft.runtime import is_sft_mode


@pytest.mark.parametrize(
    ("loss_type", "tags", "predict_interval", "has_backuper", "expected_calls"),
    [
        ("sft", ["actor"], None, True, 0),
        ("policy_loss", ["actor"], None, True, 1),
        ("sft", ["actor", "ref"], None, True, 1),
        ("sft", ["actor"], 10, True, 1),
        ("sft", ["actor"], None, False, 0),
    ],
    ids=["plain_sft", "rl", "multiple_roles", "sft_prediction", "no_backuper"],
)
def test_post_train_actor_backup(loss_type, tags, predict_interval, has_backuper, expected_calls):
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    tree = ast.parse(path.read_text())
    train_actor = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_actor"
    )
    backup_call = ast.dump(ast.parse('self.weights_backuper.backup("actor")', mode="eval").body)
    guard = next(
        node
        for node in train_actor.body
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Expr) and ast.dump(child.value) == backup_call for child in node.body)
    )
    backuper = Mock(backup_tags=tags)
    actor = SimpleNamespace(args=Namespace(loss_type=loss_type, sft_predict_interval=predict_interval))
    if has_backuper:
        actor.weights_backuper = backuper
    exec(
        compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec"),
        {"self": actor, "is_sft_mode": is_sft_mode},
    )
    assert backuper.backup.call_count == expected_calls
