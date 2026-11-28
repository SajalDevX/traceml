# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from traceml_ai.utils.ast_analysis import analyze_script, build_code_manifest


def test_hf_step_intervals_keep_fractional_ratios(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import TrainingArguments\n"
        "args = TrainingArguments(\n"
        "    output_dir='out',\n"
        "    logging_steps=0.01,\n"
        "    save_steps=0.25,\n"
        "    eval_steps=500,\n"
        ")\n",
        encoding="utf-8",
    )

    hf = build_code_manifest(analyze_script(str(script)))["huggingface"]

    assert hf["training_arguments_detected"] is True
    assert hf["logging_steps"] == 0.01
    assert hf["save_steps"] == 0.25
    assert hf["eval_steps"] == 500
    assert isinstance(hf["eval_steps"], int)


def test_cudnn_benchmark_disabled_is_not_reported_as_set(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\ntorch.backends.cudnn.benchmark = False\n",
        encoding="utf-8",
    )

    manifest = build_code_manifest(analyze_script(str(script)))

    assert manifest["precision"]["cudnn_benchmark"] is False


def test_cudnn_benchmark_enabled_is_reported(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\ntorch.backends.cudnn.benchmark = True\n",
        encoding="utf-8",
    )

    manifest = build_code_manifest(analyze_script(str(script)))

    assert manifest["precision"]["cudnn_benchmark"] is True


def test_method_calls_on_expressions_are_detected(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "def train(model, loader, optimizer, scaler, device):\n"
        "    for batch in loader:\n"
        "        x = batch['x'].to(device, non_blocking=True)\n"
        "        loss = model(x).pow(2).mean()\n"
        "        scaler.scale(loss).backward()\n"
        "        scaler.step(optimizer)\n"
        "        print(loss.detach().item())\n"
        "        preds = model(x).argmax(1).cpu().numpy()\n",
        encoding="utf-8",
    )

    manifest = build_code_manifest(analyze_script(str(script)))

    assert manifest["train_loop"]["backward_detected"] is True
    assert manifest["device_transfer"] == {
        "to_device_detected": True,
        "non_blocking_used": True,
    }
    sync = manifest["sync_calls_in_train_loop"]
    assert sync["item_calls"] == 1
    assert sync["cpu_calls"] == 1
    assert sync["numpy_calls"] == 1
