#!/usr/bin/env python3
"""全参微调 TinyLlama-1.1B-Chat：BF16、损失非有限即停、可训练参数漂移即停。

用法由同目录 train.zsh 驱动，正常不需要手工调用。
"""
import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import lora
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.utils import get_total_parameters
import yaml

EXPECTED_LAYERS = 22
EXPECTED_TRAINABLE_MILLIONS = 969.0
TRAINABLE_TOLERANCE_MILLIONS = 1.0
MIN_TRAINABLE_FRACTION = 0.879
MAX_TRAINABLE_FRACTION = 0.882
EMBEDDING_KEY = "model.embed_tokens.weight"


class FiniteLossCallback(TrainingCallback):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def on_train_loss_report(self, info):
        if not math.isfinite(float(info["train_loss"])):
            raise RuntimeError("训练损失非有限；已停止，不能使用本次检查点")
        if self.wrapped:
            self.wrapped.on_train_loss_report(info)

    def on_val_loss_report(self, info):
        if not math.isfinite(float(info["val_loss"])):
            raise RuntimeError("验证损失非有限；已停止，不能使用本次检查点")
        if self.wrapped:
            self.wrapped.on_val_loss_report(info)


def check_lengths(model_dir, data_dir, limit):
    """拒绝任何会被 max_seq_length 截断的样本；静默截断会毁掉整轮训练。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True
    )
    if not tokenizer.chat_template:
        raise SystemExit("模型没有对话模板，无法套用与推理一致的提示格式")
    longest = (0, "无")
    for name in ("train", "valid"):
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                encoded = tokenizer.apply_chat_template(
                    json.loads(line)["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                tokens = (encoded["input_ids"] if isinstance(encoded, Mapping)
                          else encoded)
                if len(tokens) > limit:
                    raise SystemExit(
                        f"{name}.jsonl 第 {index} 行有 {len(tokens)} tokens，"
                        f"超过 max_seq_length={limit}；拒绝截断训练"
                    )
                if len(tokens) > longest[0]:
                    longest = (len(tokens), f"{name}.jsonl 第 {index} 行")
    print(f"数据长度检查通过：最长 {longest[0]} tokens（{longest[1]}），"
          f"上限 {limit}", flush=True)


def run():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    known, _ = parser.parse_known_args()
    if (known.adapter_path / "adapters.safetensors").exists():
        parser.error(f"已有全参权重，拒绝覆盖：{known.adapter_path}")
    if "--resume-adapter-file" in sys.argv:
        parser.error("本次为首次全参微调，不得恢复旧 LoRA 或 27B 检查点")
    config = json.loads((known.model / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_type") != "llama"
            or config.get("num_hidden_layers") != EXPECTED_LAYERS
            or config.get("vocab_size") != 32000
            or config.get("hidden_size") != 2048):
        parser.error("需要未量化的 TinyLlama-1.1B-Chat 权重目录")
    if not (known.data / "train.jsonl").is_file() or not (
        known.data / "valid.jsonl"
    ).is_file():
        parser.error("数据目录缺少 train.jsonl 或 valid.jsonl")
    check_lengths(
        known.model,
        known.data,
        yaml.safe_load(known.config.read_text(encoding="utf-8"))["max_seq_length"],
    )

    original_load = lora.load
    original_report = lora.print_trainable_parameters
    original_callbacks = lora.get_reporting_callbacks

    def load_bf16(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)
        layers = len(model.layers)
        if layers != EXPECTED_LAYERS:
            raise RuntimeError(f"模型有 {layers} 层，与预期的 {EXPECTED_LAYERS} 层不符")
        if dict(tree_flatten(model.parameters()))[EMBEDDING_KEY].dtype != mx.bfloat16:
            model.set_dtype(mx.bfloat16)
            mx.eval(model.parameters())
            print("基础模型参数已在内存中转为 BF16", flush=True)
        else:
            print("基础模型参数已是 BF16", flush=True)
        mx.clear_cache()
        return model, tokenizer

    def checked_report(model):
        """mlx-lm 已完成 freeze/unfreeze，这里复核真正被放开的参数范围。"""
        total = get_total_parameters(model) / 1e6
        trainable = sum(
            v.size for _, v in tree_flatten(model.trainable_parameters())
        ) / 1e6
        fraction = trainable / total
        if (abs(trainable - EXPECTED_TRAINABLE_MILLIONS) > TRAINABLE_TOLERANCE_MILLIONS
                or not MIN_TRAINABLE_FRACTION < fraction < MAX_TRAINABLE_FRACTION):
            raise RuntimeError(
                f"可训练参数 {trainable:.1f}M/{total:.1f}M（{fraction:.2%}）与预期的"
                f" {EXPECTED_TRAINABLE_MILLIONS:.1f}M 不符；"
                "fine_tune_type: full 或 num_layers 未按全参微调生效，已停止"
            )
        original_report(model)

    def callbacks(*args, **kwargs):
        return FiniteLossCallback(original_callbacks(*args, **kwargs))

    lora.load = load_bf16
    lora.print_trainable_parameters = checked_report
    lora.get_reporting_callbacks = callbacks
    lora.main()


if __name__ == "__main__":
    run()
