#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local style-adaptation pipeline: augment -> train -> parse -> generate -> finish."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer
import yaml

from dependency_perturb import load_parser, perturb_text
from infer_full import CONTEXT_LIMIT, load_full
from output_control import control_errors
from style_data import prepare, write_data
from text_pipeline import build_plan, prompt_token_ids

ROOT = Path(__file__).resolve().parent


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=Path, help="训练模式：有权使用的 UTF-8 英文风格文本")
    parser.add_argument("--input", type=Path, required=True, help="待处理 UTF-8 英文文本")
    parser.add_argument("--run-dir", type=Path, required=True, help="新的独立训练运行目录")
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="模板模式：已有微调权重目录，跳过数据增强与训练")
    parser.add_argument("--steps", type=int, default=None, help="训练步数，默认按样本数调整")
    parser.add_argument("--spelling-rate", type=float, default=0.0)
    parser.add_argument("--grammar-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="只生成数据并预检，不训练")
    return parser.parse_args(argv)


def _config(steps, seed):
    config = yaml.safe_load((ROOT / "training_full.yaml").read_text(encoding="utf-8"))
    config["iters"] = steps
    config["save_every"] = steps
    config["steps_per_eval"] = max(10, steps // 2)
    config["lr_schedule"]["arguments"][1] = max(1, steps - 1)
    config["seed"] = seed
    return config


def _generate(text, model, tokenizer, nlp, seed, spelling, grammar):
    changed = perturb_text(text, nlp)
    plan = build_plan(tokenizer, changed)
    sampler = make_sampler(temp=0.9, top_p=0.95)
    mx.random.seed(seed)
    pieces = []
    for index, (unit, separator) in enumerate(plan, 1):
        ids = prompt_token_ids(tokenizer, unit)
        budget = CONTEXT_LIMIT - len(ids)
        if budget < 16:
            raise ValueError(f"第 {index} 段没有足够生成预算")
        output = []
        tokens = 0
        for response in stream_generate(
            model, tokenizer, mx.array(ids), max_tokens=budget, sampler=sampler
        ):
            output.append(response.text)
            tokens = response.generation_tokens
        if tokens >= budget:
            raise RuntimeError(f"第 {index} 段触及上下文上限；拒绝交付可能截断的结果")
        pieces.append("".join(output).strip() + separator)
    return control_errors("".join(pieces), nlp, spelling, grammar, seed)


def run(argv=None):
    args = _arguments(argv)
    if bool(args.style) == bool(args.adapter_path):
        raise ValueError("请选择 --style（重新训练）或 --adapter-path（模板推理），不可同时使用")
    if args.prepare_only and args.adapter_path:
        raise ValueError("--prepare-only 仅适用于 --style 训练模式")
    if args.steps is not None and args.adapter_path:
        raise ValueError("--steps 仅适用于 --style 训练模式")
    if args.steps is not None and not 1 <= args.steps <= 720:
        raise ValueError("--steps 须介于 1 与 720")
    if not (0 <= args.spelling_rate <= 0.05 and 0 <= args.grammar_rate <= 0.05):
        raise ValueError("错误率须介于 0 与 0.05")
    if args.run_dir.exists():
        raise ValueError("运行目录已存在，请指定新的 --run-dir，避免覆盖旧权重或用户数据")
    if not args.model.joinpath("config.json").is_file():
        raise ValueError("找不到本地 TinyLlama 模型目录")
    if args.style and not args.style.is_file():
        raise ValueError("--style 必须是现有文件")
    if not args.input.is_file():
        raise ValueError("--input 必须是现有文件")
    source = args.input.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError("待处理文本为空")
    nlp = load_parser()
    if args.adapter_path:
        if not (args.adapter_path / "adapter_config.json").is_file() or not (
            args.adapter_path / "adapters.safetensors"
        ).is_file():
            raise ValueError("模板目录缺少 adapter_config.json 或 adapters.safetensors")
        args.run_dir.mkdir(parents=True, exist_ok=False)
        model, tokenizer = load_full(args.model, args.adapter_path)
        output = _generate(
            source, model, tokenizer, nlp, args.seed,
            args.spelling_rate, args.grammar_rate
        )
        result_path = args.run_dir / "result.txt"
        result_path.write_text(output, encoding="utf-8")
        print(f"完成：{result_path}", flush=True)
        return
    target = args.style.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True
    )
    if not tokenizer.chat_template:
        raise ValueError("本地模型缺少 chat 模板")
    pairs, report = prepare(target, nlp, tokenizer, args.seed)
    steps = args.steps or min(200, max(20, len(pairs["train"]) * 2))
    # Validate the inference budget before spending time on training.
    build_plan(tokenizer, perturb_text(source, nlp))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_data(pairs, args.run_dir / "data")
    (args.run_dir / "training.yaml").write_text(
        yaml.safe_dump(_config(steps, args.seed), allow_unicode=True), encoding="utf-8"
    )
    (args.run_dir / "report.json").write_text(
        json.dumps({**report, "steps": steps}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"样本预检通过：{report['train_pairs']} 训练对、"
          f"{report['valid_pairs']} 验证对；计划 {steps} 步", flush=True)
    if args.prepare_only:
        print(f"数据已写入 {args.run_dir / 'data'}；未启动训练", flush=True)
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([
        sys.executable, str(ROOT / "train_full.py"), "--model", str(args.model),
        "--data", str(args.run_dir / "data"),
        "--adapter-path", str(args.run_dir / "weights"),
        "--config", str(args.run_dir / "training.yaml"),
    ], check=True, env=env)
    model, tokenizer = load_full(args.model, args.run_dir / "weights")
    output = _generate(
        source, model, tokenizer, nlp, args.seed,
        args.spelling_rate, args.grammar_rate
    )
    result_path = args.run_dir / "result.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"完成：{result_path}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"管道失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
