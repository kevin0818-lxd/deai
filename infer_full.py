#!/usr/bin/env python3
"""TinyLlama-1.1B-Chat 全参微调权重（output/adapters.safetensors）的推理入口。

与 mlx-lm 自带 generate 的区别，只有两点，其余保持一致：
  1. 打开权重前先核对 checkpoint 与模型逐张量相等，避免 strict=False 静默丢弃；
  2. 按训练时的提示格式（"Humanize this English paragraph:" + 段落）逐段改写。
"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

PROMPT_PREFIX = "Humanize this English paragraph:\n\n"
CONTEXT_LIMIT = 2048
MAX_PROMPT_TOKENS = 1536
CHECKPOINT = "adapters.safetensors"


def load_full(model_dir, adapter_dir):
    """加载基础模型 + 全参权重，并证明权重真的落进了模型。"""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise SystemExit(f"缺少 {config_path}；先完成训练再推理")
    fine_tune_type = json.loads(config_path.read_text(encoding="utf-8")).get(
        "fine_tune_type"
    )
    if fine_tune_type != "full":
        raise SystemExit(
            f"adapter_config.json 的 fine_tune_type 是 {fine_tune_type!r}，"
            "不是 full；这不是全参权重，拒绝加载"
        )
    if not (adapter_dir / CHECKPOINT).is_file():
        raise SystemExit(f"缺少全参权重：{adapter_dir / CHECKPOINT}")

    model, tokenizer = load(str(model_dir), adapter_path=str(adapter_dir))
    params = dict(tree_flatten(model.parameters()))
    lora_keys = [k for k in params if "lora_" in k]
    if lora_keys:
        raise SystemExit(f"模型里出现了 LoRA 层（{lora_keys[:3]}…），加载路径不对")

    checkpoint = mx.load(str(adapter_dir / CHECKPOINT))
    missing = sorted(k for k in checkpoint if k not in params)
    if missing:
        raise SystemExit(f"{len(missing)} 个权重键在模型里不存在，例如 {missing[:3]}")
    mismatched = [k for k, v in checkpoint.items() if not mx.array_equal(params[k], v)]
    if mismatched:
        raise SystemExit(f"{len(mismatched)} 个权重与模型不一致，例如 {mismatched[:3]}")
    print(
        f"全参权重核对通过：{len(checkpoint)} 个张量逐一相等，"
        f"覆盖 {sum(v.size for v in checkpoint.values()) / 1e6:.1f}M 参数",
        flush=True,
    )
    del checkpoint
    mx.clear_cache()
    return model, tokenizer


def split_blocks(text):
    """按空行切段落，保留段落之间的空行，便于原样拼回。"""
    parts = re.split(r"(\n[ \t]*\n)", text)
    blocks = []
    for index in range(0, len(parts), 2):
        body = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if body.strip():
            blocks.append((body, separator))
    return blocks


def rewrite(model, tokenizer, prompt_ids, sampler, max_tokens, out_stream):
    started = time.time()
    text = ""
    tokens = 0
    stats = None
    for response in stream_generate(
        model, tokenizer, mx.array(prompt_ids), max_tokens=max_tokens, sampler=sampler
    ):
        out_stream.write(response.text)
        out_stream.flush()
        text += response.text
        tokens = response.generation_tokens
        stats = response
    elapsed = time.time() - started
    return text, tokens, elapsed, stats


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="待改写的英文纯文本")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=root / "model")
    parser.add_argument("--adapter-path", type=Path, default=root / "output")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tokens", type=int, default=0, help="单段上限；0 表示用满 2048 上下文"
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"找不到输入文件：{args.input}")
    out = args.out or args.input.with_name(
        f"{args.input.stem}.humanized.tinyllama-full.txt"
    )
    partial = Path(f"{out}.partial.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    model, tokenizer = load_full(args.model, args.adapter_path)

    blocks = split_blocks(args.input.read_text(encoding="utf-8"))
    if not blocks:
        parser.error("输入文件没有可改写的正文")

    print(
        f"温度 {args.temperature} / top-p {args.top_p} / 种子 {args.seed}；"
        f"共 {len(blocks)} 段；输出 {out}",
        flush=True,
    )
    pieces = []
    with partial.open("w", encoding="utf-8") as stream:
        for number, (body, separator) in enumerate(blocks, 1):
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT_PREFIX + body.strip()}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            budget = args.max_tokens or CONTEXT_LIMIT - len(prompt_ids)
            if len(prompt_ids) > MAX_PROMPT_TOKENS or budget < 16:
                raise SystemExit(
                    f"第 {number} 段提示词 {len(prompt_ids)} tokens，超出可用预算；"
                    "请把该段再拆短，不要靠截断"
                )
            print(
                f"[{number}/{len(blocks)}] 提示词 {len(prompt_ids)} tokens，"
                f"可生成 {budget} tokens",
                flush=True,
            )
            text, tokens, elapsed, stats = rewrite(
                model, tokenizer, prompt_ids, sampler, budget, sys.stdout
            )
            peak = f"{stats.peak_memory:.2f} GB" if stats is not None else "未知"
            print(
                f"\n[{number}/{len(blocks)}] {tokens} tokens，"
                f"{tokens / max(elapsed, 1e-6):.1f} tok/s，{elapsed:.1f}s，"
                f"峰值内存 {peak}",
                flush=True,
            )
            if tokens >= budget:
                print(
                    f"注意：第 {number} 段生成到上限 {budget} tokens，结尾可能被截断",
                    flush=True,
                )
            pieces.append((text.strip(), separator))
            stream.write(text.strip() + separator)
            stream.flush()

    out.write_text("".join(text + separator for text, separator in pieces), encoding="utf-8")
    partial.unlink()
    print(f"完成：{out}")


if __name__ == "__main__":
    main()
