# SPDX-License-Identifier: Apache-2.0
"""Bounded, lossless paragraph packing for local TinyLlama inference."""

from collections.abc import Mapping
import re

from infer_full import PROMPT_PREFIX, split_blocks

SEGMENT_PROMPT_LIMIT = 900
CHAR_BUDGET = 3_200


def sentence_split(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def prompt_token_ids(tokenizer, body):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT_PREFIX + body}],
        tokenize=True, add_generation_prompt=True,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return list(ids)


def split_to_fit(tokenizer, text, limit):
    if len(prompt_token_ids(tokenizer, text)) <= limit:
        return [text]
    sentences = sentence_split(text)
    if len(sentences) >= 2:
        half = len(sentences) // 2
        left, right = " ".join(sentences[:half]), " ".join(sentences[half:])
    else:
        words = text.split()
        if len(words) < 2:
            raise ValueError("存在无法在不截断前提下切分的片段")
        half = len(words) // 2
        left, right = " ".join(words[:half]), " ".join(words[half:])
    return split_to_fit(tokenizer, left, limit) + split_to_fit(tokenizer, right, limit)


def pack_units(tokenizer, text):
    groups, buffer, size = [], [], 0
    for sentence in sentence_split(text):
        if buffer and size + len(sentence) > CHAR_BUDGET:
            groups.append(" ".join(buffer))
            buffer, size = [], 0
        buffer.append(sentence)
        size += len(sentence) + 1
    if buffer:
        groups.append(" ".join(buffer))
    units = []
    for group in groups or [text]:
        units.extend(split_to_fit(tokenizer, group, SEGMENT_PROMPT_LIMIT))
    return units


def build_plan(tokenizer, text):
    """Yield (unit, separator), retaining blank lines between paragraphs."""
    plan = []
    for body, separator in split_blocks(text):
        units = pack_units(tokenizer, body.strip())
        for index, unit in enumerate(units):
            plan.append((unit, separator if index == len(units) - 1 else " "))
    return plan
