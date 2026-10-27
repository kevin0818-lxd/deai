# SPDX-License-Identifier: Apache-2.0
"""Build synthetic source -> target-style chat pairs from licensed English prose.

Only the uploaded text is used as the target. Synthetic sources preserve the
original words wherever possible; a low transformation rate is a hard error.
"""

import hashlib
import json
import re
from pathlib import Path

from dependency_perturb import perturb_text
from infer_full import PROMPT_PREFIX, split_blocks

MIN_UNITS = 12
MIN_CHANGED_FRACTION = 0.15
MAX_WORDS_PER_UNIT = 120
MAX_CHAT_TOKENS = 2048

# Mild register normalization, not a claim that these are meaning-preserving
# paraphrases in every context. Ambiguous terms and proper nouns are excluded.
EXPANSIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "it's": "it is", "they're": "they are", "we're": "we are",
}
REGISTER = {
    "moreover": "also", "furthermore": "also", "nevertheless": "however",
    "approximately": "about", "numerous": "many", "utilize": "use",
    "subsequently": "later", "commence": "begin", "terminate": "end",
    "regarding": "about", "individuals": "people", "purchase": "buy",
}


def _replace_known(text, mapping):
    changed = False

    def replace(match):
        nonlocal changed
        word = match.group()
        replacement = mapping[word.lower()]
        changed = True
        return replacement.capitalize() if word[0].isupper() else replacement

    pattern = re.compile(
        r"(?<![A-Za-z])(?:"
        + "|".join(re.escape(word) for word in sorted(mapping, key=len, reverse=True))
        + r")(?![A-Za-z])", re.IGNORECASE
    )
    return pattern.sub(replace, text), changed


def synthetic_sources(target, nlp):
    """Return distinct, deterministic source variants with the same facts."""
    variants = []
    normalized, expanded = _replace_known(target, EXPANSIONS)
    normalized, registered = _replace_known(normalized, REGISTER)
    if expanded or registered:
        variants.append(normalized)
    reordered = perturb_text(target, nlp)
    if reordered != target and reordered not in variants:
        variants.append(reordered)
    if (expanded or registered) and reordered != target:
        combined, _ = _replace_known(reordered, EXPANSIONS)
        combined, _ = _replace_known(combined, REGISTER)
        if combined not in variants:
            variants.append(combined)
    return [variant for variant in variants if variant != target]


def weak_source(target):
    """A fact-preserving but weak baseline: the target inside a verbose frame."""
    return "The following passage contains the information to be conveyed:\n" + target


def _units(text, nlp):
    """Split into small units, retaining paragraph boundaries as split groups."""
    groups = []
    for body, _ in split_blocks(text):
        sentences = [sentence.text.strip() for sentence in nlp(body).sents]
        current, size = [], 0
        for sentence in sentences:
            words = len(sentence.split())
            if words > MAX_WORDS_PER_UNIT:
                raise ValueError("存在超过 120 词的单句；请先人工拆句")
            if current and size + words > MAX_WORDS_PER_UNIT:
                groups.append(" ".join(current))
                current, size = [], 0
            current.append(sentence)
            size += words
        if current:
            groups.append(" ".join(current))
    # Stable deduplication prevents copies of the same text crossing the split.
    return list(dict.fromkeys(groups))


def _chat(source, target):
    return {"messages": [
        {"role": "user", "content": PROMPT_PREFIX + source},
        {"role": "assistant", "content": target},
    ]}


def _length(tokenizer, example):
    ids = tokenizer.apply_chat_template(
        example["messages"], tokenize=True, add_generation_prompt=False
    )
    return len(ids["input_ids"] if isinstance(ids, dict) else ids)


def prepare(text, nlp, tokenizer, seed=42):
    """Create disjoint train/validation pairs and a non-sensitive report."""
    if not text.strip():
        raise ValueError("目标风格英文文本为空")
    units = _units(text, nlp)
    if len(units) < MIN_UNITS:
        raise ValueError(f"目标文本需至少 {MIN_UNITS} 个不同训练单元，当前 {len(units)} 个")
    # Split before augmenting, keyed by target content, not by variant.
    ordered = sorted(units, key=lambda x: hashlib.sha256(
        f"{seed}:{x}".encode("utf-8")).digest())
    valid_count = max(2, len(ordered) // 10)
    splits = {"valid": ordered[:valid_count], "train": ordered[valid_count:]}
    pairs = {"train": [], "valid": []}
    coverage = {}
    weak_counts = {}
    for name, targets in splits.items():
        changed = 0
        weak = 0
        for target in targets:
            sources = synthetic_sources(target, nlp)
            valid_sources = []
            for source in sources:
                item = _chat(source, target)
                if _length(tokenizer, item) <= MAX_CHAT_TOKENS:
                    valid_sources.append(item)
            if valid_sources:
                changed += 1
                pairs[name].extend(valid_sources)
            else:
                baseline = _chat(weak_source(target), target)
                if _length(tokenizer, baseline) <= MAX_CHAT_TOKENS:
                    pairs[name].append(baseline)
                    weak += 1
        coverage[name] = changed / len(targets)
        weak_counts[name] = weak
    if min(coverage.values()) < MIN_CHANGED_FRACTION or min(map(len, pairs.values())) < 2:
        raise ValueError(
            "目标文本实质增强覆盖不足（训练 %.0f%% / 验证 %.0f%%）；请增加包含可安全调整"
            "的短语或可规范化用词的英文文本" %
            (coverage["train"] * 100, coverage["valid"] * 100)
        )
    return pairs, {
        "units": len(units), "train_pairs": len(pairs["train"]),
        "valid_pairs": len(pairs["valid"]), "substantive_coverage": coverage,
        "weak_template_pairs": weak_counts,
        "trainable_fraction": "22 个 decoder block / 约 88% 参数",
    }


def write_data(pairs, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    for split, examples in pairs.items():
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
