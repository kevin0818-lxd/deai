# SPDX-License-Identifier: Apache-2.0
"""Opt-in, deterministic low-rate surface errors for the final text only."""

import hashlib

UNSAFE = set("\"'“”‘’()[]{}")


def _selected(seed, index, kind, rate):
    digest = hashlib.sha256(f"{seed}:{index}:{kind}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def control_errors(text, nlp, spelling=0.0, grammar=0.0, seed=42):
    """Each rate is an upper probability per eligible token; zero is identity."""
    if not (0 <= spelling <= 0.05 and 0 <= grammar <= 0.05):
        raise ValueError("拼写/语法错误率均须介于 0 与 0.05")
    if not spelling and not grammar:
        return text
    doc = nlp(text)
    changes = {}
    for sentence in doc.sents:
        if any(char in UNSAFE for char in sentence.text):
            continue
        if any(token.ent_type_ or token.like_num or token.dep_ == "neg"
               for token in sentence):
            continue
        for token in sentence:
            if (grammar and token.lower_ in {"a", "an", "the"}
                    and token.dep_ == "det" and token.head.pos_ == "NOUN"
                    and _selected(seed, token.i, "grammar", grammar)):
                changes[token.i] = ""
            elif (spelling and token.is_alpha and token.is_lower
                  and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                  and len(token.text) >= 6
                  and _selected(seed, token.i, "spelling", spelling)):
                word = token.text
                changes[token.i] = word[:2] + word[3] + word[2] + word[4:]
    return "".join(changes.get(token.i, token.text) + (
        "" if changes.get(token.i) == "" else token.whitespace_
    ) for token in doc)
