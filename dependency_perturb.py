"""Conservative, dependency-guided reordering for the HTTP humanize pipeline."""

import sys

from infer_full import split_blocks

CONNECTORS = {"when", "although", "if"}
PREPOSITIONS = {"after", "before", "during", "in", "on", "at"}
UNSAFE_MARKS = set("\"“”‘’()[]{};:")
FINAL_MARKS = ".!?"


def load_parser():
    """Load a locally installed English dependency parser, never downloading it."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "缺少英文依存解析器；在 .venv 中安装 spaCy 和 en_core_web_sm"
        ) from exc
    if "parser" not in nlp.pipe_names:
        raise RuntimeError("en_core_web_sm 缺少 parser 组件")
    return nlp


def _lower_initial(text):
    return text[0].lower() + text[1:]


def _upper_initial(text):
    return text[0].upper() + text[1:]


def _candidate(doc, moved):
    """Return a complete edge adjunct as (start, end, destination), or None."""
    if len(doc) < 6 or not doc.has_annotation("DEP"):
        return None
    if any(token.text in UNSAFE_MARKS or token.dep_ in {"neg", "cc", "conj"}
           for token in doc):
        return None
    last = doc[-1]
    if last.text not in FINAL_MARKS:
        return None

    # Moving trailing adjuncts first permits two independent trailing phrases
    # to be fronted in one sentence without moving either phrase twice.
    candidates = []
    for token in doc:
        if token.dep_ not in {"advcl", "prep", "advmod"}:
            continue
        subtree = list(token.subtree)
        indices = [part.i for part in subtree]
        start, end = min(indices), max(indices) + 1
        if indices != list(range(start, end)):
            continue
        if token.head.i >= start and token.head.i < end:
            continue
        while end > start and doc[end - 1].is_punct:
            end -= 1
        if start >= end or end >= len(doc):
            continue
        phrase = doc[start:end]
        signature = tuple(t.text.lower() for t in phrase)
        if signature in moved or len(phrase) < 2:
            continue
        if any(t.ent_type_ or t.like_num or t.dep_ == "neg" for t in phrase):
            continue
        if token.dep_ == "advcl":
            if phrase[0].lower_ not in CONNECTORS:
                continue
        elif token.dep_ == "prep":
            if token.lower_ not in PREPOSITIONS or not any(
                child.dep_ in {"pobj", "pcomp"} for child in token.children
            ):
                continue
            if token.head.pos_ not in {"VERB", "AUX", "ADV"}:
                continue
        else:
            # A standalone adverb is too easy to attach to the wrong clause.
            continue
        if start == 0 and end < len(doc) - 2 and doc[end].text == ",":
            candidates.append((start, end, "tail", signature))
        elif end == len(doc) - 1 and start > 1 and token.dep_ == "prep":
            candidates.append((start, end, "front", signature))
    front = [item for item in candidates if item[2] == "front"]
    tail = [item for item in candidates if item[2] == "tail"]
    return (front or tail or [None])[0]


def _move(sentence, doc, candidate, nlp):
    start, end, destination, _ = candidate
    phrase = sentence[doc[start].idx:doc[end - 1].idx + len(doc[end - 1])]
    if destination == "front":
        core = sentence[:doc[start].idx].rstrip()
        if not core:
            return None
        core = core if doc[0].pos_ == "PROPN" else _lower_initial(core)
        result = f"{_upper_initial(phrase)}, {core}{sentence[-1]}"
    else:
        comma = doc[end]
        core = sentence[comma.idx + len(comma):].strip()
        if not core or core[-1] not in FINAL_MARKS:
            return None
        core = core[:-1].rstrip()
        if not core:
            return None
        result = f"{_upper_initial(core)} {_lower_initial(phrase)}{sentence[-1]}"
    # Ignore punctuation/case changes when checking that every lexical token
    # survived exactly once; no paraphrasing happens in this stage.
    original = sorted(t.text.lower() for t in doc if not t.is_punct)
    return result if original == sorted(
        t.text.lower() for t in nlp.make_doc(result) if not t.is_punct
    ) else None


def _perturb_sentence(sentence, nlp):
    if any(char in UNSAFE_MARKS for char in sentence) or sentence.count("'") % 2:
        return sentence
    current, moved = sentence, set()
    for _ in range(2):
        doc = nlp(current)
        candidate = _candidate(doc, moved)
        if candidate is None:
            break
        changed = _move(current, doc, candidate, nlp)
        if changed is None or changed == current:
            break
        moved.add(candidate[3])
        current = changed
    return current


def perturb_text(text, nlp):
    """Preserve paragraph separators and fall back locally on parser errors."""
    pieces = []
    for body, separator in split_blocks(text):
        try:
            doc = nlp(body)
            cursor, rewritten = 0, []
            for sent in doc.sents:
                rewritten.append(body[cursor:sent.start_char])
                source = body[sent.start_char:sent.end_char]
                try:
                    rewritten.append(_perturb_sentence(source, nlp))
                except Exception as exc:
                    print(f"依存扰动句子回退：{type(exc).__name__}",
                          file=sys.stderr, flush=True)
                    rewritten.append(source)
                cursor = sent.end_char
            rewritten.append(body[cursor:])
            pieces.append("".join(rewritten) + separator)
        except Exception as exc:
            print(f"依存扰动段落回退：{type(exc).__name__}",
                  file=sys.stderr, flush=True)
            pieces.append(body + separator)
    return "".join(pieces)
