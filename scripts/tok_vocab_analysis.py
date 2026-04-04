"""
Analyze saved tokenizer vocabularies and estimate how many tokens represent complete words.

Examples:
/home/teddy/Desktop/dev/repo/nanochat/.venv-5090/bin/python -m scripts.tok_vocab_analysis \
  --tokenizer-dir ~/.cache/nanochat/tokenizer-65k-10B \
  --tokenizer-dir ~/.cache/nanochat/tokenizer-sp-unigram-262k
"""

import argparse
import os
from collections import defaultdict

import regex

from nanochat.tokenizer import load_tokenizer_from_directory, load_token_bytes_from_directory


WORD_CORE_RE = regex.compile(r"^[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*$")
LETTER_FRAGMENT_RE = regex.compile(r"^[\p{L}]+$")
NUMERIC_RE = regex.compile(r"^[\p{N}]+$")
SHORT_WORD_ALLOWLIST = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze tokenizer vocabulary composition")
    parser.add_argument(
        "--tokenizer-dir",
        action="append",
        required=True,
        help="Path to a saved tokenizer directory. Repeat to compare multiple tokenizers.",
    )
    parser.add_argument(
        "--min-word-chars",
        type=int,
        default=3,
        help="Minimum stripped character length required to count as a complete word token",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of example tokens to print per category",
    )
    return parser.parse_args()


def normalize_dir(path):
    return os.path.abspath(os.path.expanduser(path))


def classify_token(token_str, token_bytes, special_tokens, min_word_chars):
    if token_str in special_tokens or token_bytes == 0:
        return "special"

    if token_str == "":
        return "empty"

    stripped = token_str.strip()
    leading_ws = len(token_str) - len(token_str.lstrip())
    trailing_ws = len(token_str) - len(token_str.rstrip())

    if stripped == "":
        return "whitespace"

    folded = stripped.casefold()
    if NUMERIC_RE.fullmatch(stripped):
        return "numeric"

    if WORD_CORE_RE.fullmatch(stripped) and (len(stripped) >= min_word_chars or folded in SHORT_WORD_ALLOWLIST):
        if leading_ws > 0 or trailing_ws > 0:
            return "complete_word_with_space"
        return "complete_word"

    if LETTER_FRAGMENT_RE.fullmatch(stripped):
        return "alpha_fragment"

    if token_bytes <= 2:
        return "byte_fragment"

    return "mixed_fragment"


def safe_decode(tokenizer, token_id):
    try:
        return tokenizer.decode([token_id])
    except Exception as exc:
        return f"<decode_error:{token_id}:{type(exc).__name__}>"


def summarize_tokenizer(tokenizer_dir, min_word_chars, top_k):
    tokenizer = load_tokenizer_from_directory(tokenizer_dir)
    token_bytes = load_token_bytes_from_directory(tokenizer_dir, device="cpu")
    vocab_size = tokenizer.get_vocab_size()
    special_tokens = set(tokenizer.get_special_tokens())

    counts = defaultdict(int)
    examples = defaultdict(list)
    bytes_by_category = defaultdict(int)
    total_non_special = 0
    total_complete_words = 0

    for token_id in range(vocab_size):
        token_str = safe_decode(tokenizer, token_id)
        token_byte_count = int(token_bytes[token_id].item())
        category = classify_token(token_str, token_byte_count, special_tokens, min_word_chars)
        counts[category] += 1
        bytes_by_category[category] += token_byte_count
        if category not in {"special", "empty"}:
            total_non_special += 1
        if category in {"complete_word", "complete_word_with_space"}:
            total_complete_words += 1
        if len(examples[category]) < top_k:
            examples[category].append((token_id, token_str, token_byte_count))

    return {
        "tokenizer_dir": tokenizer_dir,
        "vocab_size": vocab_size,
        "counts": dict(counts),
        "examples": dict(examples),
        "bytes_by_category": dict(bytes_by_category),
        "special_tokens": sorted(special_tokens),
        "total_non_special": total_non_special,
        "total_complete_words": total_complete_words,
    }


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def format_examples(rows):
    if not rows:
        return "-"
    return ", ".join(f"{token_id}:{token_str!r}" for token_id, token_str, _ in rows)


def print_summary(summary, top_k):
    counts = summary["counts"]
    vocab_size = summary["vocab_size"]
    total_non_special = summary["total_non_special"]
    total_complete_words = summary["total_complete_words"]

    print(f"\n=== {summary['tokenizer_dir']} ===")
    print(f"vocab_size: {vocab_size:,}")
    print(f"complete_word_tokens: {total_complete_words:,} / {vocab_size:,} ({pct(total_complete_words, vocab_size):.2f}% of vocab)")
    print(f"complete_word_tokens_non_special: {total_complete_words:,} / {total_non_special:,} ({pct(total_complete_words, total_non_special):.2f}% of non-special vocab)")

    ordered_categories = [
        "complete_word",
        "complete_word_with_space",
        "alpha_fragment",
        "mixed_fragment",
        "byte_fragment",
        "numeric",
        "whitespace",
        "special",
        "empty",
    ]
    print("\ncategory counts:")
    for category in ordered_categories:
        count = counts.get(category, 0)
        if count == 0 and category == "empty":
            continue
        print(f"  {category:26s} {count:8,d}  {pct(count, vocab_size):6.2f}%")

    print(f"\nexamples (top {top_k} per category):")
    for category in ordered_categories:
        rows = summary["examples"].get(category, [])
        if not rows:
            continue
        print(f"  {category:26s} {format_examples(rows)}")


def print_comparison_table(summaries):
    if len(summaries) < 2:
        return
    print("\n=== Comparison ===")
    header = (
        f"{'tokenizer':40s} {'vocab':>10s} {'complete':>10s} {'complete%':>10s} "
        f"{'spaced%':>10s} {'alpha_frag%':>12s} {'mixed_frag%':>12s} {'byte_frag%':>11s}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        counts = summary["counts"]
        vocab_size = summary["vocab_size"]
        complete = counts.get("complete_word", 0) + counts.get("complete_word_with_space", 0)
        row = (
            f"{os.path.basename(summary['tokenizer_dir']):40.40s} "
            f"{vocab_size:10,d} "
            f"{complete:10,d} "
            f"{pct(complete, vocab_size):9.2f}% "
            f"{pct(counts.get('complete_word_with_space', 0), vocab_size):9.2f}% "
            f"{pct(counts.get('alpha_fragment', 0), vocab_size):11.2f}% "
            f"{pct(counts.get('mixed_fragment', 0), vocab_size):11.2f}% "
            f"{pct(counts.get('byte_fragment', 0), vocab_size):10.2f}%"
        )
        print(row)


def main():
    args = parse_args()
    tokenizer_dirs = [normalize_dir(path) for path in args.tokenizer_dir]
    summaries = [summarize_tokenizer(path, args.min_word_chars, args.top_k) for path in tokenizer_dirs]
    print("Heuristic: 'complete word' means the token decodes to optional surrounding whitespace plus a word-shaped core with no punctuation beyond internal apostrophes/hyphens.")
    print("Count both bare words and whitespace-prefixed/suffixed words as complete words, because both can carry full lexical meaning.")
    for summary in summaries:
        print_summary(summary, args.top_k)
    print_comparison_table(summaries)


if __name__ == "__main__":
    main()