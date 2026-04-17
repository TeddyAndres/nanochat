"""CLI analysis tool for sparse loss top-k metrics produced during training.

Usage:
    python -m scripts.analyze_loss_topk /media/teddy/Ventoy/nanochat/nanochat/topk_analysis/d6_replan10step_5kstep <subcommand> [options]

Subcommands:
    summary    overview of step files, record counts, and score distributions
    trends     per-step mean CE loss and logit margin over training
    tokens     top-N worst predicted tokens aggregated across steps
    confusion  top-N target→predicted confusion pairs
    inspect    raw record dump for a specific step or step range
    --step-range START:END can be used to filter which step files are considered by all subcommands (either bound optional, e.g. '100:500'); inspect also supports --step S to focus on a specific step.
    --window W applies a rolling mean with window size W to the score columns in trends (default: 1 = no smoothing).
    --topk N limits the number of rows displayed by tokens and confusion to the top N (default: 20).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Optional

import torch

from nanochat.sparse_analysis import (
    SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    SPARSE_LOSS_TOPK_RANKING_MODES,
    SPARSE_LOSS_TOPK_RANKING_SINGLE,
    normalize_sparse_topk_ranking_mode,
)

# ── Column indices ───────────────────────────────────────────────────────────────
# Both correct (7 cols) and incorrect (9 cols) records share the same leading cols.
_COL_STEP          = 0
_COL_MICRO         = 1
_COL_SEQ           = 2
_COL_ROW           = 3
_COL_POS           = 4
_COL_TARGET_LOCAL  = 5
_COL_TARGET_GLOBAL = 6
# Incorrect records only:
_COL_WRONG_LOCAL   = 7
_COL_WRONG_GLOBAL  = 8


# ── File discovery ───────────────────────────────────────────────────────────────

def _parse_step_range(s: str) -> tuple[Optional[int], Optional[int]]:
    """Parse 'START:END', ':END', 'START:', or ':' into an (lo, hi) tuple."""
    m = re.fullmatch(r'(\d*):(\d*)', s)
    if m is None:
        raise argparse.ArgumentTypeError(
            f"Invalid step range '{s}'; expected format START:END (either bound optional)"
        )
    lo = int(m.group(1)) if m.group(1) else None
    hi = int(m.group(2)) if m.group(2) else None
    return lo, hi


def iter_step_files(
    analysis_dir: Path,
    step_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> list[tuple[int, int, Path]]:
    """Return sorted (start_step, end_step, path) triples for legacy and batched sparse-analysis files."""
    single_pattern = re.compile(r'^step_(\d{6})\.pt$')
    batch_pattern = re.compile(r'^steps_(\d{6})_(\d{6})\.pt$')
    results = []
    for f in analysis_dir.iterdir():
        m = single_pattern.match(f.name)
        if m is not None:
            start_step = int(m.group(1))
            end_step = start_step
        else:
            m = batch_pattern.match(f.name)
            if m is None:
                continue
            start_step = int(m.group(1))
            end_step = int(m.group(2))
        if step_range is not None:
            lo, hi = step_range
            if lo is not None and end_step < lo:
                continue
            if hi is not None and start_step > hi:
                continue
        results.append((start_step, end_step, f))
    results.sort(key=lambda x: (x[0], x[1], x[2].name))
    return results


# ── Payload loading ──────────────────────────────────────────────────────────────

def _normalize_step_payload(raw: dict) -> dict:
    """Normalize one step payload and strip -inf padding rows from scores/records."""
    out: dict = {"step": int(raw["step"])}
    for prefix in ("correct", "incorrect"):
        scores: torch.Tensor = raw[f"{prefix}_scores"]    # (N,)
        records: torch.Tensor = raw[f"{prefix}_records"]  # (N, cols)
        valid = torch.isfinite(scores)
        out[f"{prefix}_scores"] = scores[valid]
        out[f"{prefix}_records"] = records[valid]
    return out


def load_payloads(path: Path, step_range: Optional[tuple[Optional[int], Optional[int]]] = None) -> list[dict]:
    """Load one analysis file and return normalized per-step payloads."""
    raw = torch.load(path, weights_only=True)
    if isinstance(raw, dict) and "step" in raw and "correct_scores" in raw and "incorrect_scores" in raw:
        payloads = [_normalize_step_payload(raw)]
    else:
        payloads = []
        if not isinstance(raw, dict):
            raise ValueError(f"Unsupported sparse analysis payload in {path}")
        for step_key, step_payload in raw.items():
            if not isinstance(step_payload, dict):
                raise ValueError(f"Unsupported sparse analysis step payload in {path} for key {step_key!r}")
            normalized_payload = _normalize_step_payload(step_payload)
            payloads.append(normalized_payload)
    if step_range is None:
        return sorted(payloads, key=lambda payload: int(payload["step"]))
    lo, hi = step_range
    filtered = []
    for payload in payloads:
        step = int(payload["step"])
        if lo is not None and step < lo:
            continue
        if hi is not None and step > hi:
            continue
        filtered.append(payload)
    return sorted(filtered, key=lambda payload: int(payload["step"]))


def iter_step_payloads(
    analysis_dir: Path,
    step_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> list[tuple[int, dict]]:
    payloads: list[tuple[int, dict]] = []
    for _start_step, _end_step, path in iter_step_files(analysis_dir, step_range):
        for payload in load_payloads(path, step_range=step_range):
            payloads.append((int(payload["step"]), payload))
    payloads.sort(key=lambda item: item[0])
    return payloads


def aggregate_ranked_records(
    scores: torch.Tensor,
    records: torch.Tensor,
    *,
    key_columns: tuple[int, ...],
    ranking_mode: str,
) -> dict[tuple[int, ...], dict[str, float | int]]:
    ranking_mode = normalize_sparse_topk_ranking_mode(ranking_mode)
    accum: dict[tuple[int, ...], dict[str, float | int]] = {}
    for idx in range(records.size(0)):
        key = tuple(int(records[idx, column].item()) for column in key_columns)
        score = float(scores[idx].item())
        stats = accum.setdefault(key, {"count": 0, "total_score": 0.0, "max_score": -float("inf")})
        stats["count"] = int(stats["count"]) + 1
        stats["total_score"] = float(stats["total_score"]) + score
        stats["max_score"] = max(float(stats["max_score"]), score)
    return accum


def sort_ranked_items(
    items: dict[tuple[int, ...], dict[str, float | int]],
    *,
    ranking_mode: str,
) -> list[tuple[tuple[int, ...], dict[str, float | int]]]:
    ranking_mode = normalize_sparse_topk_ranking_mode(ranking_mode)
    score_field = "max_score" if ranking_mode == SPARSE_LOSS_TOPK_RANKING_SINGLE else "total_score"
    return sorted(
        items.items(),
        key=lambda item: (-float(item[1][score_field]), item[0]),
    )


# ── Tokenizer ────────────────────────────────────────────────────────────────────

def try_load_tokenizer():
    try:
        from nanochat.tokenizer import get_tokenizer
        return get_tokenizer()
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); token IDs will be shown as <N>", file=sys.stderr)
        return None


def decode_token(tok, token_id: int) -> str:
    if tok is None:
        return f"<{token_id}>"
    try:
        text = tok.decode([token_id])
        return repr(text)
    except Exception:
        return f"<{token_id}>"


# ── Table printing ───────────────────────────────────────────────────────────────

def print_table(headers: list[str], rows: list[list], title: Optional[str] = None) -> None:
    if title:
        print(f"\n{title}")
        print("─" * max(len(title), 40))
    if not rows:
        print("  (no data)")
        return
    col_widths = [len(h) for h in headers]
    str_rows: list[list[str]] = []
    for row in rows:
        str_row = []
        for i, cell in enumerate(row):
            s = str(cell)
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(s))
            str_row.append(s)
        str_rows.append(str_row)
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in str_rows:
        print(fmt.format(*row))


def _fmt_float(v: float, decimals: int = 4) -> str:
    if not math.isfinite(v):
        return str(v)
    return f"{v:.{decimals}f}"


# ── Subcommand: summary ──────────────────────────────────────────────────────────

def cmd_summary(analysis_dir: Path, step_range, args) -> None:
    payloads = iter_step_payloads(analysis_dir, step_range)
    if not payloads:
        print("No matching step files found.")
        return

    steps = [step for step, _payload in payloads]
    total_correct = 0
    total_incorrect = 0
    all_correct: list[torch.Tensor] = []
    all_incorrect: list[torch.Tensor] = []

    for _step, payload in payloads:
        cs = payload["correct_scores"]
        is_ = payload["incorrect_scores"]
        total_correct += cs.numel()
        total_incorrect += is_.numel()
        if cs.numel():
            all_correct.append(cs)
        if is_.numel():
            all_incorrect.append(is_)

    print_table(
        ["metric", "value"],
        [
            ["total analyzed steps",   str(len(payloads))],
            ["step range",             f"{steps[0]} – {steps[-1]}"],
            ["total correct records",  str(total_correct)],
            ["total incorrect records", str(total_incorrect)],
        ],
        title="File / Record Counts",
    )

    def _dist_rows(label: str, tensors: list[torch.Tensor]) -> list[list]:
        if not tensors:
            return [[label, "–", "–", "–", "–", "–"]]
        t = torch.cat(tensors)
        return [[
            label,
            _fmt_float(t.min().item()),
            _fmt_float(torch.quantile(t, 0.25).item()),
            _fmt_float(t.mean().item()),
            _fmt_float(torch.quantile(t, 0.75).item()),
            _fmt_float(t.max().item()),
        ]]

    print_table(
        ["metric", "min", "p25", "mean", "p75", "max"],
        _dist_rows("correct CE loss",    all_correct)
        + _dist_rows("incorrect margin", all_incorrect),
        title="Score Distributions",
    )


# ── Subcommand: trends ───────────────────────────────────────────────────────────

def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def cmd_trends(analysis_dir: Path, step_range, args) -> None:
    payloads = iter_step_payloads(analysis_dir, step_range)
    if not payloads:
        print("No matching step files found.")
        return

    window: int = getattr(args, "window", 1)
    raw: list[tuple[int, int, float, int, float]] = []

    for step, payload in payloads:
        cs = payload["correct_scores"]
        is_ = payload["incorrect_scores"]
        mean_ce     = cs.mean().item()  if cs.numel()  else float("nan")
        mean_margin = is_.mean().item() if is_.numel() else float("nan")
        raw.append((step, cs.numel(), mean_ce, is_.numel(), mean_margin))

    steps      = [r[0] for r in raw]
    n_correct  = [r[1] for r in raw]
    mean_ce    = _rolling_mean([r[2] for r in raw], window)
    n_incorrect = [r[3] for r in raw]
    mean_margin = _rolling_mean([r[4] for r in raw], window)

    title = "Per-Step Trends"
    if window > 1:
        title += f" (window={window})"

    print_table(
        ["step", "n_correct", "mean_ce_loss", "n_incorrect", "mean_margin"],
        [
            [
                str(steps[i]),
                str(n_correct[i]),
                _fmt_float(mean_ce[i]),
                str(n_incorrect[i]),
                _fmt_float(mean_margin[i]),
            ]
            for i in range(len(steps))
        ],
        title=title,
    )


# ── Subcommand: tokens ───────────────────────────────────────────────────────────

def cmd_tokens(analysis_dir: Path, step_range, args) -> None:
    payloads = iter_step_payloads(analysis_dir, step_range)
    if not payloads:
        print("No matching step files found.")
        return

    topk = args.topk
    tok = try_load_tokenizer()
    ranking_mode = normalize_sparse_topk_ranking_mode(args.ranking_mode)

    accum: dict[tuple[int, ...], dict[str, float | int]] = {}
    for _step, payload in payloads:
        step_accum = aggregate_ranked_records(
            payload["correct_scores"],
            payload["correct_records"],
            key_columns=(_COL_TARGET_GLOBAL,),
            ranking_mode=ranking_mode,
        )
        for key, stats in step_accum.items():
            global_stats = accum.setdefault(key, {"count": 0, "total_score": 0.0, "max_score": -float("inf")})
            global_stats["count"] = int(global_stats["count"]) + int(stats["count"])
            global_stats["total_score"] = float(global_stats["total_score"]) + float(stats["total_score"])
            global_stats["max_score"] = max(float(global_stats["max_score"]), float(stats["max_score"]))

    if not accum:
        print("No correct records found.")
        return

    sorted_items = sort_ranked_items(accum, ranking_mode=ranking_mode)[:topk]

    rows = []
    for rank, ((tok_id,), stats) in enumerate(sorted_items, 1):
        count = int(stats["count"])
        total_score = float(stats["total_score"])
        max_score = float(stats["max_score"])
        mean_score = total_score / count if count > 0 else 0.0
        rows.append([
            str(rank),
            str(tok_id),
            decode_token(tok, tok_id),
            str(count),
            _fmt_float(max_score),
            _fmt_float(mean_score),
            _fmt_float(total_score),
        ])

    title = (
        f"Top-{topk} Worst Predicted Tokens (by max single-occurrence CE loss)"
        if ranking_mode == SPARSE_LOSS_TOPK_RANKING_SINGLE
        else f"Top-{topk} Worst Predicted Tokens (by total accumulated CE loss)"
    )

    print_table(
        ["rank", "token_id", "decoded", "count", "max_ce_loss", "mean_ce_loss", "total_ce_loss"],
        rows,
        title=title,
    )


# ── Subcommand: confusion ────────────────────────────────────────────────────────

def cmd_confusion(analysis_dir: Path, step_range, args) -> None:
    payloads = iter_step_payloads(analysis_dir, step_range)
    if not payloads:
        print("No matching step files found.")
        return

    topk = args.topk
    tok = try_load_tokenizer()
    ranking_mode = normalize_sparse_topk_ranking_mode(args.ranking_mode)

    accum: dict[tuple[int, ...], dict[str, float | int]] = {}
    for _step, payload in payloads:
        step_accum = aggregate_ranked_records(
            payload["incorrect_scores"],
            payload["incorrect_records"],
            key_columns=(_COL_TARGET_GLOBAL, _COL_WRONG_GLOBAL),
            ranking_mode=ranking_mode,
        )
        for key, stats in step_accum.items():
            global_stats = accum.setdefault(key, {"count": 0, "total_score": 0.0, "max_score": -float("inf")})
            global_stats["count"] = int(global_stats["count"]) + int(stats["count"])
            global_stats["total_score"] = float(global_stats["total_score"]) + float(stats["total_score"])
            global_stats["max_score"] = max(float(global_stats["max_score"]), float(stats["max_score"]))

    if not accum:
        print("No incorrect records found.")
        return

    sorted_items = sort_ranked_items(accum, ranking_mode=ranking_mode)[:topk]

    rows = []
    for rank, ((target_id, wrong_id), stats) in enumerate(sorted_items, 1):
        count = int(stats["count"])
        total_margin = float(stats["total_score"])
        max_margin = float(stats["max_score"])
        mean_margin = total_margin / count if count > 0 else 0.0
        rows.append([
            str(rank),
            str(target_id),
            decode_token(tok, target_id),
            str(wrong_id),
            decode_token(tok, wrong_id),
            str(count),
            _fmt_float(max_margin),
            _fmt_float(mean_margin),
            _fmt_float(total_margin),
        ])

    title = (
        f"Top-{topk} Confusion Pairs (by max single-occurrence margin)"
        if ranking_mode == SPARSE_LOSS_TOPK_RANKING_SINGLE
        else f"Top-{topk} Confusion Pairs (by total accumulated margin)"
    )

    print_table(
        ["rank", "target_id", "target", "wrong_id", "wrong", "count", "max_margin", "mean_margin", "total_margin"],
        rows,
        title=title,
    )


# ── Subcommand: inspect ──────────────────────────────────────────────────────────

def cmd_inspect(analysis_dir: Path, step_range, args) -> None:
    specific_step: Optional[int] = getattr(args, "step", None)
    if specific_step is not None:
        payloads = iter_step_payloads(analysis_dir, (specific_step, specific_step))
    else:
        payloads = iter_step_payloads(analysis_dir, step_range)

    if not payloads:
        print("No matching step files found.")
        return

    tok = try_load_tokenizer()

    for step_num, payload in payloads:

        # Correct records
        c_scores  = payload["correct_scores"]
        c_records = payload["correct_records"]
        correct_rows = []
        for i in range(c_records.size(0)):
            r = c_records[i]
            correct_rows.append([
                str(r[_COL_STEP].item()),
                str(r[_COL_MICRO].item()),
                str(r[_COL_SEQ].item()),
                str(r[_COL_ROW].item()),
                str(r[_COL_POS].item()),
                str(r[_COL_TARGET_LOCAL].item()),
                str(r[_COL_TARGET_GLOBAL].item()),
                decode_token(tok, int(r[_COL_TARGET_GLOBAL].item())),
                _fmt_float(c_scores[i].item()),
            ])

        print_table(
            ["step", "micro", "seq", "row", "pos", "target_local", "target_global", "decoded", "ce_loss"],
            correct_rows,
            title=f"Step {step_num} — Correct Records ({len(correct_rows)} rows)",
        )

        # Incorrect records
        i_scores  = payload["incorrect_scores"]
        i_records = payload["incorrect_records"]
        incorrect_rows = []
        for i in range(i_records.size(0)):
            r = i_records[i]
            incorrect_rows.append([
                str(r[_COL_STEP].item()),
                str(r[_COL_MICRO].item()),
                str(r[_COL_SEQ].item()),
                str(r[_COL_ROW].item()),
                str(r[_COL_POS].item()),
                str(r[_COL_TARGET_GLOBAL].item()),
                decode_token(tok, int(r[_COL_TARGET_GLOBAL].item())),
                str(r[_COL_WRONG_GLOBAL].item()),
                decode_token(tok, int(r[_COL_WRONG_GLOBAL].item())),
                _fmt_float(i_scores[i].item()),
            ])

        print_table(
            ["step", "micro", "seq", "row", "pos", "target_global", "target", "wrong_global", "wrong", "margin"],
            incorrect_rows,
            title=f"Step {step_num} — Incorrect Records ({len(incorrect_rows)} rows)",
        )


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="analyze_loss_topk",
        description="Analyse sparse loss top-k metrics from a *_sparse_analysis/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "analysis_dir",
        type=Path,
        help="Path to *_sparse_analysis/ directory",
    )
    parser.add_argument(
        "--step-range",
        metavar="START:END",
        type=_parse_step_range,
        default=None,
        dest="step_range",
        help="Only consider steps in [START, END] (either bound optional, e.g. '100:500')",
    )
    parser.add_argument(
        "--ranking-mode",
        type=str,
        default=SPARSE_LOSS_TOPK_RANKING_SINGLE,
        choices=SPARSE_LOSS_TOPK_RANKING_MODES,
        help="Ranking mode for aggregated tokens/confusion views: 'single' ranks by max single occurrence, 'accumulated' ranks by total score",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("summary",   help="File/record counts and score distribution statistics")

    trends_p = sub.add_parser("trends", help="Per-step mean CE loss and logit margin over training")
    trends_p.add_argument(
        "--window",
        type=int,
        default=1,
        metavar="W",
        help="Rolling mean window size for score columns (default: 1 = no smoothing)",
    )

    tokens_p = sub.add_parser("tokens",    help="Top-N worst predicted tokens aggregated across steps")
    tokens_p.add_argument(
        "--topk",
        type=int,
        default=20,
        metavar="N",
        help="Number of tokens to display (default: 20)",
    )

    confusion_p = sub.add_parser("confusion", help="Top-N target→predicted confusion pairs")
    confusion_p.add_argument(
        "--topk",
        type=int,
        default=20,
        metavar="N",
        help="Number of confusion pairs to display (default: 20)",
    )

    inspect_p = sub.add_parser("inspect", help="Raw record dump for a specific step or step range")
    inspect_p.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="S",
        help="Specific step to inspect; if omitted, uses top-level --step-range",
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    analysis_dir: Path = args.analysis_dir

    if not analysis_dir.exists():
        print(f"error: directory does not exist: {analysis_dir}", file=sys.stderr)
        sys.exit(1)
    if not analysis_dir.is_dir():
        print(f"error: not a directory: {analysis_dir}", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "summary":   cmd_summary,
        "trends":    cmd_trends,
        "tokens":    cmd_tokens,
        "confusion": cmd_confusion,
        "inspect":   cmd_inspect,
    }
    dispatch[args.subcommand](analysis_dir, args.step_range, args)


if __name__ == "__main__":
    main()
