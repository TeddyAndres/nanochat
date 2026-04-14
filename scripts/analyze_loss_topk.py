"""CLI analysis tool for sparse loss top-k metrics produced during training.

Usage:
    python -m scripts.analyze_loss_topk /media/teddy/Ventoy/nanochat/nanochat/base_data_climbmix_token_cache_v3_sparse_analysis <subcommand> [options]

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
) -> list[tuple[int, Path]]:
    """Return sorted (step_int, path) pairs for all step_XXXXXX.pt files in analysis_dir."""
    pattern = re.compile(r'^step_(\d{6})\.pt$')
    results = []
    for f in analysis_dir.iterdir():
        m = pattern.match(f.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step_range is not None:
            lo, hi = step_range
            if lo is not None and step < lo:
                continue
            if hi is not None and step > hi:
                continue
        results.append((step, f))
    results.sort(key=lambda x: x[0])
    return results


# ── Payload loading ──────────────────────────────────────────────────────────────

def load_payload(path: Path) -> dict:
    """Load a step .pt file; strip -inf padding rows from scores/records."""
    raw = torch.load(path, weights_only=True)
    out: dict = {"step": int(raw["step"])}
    for prefix in ("correct", "incorrect"):
        scores: torch.Tensor = raw[f"{prefix}_scores"]    # (N,)
        records: torch.Tensor = raw[f"{prefix}_records"]  # (N, cols)
        valid = torch.isfinite(scores)
        out[f"{prefix}_scores"] = scores[valid]
        out[f"{prefix}_records"] = records[valid]
    return out


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
    files = iter_step_files(analysis_dir, step_range)
    if not files:
        print("No matching step files found.")
        return

    steps = [s for s, _ in files]
    total_correct = 0
    total_incorrect = 0
    all_correct: list[torch.Tensor] = []
    all_incorrect: list[torch.Tensor] = []

    for _step, path in files:
        payload = load_payload(path)
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
            ["total step files",       str(len(files))],
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
    files = iter_step_files(analysis_dir, step_range)
    if not files:
        print("No matching step files found.")
        return

    window: int = getattr(args, "window", 1)
    raw: list[tuple[int, int, float, int, float]] = []

    for step, path in files:
        payload = load_payload(path)
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
    files = iter_step_files(analysis_dir, step_range)
    if not files:
        print("No matching step files found.")
        return

    topk = args.topk
    tok = try_load_tokenizer()

    # {token_id: [count, sum_score]}
    accum: dict[int, list] = {}
    for _step, path in files:
        payload = load_payload(path)
        scores  = payload["correct_scores"]
        records = payload["correct_records"]
        for i in range(records.size(0)):
            tok_id = int(records[i, _COL_TARGET_GLOBAL].item())
            score  = float(scores[i].item())
            if tok_id not in accum:
                accum[tok_id] = [0, 0.0]
            accum[tok_id][0] += 1
            accum[tok_id][1] += score

    if not accum:
        print("No correct records found.")
        return

    sorted_items = sorted(accum.items(), key=lambda kv: kv[1][1], reverse=True)[:topk]

    rows = []
    for rank, (tok_id, (count, total_score)) in enumerate(sorted_items, 1):
        mean_score = total_score / count if count > 0 else 0.0
        rows.append([
            str(rank),
            str(tok_id),
            decode_token(tok, tok_id),
            str(count),
            _fmt_float(mean_score),
            _fmt_float(total_score),
        ])

    print_table(
        ["rank", "token_id", "decoded", "count", "mean_ce_loss", "total_ce_loss"],
        rows,
        title=f"Top-{topk} Worst Predicted Tokens (by total CE loss)",
    )


# ── Subcommand: confusion ────────────────────────────────────────────────────────

def cmd_confusion(analysis_dir: Path, step_range, args) -> None:
    files = iter_step_files(analysis_dir, step_range)
    if not files:
        print("No matching step files found.")
        return

    topk = args.topk
    tok = try_load_tokenizer()

    # {(target_global, wrong_global): [count, sum_margin]}
    accum: dict[tuple[int, int], list] = {}
    for _step, path in files:
        payload = load_payload(path)
        scores  = payload["incorrect_scores"]
        records = payload["incorrect_records"]
        for i in range(records.size(0)):
            target_id = int(records[i, _COL_TARGET_GLOBAL].item())
            wrong_id  = int(records[i, _COL_WRONG_GLOBAL].item())
            score     = float(scores[i].item())
            key       = (target_id, wrong_id)
            if key not in accum:
                accum[key] = [0, 0.0]
            accum[key][0] += 1
            accum[key][1] += score

    if not accum:
        print("No incorrect records found.")
        return

    sorted_items = sorted(accum.items(), key=lambda kv: kv[1][0], reverse=True)[:topk]

    rows = []
    for rank, ((target_id, wrong_id), (count, total_margin)) in enumerate(sorted_items, 1):
        mean_margin = total_margin / count if count > 0 else 0.0
        rows.append([
            str(rank),
            str(target_id),
            decode_token(tok, target_id),
            str(wrong_id),
            decode_token(tok, wrong_id),
            str(count),
            _fmt_float(mean_margin),
            _fmt_float(total_margin),
        ])

    print_table(
        ["rank", "target_id", "target", "wrong_id", "wrong", "count", "mean_margin", "total_margin"],
        rows,
        title=f"Top-{topk} Confusion Pairs (by count)",
    )


# ── Subcommand: inspect ──────────────────────────────────────────────────────────

def cmd_inspect(analysis_dir: Path, step_range, args) -> None:
    specific_step: Optional[int] = getattr(args, "step", None)
    if specific_step is not None:
        files = iter_step_files(analysis_dir, (specific_step, specific_step))
    else:
        files = iter_step_files(analysis_dir, step_range)

    if not files:
        print("No matching step files found.")
        return

    tok = try_load_tokenizer()

    for step_num, path in files:
        payload = load_payload(path)

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
