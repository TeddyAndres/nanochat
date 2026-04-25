"""Train a tokenizer for nanochat."""
import os
import time
import argparse
from typing import cast
import torch
from nanochat.tokenizer import RustBPETokenizer, SentencePieceTokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a tokenizer')
parser.add_argument('--max-chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=100_000, help='Maximum characters per document (default: 20,000)')
parser.add_argument('--vocab-size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
parser.add_argument('--backend', choices=['rustbpe', 'sentencepiece'], default='rustbpe', help='Tokenizer backend to train')
parser.add_argument('--sentencepiece-model-type', choices=['unigram', 'bpe', 'char', 'word'], default='unigram', help='SentencePiece model type (only used when --backend=sentencepiece)')
parser.add_argument('--sentencepiece-input-sentence-size', type=int, default=250_000, help='Maximum number of training documents sampled by SentencePiece (0 = use all, larger uses more RAM)')
parser.add_argument('--sentencepiece-max-sentence-length', type=int, default=-1, help='Maximum per-document length SentencePiece sees after doc_cap (-1 = use doc_cap)')
parser.add_argument('--sentencepiece-num-threads', type=int, default=4, help='SentencePiece trainer threads')
parser.add_argument('--sentencepiece-shuffle-input-sentence', action=argparse.BooleanOptionalAction, default=True, help='Shuffle/sample documents before SentencePiece training when input_sentence_size is set')
parser.add_argument('--sentencepiece-split-by-whitespace', action=argparse.BooleanOptionalAction, default=True, help='Prevent SentencePiece pieces from spanning whitespace boundaries')
parser.add_argument('--sentencepiece-treat-whitespace-as-suffix', action=argparse.BooleanOptionalAction, default=True, help='Attach whitespace marker to the end of pieces instead of the beginning')
parser.add_argument('--sentencepiece-allow-whitespace-only-pieces', action=argparse.BooleanOptionalAction, default=True, help='Allow standalone whitespace-only pieces in SentencePiece')
parser.add_argument('--sentencepiece-add-dummy-prefix', action=argparse.BooleanOptionalAction, default=False, help='Add the standard SentencePiece dummy prefix so sentence-initial words share pieces with post-space words')
parser.add_argument('--sentencepiece-remove-extra-whitespaces', action=argparse.BooleanOptionalAction, default=True, help='Collapse repeated whitespace during SentencePiece normalization')
parser.add_argument('--train-extremely-large-corpus', '--train_extremely_large_corpus', dest='train_extremely_large_corpus', action=argparse.BooleanOptionalAction, default=False, help='Enable SentencePiece\'s large-corpus training mode')
parser.add_argument('--tokenizer-dir', default='tokenizer', help='Output tokenizer directory; relative paths are resolved under the nanochat base dir')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"backend: {args.backend}")
if args.backend == 'sentencepiece':
    print(f"sentencepiece_model_type: {args.sentencepiece_model_type}")
    print(f"sentencepiece_input_sentence_size: {args.sentencepiece_input_sentence_size:,}")
    print(f"sentencepiece_max_sentence_length: {args.doc_cap if args.sentencepiece_max_sentence_length < 0 else args.sentencepiece_max_sentence_length:,}")
    print(f"sentencepiece_num_threads: {args.sentencepiece_num_threads}")
    print(f"sentencepiece_shuffle_input_sentence: {args.sentencepiece_shuffle_input_sentence}")
    print(f"sentencepiece_split_by_whitespace: {args.sentencepiece_split_by_whitespace}")
    print(f"sentencepiece_treat_whitespace_as_suffix: {args.sentencepiece_treat_whitespace_as_suffix}")
    print(f"sentencepiece_allow_whitespace_only_pieces: {args.sentencepiece_allow_whitespace_only_pieces}")
    print(f"sentencepiece_add_dummy_prefix: {args.sentencepiece_add_dummy_prefix}")
    print(f"sentencepiece_remove_extra_whitespaces: {args.sentencepiece_remove_extra_whitespaces}")
    print(f"train_extremely_large_corpus: {args.train_extremely_large_corpus}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
if args.backend == 'rustbpe':
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
else:
    sentencepiece_max_sentence_length = args.doc_cap if args.sentencepiece_max_sentence_length < 0 else args.sentencepiece_max_sentence_length
    tokenizer = SentencePieceTokenizer.train_from_iterator(
        text_iter,
        args.vocab_size,
        model_type=args.sentencepiece_model_type,
        input_sentence_size=args.sentencepiece_input_sentence_size,
        shuffle_input_sentence=args.sentencepiece_shuffle_input_sentence,
        max_sentence_length=sentencepiece_max_sentence_length,
        num_threads=args.sentencepiece_num_threads,
        split_by_whitespace=args.sentencepiece_split_by_whitespace,
        treat_whitespace_as_suffix=args.sentencepiece_treat_whitespace_as_suffix,
        allow_whitespace_only_pieces=args.sentencepiece_allow_whitespace_only_pieces,
        add_dummy_prefix=args.sentencepiece_add_dummy_prefix,
        remove_extra_whitespaces=args.sentencepiece_remove_extra_whitespaces,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
    )
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = cast(str, get_base_dir())
tokenizer_dir = cast(str, args.tokenizer_dir)
if not os.path.isabs(tokenizer_dir):
    tokenizer_dir = os.path.join(base_dir, tokenizer_dir)
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
