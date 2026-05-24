# Running the sparse + manifest training path

Use the following steps to run the sparse + manifest path on branch `feature/upstream-sparse-runtime`.

## 1. Clone the repository

```bash
git clone https://github.com/TeddyAndres/nanochat.git
cd nanochat
git checkout feature/upstream-sparse-runtime
```

## 2. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

If the repository includes a requirements or project configuration file, install the dependencies from it. For example:

```bash
pip install -r requirements.txt
```

If the project uses a different install flow, follow the repository’s standard setup instructions first.

## 4. Build the sparse manifest

Most people will need to create the manifest before training. The repository includes a builder at `scripts/build_sparse_manifest.py`.

Create the output directory first:

```bash
mkdir -p manifests
```

Then build the manifest from the repository root:

```bash
python -m scripts.build_sparse_manifest \
  --output manifests/65kvocab_2kseq_4batch_32accum_20kstep.json \
  --num-iterations 20000 \
  --device-batch-size 4 \
  --total-batch-size 262144
```

### Optional manifest-builder flags

Depending on your setup, you may also want to set:

- `--ddp-world-size`: the target world size the manifest should represent.
- `--max-seq-len`: sequence length, default `2048`.
- `--split`: dataset split, default `train`.
- `--token-cache-dir`: where pre-tokenized data should be written.
- `--token-cache-workers`: worker count for token-cache creation.
- `--token-cache-shard-batches`: how many tokenized batches to store per cache shard.
- `--buffer-size`: best-fit document buffer size.
- `--shard-steps`: how many completed optimizer steps to buffer before flushing shard JSON files.

If you are generating the manifest for the exact training run below, keep the batch geometry aligned with training so the manifest matches the run configuration.

## 5. Disk-space and preprocessing considerations

The sparse manifest builder pre-tokenizes data and writes a token cache to disk before or while constructing the manifest. That means manifest generation can require substantial local storage in addition to the final JSON manifest.

Before running the builder, make sure you have enough disk space for:

- the original dataset,
- the pre-tokenized cache,
- the generated manifest JSON and shard files,
- temporary growth while shards are being written.

Practical tips:

- Prefer a fast local SSD for the token cache.
- Use `--token-cache-dir` if your default dataset location does not have enough free space.
- Expect the token cache to be much larger than the final manifest file.
- Do not assume manifest generation is lightweight just because the output is JSON; the expensive part is pre-tokenizing and caching the corpus.
- If space is tight, point the cache to a larger volume and clean up old cache directories when they are no longer needed.

## 6. Confirm the sparse manifest file exists

Make sure this file is present relative to the repository root after the builder finishes:

```text
manifests/65kvocab_2kseq_4batch_32accum_20kstep.json
```

## 7. Run the sparse + manifest training command

From the repository root, run:

```bash
python -m scripts.base_train \
  --sparse-mode \
  --depth 24 \
  --window-pattern L \
  --fp8 \
  --total-batch-size 262144 \
  --device-batch-size 4 \
  --num-iterations 20000 \
  --sparse-manifest manifests/65kvocab_2kseq_4batch_32accum_20kstep.json \
  --warmup-ratio 0 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.1 \
  --log-every 10 \
  --core-metric-every 5000 \
  --run "d24 dynamic false 20ksteps 262keffbatch"
```

## 8. What the key flags do

- `--sparse-mode`: enables the sparse training path.
- `--sparse-manifest ...json`: loads the manifest that defines the sparse schedule/configuration.
- `--depth 24`: trains a 24-layer model.
- `--window-pattern L`: uses the `L` window pattern.
- `--fp8`: enables FP8 mode if your hardware/software stack supports it.
- `--total-batch-size 262144`: sets the effective global batch size.
- `--device-batch-size 4`: sets the per-device batch size.
- `--num-iterations 20000`: runs training for 20,000 iterations.
- `--warmup-ratio 0`, `--warmdown-ratio 0.65`, `--final-lr-frac 0.1`: configure the learning-rate schedule.
- `--log-every 10`: logs every 10 iterations.
- `--core-metric-every 5000`: reports core metrics every 5,000 iterations.
- `--run "..."`: sets the run name for tracking/logging.

## 9. Notes and troubleshooting

- Run both the manifest builder and training command from the repository root so relative paths resolve correctly.
- The manifest builder is single-process only; do not run it with `torchrun`.
- If you are training with multiple GPUs, set `--ddp-world-size` when building the manifest so it matches the intended distributed batch geometry.
- If `--fp8` fails, verify your GPU, drivers, PyTorch, and CUDA stack support FP8.
- If the manifest path is different in your checkout, update `--sparse-manifest` accordingly.
- If manifest creation is slow, that is expected: tokenization and cache generation are doing most of the work.
