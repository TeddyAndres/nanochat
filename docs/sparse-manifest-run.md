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

## 4. Confirm the sparse manifest file exists

Make sure this file is present relative to the repository root:

```text
manifests/65kvocab_2kseq_4batch_32accum_20kstep.json
```

## 5. Run the sparse + manifest training command

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

## 6. What the key flags do

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

## 7. Notes and troubleshooting

- Run the command from the repository root so the relative manifest path resolves correctly.
- If `--fp8` fails, verify your GPU, drivers, PyTorch, and CUDA stack support FP8.
- If you are using multiple GPUs or distributed launch tooling, adapt the command to your environment while keeping the same training arguments.
- If the manifest path is different in your checkout, update `--sparse-manifest` accordingly.
