# RunPod GPU Training Automation

Programmatic RunPod pod management for training jobs: spin up a GPU, upload code,
run training, download outputs, and terminate — all from Python.

Includes a watchdog that kills pods older than N minutes (safety net against idle costs).

## Files

| File | Purpose |
|---|---|
| `runpod_helpers.py` | Core library: pod CRUD, SSH upload/exec/download, training pipeline |
| `main.py` | Orchestrator CLI: one-command training runs |
| `watchdog.py` | Background process that terminates stale pods |
| `training_code/train_mnist.py` | Demo MNIST training script |

## Setup

```bash
pip install requests
export RUNPOD_API_KEY="your_api_key_here"
```

No other dependencies needed — SSH/SCP use the system `ssh` and `scp` commands.

## Usage

### Run a training job (full pipeline)

```bash
# Default: spot instance, cheapest available GPU, auto-fallback
python main.py

# Specify GPU (for your H100 challenge runs)
python main.py --gpu "NVIDIA H100 80GB HBM3"

# On-demand instead of spot
python main.py --on-demand

# Custom training code
python main.py --code-dir /path/to/your/code --script your_train.py
```

### Use the library directly

```python
from runpod_helpers import (
    list_pods,
    create_pod,
    get_pod,
    terminate_pod,
    terminate_all_pods,
    terminate_stale_pods,
    run_training,
    ssh_exec, ssh_upload, ssh_download,
)

# List all running pods
pods = list_pods()

# Create a spot pod with auto-fallback (tries cheap GPUs first)
pod = create_pod(
    name="my-training",
    gpu_type="NVIDIA H100 80GB HBM3",
    spot=True,
    ssh_public_key=pub_key,
    auto_fallback=True,  # falls back to cheaper GPUs if H100 unavailable
)

# Full training pipeline (create → upload → train → download → terminate)
result = run_training(
    training_code_dir="./my_code",
    train_script="train.py",
    output_dir="./outputs",
    gpu_type="NVIDIA H100 80GB HBM3",
    spot=True,
)

# Terminate all pods older than 40 minutes
terminated = terminate_stale_pods(max_age_minutes=40)

# Terminate a specific pod
terminate_pod("pod_id_here")

# Terminate all pods
terminate_all_pods()
```

### Watchdog (background safety net)

```bash
# Run every 10 minutes, kill pods older than 40 min
python watchdog.py --interval 600 --max-age 40

# Run once (e.g. from cron)
python watchdog.py --once --max-age 40

# Example crontab entry (every 10 minutes):
# */10 * * * * RUNPOD_API_KEY="your_key" python /path/to/watchdog.py --once --max-age 40
```

### CLI commands

```bash
# List pods
python runpod_helpers.py list

# Create a pod
python runpod_helpers.py create --name my-pod --gpu "NVIDIA RTX A4000"

# Terminate a pod
python runpod_helpers.py terminate <pod_id>

# Terminate all pods
python runpod_helpers.py terminate-all

# Terminate stale pods (older than 40 minutes)
python runpod_helpers.py terminate-stale --max-age 40
```

## How it works

1. **SSH key**: Auto-generates an ed25519 key pair (or uses yours)
2. **Pod creation**: Injects the public key via `PUBLIC_KEY` env var so the pod trusts it
3. **SSH readiness**: Polls the API until the pod has a public IP + SSH port mapping, then pings SSH
4. **Upload**: Uses `scp` to copy your training code into `/workspace/code/`
5. **Execute**: Runs your training script over SSH
6. **Download**: Copies `/workspace/outputs/` back to your machine via `scp`
7. **Terminate**: Deletes the pod via REST API (stops billing immediately)

## GPU Fallback

When `auto_fallback=True`, the system tries GPUs from cheapest to most expensive:

1. RTX 2000 Ada → RTX A2000 → RTX 3070 → RTX 3080 → RTX A4000 → ...
2. For each GPU, tries: spot+community → spot+secure → on-demand+community → on-demand+secure

## Adapting for your challenge

Replace `training_code/train_mnist.py` with your actual training script. The script should:

1. Save outputs to `/workspace/outputs/` (model weights, metrics, etc.)
2. Complete within your time limit (10 minutes)

Then run:

```bash
python main.py --code-dir ./your_training_code --script your_script.py --gpu "NVIDIA H100 80GB HBM3"
```
