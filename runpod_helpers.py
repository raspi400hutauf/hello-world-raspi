"""
runpod_helpers.py — Programmatic RunPod pod management + SSH operations.

Functions:
    list_pods()                  — List all running pods
    create_pod(...)              — Create a new pod (on-demand or spot)
    get_pod(pod_id)              — Get details for a specific pod
    get_pod_ssh_info(pod_id)     — Wait for pod to be ready, return SSH host/port
    terminate_pod(pod_id)        — Terminate a specific pod
    terminate_all_pods()         — Terminate every running pod
    terminate_stale_pods(max_age_minutes=40) — Terminate pods older than X minutes
    ssh_upload(host, port, key, local, remote) — SCP upload
    ssh_download(host, port, key, remote, local) — SCP download
    ssh_exec(host, port, key, cmd) — Run a command via SSH
    run_training(...)            — Full pipeline: create pod → upload → train → download → terminate
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_REST_URL = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Default pod settings
DEFAULT_GPU_TYPE = "NVIDIA RTX 2000 Ada Generation"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_CONTAINER_DISK_GB = 20
DEFAULT_VOLUME_GB = 20
DEFAULT_CLOUD_TYPE = "COMMUNITY"  # SECURE or COMMUNITY (REST API only supports these two)


def _headers():
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

def list_pods(desired_status: Optional[str] = None) -> list[dict]:
    """
    List all pods. Optionally filter by desiredStatus (RUNNING, EXITED, TERMINATED).
    Returns a list of pod dicts.
    """
    params = {}
    if desired_status:
        params["desiredStatus"] = desired_status
    r = requests.get(f"{RUNPOD_REST_URL}/pods", headers=_headers(), params=params)
    r.raise_for_status()
    return r.json()


def get_pod(pod_id: str) -> dict:
    """Get full details for a single pod."""
    r = requests.get(
        f"{RUNPOD_REST_URL}/pods/{pod_id}",
        headers=_headers(),
        params={"includeMachine": "true"},
    )
    r.raise_for_status()
    return r.json()


# Cheap-to-expensive GPU fallback order for automatic retries
GPU_FALLBACK_ORDER = [
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX A2000",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
]


def _try_create_pod(body: dict) -> dict:
    """Attempt to create a pod with the given body. Returns pod dict or raises."""
    r = requests.post(f"{RUNPOD_REST_URL}/pods", headers=_headers(), json=body)
    data = r.json()
    if r.status_code in (200, 201) and isinstance(data, dict) and "id" in data:
        return data
    error_msg = data.get("error", r.text) if isinstance(data, dict) else r.text
    raise RuntimeError(f"HTTP {r.status_code}: {error_msg}")


def create_pod(
    name: str = "training-pod",
    gpu_type: str = DEFAULT_GPU_TYPE,
    image: str = DEFAULT_IMAGE,
    gpu_count: int = 1,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    spot: bool = True,
    ssh_public_key: Optional[str] = None,
    bid_per_gpu: Optional[float] = None,
    auto_fallback: bool = False,
) -> dict:
    """
    Create a new pod via REST API.

    Args:
        spot: If True, create an interruptible (spot) pod — much cheaper.
        ssh_public_key: SSH public key string. If provided, injected as PUBLIC_KEY env var
                        so the pod's SSH daemon trusts it.
        bid_per_gpu: Max bid per GPU/hr for spot. If None, RunPod picks the market rate.
        auto_fallback: If True, try multiple GPU types and cloud configs until one works.

    Returns:
        Pod dict from the API (contains 'id', 'publicIp', 'portMappings', etc.)
    """
    env = {}
    if ssh_public_key:
        env["PUBLIC_KEY"] = ssh_public_key

    def _build_body(gpu: str, cloud: str, interruptible: bool) -> dict:
        return {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [gpu],
            "gpuCount": gpu_count,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "cloudType": cloud,
            "supportPublicIp": True,
            "ports": ["22/tcp"],
            "interruptible": interruptible,
            "env": env,
        }

    if not auto_fallback:
        # Single attempt with exact parameters
        body = _build_body(gpu_type, cloud_type, spot)
        return _try_create_pod(body)

    # Auto-fallback: try the requested GPU first, then fall through the list
    gpu_list = [gpu_type] + [g for g in GPU_FALLBACK_ORDER if g != gpu_type]
    # Try combos: (spot+COMMUNITY), (spot+SECURE), (on-demand+COMMUNITY), (on-demand+SECURE)
    configs = []
    if spot:
        configs += [(True, "COMMUNITY"), (True, "SECURE")]
    configs += [(False, "COMMUNITY"), (False, "SECURE")]

    last_error = None
    for gpu in gpu_list:
        for interruptible, cloud in configs:
            mode = "spot" if interruptible else "on-demand"
            try:
                body = _build_body(gpu, cloud, interruptible)
                pod = _try_create_pod(body)
                print(f"  ✓ Created {mode} pod with {gpu} on {cloud} cloud")
                return pod
            except RuntimeError as e:
                last_error = e
                # Only print if it's a real attempt (not "no spot price" type errors)
                continue

    raise RuntimeError(f"No GPU available after trying all fallback options. Last error: {last_error}")


def terminate_pod(pod_id: str) -> bool:
    """Terminate (permanently delete) a single pod. Returns True on success."""
    r = requests.delete(f"{RUNPOD_REST_URL}/pods/{pod_id}", headers=_headers())
    if r.status_code in (200, 204, 404):
        return True
    r.raise_for_status()
    return False


def terminate_all_pods() -> list[str]:
    """Terminate every pod in the account. Returns list of terminated pod IDs."""
    pods = list_pods()
    terminated = []
    for pod in pods:
        pid = pod["id"]
        try:
            terminate_pod(pid)
            terminated.append(pid)
            print(f"  Terminated pod {pid} ({pod.get('name', '?')})")
        except Exception as e:
            print(f"  Failed to terminate pod {pid}: {e}")
    return terminated


def terminate_stale_pods(max_age_minutes: int = 40) -> list[str]:
    """
    Terminate all pods whose uptime exceeds max_age_minutes.
    Uses lastStartedAt from the REST API to compute age.
    Returns list of terminated pod IDs.
    """
    pods = list_pods()
    now = datetime.now(timezone.utc)
    terminated = []

    for pod in pods:
        pid = pod["id"]
        started_str = pod.get("lastStartedAt")
        if not started_str:
            print(f"  Pod {pid}: no lastStartedAt — skipping")
            continue

        # Parse ISO timestamp
        started_at = datetime.fromisoformat(started_str.replace("Z", "+00:00"))
        age_minutes = (now - started_at).total_seconds() / 60.0

        print(f"  Pod {pid} ({pod.get('name', '?')}): age={age_minutes:.1f}min", end="")
        if age_minutes > max_age_minutes:
            try:
                terminate_pod(pid)
                terminated.append(pid)
                print(" → TERMINATED")
            except Exception as e:
                print(f" → FAILED to terminate: {e}")
        else:
            print(" → OK")

    return terminated


# ---------------------------------------------------------------------------
# SSH key management
# ---------------------------------------------------------------------------

def ensure_ssh_key(key_path: Optional[str] = None) -> tuple[str, str]:
    """
    Ensure an SSH key pair exists. Returns (private_key_path, public_key_string).
    If key_path is given, use that. Otherwise generate a temporary one.
    """
    if key_path and os.path.exists(key_path):
        pub_path = key_path + ".pub"
        if not os.path.exists(pub_path):
            raise FileNotFoundError(f"Public key not found: {pub_path}")
        with open(pub_path) as f:
            pub_key = f.read().strip()
        return key_path, pub_key

    # Generate a new ephemeral key pair
    key_dir = os.path.join(tempfile.gettempdir(), "runpod_ssh")
    os.makedirs(key_dir, exist_ok=True)
    priv = os.path.join(key_dir, "id_ed25519")
    if not os.path.exists(priv):
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", priv, "-N", "", "-q"],
            check=True,
        )
        os.chmod(priv, 0o600)
    with open(priv + ".pub") as f:
        pub_key = f.read().strip()
    return priv, pub_key


# ---------------------------------------------------------------------------
# Wait for pod to be SSH-ready
# ---------------------------------------------------------------------------

def get_pod_ssh_info(pod_id: str, timeout: int = 300, poll_interval: int = 10) -> dict:
    """
    Poll until the pod is RUNNING and has SSH port mapping.
    Returns dict with keys: host, port, pod_id.
    """
    print(f"Waiting for pod {pod_id} to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = get_pod(pod_id)
        status = pod.get("desiredStatus", "")
        public_ip = pod.get("publicIp")
        port_map = pod.get("portMappings") or {}

        if status == "RUNNING" and public_ip and port_map.get("22"):
            ssh_port = port_map["22"]
            print(f"  Pod ready: {public_ip}:{ssh_port}")
            return {"host": public_ip, "port": int(ssh_port), "pod_id": pod_id}

        state_info = f"status={status}, ip={public_ip}, ports={port_map}"
        print(f"  Not ready yet ({state_info}), retrying in {poll_interval}s...")
        time.sleep(poll_interval)

    raise TimeoutError(f"Pod {pod_id} did not become SSH-ready within {timeout}s")


# ---------------------------------------------------------------------------
# SSH operations  (uses subprocess — no paramiko dependency needed)
# ---------------------------------------------------------------------------

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
]


def ssh_exec(host: str, port: int, key_path: str, cmd: str, timeout: int = 600) -> str:
    """Run a command on the pod via SSH. Returns stdout."""
    full_cmd = [
        "ssh", *_SSH_OPTS,
        "-i", key_path,
        "-p", str(port),
        f"root@{host}",
        cmd,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}):\n"
            f"  cmd: {cmd}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout


def ssh_upload(host: str, port: int, key_path: str, local_path: str, remote_path: str):
    """Upload a file or directory to the pod via SCP."""
    scp_flags = ["-r"] if os.path.isdir(local_path) else []
    full_cmd = [
        "scp", *_SSH_OPTS, *scp_flags,
        "-i", key_path,
        "-P", str(port),
        local_path,
        f"root@{host}:{remote_path}",
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"SCP upload failed:\n  {result.stderr.strip()}")
    print(f"  Uploaded {local_path} → {remote_path}")


def ssh_download(host: str, port: int, key_path: str, remote_path: str, local_path: str):
    """Download a file or directory from the pod via SCP."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    full_cmd = [
        "scp", *_SSH_OPTS, "-r",
        "-i", key_path,
        "-P", str(port),
        f"root@{host}:{remote_path}",
        local_path,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"SCP download failed:\n  {result.stderr.strip()}")
    print(f"  Downloaded {remote_path} → {local_path}")


def wait_ssh_ready(host: str, port: int, key_path: str, retries: int = 12, delay: int = 10):
    """
    Wait until SSH is actually accepting connections (pod networking can lag behind API status).
    """
    for i in range(retries):
        try:
            out = ssh_exec(host, port, key_path, "echo SSH_OK", timeout=20)
            if "SSH_OK" in out:
                print("  SSH connection established.")
                return
        except Exception:
            pass
        print(f"  SSH not ready yet, retry {i+1}/{retries}...")
        time.sleep(delay)
    raise TimeoutError("SSH never became reachable")


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training(
    training_code_dir: str,
    train_script: str = "train_mnist.py",
    output_dir: str = "./outputs",
    gpu_type: str = DEFAULT_GPU_TYPE,
    spot: bool = True,
    ssh_key_path: Optional[str] = None,
    pod_name: str = "training-pod",
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    image: str = DEFAULT_IMAGE,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    auto_fallback: bool = True,
) -> dict:
    """
    End-to-end training pipeline:
      1. Generate/load SSH key
      2. Create a RunPod pod (spot by default)
      3. Wait for SSH
      4. Upload training code
      5. Execute training script
      6. Download outputs
      7. Terminate pod

    Returns a dict with metrics and paths.
    """
    pod_id = None
    try:
        # 1. SSH key
        print("\n[1/7] Preparing SSH key...")
        key_path, pub_key = ensure_ssh_key(ssh_key_path)
        print(f"  Using key: {key_path}")

        # 2. Create pod
        print(f"\n[2/7] Creating {'spot' if spot else 'on-demand'} pod ({gpu_type})...")
        pod = create_pod(
            name=pod_name,
            gpu_type=gpu_type,
            spot=spot,
            ssh_public_key=pub_key,
            cloud_type=cloud_type,
            image=image,
            container_disk_gb=container_disk_gb,
            volume_gb=volume_gb,
            auto_fallback=auto_fallback,
        )
        pod_id = pod["id"]
        print(f"  Pod created: {pod_id}")

        # 3. Wait for SSH
        print("\n[3/7] Waiting for pod to become SSH-ready...")
        ssh = get_pod_ssh_info(pod_id, timeout=300)
        host, port = ssh["host"], ssh["port"]
        wait_ssh_ready(host, port, key_path)

        # 4. Upload code
        print("\n[4/7] Uploading training code...")
        ssh_exec(host, port, key_path, "mkdir -p /workspace/code")
        ssh_upload(host, port, key_path, training_code_dir, "/workspace/code/")

        # 5. Run training
        print("\n[5/7] Running training script...")
        remote_script = f"/workspace/code/{os.path.basename(training_code_dir)}/{train_script}"
        output = ssh_exec(
            host, port, key_path,
            f"cd /workspace && python {remote_script}",
            timeout=1200,  # 20 min hard cap
        )
        print(output)

        # 6. Download outputs
        print("\n[6/7] Downloading outputs...")
        os.makedirs(output_dir, exist_ok=True)
        # SCP the contents: list remote files, download each
        file_list = ssh_exec(host, port, key_path, "ls /workspace/outputs/").strip().split()
        for fname in file_list:
            ssh_download(host, port, key_path, f"/workspace/outputs/{fname}", os.path.join(output_dir, fname))

        # 7. Terminate
        print("\n[7/7] Terminating pod...")
        terminate_pod(pod_id)
        print(f"  Pod {pod_id} terminated.")

        # Load metrics if available
        metrics_file = os.path.join(output_dir, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)

        return {
            "success": True,
            "pod_id": pod_id,
            "output_dir": output_dir,
            "metrics": metrics,
        }

    except Exception as e:
        print(f"\n*** ERROR: {e}")
        # Safety: always try to terminate the pod
        if pod_id:
            print(f"  Cleaning up pod {pod_id}...")
            try:
                terminate_pod(pod_id)
                print(f"  Pod {pod_id} terminated.")
            except Exception as te:
                print(f"  WARNING: Failed to terminate pod: {te}")
        return {
            "success": False,
            "pod_id": pod_id,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _print_pod_table(pods):
    """Pretty-print a list of pods."""
    if not pods:
        print("  No pods found.")
        return
    print(f"  {'ID':<20} {'Name':<25} {'Status':<12} {'GPU':<35} {'Cost/hr':<10} {'Started'}")
    print(f"  {'─'*20} {'─'*25} {'─'*12} {'─'*35} {'─'*10} {'─'*25}")
    for p in pods:
        gpu_name = ""
        if p.get("gpu"):
            gpu_name = p["gpu"].get("displayName", "")
        print(
            f"  {p['id']:<20} "
            f"{(p.get('name') or '?'):<25} "
            f"{p.get('desiredStatus', '?'):<12} "
            f"{gpu_name:<35} "
            f"${p.get('costPerHr', '?'):<9} "
            f"{p.get('lastStartedAt', '?')}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RunPod pod management helpers")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all pods")

    # create
    p_create = sub.add_parser("create", help="Create a pod")
    p_create.add_argument("--name", default="training-pod")
    p_create.add_argument("--gpu", default=DEFAULT_GPU_TYPE)
    p_create.add_argument("--spot", action="store_true", default=True)
    p_create.add_argument("--on-demand", action="store_true")

    # terminate
    p_term = sub.add_parser("terminate", help="Terminate a pod by ID")
    p_term.add_argument("pod_id")

    # terminate-all
    sub.add_parser("terminate-all", help="Terminate all pods")

    # terminate-stale
    p_stale = sub.add_parser("terminate-stale", help="Terminate pods older than X minutes")
    p_stale.add_argument("--max-age", type=int, default=40)

    # train
    p_train = sub.add_parser("train", help="Run full training pipeline")
    p_train.add_argument("--code-dir", required=True, help="Path to training code directory")
    p_train.add_argument("--script", default="train_mnist.py")
    p_train.add_argument("--output-dir", default="./outputs")
    p_train.add_argument("--gpu", default=DEFAULT_GPU_TYPE)
    p_train.add_argument("--spot", action="store_true", default=True)
    p_train.add_argument("--on-demand", action="store_true")

    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable.")
        sys.exit(1)

    if args.command == "list":
        pods = list_pods()
        _print_pod_table(pods)

    elif args.command == "create":
        spot = not args.on_demand
        pod = create_pod(name=args.name, gpu_type=args.gpu, spot=spot)
        print(f"Created pod: {pod['id']}")

    elif args.command == "terminate":
        terminate_pod(args.pod_id)
        print(f"Terminated pod: {args.pod_id}")

    elif args.command == "terminate-all":
        terminated = terminate_all_pods()
        print(f"Terminated {len(terminated)} pod(s).")

    elif args.command == "terminate-stale":
        terminated = terminate_stale_pods(args.max_age)
        print(f"Terminated {len(terminated)} stale pod(s).")

    elif args.command == "train":
        spot = not args.on_demand
        result = run_training(
            training_code_dir=args.code_dir,
            train_script=args.script,
            output_dir=args.output_dir,
            gpu_type=args.gpu,
            spot=spot,
        )
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
