"""
Execute a shell command on a persistent Google Colab GPU session.

Usage:
    python3 colab/colab_exec.py --cmd "python3 benchmarks/colab_benchmark.py"
    python3 colab/colab_exec.py --cmd "python3 -m pytest tests/ -v"

Manages a persistent session named 'lp-autoresearch' with a T4 GPU.
"""

import subprocess, sys, os, time, argparse, base64

COLAB_SESSION = "lp-autoresearch"
REPO_URL = "https://github.com/msprengholz/lp-transformation.git"
REPO_DIR = "/content/lp-transformation"
UVX_BASE = ["colab", "--auth", "oauth2"]


def log(msg):
    print(f"[colab-exec] {msg}", file=sys.stderr)


def colab(*args, timeout=120, input_text=None):
    cmd = UVX_BASE + list(args)
    log(f"colab {' '.join(args)[:120]}")
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       input=input_text, encoding='utf-8', errors='replace',
                       env=env)
    return r


def ensure_session():
    """Create GPU session if needed, with retry on capacity errors."""
    r = colab("sessions", timeout=30)
    if COLAB_SESSION in r.stdout:
        log(f"Session '{COLAB_SESSION}' ready")
        return True
    
    for attempt in range(5):
        log(f"Creating session '{COLAB_SESSION}' (T4 GPU)...")
        r = colab("new", "-s", COLAB_SESSION, "--gpu", "T4", timeout=120)
        if r.returncode == 0:
            colab("install", "-s", COLAB_SESSION, "numpy", "pytest", timeout=120)
            return True
        if "Service Unavailable" in r.stderr:
            wait = 30 * (attempt + 1)
            log(f"GPU capacity full, retrying in {wait}s...")
            time.sleep(wait)
        else:
            log(f"FAILED: {r.stderr[:200]}")
            return False
    log("All GPU retries exhausted, falling back to CPU")
    r = colab("new", "-s", COLAB_SESSION, timeout=120)  # CPU runtime
    if r.returncode == 0:
        colab("install", "-s", COLAB_SESSION, "numpy", "pytest", timeout=120)
        return True
    return False


def ensure_repo():
    """Clone/pull repo on Colab VM."""
    r = colab("exec", "-s", COLAB_SESSION, timeout=30,
              input_text=f"import os; print(os.path.isdir('{REPO_DIR}'))")
    if "True" in r.stdout:
        log("Repo exists, updating...")
        colab("exec", "-s", COLAB_SESSION, timeout=60,
              input_text=f"import subprocess; subprocess.run(['git', '-C', '{REPO_DIR}', 'pull'], capture_output=True)")
    else:
        log("Cloning repo...")
        colab("exec", "-s", COLAB_SESSION, timeout=120,
              input_text=f"import subprocess; subprocess.run(['git', 'clone', '{REPO_URL}', '{REPO_DIR}'], capture_output=True)")


def run_command(cmd, timeout=600):
    """
    Run a shell command on the Colab VM.
    Returns (returncode, stdout_text).
    Prints stdout as it comes (for real-time METRIC parsing).
    """
    # Base64-encode the command to avoid quoting issues
    cmd_b64 = base64.b64encode(cmd.encode()).decode()
    
    bootstrap = (
        "import subprocess, sys, os, base64; "
        f"os.chdir('{REPO_DIR}'); "
        f"sys.path.insert(0, '.'); "
        f"cmd = base64.b64decode('{cmd_b64}').decode(); "
        "r = subprocess.run(cmd, shell=True, capture_output=True, "
        "timeout=None); "
        "print(r.stdout.decode() if isinstance(r.stdout, bytes) else r.stdout, end=''); "
        "sys.exit(r.returncode)"
    )
    
    r = colab("exec", "-s", COLAB_SESSION, timeout=timeout + 30,
              input_text=bootstrap)
    # Print stdout from Colab so run_experiment can see METRIC lines
    if r.stdout:
        try:
            print(r.stdout, end="")
        except UnicodeEncodeError:
            print(r.stdout.encode('utf-8', errors='replace').decode('utf-8'), end="")
    if r.stderr and "Traceback" in r.stderr:
        try:
            print(r.stderr, end="", file=sys.stderr)
        except UnicodeEncodeError:
            print(r.stderr.encode('utf-8', errors='replace').decode('utf-8'), end="", file=sys.stderr)
    return r.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="Shell command to run on Colab")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not ensure_session():
        sys.exit(1)
    ensure_repo()
    rc = run_command(args.cmd, timeout=args.timeout)
    sys.exit(rc)


if __name__ == "__main__":
    main()
