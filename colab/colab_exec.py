#!/usr/bin/env python3
"""
Execute a shell command on a persistent Google Colab GPU session.

Usage:
    python3 colab/colab_exec.py --cmd "python3 -m pytest tests/ -v"
    python3 colab/colab_exec.py --cmd "python3 benchmarks/colab_benchmark.py"
    python3 colab/colab_exec.py --cmd "python3 -c \"print('hello')\""

Manages a persistent session named 'lp-autoresearch'.
Creates it on first use with a T4 GPU; reuses it on subsequent calls.
"""

import subprocess, sys, os, time, argparse, textwrap

COLAB_SESSION = "lp-autoresearch"
REPO_URL = "https://github.com/msprengholz/lp-transformation.git"
REPO_DIR = "/content/lp-transformation"
UVX_BASE = ["uvx", "--from", "git+https://github.com/monatis/google-colab-cli",
            "colab", "--auth", "oauth2"]


def log(msg):
    print(f"[colab-exec] {msg}", file=sys.stderr)


def colab(*args, timeout=120, input_text=None):
    """Run a colab-cli command and return CompletedProcess."""
    cmd = UVX_BASE + list(args)
    log(f"Running: colab {' '.join(args)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       input=input_text)
    if r.returncode != 0 and not args[0] in ("sessions", "status"):
        log(f"colab stderr: {r.stderr[:300]}")
    return r


def ensure_session():
    """Create the GPU session if it doesn't already exist."""
    r = colab("sessions", timeout=30)
    if COLAB_SESSION in r.stdout:
        log(f"Session '{COLAB_SESSION}' already exists")
        return True
    log(f"Creating session '{COLAB_SESSION}' with T4 GPU...")
    r = colab("new", "-s", COLAB_SESSION, "--gpu", "T4", timeout=120)
    if r.returncode != 0:
        log(f"Failed to create session: {r.stderr[:300]}")
        return False
    log("Session created")
    # Ensure base packages are installed (first run only)
    colab("install", "-s", COLAB_SESSION, "-q", "numpy", "pytest", timeout=120)
    return True


def ensure_repo():
    """Clone or pull the repo on the Colab VM."""
    # Check if repo exists
    check = colab("exec", "-s", COLAB_SESSION, timeout=30,
                  input_text=f"import os; print(os.path.isdir('{REPO_DIR}'))")
    if "True" in check.stdout:
        log("Repo exists, pulling...")
        colab("exec", "-s", COLAB_SESSION, timeout=30,
              input_text=f"import subprocess; subprocess.run(['git', '-C', '{REPO_DIR}', 'pull'], capture_output=True)")
    else:
        log("Cloning repo...")
        colab("exec", "-s", COLAB_SESSION, timeout=60,
              input_text=f"import subprocess; subprocess.run(['git', 'clone', '{REPO_URL}', '{REPO_DIR}'], capture_output=True)")


def run_command(cmd, timeout=600):
    """Run a command on the Colab VM and return (returncode, stdout, stderr)."""
    # Escape the command for Python string
    escaped_cmd = cmd.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    
    # Write command to a temp file on Colab
    setup = textwrap.dedent(f'''
    import subprocess, sys, os
    os.chdir("{REPO_DIR}")
    sys.path.insert(0, ".")
    cmd = """{cmd}"""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout={timeout})
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr, file=sys.stderr)
    sys.exit(r.returncode)
    ''').strip()
    
    r = colab("exec", "-s", COLAB_SESSION, timeout=timeout + 30,
              input_text=setup)
    return r.returncode, r.stdout, r.stderr


def main():
    parser = argparse.ArgumentParser(description="Run command on Colab GPU session")
    parser.add_argument("--cmd", required=True, help="Shell command to run")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Command timeout in seconds")
    args = parser.parse_args()

    if not ensure_session():
        sys.exit(1)
    ensure_repo()
    
    code, out, err = run_command(args.cmd, timeout=args.timeout)
    print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
