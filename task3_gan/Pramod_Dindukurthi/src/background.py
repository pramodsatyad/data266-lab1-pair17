"""Detach a training command from the notebook; retain logs and exit status."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


@contextmanager
def run_lock(path):
    # Kernel-managed lock is released even if the supervisor crashes.
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def write_status(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "status", "worker"])
    parser.add_argument("--run-dir", required=True)
    args, command = parser.parse_known_args()
    if command[:1] == ["--"]:
        command = command[1:]
    folder = Path(args.run_dir).resolve()/"reproducibility"/"background"
    folder.mkdir(parents=True, exist_ok=True)
    lock, status, log = folder/"run.lock", folder/"status.json", folder/"console.log"
    if args.action == "status":
        try:
            with run_lock(lock):
                active = False
        except OSError:
            active = True
        print("Background supervisor active:", active)
        print(status.read_text() if status.exists() else "No recorded job.")
        if not active and status.exists() and json.loads(status.read_text()).get("state") == "running":
            print("Supervisor exited unexpectedly. Inspect logs/checkpoint before resuming.")
        if log.exists():
            with open(log, "rb") as f:
                f.seek(max(0, log.stat().st_size - 12000))
                print(f.read().decode(errors="replace"))
        return 0
    if not command:
        parser.error("Provide the training command after --")
    if args.action == "launch":
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        with open(log, "ab", buffering=0) as output:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                 "--run-dir", str(Path(args.run_dir).resolve()), "--", *command],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                close_fds=True, **options)
        try:
            result = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            result = 0
        print("Supervisor PID:", process.pid, "Log:", log)
        print("Check status and confirm checkpoint progress before leaving.")
        return result
    try:
        with run_lock(lock):
            info = {"state": "running", "supervisor_pid": os.getpid(),
                    "started_utc": datetime.now(timezone.utc).isoformat(), "command": command}
            write_status(status, info)
            try:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
                info["training_pid"] = child.pid
                write_status(status, info)
                result = child.wait()
            except Exception as exc:
                info["error"] = repr(exc)
                result = 1
            info.update(state="finished" if result == 0 else "failed", exit_code=result,
                        ended_utc=datetime.now(timezone.utc).isoformat())
            write_status(status, info)
            return result
    except OSError as exc:
        print("Cannot acquire background run lock; another job may be running:", exc, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
