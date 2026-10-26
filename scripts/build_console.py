"""Install locked frontend dependencies, check, test and build shipped assets."""
from pathlib import Path
import shutil
import subprocess


def main():
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("Node.js and npm are required to build the console.")
    subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "check"], cwd=frontend, check=True)


if __name__ == "__main__":
    main()
