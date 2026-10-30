#!/usr/bin/env python3
"""One-file DLEF project bootstrap.

Use it on any machine with Python 3.10+:

    python init_dlef.py
    python init_dlef.py ~/code/my-project --project my-project
    python init_dlef.py ~/code/my-project --source /path/to/DLEF
    python init_dlef.py ~/code/my-project --repo git+ssh://git@github.com/AceWen-dev/GDLF.git

It installs the `dlef` core if needed, then runs `dlef init`. It does not
install torch/tensorboard unless you pass --install-project.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ask(prompt: str, default: str) -> str:
    try:
        answer = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return answer or default


def confirm(prompt: str, default: bool = False) -> bool:
    marker = "Y/n" if default else "y/N"
    try:
        answer = input(f"{prompt} [{marker}]: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(str(item) for item in argv))
    return subprocess.run(argv, cwd=cwd, env=env, check=False).returncode


def dlef_importable() -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", "import dlef"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def local_repo() -> Path | None:
    candidate = Path(__file__).resolve().parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "dlef").is_dir():
        return candidate
    return None


def install_dlef(source: Path | None, repo: str, ref: str) -> int:
    pip = [sys.executable, "-m", "pip", "install"]
    if source is not None:
        return run([*pip, "-e", str(source)])
    if repo.startswith("git@"):
        host, path = repo.split(":", 1)
        repo = f"git+ssh://{host}/{path}"
    spec = repo if repo.startswith("git+") else f"git+{repo}"
    if ref:
        spec = f"{spec}@{ref}"
    return run([*pip, spec])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="new project directory")
    parser.add_argument("--project", help="project id (default: target directory name)")
    parser.add_argument(
        "--source",
        help="local DLEF repository; install it editable instead of downloading",
    )
    parser.add_argument(
        "--repo",
        default="https://github.com/AceWen-dev/GDLF.git",
        help="DLEF git URL (default: GitHub HTTPS)",
    )
    parser.add_argument("--ref", default="main", help="git branch/tag/commit")
    parser.add_argument("--template", help="explicit template directory")
    parser.add_argument("--data-root", help="DLEF_DATA_ROOT value to print/use")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument("--install-project", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--yes", action="store_true", help="non-interactive defaults")
    args = parser.parse_args(argv)

    target_raw = args.target
    if not target_raw:
        target_raw = "./my-project" if args.yes else ask("Project directory", "./my-project")
    target = Path(target_raw).expanduser().resolve()
    project = args.project or target.name
    if not args.yes and not args.project:
        project = ask("Project id", project)

    if target.exists() and not target.is_dir():
        print(f"destination is not a directory: {target}", file=sys.stderr)
        return 2
    if target.exists() and any(target.iterdir()) and not args.force:
        if args.yes or not confirm(f"{target} is not empty; write into it?", False):
            print(f"destination is not empty: {target}; pass --force to overwrite", file=sys.stderr)
            return 2
        args.force = True

    source = Path(args.source).expanduser().resolve() if args.source else local_repo()
    if args.skip_install:
        if not dlef_importable():
            print("--skip-install was given but `import dlef` fails", file=sys.stderr)
            return 2
    elif source is not None:
        code = install_dlef(source, args.repo, args.ref)
        if code:
            return code
    elif not dlef_importable():
        code = install_dlef(None, args.repo, args.ref)
        if code:
            print(
                "could not install dlef; pass --source /path/to/DLEF or --repo <git-url>",
                file=sys.stderr,
            )
            return code

    env = dict(os.environ)
    if args.data_root:
        env["DLEF_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    command = [
        sys.executable,
        "-m",
        "dlef.cli",
        "init",
        str(target),
        "--project",
        project,
    ]
    if args.template:
        command.extend(["--template", str(Path(args.template).expanduser().resolve())])
    if args.force:
        command.append("--force")
    if args.no_git or shutil.which("git") is None:
        command.append("--no-git")
    code = run(command, env=env)
    if code:
        return code

    if args.install_project:
        code = run(
            [sys.executable, "-m", "pip", "install", "-e", str(target)],
            env=env,
        )
        if code:
            return code

    print()
    print("done.")
    print(f"  cd {target}")
    if not args.install_project:
        print("  pip install -e .")
    if args.data_root:
        print(f"  export DLEF_DATA_ROOT={env['DLEF_DATA_ROOT']}")
    else:
        print("  export DLEF_DATA_ROOT=/path/to/data")
    print("  # edit PROJECT.yaml, STATE.md, DECISIONS.md")
    print("  make test")
    print("  make audit")
    print("  make check")
    print("  # then tell your coding agent: read AGENTS.md and take over")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
