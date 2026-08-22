#!/usr/bin/env python3
"""One-shot exact-object publisher for NousResearch/hermes-agent#91906."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

FORK_REPO = "andrexibiza/hermes-agent"
UPSTREAM_REPO = "NousResearch/hermes-agent"
TARGET_BRANCH = "restack/pr89479-current-main"
EXPECTED_TARGET_HEAD = "18eb288092b3230911454e822f10d93aef190d91"
VERIFIED_BASE = "7d6db4efb885856078e4d19f804035226df81e0d"
ARTIFACT_ID = "9469099811"
ARTIFACT_ZIP_SHA256 = "ba96d0e59014e0836f1fa0ccf5e3f1a81342f35c07bbb3e5a99e9f2fe27593f4"

EXPECTED_FILES = {
    ".npmrc": "46e922c1fc1dd16ee9a1623a2b76d1f4e4a11ab03f225803beed058356535eda",
    "package.json": "9cd7dd3823c2af968b66178efc7dc7f489ac9e72fd6dbaaa7e875494a6cd7e4d",
    "package-lock.json": "025cdcc245dbccda191c41c001a57f6aa9dfcaefa4ccc19aed418f28d918e84a",
    "uv.lock": "c72c4fe0e636f971564432599f60257e040b0d3396061fe733610575a48aaaa3",
    "apps/desktop/package.json": "56f4cdb46f65a4a075e570f6acdad1f6c05fb903b37d9d01b1323842ef62bb2f",
    "apps/desktop/scripts/packaged-app-layout.mjs": "9838203f20c16bdf38484cc605458bcd85181bbf2bcc37afbb076a5569a2f54f",
    "apps/desktop/scripts/packaged-app-layout.test.mjs": "b1c90a8f0ca86f499985dd5e9837b27302eaf1af9f56671aabaaa530b4841d36",
    "apps/desktop/scripts/test-desktop.mjs": "9dc343e3e40b6aad5c479f48b879afef8c243a8e3e87727d0f85d438af0cee61",
}
PACKAGE_LOCK_SHA256 = EXPECTED_FILES["package-lock.json"]
UV_LOCK_SHA256 = EXPECTED_FILES["uv.lock"]


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def output(args: list[str], *, cwd: Path | None = None) -> str:
    return run(args, cwd=cwd, capture=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comment(body: str) -> None:
    run(
        ["gh", "pr", "comment", "91906", "--repo", UPSTREAM_REPO, "--body", body],
        check=False,
    )


def run_url() -> str:
    repository = os.environ.get("GITHUB_REPOSITORY", "andrexibiza/kaggle-titanic-competition")
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def assert_toolchain() -> None:
    actual = {
        "node": output(["node", "--version"]),
        "npm": output(["npm", "--version"]),
        "uv": output(["uv", "--version"]),
    }
    if actual["node"] != "v22.22.0" or actual["npm"] != "11.17.0" or not actual["uv"].startswith("uv 0.12.5"):
        raise RuntimeError(f"toolchain mismatch: {actual}")


def clone_and_resolve(root: Path) -> tuple[Path, str]:
    run(["gh", "auth", "setup-git"])
    repo = root / "hermes-agent"
    run(
        [
            "git",
            "clone",
            "--no-tags",
            "--single-branch",
            "--branch",
            "main",
            f"https://github.com/{FORK_REPO}.git",
            str(repo),
        ]
    )
    run(["git", "remote", "add", "upstream", f"https://github.com/{UPSTREAM_REPO}.git"], cwd=repo)
    run(
        [
            "git",
            "fetch",
            "--no-tags",
            "upstream",
            "refs/heads/main:refs/remotes/upstream/main",
        ],
        cwd=repo,
    )
    run(
        [
            "git",
            "fetch",
            "--no-tags",
            "origin",
            f"refs/heads/{TARGET_BRANCH}:refs/remotes/origin/{TARGET_BRANCH}",
        ],
        cwd=repo,
    )
    run(["git", "cat-file", "-e", f"{VERIFIED_BASE}^{{commit}}"], cwd=repo)
    run(["git", "cat-file", "-e", f"{EXPECTED_TARGET_HEAD}^{{commit}}"], cwd=repo)

    remote_target = output(["git", "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}"], cwd=repo).split()[0]
    if remote_target != EXPECTED_TARGET_HEAD:
        raise RuntimeError(
            f"target lease moved: expected {EXPECTED_TARGET_HEAD}, got {remote_target}"
        )

    base_sha = output(["git", "rev-parse", "refs/remotes/upstream/main"], cwd=repo)
    run(["git", "merge-base", "--is-ancestor", VERIFIED_BASE, base_sha], cwd=repo)
    print(f"Live upstream authority: {base_sha}")
    return repo, base_sha


def reject_dependency_drift(repo: Path, base_sha: str) -> None:
    changed = [
        line
        for line in output(
            ["git", "diff", "--name-only", VERIFIED_BASE, base_sha], cwd=repo
        ).splitlines()
        if line
    ]
    exact = {
        ".npmrc",
        "package.json",
        "package-lock.json",
        "uv.lock",
        "pyproject.toml",
        "uv.toml",
        *EXPECTED_FILES.keys(),
    }
    patterns = (
        re.compile(r"(^|/)package(?:-lock)?\.json$"),
        re.compile(r"(^|/)(?:pyproject\.toml|uv\.toml|uv\.lock)$"),
        re.compile(r"(^|/)(?:requirements|constraints)[^/]*\.(?:txt|in)$"),
    )
    blocked = sorted(
        path
        for path in changed
        if path in exact or any(pattern.search(path) for pattern in patterns)
    )
    print("Changes since verified base:")
    print("\n".join(changed) if changed else "(none)")
    if blocked:
        raise RuntimeError(
            "dependency inputs moved after verification; regenerate instead:\n"
            + "\n".join(blocked)
        )


def download_and_verify_artifact(root: Path) -> Path:
    archive = root / "dependency-security-restack.zip"
    with archive.open("wb") as handle:
        print(f"+ gh api /repos/{UPSTREAM_REPO}/actions/artifacts/{ARTIFACT_ID}/zip")
        subprocess.run(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"/repos/{UPSTREAM_REPO}/actions/artifacts/{ARTIFACT_ID}/zip",
            ],
            check=True,
            stdout=handle,
        )
    actual_zip = sha256(archive)
    if actual_zip != ARTIFACT_ZIP_SHA256:
        raise RuntimeError(
            f"artifact ZIP digest mismatch: expected {ARTIFACT_ZIP_SHA256}, got {actual_zip}"
        )

    payload = root / "payload"
    payload.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(payload)

    actual_files = {
        path.relative_to(payload).as_posix()
        for path in payload.rglob("*")
        if path.is_file()
    }
    expected_files = set(EXPECTED_FILES) | {"SHA256SUMS"}
    if actual_files != expected_files:
        raise RuntimeError(
            f"artifact file set mismatch: expected {sorted(expected_files)}, "
            f"got {sorted(actual_files)}"
        )
    for relative, expected in EXPECTED_FILES.items():
        actual = sha256(payload / relative)
        if actual != expected:
            raise RuntimeError(
                f"{relative}: expected sha256 {expected}, got {actual}"
            )
    print("Exact eight-file artifact graph verified.")
    return payload


def materialize_and_verify(repo: Path, payload: Path, base_sha: str) -> None:
    run(["git", "checkout", "--detach", base_sha], cwd=repo)
    for relative in EXPECTED_FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload / relative, target)

    verifier = repo / ".github/scripts/restack_dependency_security.py"
    if verifier.exists():
        raise RuntimeError(f"temporary verifier path unexpectedly exists on main: {verifier}")
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text(
        output(
            [
                "git",
                "show",
                f"{EXPECTED_TARGET_HEAD}:.github/scripts/restack_dependency_security.py",
            ],
            cwd=repo,
        )
        + "\n",
        encoding="utf-8",
    )

    run(["npm", "install", "--package-lock-only", "--ignore-scripts", "--include=optional"], cwd=repo)
    run(["npm", "update", "nanoid", "--package-lock-only", "--ignore-scripts", "--include=optional"], cwd=repo)
    if sha256(repo / "package-lock.json") != PACKAGE_LOCK_SHA256:
        raise RuntimeError("package-lock.json changed under the pinned regeneration command")

    run(["uv", "lock", "--upgrade-package", "h2==4.4.1"], cwd=repo)
    if sha256(repo / "uv.lock") != UV_LOCK_SHA256:
        raise RuntimeError("uv.lock changed under the pinned regeneration command")

    run([sys.executable, str(verifier), "verify"], cwd=repo)
    verifier.unlink()
    run(["node", "--test", "apps/desktop/scripts/packaged-app-layout.test.mjs"], cwd=repo)
    run(["uv", "lock", "--check"], cwd=repo)
    run(["git", "diff", "--check"], cwd=repo)
    run(["npm", "audit", "--workspaces=false", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "web", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "ui-tui", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "apps/desktop", "--audit-level=high"], cwd=repo)


def commit_and_publish(repo: Path, base_sha: str) -> str:
    expected = sorted(EXPECTED_FILES)
    run(["git", "add", "--", *expected], cwd=repo)
    staged = sorted(
        line
        for line in output(["git", "diff", "--cached", "--name-only"], cwd=repo).splitlines()
        if line
    )
    if staged != expected:
        raise RuntimeError(f"staged file set mismatch: expected {expected}, got {staged}")
    if output(["git", "diff", "--name-only"], cwd=repo):
        raise RuntimeError("unexpected unstaged tracked changes remain")
    if output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo):
        raise RuntimeError("unexpected untracked files remain")

    run(
        [
            "git",
            "fetch",
            "--no-tags",
            "upstream",
            "refs/heads/main:refs/remotes/upstream/main",
        ],
        cwd=repo,
    )
    current_base = output(["git", "rev-parse", "refs/remotes/upstream/main"], cwd=repo)
    if current_base != base_sha:
        raise RuntimeError(f"upstream main moved during verification: {base_sha} -> {current_base}")
    current_target = output(
        ["git", "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}"], cwd=repo
    ).split()[0]
    if current_target != EXPECTED_TARGET_HEAD:
        raise RuntimeError(
            f"target branch moved during verification: {EXPECTED_TARGET_HEAD} -> {current_target}"
        )

    run(["git", "diff", "--cached", "--check"], cwd=repo)
    run(["git", "config", "user.name", "Axl Ibiza, MBA"], cwd=repo)
    run(["git", "config", "user.email", "andrexibiza@gmail.com"], cwd=repo)
    message = (
        "Restack the active Electron, Nano ID, and h2 advisory remediation on the "
        "exact current main object while preserving the React Compiler graph and "
        "source attribution.\n\n"
        "Co-authored-by: schmitzi8 <schmitzi8@users.noreply.github.com>\n"
        "Co-authored-by: orcaspainting-dev "
        "<264355715+orcaspainting-dev@users.noreply.github.com>"
    )
    run(
        [
            "git",
            "commit",
            "-m",
            "fix(deps): close current advisory graph on current main",
            "-m",
            message,
        ],
        cwd=repo,
    )
    head_sha = output(["git", "rev-parse", "HEAD"], cwd=repo)
    parent = output(["git", "rev-parse", "HEAD^"], cwd=repo)
    if parent != base_sha:
        raise RuntimeError(f"commit parent mismatch: expected {base_sha}, got {parent}")

    run(
        [
            "git",
            "push",
            f"--force-with-lease=refs/heads/{TARGET_BRANCH}:{EXPECTED_TARGET_HEAD}",
            "origin",
            f"HEAD:refs/heads/{TARGET_BRANCH}",
        ],
        cwd=repo,
    )
    remote = output(["git", "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}"], cwd=repo).split()[0]
    if remote != head_sha:
        raise RuntimeError(f"remote publication mismatch: expected {head_sha}, got {remote}")
    return head_sha


def main() -> int:
    try:
        if not os.environ.get("GH_TOKEN"):
            raise RuntimeError("GH_TOKEN is unavailable")
        assert_toolchain()
        with tempfile.TemporaryDirectory(prefix="hermes-dependency-security-") as temp:
            root = Path(temp)
            repo, base_sha = clone_and_resolve(root)
            reject_dependency_drift(repo, base_sha)
            payload = download_and_verify_artifact(root)
            materialize_and_verify(repo, payload, base_sha)
            head_sha = commit_and_publish(repo, base_sha)

        body = (
            f"Branch truth is now published at `{head_sha}`, parented directly on live "
            f"upstream main `{base_sha}`. Exact generated lock digests: "
            f"`package-lock.json {PACKAGE_LOCK_SHA256}`; `uv.lock {UV_LOCK_SHA256}`. "
            "The publisher re-ran the pinned npm/uv generation, semantic verifier, "
            "x64/ARM64 layout regression, lock checks, diff checks, and all four npm "
            f"audit scopes before the lease-guarded push. Publication run: {run_url()}. "
            "Draft remains until exact-head upstream CI, Docker, Nix, and OSV evidence complete."
        )
        comment(body)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            Path(summary).write_text(
                f"# Published #91906\n\n- head: `{head_sha}`\n- parent: `{base_sha}`\n",
                encoding="utf-8",
            )
        return 0
    except Exception as exc:
        message = (
            "The exact-object publisher stopped at a verification or authority gate "
            f"before moving branch truth. Run: {run_url()}. Error: `{type(exc).__name__}: "
            f"{str(exc)[:1200]}`."
        )
        print(message, file=sys.stderr)
        try:
            comment(message)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
