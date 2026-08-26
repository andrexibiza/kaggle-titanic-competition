#!/usr/bin/env python3
"""Publish the exact current-main dependency product for NousResearch/hermes-agent#91906."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

FORK_REPO = "andrexibiza/hermes-agent"
UPSTREAM_REPO = "NousResearch/hermes-agent"
TARGET_BRANCH = "restack/pr89479-current-main"
BASE_SHA = "f751a8c5467c41500e505d90cb0eb8b70929080f"
EXPECTED_TARGET_HEAD = "dbdda7ff9943ab442a9620a3e18ae90de5c182b6"
MATERIALIZER_RUN = "32921389471"
ARTIFACT_ID = "9589956219"
ARTIFACT_ZIP_SHA256 = "d43b3bf5b0e8ddef74cd660e3334bbb8a9a8cb3833cbe09cad1e5611459f0e18"

EXPECTED_FILES = {
    ".npmrc": "46e922c1fc1dd16ee9a1623a2b76d1f4e4a11ab03f225803beed058356535eda",
    "apps/desktop/package.json": "a5e2e86bd17f87db26f09e172be01db9e1175fbe423b8dad993d93af671e0c7d",
    "apps/desktop/scripts/packaged-app-layout.mjs": "9838203f20c16bdf38484cc605458bcd85181bbf2bcc37afbb076a5569a2f54f",
    "apps/desktop/scripts/packaged-app-layout.test.mjs": "b1c90a8f0ca86f499985dd5e9837b27302eaf1af9f56671aabaaa530b4841d36",
    "apps/desktop/scripts/test-desktop.mjs": "9dc343e3e40b6aad5c479f48b879afef8c243a8e3e87727d0f85d438af0cee61",
    "package-lock.json": "c8b3d110ea4d98f69ce06a7c4965e2916030b7a1e959218584abad5312edeed4",
    "package.json": "9cd7dd3823c2af968b66178efc7dc7f489ac9e72fd6dbaaa7e875494a6cd7e4d",
    "uv.lock": "8ae8a064108e8908dcd947b67a74268f29ac2b56c6f5a9c957fb9cc5d2ad3e68",
    "website/.npmrc": "15124301297f7d5ddae9162b61eef79e41c2887a10cb2df0b61960ccf635baab",
    "website/package-lock.json": "d14972563d599e251d4a6269fc78124e011ba45e33fe9b1de8612584ab6daa29",
    "website/package.json": "c5ef11e7d257b15ce530861037bc3609c928131ecfc9142fcbb0e0b10ef116b7",
}
EXPECTED_RECEIPT = {
    "source_main": BASE_SHA,
    "control_head": EXPECTED_TARGET_HEAD,
    "control_parent": BASE_SHA,
    "materializer_root_blob": "c092011a7135296b0e9c7c672810ca0f45dcac7d",
    "materializer_website_blob": "d73cd5151a2378d2e7f262e2dd4974470b809c50",
    "node": "v22.22.0",
    "npm": "11.17.0",
    "uv": "uv 0.12.5",
}


def run(args: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
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


def parse_receipt(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        key, separator, value = raw.partition("=")
        if not separator or not key or key in values:
            raise RuntimeError(f"invalid receipt line: {raw!r}")
        values[key] = value
    return values


def download_artifact(root: Path) -> Path:
    archive = root / "dependency-security-current-main.zip"
    with archive.open("wb") as handle:
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
    actual = sha256(archive)
    if actual != ARTIFACT_ZIP_SHA256:
        raise RuntimeError(f"artifact digest mismatch: expected {ARTIFACT_ZIP_SHA256}, got {actual}")

    payload = root / "payload"
    payload.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (payload / member.filename).resolve()
            if payload.resolve() not in target.parents and target != payload.resolve():
                raise RuntimeError(f"unsafe artifact path: {member.filename}")
        bundle.extractall(payload)

    actual_paths = {
        path.relative_to(payload).as_posix()
        for path in payload.rglob("*")
        if path.is_file()
    }
    expected_paths = set(EXPECTED_FILES) | {"MATERIALIZATION_RECEIPT.txt", "SHA256SUMS"}
    if actual_paths != expected_paths:
        raise RuntimeError(f"artifact path set mismatch: {sorted(actual_paths)}")
    if parse_receipt(payload / "MATERIALIZATION_RECEIPT.txt") != EXPECTED_RECEIPT:
        raise RuntimeError("materialization receipt does not bind the expected source, tools, and materializers")
    for relative, expected in EXPECTED_FILES.items():
        actual = sha256(payload / relative)
        if actual != expected:
            raise RuntimeError(f"{relative}: expected {expected}, got {actual}")
    return payload


def clone_and_gate(root: Path) -> Path:
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN is required")
    run(["gh", "auth", "setup-git"])
    repo = root / "hermes-agent"
    run(["git", "clone", "--no-tags", f"https://github.com/{FORK_REPO}.git", str(repo)])
    run(["git", "remote", "add", "upstream", f"https://github.com/{UPSTREAM_REPO}.git"], cwd=repo)
    run(["git", "fetch", "--no-tags", "upstream", "refs/heads/main:refs/remotes/upstream/main"], cwd=repo)
    run(["git", "fetch", "--no-tags", "origin", f"refs/heads/{TARGET_BRANCH}:refs/remotes/origin/{TARGET_BRANCH}"], cwd=repo)

    actual_base = output(["git", "rev-parse", "refs/remotes/upstream/main"], cwd=repo)
    actual_target = output(["git", "rev-parse", f"refs/remotes/origin/{TARGET_BRANCH}"], cwd=repo)
    if actual_base != BASE_SHA:
        raise RuntimeError(f"upstream main moved: expected {BASE_SHA}, got {actual_base}")
    if actual_target != EXPECTED_TARGET_HEAD:
        raise RuntimeError(f"target lease moved: expected {EXPECTED_TARGET_HEAD}, got {actual_target}")
    if output(["git", "rev-parse", f"{EXPECTED_TARGET_HEAD}^"], cwd=repo) != BASE_SHA:
        raise RuntimeError("materialization control object is not one child of the bound main")
    return repo


def publish(repo: Path, payload: Path) -> str:
    run(["git", "checkout", "--detach", BASE_SHA], cwd=repo)
    for relative in EXPECTED_FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload / relative, target)

    expected = sorted(EXPECTED_FILES)
    run(["git", "add", "--", *expected], cwd=repo)
    staged = sorted(filter(None, output(["git", "diff", "--cached", "--name-only"], cwd=repo).splitlines()))
    if staged != expected:
        raise RuntimeError(f"staged path set mismatch: expected {expected}, got {staged}")
    if output(["git", "diff", "--name-only"], cwd=repo):
        raise RuntimeError("unexpected unstaged tracked changes")
    if output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo):
        raise RuntimeError("unexpected untracked files")
    run(["git", "diff", "--cached", "--check"], cwd=repo)

    run(["node", "--test", "apps/desktop/scripts/packaged-app-layout.test.mjs"], cwd=repo)
    run(["uv", "lock", "--check"], cwd=repo)
    run(["npm", "audit", "--workspaces=false", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "web", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "ui-tui", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--workspace", "apps/desktop", "--audit-level=high"], cwd=repo)
    run(["npm", "audit", "--audit-level=high"], cwd=repo / "website")

    run(["git", "fetch", "--no-tags", "upstream", "refs/heads/main:refs/remotes/upstream/main"], cwd=repo)
    if output(["git", "rev-parse", "refs/remotes/upstream/main"], cwd=repo) != BASE_SHA:
        raise RuntimeError("upstream main moved during verification")
    remote_target = output(["git", "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}"], cwd=repo).split()[0]
    if remote_target != EXPECTED_TARGET_HEAD:
        raise RuntimeError("target branch moved during verification")

    run(["git", "config", "user.name", "Axl Ibiza, MBA"], cwd=repo)
    run(["git", "config", "user.email", "andrexibiza@gmail.com"], cwd=repo)
    message = """fix(deps): close current advisory graph on landing main

Consolidate the verified Electron, root and standalone-website Nano ID,
and h2 advisory closure on exact landing main. Preserve React Compiler
movement, Nano ID v6 isolation, native Desktop packaging, and Linux
x64/ARM64 packaged-app layout semantics.

Refs #85916
Refs #89335
Refs #89479
Refs #90486
Refs #91042
Refs #92543
Refs #92573

Co-authored-by: hdy2001 <hdy2001@users.noreply.github.com>
Co-authored-by: SovereignSignal <SovereignSignal@users.noreply.github.com>
Co-authored-by: schmitzi8 <schmitzi8@users.noreply.github.com>
Co-authored-by: orcaspainting-dev <264355715+orcaspainting-dev@users.noreply.github.com>
Co-authored-by: mrxmoex <280798547+mrxmoex@users.noreply.github.com>
Signed-off-by: Axl Ibiza, MBA <andrexibiza@gmail.com>
"""
    message_path = repo / ".git" / "PR91906_COMMIT_MESSAGE"
    message_path.write_text(message, encoding="utf-8")
    run(["git", "commit", "--file", str(message_path)], cwd=repo)
    product_head = output(["git", "rev-parse", "HEAD"], cwd=repo)
    if output(["git", "rev-parse", "HEAD^"], cwd=repo) != BASE_SHA:
        raise RuntimeError("product commit parent mismatch")
    if output(["git", "rev-list", "--count", f"{BASE_SHA}..HEAD"], cwd=repo) != "1":
        raise RuntimeError("product branch is not exactly one commit ahead")
    final_paths = sorted(filter(None, output(["git", "diff", "--name-only", f"{BASE_SHA}..HEAD"], cwd=repo).splitlines()))
    if final_paths != expected:
        raise RuntimeError(f"final path set mismatch: {final_paths}")
    for relative, expected_sha in EXPECTED_FILES.items():
        if sha256(repo / relative) != expected_sha:
            raise RuntimeError(f"post-commit byte mismatch: {relative}")

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
    return product_head


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pr91906-") as temporary:
        root = Path(temporary)
        payload = download_artifact(root)
        repo = clone_and_gate(root)
        product_head = publish(repo, payload)

    run_url = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'andrexibiza/kaggle-titanic-competition')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', 'unknown')}"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    receipt = (
        "## PR 91906 product publication\n\n"
        f"- source main: `{BASE_SHA}`\n"
        f"- materializer object: `{EXPECTED_TARGET_HEAD}`\n"
        f"- materializer run/artifact: `{MATERIALIZER_RUN}` / `{ARTIFACT_ID}`\n"
        f"- product head: `{product_head}`\n"
        f"- product paths: `{len(EXPECTED_FILES)}`\n"
        f"- publisher run: {run_url}\n"
    )
    print(receipt)
    if summary:
        Path(summary).write_text(receipt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
