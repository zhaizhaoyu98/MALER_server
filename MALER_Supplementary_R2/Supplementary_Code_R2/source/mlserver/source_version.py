"""Verify the running checkout rather than trusting a version environment label."""

import os
import subprocess

from django.conf import settings


def source_identity():
    directory = os.path.realpath(settings.BASE_DIR)
    try:
        commit = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5).decode("ascii").strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", directory, "status", "--porcelain=v1", "--untracked-files=no"],
            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unverified", None
    declared = getattr(settings, "MALER_SOFTWARE_COMMIT", "unverified")
    matched = declared in ("", "unverified", commit)
    verified = commit != "unverified" and dirty is False and matched
    if getattr(settings, "MALER_REQUIRE_VERIFIED_RELEASE", False) and not verified:
        raise ValueError("Release checks require a clean, verifiable source commit matching the declared version.")
    return {"commit": commit, "dirty": dirty, "declared_commit_matches": matched,
            "verified": verified}
