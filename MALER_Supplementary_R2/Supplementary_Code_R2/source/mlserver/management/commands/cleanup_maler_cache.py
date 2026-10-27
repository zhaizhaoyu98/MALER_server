"""Remove expired private task data using the token's creation/expiry record."""

from __future__ import absolute_import

import calendar
import hashlib
import json
import os
import shutil
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


def _utc_epoch(value):
    return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))


def _anonymous_id(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


class Command(BaseCommand):
    help = "List expired private MALER tasks, or physically delete them with --apply."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", dest="apply_changes")
        parser.add_argument("--hours", type=float, default=None)

    def handle(self, *args, **options):
        hours = options["hours"]
        if hours is None:
            hours = float(settings.MALER_CACHE_RETENTION_HOURS)
        if hours < 0:
            raise CommandError("Retention hours must be zero or greater.")
        root = os.path.realpath(settings.MALER_TASK_ROOT)
        if not os.path.isdir(root):
            self.stdout.write("Private task directory does not exist.")
            return
        now = time.time()
        expired, deleted, failed = 0, 0, 0
        for entry in os.scandir(root):
            if entry.is_symlink():
                continue
            path = os.path.realpath(entry.path)
            if os.path.dirname(path) != root:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    access_path = os.path.join(path, "access.json")
                    if os.path.isfile(access_path):
                        with open(access_path, "r", encoding="utf-8") as handle:
                            access = json.load(handle)
                        # --hours is for controlled audits; routine deletion uses
                        # the fixed expiry written when the token was issued.
                        deadline = (_utc_epoch(access["expires_utc"]) if options["hours"] is None
                                    else _utc_epoch(access["created_utc"]) + hours * 3600)
                    else:
                        deadline = entry.stat(follow_symlinks=False).st_mtime + hours * 3600
                elif entry.is_file(follow_symlinks=False):
                    # Interrupted Predict uploads leave random root-level files.
                    if not entry.name.endswith((".maler", ".data", ".tmp")):
                        continue
                    deadline = entry.stat(follow_symlinks=False).st_mtime + hours * 3600
                else:
                    continue
                if now < deadline:
                    continue
                expired += 1
                if options["apply_changes"]:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    deleted += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failed += 1
                self.stderr.write("cleanup_failed id=%s reason=%s" %
                                  (_anonymous_id(entry.name), exc.__class__.__name__))
        self.stdout.write("cleanup mode=%s expired=%d deleted=%d failed=%d" %
                          ("apply" if options["apply_changes"] else "dry-run",
                           expired, deleted, failed))
        if failed:
            raise CommandError("Some task data could not be assessed or deleted.")
