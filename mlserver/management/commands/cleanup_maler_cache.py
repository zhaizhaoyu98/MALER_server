"""Safely remove expired MALER project-cache directories."""

from __future__ import absolute_import

import os
import shutil
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "List expired MALER cache directories, or delete them with --apply."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", dest="apply_changes")
        parser.add_argument("--hours", type=float, default=None)

    def handle(self, *args, **options):
        hours = options["hours"]
        if hours is None:
            hours = float(settings.MALER_CACHE_RETENTION_HOURS)
        if hours < 0:
            raise CommandError("Retention hours must be zero or greater.")
        cache_root = os.path.realpath(os.path.join(settings.STATIC_ROOT, "cache"))
        if not os.path.isdir(cache_root):
            self.stdout.write("Cache directory does not exist: %s" % cache_root)
            return
        cutoff = time.time() - hours * 3600.0
        expired = []
        for entry in os.scandir(cache_root):
            if not entry.is_dir(follow_symlinks=False) or entry.is_symlink():
                continue
            resolved = os.path.realpath(entry.path)
            if os.path.dirname(resolved) != cache_root:
                continue
            if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                expired.append(resolved)
        for path in sorted(expired):
            if options["apply_changes"]:
                shutil.rmtree(path)
                self.stdout.write("deleted\t%s" % path)
            else:
                self.stdout.write("expired\t%s" % path)
        self.stdout.write(self.style.SUCCESS(
            "%d expired cache director%s %s." % (
                len(expired), "y" if len(expired) == 1 else "ies",
                "deleted" if options["apply_changes"] else "listed")))
