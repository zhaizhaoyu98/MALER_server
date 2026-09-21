from __future__ import absolute_import

import json
import os

from django.core.management.base import BaseCommand

from mlserver.model_registry import FEATURE_REDUCTION, MODEL_REGISTRY, MODEL_REGISTRY_VERSION


class Command(BaseCommand):
    help = "Export the canonical MALER estimator and parameter registry as JSON."

    def add_arguments(self, parser):
        parser.add_argument("output", help="Destination JSON path")

    def handle(self, *args, **options):
        path = os.path.abspath(options["output"])
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = {
            "registry_version": MODEL_REGISTRY_VERSION,
            "model_registry": MODEL_REGISTRY,
            "feature_reduction": FEATURE_REDUCTION,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.stdout.write("Exported model registry to %s" % path)
