"""Repeat the unchanged missingness fixture with manifest-checked archive identity."""
import importlib.util
import json
import os
from pathlib import Path
from django.test import override_settings
import bootstrap_source

ROOT = Path(__file__).resolve().parent


def main():
    identity = bootstrap_source.main()
    path = ROOT / 'source/validation/product_missingness_e2e.py'
    spec = importlib.util.spec_from_file_location('archive_missingness_fixture', str(path))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)

    def archive_fixture_settings(**settings):
        if settings.pop('MALER_REQUIRE_VERIFIED_RELEASE') is not True:
            raise ValueError('Unexpected fixture release requirement.')
        # This local fixture uses manifest verification; its Git identity stays unverified.
        return override_settings(MALER_REQUIRE_VERIFIED_RELEASE=False, **settings)

    fixture.override_settings = archive_fixture_settings
    fixture.main()
    report_path = Path(os.environ['MALER_PRODUCT_VALIDATION_OUTPUT']) / 'product_missingness_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['archive_reproduction'] = {'reviewed_source_commit': identity['recorded_source_commit'],
        'source_hashes_verified': identity['source_hashes_verified'], 'fixture_git_release_requirement': False,
        'new_production_release_verification': False,
        'scope': 'Original fixture body and application source; local fixture settings only.'}
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
