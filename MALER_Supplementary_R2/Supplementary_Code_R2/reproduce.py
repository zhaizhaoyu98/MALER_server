"""Run offline review checks; preserved evidence is never replaced."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import bootstrap_source

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['verify', 'parity', 'missingness', 'tests', 'browser-evidence', 'figures'])
    args = parser.parse_args()
    bootstrap_source.main()
    out = ROOT / 'reproduction_runs' / args.check
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', DJANGO_SETTINGS_MODULE='ML_WebServer.settings')
    # An archive must not inherit an unrelated Git identity from its extraction directory.
    env['GIT_CEILING_DIRECTORIES'] = str(ROOT.parent)
    env['MALER_SOFTWARE_COMMIT'] = 'unverified'
    env['MALER_REQUIRE_VERIFIED_RELEASE'] = 'False'
    env['MALER_REPRODUCTION_OUTPUT'] = str(out)
    env['MALER_SOURCE_DIR'] = str(SOURCE)

    def run(path, extra=None):
        subprocess.check_call([sys.executable, str(path)] + (extra or []), cwd=str(SOURCE), env=env)

    if args.check == 'verify':
        run(ROOT / 'verify_attachment.py')
    elif args.check == 'parity':
        for p in (ROOT / 'evidence/primary_results').iterdir():
            if p.is_file():
                shutil.copy2(str(p), str(out / p.name))
        env['MALER_VALIDATION_OUTPUT_DIR'] = str(out)
        env['MALER_FIGURE_OUTPUT_DIR'] = str(ROOT / 'evidence/primary_figures')
        run(SOURCE / 'validation/reference_parity_check.py')
        run(ROOT / 'reproduction/current_fixed_refit_parity.py')
    elif args.check == 'missingness':
        env['MALER_PRODUCT_VALIDATION_OUTPUT'] = str(out)
        run(ROOT / 'missingness_from_archive.py')
    elif args.check == 'tests':
        run(SOURCE / 'manage.py', ['test', 'mlserver.tests', 'mlserver.tests_submission',
                                 'mlserver.tests_audit', 'mlserver.tests_public_online',
                                 'mlserver.tests_xgboost', '--noinput'])
    elif args.check == 'browser-evidence':
        run(ROOT / 'verify_attachment.py')
        run(ROOT / 'reproduction/verify_final_browser_predictions.py')
    else:
        env['MALER_VALIDATION_OUTPUT_DIR'] = str(ROOT / 'evidence/primary_results')
        env['MALER_FIGURE_OUTPUT_DIR'] = str(out)
        for name in ['make_workflow_figure.py', 'make_interface_figure.py',
                     'make_main_figure3_revised.py', 'restore_r1_supplementary_figures.py']:
            run(ROOT / 'figure_sources' / name)


if __name__ == '__main__':
    main()
