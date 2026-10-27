"""Check attachment hashes and links in retained production evidence, offline."""
import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path
from bootstrap_source import checked_path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
CLOUD = ROOT / 'Cloud_Evidence_R2/production_validation'


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = checked_path(ROOT, row['file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Attachment checksum mismatch: ' + row['file'])
    with (ROOT / 'reviewer_evidence_index.csv').open(encoding='utf-8', newline='') as handle:
        index = list(csv.DictReader(handle))
    for row in index:
        if not row['location'].startswith('Submitted separately: '):
            if not checked_path(ROOT, row['location']).is_file():
                raise ValueError('Missing reviewer evidence: ' + row['location'])
    assert len(index) == 77 and len({r['reviewer_subitem'] for r in index}) == 31
    report = json.loads((CLOUD / 'workflow_verification.json').read_text(encoding='utf-8'))
    assert report['source_commit'] == manifest['reviewed_source_commit']
    assert len(report['records']) == 8
    for row in report['records']:
        assert row['prediction_rows'] == 40
        assert row['all_rows_match_saved_pipeline'] and row['imputer_unchanged']
        assert row['independent_pipeline_fit_calls'] == row['independent_imputer_fit_calls'] == 0
        model_path = CLOUD / 'browser' / row['name'] / 'browser_model.maler'
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == row['model_bundle_sha256']
        with zipfile.ZipFile(model_path) as model:
            metadata = json.loads(model.read('manifest.json'))['metadata']
        assert metadata['selection_lock_sha256'] == row['selection_lock_sha256']
        assert metadata['software_commit'] == report['source_commit']
    result = {'payload_files_verified': len(manifest['files']), 'reviewer_index_entries': len(index),
              'reviewer_subitems': 31, 'retained_public_workflows_checked': 8,
              'predictions_per_retained_workflow': 40, 'scope': __doc__,
              'new_live_server_validation': False, 'production_signatures_authenticated_offline': False}
    output = Path(os.environ.get('MALER_REPRODUCTION_OUTPUT', str(CODE / 'reproduction_runs/verify')))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'attachment_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
