import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
layout = json.loads((HERE/'browser_layout.json').read_text(encoding='utf-8'))
allowed = {'analysis.csv', 'predict_reordered.csv', 'predict.csv', 'input_manifest.json',
    'browser_result.json', 'browser_model.maler', 'browser_e2e_report.json', 'browser_report.json',
    'browser_prediction_rows.json', 'browser_rows.json', 'binary_preview_cutoff.png',
    'binary_result.png', 'analysis_preview.png', 'analysis_result.png', 'predict_preview.png', 'predict_result.png'}
with zipfile.ZipFile(HERE/'browser_packet.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for row in layout:
        folder = HERE/'browser'/row['root']/('browser_'+row['mode'])
        report_name = 'browser_e2e_report.json' if row['binary_harness'] else 'browser_report.json'
        report = json.loads((folder/report_name).read_text(encoding='utf-8'))
        assert not report['page_errors'] and not report['unexpected_failed_requests']
        for path in folder.iterdir():
            assert path.name in allowed and path.is_file()
            archive.write(path, row['name']+'/'+path.name)
    archive.write(HERE.parent/'completion_checks_20261022/final_evidence/browser/custom/browser_prediction_rows.json', 'legacy_prediction_rows.json')
    archive.write(HERE/'prior_release_delete_report.json', 'prior_release_delete_report.json')
print('Eight workflow packet ready; private capabilities excluded.')
