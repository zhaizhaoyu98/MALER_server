"""Verify every real-browser prediction against the signed fitted pipeline."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'source'))
from mlserver.safe_ml import load_signed_model_bundle, classification_prediction

out = ROOT / 'evidence' / 'final_source_5d3d1a3' / 'browser'
report = json.loads((out / 'browser_e2e_report.json').read_text())
rows = json.loads((out / 'browser_prediction_rows.json').read_text())
model, manifest = load_signed_model_bundle(out / 'browser_model.maler', 'synthetic-browser-only-key')
policy = manifest['metadata']
frame = pd.read_csv(ROOT / 'evidence' / 'product_missingness' / 'rate_10' / 'predict_reordered.csv', index_col=0).T
features = manifest['feature_names']
X = frame[features].astype(float)
before = model.named_steps['imputer'].statistics_.copy()
labels, _, _ = classification_prediction(model, X, policy['positive_class'], policy['threshold_policy'], policy['threshold_value'])
expected = [{'Name': str(name), 'Label': str(label)} for name, label in zip(X.index, labels)]
assert rows == expected, 'Browser table differs from saved pipeline predictions'
np.testing.assert_array_equal(before, model.named_steps['imputer'].statistics_)
assert manifest['metadata']['selection_lock_sha256'] == report['selection_lock_sha256']
assert policy['software_commit'] == report['source']['commit']
result = {'prediction_count': len(rows), 'all_browser_rows_match_saved_pipeline': True,
          'positive_class': policy['positive_class'], 'threshold_value': policy['threshold_value'],
          'imputer_unchanged': True, 'manifest_lock_hash_matches': True,
          'source_commit': report['source']['commit']}
(Path(os.environ['MALER_REPRODUCTION_OUTPUT']) / 'browser_prediction_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
