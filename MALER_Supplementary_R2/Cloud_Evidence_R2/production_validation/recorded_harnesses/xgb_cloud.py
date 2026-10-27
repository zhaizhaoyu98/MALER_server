"""Counts/config-only host audit and identified synthetic-task verification."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import shutil
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
SERVICE = 'maler-r2-restricted.service'
PY = '/mnt/home/liqiang/.conda/envs/gene_edit/bin/python3.7'
ENV = Path('/mnt/home/liqiang/maler_revision2_20260930/runtime/public_online_environment.env')
PRIOR = 'b87cae92d696798604beab228c082d713f2a0e63'


def now(): return datetime.datetime.utcnow().isoformat()+'Z'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name, value): (HERE/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def run(args, timeout=40, user=False, env=None, cwd=None):
    if user: args = ['runuser', '-u', 'liqiang', '--']+args
    process = subprocess.run(args, env=env, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=timeout)
    output = process.stdout.decode('utf-8', errors='replace').strip()
    assert process.returncode == 0, 'Command failed: '+args[0]+'\n'+output
    return output


def env_values():
    values = {}
    for line in ENV.read_text(encoding='utf-8').splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and '=' in tokens[0]:
            key, value = tokens[0].split('=', 1); values[key] = value
    return values


def identity():
    props = dict(line.split('=', 1) for line in run(['systemctl', 'show', SERVICE,
        '--property=ActiveState,MainPID,WorkingDirectory', '--no-pager']).splitlines())
    assert props['ActiveState'] == 'active'
    pid = int(props['MainPID']); source = Path(props['WorkingDirectory'])
    assert pid > 0 and Path('/proc/%d/cwd' % pid).resolve() == source
    assert str(Path('/proc/%d/exe' % pid).resolve()) == PY
    values = dict(row.decode().split('=', 1) for row in
                  Path('/proc/%d/environ' % pid).read_bytes().split(b'\0') if b'=' in row)
    return {'server_utc': now(), 'source_commit': run(['git', '-C', str(source), 'rev-parse', 'HEAD'], user=True),
        'source_clean': not bool(run(['git', '-C', str(source), 'status', '--porcelain=v1'], user=True)),
        'source_directory': str(source), 'process_exe': PY, 'master_pid': pid,
        'public_uploads_enabled': values.get('MALER_PUBLIC_UPLOADS_ENABLED', '').lower() == 'true',
        'task_root': values.get('MALER_TASK_ROOT')}


def resolved_settings(current=None):
    current = current or identity(); env = dict(os.environ, **env_values())
    code = ('import json,django;django.setup();from django.conf import settings as s;'
            'print(json.dumps({"settings_module":s.SETTINGS_MODULE,"database_path":str(s.DATABASES["default"]["NAME"]),'
            '"task_root":s.MALER_TASK_ROOT,"upload_tmp":s.FILE_UPLOAD_TEMP_DIR,"static_root":str(s.STATIC_ROOT),'
            '"ssl_redirect":s.SECURE_SSL_REDIRECT,"hsts_seconds":s.SECURE_HSTS_SECONDS}))')
    return json.loads(run([PY, '-c', code], user=True, env=env, cwd=current['source_directory']))


def inspect():
    current = identity(); actual = resolved_settings(current)
    assert current['source_commit'] == PRIOR and current['source_clean'] and current['public_uploads_enabled']
    assert actual['database_path'] == '/var/lib/maler_public_r2/state/public.sqlite3'
    result = {'server_utc': now(), 'identity': current, 'resolved_settings': actual,
        'mount': run(['findmnt', '-n', '-o', 'TARGET,FSTYPE,OPTIONS', '-T', actual['task_root']]),
        'no_swap': not bool(run(['swapon', '--show', '--noheadings'])),
        'ops_health': json.loads(Path('/var/log/maler_ops/health.json').read_text()),
        'scope': 'Read-only actual process/config audit; no task contents, secrets or backups returned'}
    save('before_identity.json', result); print(json.dumps(result, indent=2))


def safe_id(value):
    assert re.fullmatch('[A-Za-z0-9_-]+', value)
    return value


FLOW_NAMES = ['binary_custom', 'oneclick', 'multiclass', 'regression', 'survival',
              'xgb_binary', 'xgb_multiclass', 'oneclick_repeat']
ALLOWED = {'analysis.csv', 'predict_reordered.csv', 'predict.csv', 'input_manifest.json',
           'browser_result.json', 'browser_model.maler', 'browser_e2e_report.json',
           'browser_report.json', 'browser_prediction_rows.json', 'browser_rows.json',
           'binary_preview_cutoff.png', 'binary_result.png', 'analysis_preview.png',
           'analysis_result.png', 'predict_preview.png', 'predict_result.png'}


def verify_as_service():
    assert os.geteuid() != 0
    expected = os.environ['MALER_VERIFY_EXPECTED']
    tasks = Path(os.environ['MALER_TASK_ROOT']); assert str(tasks) == '/var/lib/maler_public_r2/tasks'
    sys.path.insert(0, os.environ['MALER_VERIFY_SOURCE'])
    import numpy as np
    import pandas as pd
    import pickle
    from unittest.mock import patch
    from mlserver.safe_ml import FoldImputer, load_signed_model_bundle, classification_prediction
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sksurv.metrics import concordance_index_censored
    key = os.environ['MALER_MODEL_SIGNING_KEY']
    records = []; results = {}
    for name in FLOW_NAMES:
        folder = HERE/'browser'/name
        binary = name in ['binary_custom', 'oneclick', 'xgb_binary', 'oneclick_repeat']
        report = json.loads((folder/('browser_e2e_report.json' if binary else 'browser_report.json')).read_text())
        assert report['source']['commit'] == expected and report['source']['verified'] and not report['source']['dirty']
        assert report['origin'] == 'https://www.inbirg.com' and not report['tls_verification_disabled']
        assert not report['page_errors'] and not report['unexpected_failed_requests']
        posts = report['successful_form_posts' if binary else 'successful_posts']
        assert len(posts) == 4 and all(row['status'] == 200 for row in posts)
        assert report['actual_budget']['actual_outer_evaluations'] == 10 and report['actual_budget']['actual_inner_splits'] == 3
        project = tasks/safe_id(report['analysis_project_id']); prediction = tasks/safe_id(report['prediction_project_id'])
        predict_file = 'predict_reordered.csv' if binary else 'predict.csv'
        assert sha(project/'data.csv') == sha(folder/'analysis.csv')
        assert sha(prediction/'data.csv') == sha(folder/predict_file)
        filename = report['model_file_name']; assert re.fullmatch(r'[A-Za-z0-9_.-]+\.maler', filename) and '..' not in filename
        assert sha(project/filename) == sha(folder/'browser_model.maler') == report['model_bundle_sha256']
        assert sha(project/'selection_lock.json') == report['selection_lock_sha256']
        result = json.loads((folder/'browser_result.json').read_text()); results[name] = result
        assert result['registry_version'] == 4
        model, manifest = load_signed_model_bundle(folder/'browser_model.maler', key)
        policy = manifest['metadata']; assert policy['software_commit'] == expected
        frame = pd.read_csv(folder/predict_file, index_col=0).T
        X = frame[manifest['feature_names']].astype(float)
        stats = model.named_steps['imputer'].statistics_.copy()
        survival = name == 'survival'; regression = name == 'regression'
        with patch.object(FoldImputer, 'fit', side_effect=AssertionError('Unexpected imputer refit')) as imputer_fit, \
             patch.object(type(model), 'fit', side_effect=AssertionError('Unexpected pipeline refit')) as pipeline_fit:
            if regression or survival:
                labels = model.predict(X); scores = None
            else:
                labels, scores, _ = classification_prediction(model, X, policy.get('positive_class'),
                    policy.get('threshold_policy') or 'estimator_native_predict', policy.get('threshold_value') or .5)
            if survival:
                valid = frame['status'].notna(); transformed = X.loc[~valid]
                for _, step in model.steps[:-1]: transformed = step.transform(transformed)
                curves = model.named_steps['model'].predict_survival_function(transformed)
                hazards = model.named_steps['model'].predict_cumulative_hazard_function(transformed)
            assert imputer_fit.call_count == pipeline_fit.call_count == 0
        np.testing.assert_array_equal(stats, model.named_steps['imputer'].statistics_)
        row_data = json.loads((folder/('browser_prediction_rows.json' if binary else 'browser_rows.json')).read_text())
        rows = row_data if binary else row_data['blind']+row_data['validation']
        assert len(rows) == len(X) == 40
        lookup = {row['Name']: row['Label'] for row in rows}; assert set(lookup) == set(X.index)
        actual = [lookup[str(index)] for index in X.index]
        if regression or survival: np.testing.assert_allclose(np.asarray(actual, dtype=float), labels, rtol=0, atol=1e-10)
        else: assert list(map(str, actual)) == list(map(str, labels))
        independent = {}
        if not binary:
            valid = frame['status' if survival else 'label'].notna()
            with (prediction/'predict_pickle.pkl').open('rb') as stream: payload = pickle.load(stream)
            assert 'access_token' not in payload
            metrics = payload['signed_validation_metrics']; target = frame.loc[valid]
            if regression:
                observed = pd.to_numeric(target['label']).to_numpy()
                independent = {'r2': r2_score(observed, labels[valid]), 'mae': mean_absolute_error(observed, labels[valid]),
                    'mse': mean_squared_error(observed, labels[valid]), 'rmse': np.sqrt(mean_squared_error(observed, labels[valid]))}
            elif survival:
                independent = {'c_index': concordance_index_censored(target['status'].astype(int).astype(bool).to_numpy(), target['time'].astype(float).to_numpy(), labels[valid])[0]}
                for position, sample in enumerate(X.index[~valid]):
                    for curve_index, function in [(0, curves[position]), (1, hazards[position])]:
                        trace = row_data['survival_curves'][2*position+curve_index]
                        assert trace['name'] == str(sample)
                        np.testing.assert_allclose(trace['x'], function.x, rtol=0, atol=1e-10)
                        np.testing.assert_allclose(trace['y'], function.y, rtol=0, atol=1e-10)
                assert len(row_data['survival_curves']) == 20
            else:
                observed = target['label'].astype(str).to_numpy()
                independent = {'balanced_accuracy': balanced_accuracy_score(observed, labels[valid]),
                    'roc_auc_ovr_macro': roc_auc_score(observed, scores[valid], multi_class='ovr', average='macro', labels=model.classes_)}
            assert all(abs(metrics[k]-v) <= 1e-10 for k, v in independent.items())
        encoded_mapping = None
        if not regression and not survival:
            assert list(model.classes_) == policy['class_order']
            classifier = model.named_steps['model']
            if name in ['xgb_binary', 'xgb_multiclass']:
                assert type(classifier).__name__ == 'LabelledXGBClassifier'
                np.testing.assert_array_equal(classifier.label_encoder_.classes_, model.classes_)
                encoded_mapping = [{'original': str(value), 'native_index': i} for i, value in enumerate(model.classes_)]
                if binary:
                    assert list(model.classes_).index('case') == 0 and policy['threshold_value'] == .65
                else:
                    assert all(value.startswith('group') for value in model.classes_)
            if binary:
                # The binary Predict fixture is blind. Check labelled metrics on the original Analysis heldout partition.
                analysis = pd.read_csv(folder/'analysis.csv', index_col=0).T
                held = analysis.loc[analysis.iloc[:, 1].astype(str).str.lower().isin(['test', 'testing', 'validation', 'external'])]
                with patch.object(FoldImputer, 'fit', side_effect=AssertionError('Unexpected heldout imputer refit')), \
                     patch.object(type(model), 'fit', side_effect=AssertionError('Unexpected heldout pipeline refit')):
                    held_labels, held_scores, _ = classification_prediction(model, held[manifest['feature_names']].astype(float),
                        policy['positive_class'], policy['threshold_policy'], policy['threshold_value'])
                observed = held.iloc[:, 0].astype(str).to_numpy(); positive = list(model.classes_).index(policy['positive_class'])
                independent = {'balanced_accuracy': balanced_accuracy_score(observed, held_labels),
                    'roc_auc': roc_auc_score(observed == policy['positive_class'], held_scores[:, positive]),
                    'pr_auc': average_precision_score(observed == policy['positive_class'], held_scores[:, positive])}
                heldout = next(row for row in result['models'] if row['key'] == result['selected_family'])['heldout_test']
                assert all(abs(heldout[k]-v) <= 1e-10 for k, v in independent.items())
        (folder/'selection_lock.json').write_bytes((project/'selection_lock.json').read_bytes())
        records.append({'name': name, 'prediction_rows': 40, 'missing_feature_cells': int(X.isna().sum().sum()),
            'analysis_project_id': report['analysis_project_id'], 'prediction_project_id': report['prediction_project_id'],
            'model_bundle_sha256': sha(folder/'browser_model.maler'), 'selection_lock_sha256': report['selection_lock_sha256'],
            'all_rows_match_saved_pipeline': True, 'imputer_unchanged': True,
            'independent_pipeline_fit_calls': 0, 'independent_imputer_fit_calls': 0,
            'original_label_mapping': encoded_mapping, 'independently_recomputed_metrics': independent,
            'metric_scope': 'Analysis heldout partition' if binary else 'labelled Predict partition'})
    one, two = results['oneclick'], results['oneclick_repeat']
    assert one['selection']['outer_split_ids_hash'] == two['selection']['outer_split_ids_hash']
    assert one['selected_family'] == two['selected_family'] and one['selection']['best_params'] == two['selection']['best_params']
    scores = []
    for result in [one, two]:
        assert any(row['key'] == 'xgboost' for row in result['models'])
        assert not any(row['key'] == 'xgboost' for row in result['failures'])
        scores.append({row['key']: [fold['balanced_accuracy'] for fold in row['nested_cv']['folds']] for row in result['models']})
    assert scores[0] == scores[1]
    assert (HERE/'browser/oneclick/browser_prediction_rows.json').read_bytes() == (HERE/'browser/oneclick_repeat/browser_prediction_rows.json').read_bytes()
    legacy, legacy_manifest = load_signed_model_bundle(HERE/'legacy_model.maler', key)
    legacy_frame = pd.read_csv(HERE/'legacy_predict.csv', index_col=0).T
    policy = legacy_manifest['metadata']; old_rows = json.loads((HERE/'legacy_prediction_rows.json').read_text())
    with patch.object(FoldImputer, 'fit', side_effect=AssertionError('Unexpected legacy imputer refit')), \
         patch.object(type(legacy), 'fit', side_effect=AssertionError('Unexpected legacy pipeline refit')):
        old_labels, _, _ = classification_prediction(legacy, legacy_frame[legacy_manifest['feature_names']].astype(float),
            policy['positive_class'], policy['threshold_policy'], policy['threshold_value'])
    assert old_rows == [{'Name': str(index), 'Label': str(label)} for index, label in zip(legacy_frame.index, old_labels)]
    proof = {'server_utc': now(), 'source_commit': expected, 'records': records,
        'verification_account_uid': os.geteuid(), 'root_application_imports_performed': False,
        'paired_oneclick': {'xgboost_successfully_compared': True, 'same_all_family_fold_scores': True,
            'same_selected_family': one['selected_family'], 'same_final_params': True, 'all_40_labels_equal': True,
            'balanced_accuracy_by_family': scores[0]},
        'legacy_signed_model': {'source_commit': policy['software_commit'], 'prediction_rows': len(old_rows),
            'same_predictions_as_previous_browser': True, 'new_fits': 0},
        'scope': 'Eight actual public workflows; independent saved-pipeline reuse, labelled metrics and curves. Binary fixtures share one input; no new independent clinical cohorts.'}
    save('workflow_verification.json', proof)
    print(json.dumps({'verified_workflows': len(records), 'source_commit': expected, 'application_checks_ran_as_root': False}))


def verify():
    current = identity(); release = json.loads(zipfile.ZipFile(HERE/'release.zip').read('release.json'))
    assert current['source_commit'] == release['source_commit'] and current['source_clean'] and current['public_uploads_enabled']
    scratch = Path(tempfile.mkdtemp(prefix='maler_xgb_verify_', dir='/run')); scratch.chmod(0o755)
    assert scratch.resolve().parent == Path('/run') and scratch.name.startswith('maler_xgb_verify_')
    try:
        target = scratch/'browser'; target.mkdir()
        with zipfile.ZipFile(HERE/'browser.zip') as archive:
            assert archive.testzip() is None and sum(info.file_size for info in archive.infolist()) < 22*1024*1024
            for name in archive.namelist():
                if name in ['legacy_prediction_rows.json', 'prior_release_delete_report.json']: destination = scratch/name
                else:
                    parts = name.split('/'); assert len(parts) == 2 and parts[0] in FLOW_NAMES and parts[1] in ALLOWED
                    destination = target/name; destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(archive.read(name))
        preliminary = json.loads((scratch/'prior_release_delete_report.json').read_text())
        assert len(preliminary['records']) == 16
        assert all(row['missing_token_status'] == 403 and row['delete_final_status'] == 200 and row['after_delete_status'] == 403 for row in preliminary['records'])
        prior_absent = [safe_id(row['project_id']) for row in preliminary['records']]
        assert all(not (Path(current['task_root'])/name).exists() for name in prior_absent)
        save('prior_release_cleanup_verified.json', {'server_utc': now(), 'absent_prior_release_synthetic_tasks': prior_absent, 'public_delete_report': preliminary})
        with zipfile.ZipFile(HERE/'release.zip') as archive:
            for name in ['legacy_model.maler', 'legacy_predict.csv']: (scratch/name).write_bytes(archive.read(name))
        script = scratch/'check.py'; script.write_bytes(Path(__file__).read_bytes()); script.chmod(0o644)
        import pwd
        account = pwd.getpwnam('liqiang')
        for name in ['legacy_model.maler', 'legacy_predict.csv', 'legacy_prediction_rows.json']:
            path = scratch/name
            os.chown(str(path), account.pw_uid, account.pw_gid); path.chmod(0o600)
        for path in target.rglob('*'):
            os.chown(str(path), account.pw_uid, account.pw_gid); path.chmod(0o700 if path.is_dir() else 0o600)
        os.chown(str(target), account.pw_uid, account.pw_gid); target.chmod(0o700)
        # The service writes only its synthetic verification receipt in this isolated directory.
        os.chown(str(scratch), account.pw_uid, account.pw_gid); scratch.chmod(0o700)
        env = dict(os.environ, **env_values()); env.update(MALER_VERIFY_SOURCE=current['source_directory'], MALER_VERIFY_EXPECTED=release['source_commit'])
        for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: env[name] = '1'
        output = run([PY, str(script), '_service_verify'], user=True, env=env, cwd=str(scratch), timeout=150)
        save('verification_execution.json', {'server_utc': now(), 'stdout': output, 'run_as': 'liqiang'})
        shutil.copytree(str(target), str(HERE/'verified_browser'))
        shutil.copy2(str(scratch/'workflow_verification.json'), str(HERE/'workflow_verification.json'))
    finally:
        assert scratch.resolve().parent == Path('/run') and scratch.name.startswith('maler_xgb_verify_')
        shutil.rmtree(str(scratch))
    save('current_identity.json', {'identity': identity(), 'resolved_settings': resolved_settings(), 'scope': 'Post-browser exact production identity'})
    print((HERE/'workflow_verification.json').read_text())


def export():
    current = identity(); proof = json.loads((HERE/'workflow_verification.json').read_text())
    absent = []
    for row in proof['records']:
        for key in ['analysis_project_id', 'prediction_project_id']:
            name = safe_id(row[key]); assert not (Path(current['task_root'])/name).exists(); absent.append(name)
    assert len(set(absent)) == 16
    run(['systemctl', 'start', 'maler-ops-health.service'], timeout=45)
    health = json.loads(Path('/var/log/maler_ops/health.json').read_text()); assert health['healthy']
    save('task_cleanup_confirmed.json', {'server_utc': now(), 'absent_synthetic_tasks': absent, 'identity': current, 'ops_health': health})
    names = ['before_identity.json', 'current_identity.json', 'release_precheck.json', 'release_applied.json',
        'cloud_tests.log', 'workflow_verification.json', 'verification_execution.json', 'task_cleanup_confirmed.json', 'prior_release_cleanup_verified.json']
    for path in sorted((HERE/'history_preliminary_1d3e42a').rglob('*')):
        assert path.name in ['release_precheck.json', 'release_applied.json', 'cloud_tests.log']
        if path.is_file(): names.append(path.relative_to(HERE).as_posix())
    files = {name: (HERE/name).read_bytes() for name in names}
    for path in sorted((HERE/'verified_browser').rglob('*')):
        if path.is_file(): files['browser/'+path.relative_to(HERE/'verified_browser').as_posix()] = path.read_bytes()
    manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in files.items()}
    files['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    with zipfile.ZipFile(HERE/'public_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items(): archive.writestr(name, data)
    print(json.dumps({'source_commit': current['source_commit'], 'files': len(files), 'sha256': sha(HERE/'public_evidence.zip')}))


if __name__ == '__main__':
    if sys.argv[1:] == ['_service_verify']:
        verify_as_service()
    else:
        assert os.geteuid() == 0
        os.umask(0o077)
        {'inspect': inspect, 'verify': verify, 'export': export}[sys.argv[1]]()
