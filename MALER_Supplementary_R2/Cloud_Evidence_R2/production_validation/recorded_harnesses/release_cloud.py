"""Guarded exact-source release; retain interpreter, storage policy and rollback."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

from xgb_cloud import HERE, PY, ENV, PRIOR, SERVICE, identity, resolved_settings, env_values, run, save, now
INITIAL = 'b87cae92d696798604beab228c082d713f2a0e63'


def plan():
    with zipfile.ZipFile(HERE/'release.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'release.json', 'legacy_model.maler', 'legacy_predict.csv'}
        return json.loads(archive.read('release.json'))


def precheck():
    current = identity(); actual = resolved_settings(current); release = plan()
    assert current['source_commit'] == release['prior_commit'] == INITIAL
    assert current['source_clean'] and current['public_uploads_enabled']
    assert actual['database_path'] == '/var/lib/maler_public_r2/state/public.sqlite3'
    versions = json.loads(run([PY, '-c', 'import json,platform,django,xgboost,sklearn;print(json.dumps({"python":platform.python_version(),"django":django.get_version(),"xgboost":xgboost.__version__,"sklearn":sklearn.__version__}))'],
                             user=True, env=dict(os.environ, **env_values()), cwd=current['source_directory']))
    assert versions['python'] == '3.7.11' and versions['django'] == '2.1.8' and versions['xgboost'] == '1.6.2'
    history = HERE/'history_preliminary_1d3e42a'; history.mkdir(exist_ok=True)
    import shutil
    for name in ['release_precheck.json', 'release_applied.json', 'cloud_tests.log']:
        if (HERE/name).is_file() and not (history/name).exists(): shutil.copy2(str(HERE/name), str(history/name))
    result = {'server_utc': now(), 'current': current, 'resolved_settings': actual,
              'versions': versions, 'next_commit': release['source_commit'],
              'runtime_upgrade_planned': False, 'nas_policy_change_planned': False}
    save('release_precheck.json', result); print(json.dumps(result, indent=2))


def apply():
    release = plan(); old = identity(); actual_before = resolved_settings(old)
    assert old['source_commit'] == release['prior_commit'] == INITIAL and old['source_clean'] and old['public_uploads_enabled']
    assert set(release['changed_source_files']) == {'README.md', 'mlserver/templates/analysis.html', 'mlserver/templates/predict.html'}
    assert release['tests'] == 95 and release['registry_version'] == 4
    source = Path(old['source_directory']).parent/('source_public_'+release['source_commit'][:7])
    if not source.exists():
        run(['git', 'clone', '--local', '--no-hardlinks', old['source_directory'], str(source)], user=True)
        for obj in release['objects']:
            assert re.fullmatch('[a-f0-9]{40}', obj['id']) and obj['kind'] in ['tree', 'blob', 'commit']
            process = subprocess.run(['runuser', '-u', 'liqiang', '--', 'git', '-C', str(source), 'hash-object', '-t', obj['kind'], '-w', '--stdin'],
                                     input=base64.b64decode(obj['data']), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.returncode == 0 and process.stdout.decode().strip() == obj['id']
        run(['git', '-C', str(source), 'checkout', '-f', release['source_commit']], user=True)
    assert run(['git', '-C', str(source), 'rev-parse', 'HEAD'], user=True) == release['source_commit']
    assert not run(['git', '-C', str(source), 'status', '--porcelain=v1'], user=True)
    assert set(run(['git', '-C', str(source), 'diff', '--name-only', release['prior_commit'], 'HEAD'], user=True).splitlines()) == set(release['changed_source_files'])
    env = dict(os.environ, **env_values())
    env.update({'DJANGO_SECURE_SSL': 'False', 'MALER_MIN_FREE_TASK_BYTES': '0',
                'MALER_SOFTWARE_COMMIT': release['source_commit'], 'PYTHONPATH': str(ENV.parent)+':'+str(source)})
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: env[key] = '1'
    process = subprocess.run(['runuser', '-u', 'liqiang', '--', PY, 'manage.py', 'test',
        'mlserver.tests', 'mlserver.tests_submission', 'mlserver.tests_audit', 'mlserver.tests_public_online', 'mlserver.tests_xgboost', '--noinput'],
        cwd=str(source), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=165)
    (HERE/'cloud_tests.log').write_bytes(process.stdout)
    output = process.stdout.decode(errors='replace')
    assert process.returncode == 0 and 'Ran 95 tests' in output and '\nOK' in output, output
    saved = HERE/('private_release_backup_'+release['source_commit'][:7]); saved.mkdir(mode=0o700)
    paths = [ENV, Path('/etc/systemd/system')/SERVICE,
             Path('/etc/systemd/system/maler-public-cleanup.service'), Path('/usr/local/libexec/maler_ops_health.py')]
    snapshots = {path: path.read_bytes() for path in paths}
    for index, (path, data) in enumerate(snapshots.items()): (saved/str(index)).write_bytes(data)
    try:
        lines = []; seen = set()
        for line in ENV.read_text().splitlines():
            key = line.split('=', 1)[0]
            if key == 'MALER_SOFTWARE_COMMIT': line = key+'='+json.dumps(release['source_commit']); seen.add(key)
            elif key == 'PYTHONPATH': line = key+'='+json.dumps(str(ENV.parent)+':'+str(source)); seen.add(key)
            lines.append(line)
        assert seen == {'MALER_SOFTWARE_COMMIT', 'PYTHONPATH'}
        ENV.write_text('\n'.join(lines)+'\n'); ENV.chmod(0o600)
        for path in paths[1:]:
            text = path.read_text(); assert old['source_directory'] in text
            path.write_text(text.replace(old['source_directory'], str(source)))
        run(['systemctl', 'daemon-reload']); run(['systemctl', 'restart', SERVICE], timeout=85)
        attempts = []; deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            try:
                with urlopen('https://www.inbirg.com/maler/help', context=ssl.create_default_context(), timeout=5) as response:
                    body = response.read().decode(); status = response.status
                attempts.append(status)
                if status == 200 and 'Registry version 4' in body: break
            except Exception as exc: attempts.append(type(exc).__name__)
            time.sleep(1)
        assert attempts[-1] == 200 and 'Registry version 4' in body
    except Exception:
        for path, data in snapshots.items(): path.write_bytes(data)
        run(['systemctl', 'daemon-reload']); run(['systemctl', 'restart', SERVICE], timeout=85)
        raise
    current = identity(); actual = resolved_settings(current)
    assert current['source_commit'] == release['source_commit'] and current['source_clean'] and current['public_uploads_enabled']
    assert actual == actual_before
    run(['systemctl', 'start', 'maler-ops-health.service'], timeout=45)
    health = json.loads(Path('/var/log/maler_ops/health.json').read_text())
    assert health['healthy']
    result = {'server_utc': now(), 'source_commit': release['source_commit'], 'identity': current,
              'resolved_settings': actual, 'tests_passed': 95, 'registry_version': 4,
              'readiness_attempts': attempts, 'health': health, 'runtime_upgraded': False,
              'nas_policy_changed': False, 'hsts_changed': False, 'changed_source_files': release['changed_source_files']}
    save('release_applied.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    assert os.geteuid() == 0
    os.umask(0o077)
    {'precheck': precheck, 'apply': apply}[sys.argv[1]]()
