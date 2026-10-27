import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
NODE = r'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
selected = set(sys.argv[1:])
for row in json.loads((HERE/'browser_layout.json').read_text(encoding='utf-8')):
    if selected and row['name'] not in selected: continue
    root = HERE/'browser'/row['root']
    receipt = root/('browser_'+row['mode'])/('browser_e2e_report.json' if row['binary_harness'] else 'browser_report.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        assert not prior['page_errors'] and not prior['unexpected_failed_requests']
        print(row['name']+': retaining completed current run', flush=True); continue
    script = 'public_browser.cjs' if row['binary_harness'] else 'public_three_tasks.cjs'
    print(row['name']+': starting', flush=True)
    process = subprocess.run([NODE, str(root/script), row['mode']], cwd=root,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
    (root/(row['name']+'_execution.log')).write_bytes(process.stdout)
    print(process.stdout.decode(errors='replace'), flush=True)
    assert process.returncode == 0, row['name']+': failed; identified task capabilities retained privately'
print('Selected production workflows completed.', flush=True)
