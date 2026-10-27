import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'completion_checks_20261022'
FINAL = json.loads((HERE/'release_plan.json').read_text(encoding='utf-8'))['source_commit']
ROOT = HERE/'browser'
ROOT.mkdir(exist_ok=True)

binary = (PREVIOUS/'final_source_browser/public_browser.cjs').read_text(encoding='utf-8')
binary = binary.replace(str(HERE.parent/'public_online_20261022/web_source/mlserver/static/cdn/js/plotly-2.8.3.min.js'), str(HERE/'web_source/mlserver/static/cdn/js/plotly-2.8.3.min.js'))
three = (PREVIOUS/'final_source_browser/public_three_tasks.cjs').read_text(encoding='utf-8').replace('014d65b4e0298dcab5f401ea9eabe0de3fe9a898', FINAL)
layout = []
for folder in ['principal', 'xgb_binary', 'xgb_multiclass', 'oneclick_repeat']:
    root = ROOT/folder; root.mkdir(exist_ok=True); (root/'private_access').mkdir(exist_ok=True)
    (root/'release_plan.json').write_text(json.dumps({'source_commit': FINAL})+'\n', encoding='utf-8')
    if folder in ['principal', 'oneclick_repeat', 'xgb_binary']:
        code = binary
        if folder == 'xgb_binary':
            first = code.index("      await page.locator('#task_logistic').click();")
            last = code.index("    }else await page.locator('#button_oc').click();", first)
            code = code[:first]+"      await page.locator('#task_xgboost').click();\n      await page.locator('#xgboost_n_estimators').fill('100');\n"+code[last:]
        (root/'public_browser.cjs').write_text(code, encoding='utf-8')
    if folder in ['principal', 'xgb_multiclass']:
        shutil.copytree(PREVIOUS/'inputs', root/'inputs', dirs_exist_ok=True)
        code = three
        if folder == 'xgb_multiclass':
            code = code.replace("family:'task_logistic'", "family:'task_xgboost'")
            code = code.replace("await page.locator('#logistic_penalty').selectOption('l2');await page.locator('#logistic_solver').selectOption('liblinear');await page.locator('#logistic_C').fill('1.0');", "await page.locator('#xgboost_n_estimators').fill('100');")
        (root/'public_three_tasks.cjs').write_text(code, encoding='utf-8')
    if folder == 'principal':
        for task in ['custom', 'oneclick', 'multiclass', 'regression', 'survival']:
            layout.append({'name': 'binary_custom' if task == 'custom' else task, 'root': folder,
                'mode': task, 'binary_harness': task in ['custom', 'oneclick']})
    else:
        layout.append({'name': folder, 'root': folder, 'mode': 'multiclass' if folder == 'xgb_multiclass' else 'oneclick' if folder == 'oneclick_repeat' else 'custom',
                       'binary_harness': folder != 'xgb_multiclass'})
(HERE/'browser_layout.json').write_text(json.dumps(layout, indent=2)+'\n', encoding='utf-8')
print('Prepared 8 final-source production workflows; no CV-budget override.')
