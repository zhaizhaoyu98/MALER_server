"""R1 native plots; only S1 scope and S6 memory-unit labels change."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'source/validation/make_validation_figures.py'
sys.path[:0]=[str(SOURCE.parents[1]),str(SOURCE.parent)]
spec=importlib.util.spec_from_file_location('r1_native_figures',str(SOURCE))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT=Path(os.environ['MALER_FIGURE_OUTPUT_DIR']); OUT.mkdir(parents=True,exist_ok=True)
m.FIGURE_DIR=str(OUT)
load=lambda n:json.loads((SOURCE.parent/'results'/n).read_text(encoding='utf-8'))
original_save=Figure.savefig
records=[]; seen=set()

def numeric(fig):
    rows=[]
    a=lambda x:np.asarray(x).tolist()
    for ax in fig.axes:
        rows.append({'lines':[(a(l.get_xdata()),a(l.get_ydata())) for l in ax.lines],
                     'collections':[(a(c.get_offsets()),[a(p.vertices) for p in c.get_paths()]) for c in ax.collections],
                     'images':[a(i.get_array()) for i in ax.images],
                     'bars':[(p.get_x(),p.get_y(),p.get_width(),p.get_height()) for p in ax.patches]})
    return hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=True,default=lambda x:x.item()).encode()).hexdigest()

def save(fig,path,*args,**kwargs):
    stem=Path(path).stem
    before=numeric(fig)
    if stem=='Figure_S1_internal_validation':
        fig.axes[0].set_title('A  Development and held-out checks',loc='left',color=m.INK,fontweight='bold',y=1.10)
    elif stem=='Figure_S6_capacity':
        fig.axes[0].set_xlabel('Peak working set (MiB)',color=m.INK)
    assert before==numeric(fig),stem
    if stem not in seen:
        seen.add(stem)
        records.append({'figure':stem,'numeric_artist_sha256':before,
                        'numeric_values_unchanged':True,
                        'local_label_edit':{'Figure_S1_internal_validation':'Panel A scope title',
                                            'Figure_S6_capacity':'MB to MiB axis label'}.get(stem),
                        'r1_native_style_preserved':True})
    return original_save(fig,path,*args,**kwargs)

Figure.savefig=save
try:
    external=load('external_gse37745.json'); gse50081=load('external_gse50081.json')
    m.internal_performance_figure(load('validation_results.json'),load('equal_budget_no_selection_baselines.json'),load('pca_sensitivity.json'))
    m.external_validation_figure(external)
    m.external_stability_figure(external)
    m.survival_expanded_figure(load('survival_expanded_validation.json'))
    m.external_calibration_figure(external,gse50081)
    m.capacity_figure(load('capacity_benchmark.json'))
finally: Figure.savefig=original_save
for r in records:
    p=OUT/(r['figure']+'.png'); ref=SOURCE.parent/'figures'/p.name
    r['png_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    r['byte_identical_to_r1']=p.read_bytes()==ref.read_bytes()
    if not r['local_label_edit']: assert r['byte_identical_to_r1'],r['figure']
assert len(records)==6 and sum(r['byte_identical_to_r1'] for r in records)==4
(OUT/'Supplementary_R1_restoration_audit.json').write_text(json.dumps({'figures':records,'all_numeric_values_preserved':True},indent=2)+'\n',encoding='utf-8')
print('Four R1 supplementary PNGs byte-identical; two plots retain only local label edits.')
