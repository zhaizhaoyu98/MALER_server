"""R1 A/B, C/D, E/F, full-width G layout with authentic current captures."""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
CAPTURE=HERE.parent.parent/'Cloud_Evidence_R2/production_validation/interface_capture'
OUT=Path(os.environ['MALER_FIGURE_OUTPUT_DIR']); OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((CAPTURE/'capture_manifest.json').read_text(encoding='utf-8'))
assert not manifest['page_content_or_style_modified'] and not manifest['form_submitted']
assert len(manifest['panels'])==7 and not manifest['page_errors'] and not manifest['unexpected_failed_requests']
images={}
for row in manifest['panels']:
    p=CAPTURE/(row['letter']+'.png')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    images[row['letter']]=Image.open(p).convert('RGB')

width=4204; outer=120; gap=128; header=105; rowgap=128; pad=8
col=(width-2*outer-gap)/2
widths={letter:col for letter in images}; widths['G']=2*col+gap
heights={letter:header+2*pad+im.height/im.width*(widths[letter]-2*pad) for letter,im in images.items()}
pairs=[('A','B'),('C','D'),('E','F')]
rowheights=[max(heights[a],heights[b]) for a,b in pairs]
height=round(2*outer+sum(rowheights)+rowgap*3+heights['G'])
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(width/300,height/300),facecolor='white')
titles={'A':'Home page and validated workflow','B':'Analysis task selection',
        'C':'Upload, privacy and access controls','D':'Feature-scaling guidance',
        'E':'Feature ranking and subset search','F':'Analysis strategy and validation',
        'G':'Signed model and prediction'}
top=outer; positions=[]
for pair,rowheight in list(zip(pairs,rowheights))+[(('G',),heights['G'])]:
    for column,letter in enumerate(pair):
        x=outer+column*(col+gap); w=widths[letter]; h=heights[letter]
        ax=fig.add_axes([x/width,1-(top+h)/height,w/width,h/height])
        ax.set(xlim=(0,w),ylim=(h,0)); ax.axis('off')
        ax.add_patch(Rectangle((0,0),w,h,linewidth=.7,edgecolor='#C8CDD1',facecolor='none'))
        ax.text(52,header*.54,letter,fontsize=17,fontweight='bold',va='center',color='black')
        ax.text(160,header*.54,titles[letter],fontsize=14.5,va='center',color='black')
        ax.plot([0,w],[header,header],color='#D7DDE1',lw=.6)
        ax.imshow(images[letter],extent=(pad,w-pad,h-pad,header+pad),aspect='auto',interpolation='none')
        positions.append({'panel':letter,'left':x,'top':top,'width':w,'height':h,
                          'raw_pixels':list(images[letter].size),'full_capture_used':True})
    top+=rowheight+rowgap
for ext in ('png','pdf'):
    fig.savefig(OUT/('Figure_2_interface_R2.'+ext),dpi=300,facecolor='white')
plt.close(fig)
audit={'source_commit':manifest['source_commit'],'capture_utc':manifest['captured_utc'],
       'layout':'R1 rows A/B, C/D, E/F, and full-width G','panels':positions,
       'website_text_or_style_modified':False,'nonadjacent_detail_montage':False,
       'raw_capture_manifest_sha256':hashlib.sha256((CAPTURE/'capture_manifest.json').read_bytes()).hexdigest(),
       'figure_pixels':[width,height],
       'note':'Panel heights expand to accommodate the current governance and validation text.'}
(OUT/'Figure_2_R1_layout_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'interface_pixels':[width,height],'r1_panel_order_preserved':True}))
