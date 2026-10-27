"""Copy the fixed SVG-derived diagram exports; no model analysis is involved."""
import hashlib
import os
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['MALER_FIGURE_OUTPUT_DIR']); OUT.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf'):
    p=HERE/'workflow_exports'/('Figure_1_workflow_R2.'+ext)
    shutil.copyfile(str(p),str(OUT/p.name))
shutil.copyfile(str(HERE/'Figure_1_workflow_R2.svg'),str(OUT/'Figure_1_workflow_R2.svg'))
print('Static workflow: R1 SVG with local text/path edits; archived exports copied.')
