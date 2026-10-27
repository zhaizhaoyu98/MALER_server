"""Actual MALER upload, nested fit, signed export and Predict under synthetic missingness."""
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=next(base for base in [ROOT/'site_source']+list(ROOT.parents)
            if (base/'manage.py').is_file() and (base/'mlserver').is_dir())
OUT=Path(os.environ.get('MALER_PRODUCT_VALIDATION_OUTPUT') or (
    ROOT/'validation'/'product_missingness' if (ROOT/'site_source').is_dir()
    else SOURCE/'validation'/'submission_evidence'/'product_missingness'))
sys.path.insert(0,str(SOURCE))
os.environ.setdefault('DJANGO_SETTINGS_MODULE','ML_WebServer.settings')
import django
django.setup()
from django.test import Client, override_settings
from django.test.utils import setup_test_environment
from django.core.files.uploadedfile import SimpleUploadedFile
from mlserver.safe_ml import FoldImputer, load_signed_model_bundle, classification_prediction
from mlserver.source_version import source_identity

SEED=20260929
KEY='synthetic-only-validation-key'

def digest(value): return hashlib.sha256(value).hexdigest()
def context(response): return response.context[-1] if isinstance(response.context,list) else response.context
def require(response,label):
    if response.status_code != 200:
        raise AssertionError('%s: HTTP %s %s'%(label,response.status_code,response.content[:1000]))

def run_one(rate):
    rng=np.random.RandomState(SEED)
    train=rng.normal(size=(80,8)); heldout=rng.normal(size=(40,8))
    train[:,0]=np.tile(np.arange(5,dtype=float),16); heldout[:,0]=1000.
    labels=np.where(train[:,1]+.6*train[:,2]>0,'case','control')
    heldout_labels=np.where(heldout[:,1]+.6*heldout[:,2]>0,'case','control')
    train_mask=rng.rand(*train.shape)<rate; test_mask=rng.rand(*heldout.shape)<rate
    train[train_mask]=np.nan; heldout[test_mask]=np.nan; heldout[0,0]=np.nan
    ids=['T%03d'%i for i in range(80)]+['H%03d'%i for i in range(40)]
    columns=['gene%d'%i for i in range(8)]
    features=pd.DataFrame(np.r_[train,heldout],index=ids,columns=columns)
    raw=features.T.copy(); raw.loc['label']=np.r_[labels,heldout_labels]
    raw.loc['set']=['training']*80+['testing']*40
    analysis_bytes=raw.loc[['label','set']+columns].to_csv().encode()
    predict_bytes=features.iloc[80:,::-1].T.to_csv().encode()
    fits=[]
    original_fit=FoldImputer.fit
    def audited_fit(self,X,y=None):
        if not isinstance(X,pd.DataFrame) or not all(str(i).startswith('T') for i in X.index):
            raise AssertionError('A fitting operation used non-development rows')
        value=original_fit(self,X,y)
        expected=np.nanmedian(X.to_numpy(dtype=float),axis=0)
        np.testing.assert_allclose(self.statistics_,expected,rtol=0,atol=0)
        fits.append({'n_train':len(X),'training_rows_sha256':digest('\n'.join(X.index).encode()),
                     'sentinel_median':float(self.statistics_[0]),
                     'statistics_sha256':digest(self.statistics_.tobytes())})
        return value
    target=OUT/('rate_%02d'%round(100*rate)); target.mkdir(parents=True,exist_ok=True)
    (target/'analysis.csv').write_bytes(analysis_bytes)
    (target/'predict_reordered.csv').write_bytes(predict_bytes)
    repeated=[]
    for repetition in range(2):
        with tempfile.TemporaryDirectory(prefix='maler-product-check-') as temporary:
            with override_settings(MALER_TASK_ROOT=str(Path(temporary)/'tasks'),
                    MALER_PUBLIC_UPLOADS_ENABLED=True,MALER_MODEL_SIGNING_KEY=KEY,
                    MALER_REQUIRE_VERIFIED_RELEASE=True,
                    MALER_WEB_MODELS={'classification':['logistic_regression']},
                    MALER_WEB_OUTER_SPLITS=3,MALER_WEB_OUTER_REPEATS=1,
                    MALER_WEB_INNER_SPLITS=2,MALER_WEB_N_JOBS=1):
                client=Client()
                preview=client.post('/maler/preview',{
                    'projectid':'BCO-AZ-synthetic-TopK','file_upload_type':'user_data',
                    'select_model':'model_bclass','strategy':'O','feature_select_method':'TopK',
                    'fsm':'A','feature_norm':'Z','data_consent':'confirmed',
                    'upload_file':SimpleUploadedFile('analysis.csv',analysis_bytes,content_type='text/csv')})
                require(preview,'upload/parse'); ctx=context(preview)
                pid,token=ctx['projectid'],ctx['access_token']
                task=Path(temporary)/'tasks'/pid
                with patch.object(FoldImputer,'fit',audited_fit):
                    response=client.post('/maler/classification_oc_result/'+pid,{
                        'access_token':token,'feature_select_method':'TopK','feature_norm':'Z','fsm':'A',
                        'positive_class':'case','threshold_policy':'explicit_probability_threshold',
                        'threshold_value':'.65'})
                require(response,'nested analysis')
                result=json.loads((task/'validated_result.json').read_text())
                selected=result['models'][0]
                model,manifest=load_signed_model_bundle(str(task/selected['model_bundle']),KEY)
                expected_median=np.nanmedian(train,axis=0)
                # CSV serialization/parsing can round original binary floats.
                # Every actual fit above was independently checked at zero tolerance.
                np.testing.assert_allclose(model.named_steps['imputer'].statistics_,expected_median,rtol=0,atol=1e-12)
                snapshot=model.named_steps['imputer'].statistics_.copy()
                actual_imputed=model.named_steps['imputer'].transform(features.iloc[80:])
                if actual_imputed[0,0]!=expected_median[0]: raise AssertionError('Incorrect heldout imputation')
                prediction,score,_=classification_prediction(model,features.iloc[80:],'case',
                                                            'explicit_probability_threshold',.65)
                lock_before=(task/'selection_lock.json').read_bytes()
                bundle=(task/selected['model_bundle']).read_bytes()
                predict_preview=client.post('/maler/predict_preview',{
                    'file_upload_type':'user_data','select_model':'model_bclass','data_consent':'confirmed',
                    'upload_model':SimpleUploadedFile('model.maler',bundle),
                    'upload_file':SimpleUploadedFile('predict.csv',predict_bytes,content_type='text/csv')})
                require(predict_preview,'Predict upload'); pc=context(predict_preview)
                count_before=len(fits)
                with patch.object(FoldImputer,'fit',side_effect=AssertionError('Predict refit the imputer')):
                    response=client.post('/maler/predict_result/'+pc['projectid'],{'access_token':pc['access_token']})
                require(response,'Predict reuse'); records=json.loads(context(response)['predict_reports_dict'])
                actual={row['Name']:row['Label'] for row in records}
                if [actual[i] for i in ids[80:]] != prediction.tolist(): raise AssertionError('Predict label parity failed')
                np.testing.assert_array_equal(snapshot,model.named_steps['imputer'].statistics_)
                if (task/'selection_lock.json').read_bytes()!=lock_before: raise AssertionError('Selection lock changed')
                if manifest['metadata']['selection_lock_sha256']!=digest(lock_before): raise AssertionError('Lock hash mismatch')
                repeated.append(digest(score.tobytes()))
                if repetition==0:
                    for name in ['validated_result.json','selection_lock.json',selected['model_bundle']]:
                        (target/name).write_bytes((task/name).read_bytes())
                    (target/'predict_labels.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    if repeated[0]!=repeated[1]: raise AssertionError('Seeded rerun probabilities changed')
    pooled=float(np.nanmedian(np.r_[train[:,0],heldout[:,0]]))
    if pooled==float(expected_median[0]): raise AssertionError('Leakage sentinel has no contrast')
    return {'rate_random_missingness':rate,'forced_heldout_sentinel_missing':True,
            'input_sha256':digest(analysis_bytes),'prediction_input_sha256':digest(predict_bytes),
            'training_mask_sha256':digest(train_mask.tobytes()),'heldout_mask_sha256':digest(test_mask.tobytes()),
            'train_sentinel_median':float(expected_median[0]),'pooled_leakage_sentinel_median':pooled,
            'heldout_imputed_sentinel':float(actual_imputed[0,0]),'prediction_hashes':repeated,
            'audited_fit_calls':len(fits),'training_only_fit_audit':fits,
            'predict_rows':40,'predict_refit_calls':0,'export_and_predict_label_parity':True,
            'positive_class':'case','class_order':manifest['metadata']['class_order'],
            'threshold_policy':'explicit_probability_threshold','threshold_value':.65}

def main():
    OUT.mkdir(parents=True,exist_ok=True); setup_test_environment()
    report={'scope':'local Django Client actual MALER route integration, not a browser or Aliyun test',
            'source':source_identity(),'python':sys.version.split()[0],'django':django.get_version(),
            'completed_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
            'training_n':80,'heldout_n':40,'features':8,
            'budget':{'outer_splits':3,'outer_repeats':1,'inner_splits':2},
            'runs':[run_one(rate) for rate in [0,.1,.3]]}
    (OUT/'product_missingness_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
    print('All three actual-product missingness cases and reruns passed.')

if __name__=='__main__': main()
