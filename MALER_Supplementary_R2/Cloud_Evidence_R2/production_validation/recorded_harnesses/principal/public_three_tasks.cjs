const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const task=process.argv[2];
const settings={multiclass:{card:'model_mclass',family:'task_logistic',result:'classification_cp_result'},regression:{card:'model_reg',family:'task_ridge',result:'regression_cp_result'},survival:{card:'model_sur',family:'task_survivaltree',result:'survival_cp_result'}};
const cfg=settings[task];if(!cfg)throw Error('Task must be multiclass, regression or survival');
const base='https://www.inbirg.com',out=path.join(__dirname,'browser_'+task),privateDir=path.join(__dirname,'private_access');
fs.mkdirSync(out,{recursive:true});
for(const name of ['analysis.csv','predict.csv','input_manifest.json'])fs.copyFileSync(path.join(__dirname,'inputs',task,name),path.join(out,name));
const assert=(x,m)=>{if(!x)throw Error(m)},sha=d=>crypto.createHash('sha256').update(d).digest('hex');
const expectedSource='961b45a0c1cee66c91c15efac1546af3262cc7db';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true,ignoreHTTPSErrors:false});
  context.setDefaultTimeout(30000);context.setDefaultNavigationTimeout(950000);
  const page=await context.newPage(),errors=[],failed=[],posts=[],targets=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failed.push({url:r.url().split('?')[0],error:r.failure()?.errorText}));
  page.on('response',r=>{if(r.request().method()==='POST')posts.push({path:new URL(r.url()).pathname,status:r.status(),origin_only_referer:r.request().headers()['referer']===base+'/'})});
  page.on('dialog',d=>d.accept());
  const progress=stage=>{const record={task,stage,client_utc:new Date().toISOString()};fs.writeFileSync(path.join(__dirname,'progress_'+task+'.json'),JSON.stringify(record));console.log(task+': '+stage)};
  const retainDelete=async()=>{
   targets.push(await page.locator('form[action$="/delete"]').evaluate(e=>({action:e.getAttribute('action'),form:Object.fromEntries(new FormData(e).entries())})));
   await context.storageState({path:path.join(privateDir,'browser_state_'+task+'.json')});
   fs.writeFileSync(path.join(privateDir,'cleanup_'+task+'.json'),JSON.stringify(targets,null,2));
  };
  progress('opening public analysis');
  await page.goto(base+'/maler/analysis',{waitUntil:'load',timeout:60000});
  assert((await page.locator('body').innerText()).includes('Public research uploads are available.'),'Uploads unavailable');
  await page.locator('#'+cfg.card).click();await page.locator('#upload_file').setInputFiles(path.join(out,'analysis.csv'));
  await page.locator('#data_consent').check();await page.locator('#button_cp').click();await page.locator('#'+cfg.family).click();
  if(task==='multiclass'){
   await page.locator('#logistic_penalty').selectOption('l2');await page.locator('#logistic_solver').selectOption('liblinear');await page.locator('#logistic_C').fill('1.0');
  }else if(task==='regression')await page.locator('#ridge_alphas').fill('[1.0]');
  else{
   await page.locator('#survivaltree_max_depth').fill('3');await page.locator('#survivaltree_min_samples_leaf').fill('4');
  }
  assert(await page.locator('#to_mail').inputValue()==='','Email address must remain empty');
  await page.waitForFunction(()=>document.querySelector('#file_md5').value.length>0);
  await Promise.all([page.waitForURL('**/maler/preview'),page.locator('#button_submit').click()]);await page.waitForLoadState('load');
  await page.screenshot({path:path.join(out,'analysis_preview.png')});progress('analysis preview');
  await Promise.all([page.waitForURL('**/maler/'+cfg.result+'/**'),page.locator('#button_submit').click()]);await page.waitForLoadState('load');
  assert((await page.locator('body').innerText()).includes('Selected family:'),'Analysis failed or selected-family result absent');
  await retainDelete();await page.screenshot({path:path.join(out,'analysis_result.png')});progress('development result');
  const jsonHref=await page.getByRole('link',{name:'Download complete JSON result'}).getAttribute('href');
  const res=await context.request.get(new URL(jsonHref,base).href);assert(res.status()===200,'Result JSON unavailable');
  const result=await res.json();fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(result,null,2)+'\n');
  assert(result.software.commit===expectedSource&&result.software.verified&&!result.software.dirty,'Wrong production source');
  const modelLink=page.locator('a[href*="/model/"]'),modelHref=await modelLink.getAttribute('href'),modelName=path.posix.basename(new URL(modelHref,base).pathname);
  const download=page.waitForEvent('download');await modelLink.click();const bundle=path.join(out,'browser_model.maler');await(await download).saveAs(bundle);
  await page.goto(base+'/maler/predict',{waitUntil:'load'});await page.locator('#'+cfg.card).click();
  await page.locator('#upload_model').setInputFiles(bundle);await page.locator('#upload_file').setInputFiles(path.join(out,'predict.csv'));await page.locator('#data_consent').check();
  await Promise.all([page.waitForURL('**/maler/predict_preview'),page.locator('#button_submit').click()]);await page.waitForLoadState('load');
  const predictionId=await page.locator('#projectid').inputValue();await page.screenshot({path:path.join(out,'predict_preview.png')});progress('Predict preview');
  await Promise.all([page.waitForURL('**/maler/predict_result/**'),page.locator('#button_submit').click()]);await page.waitForLoadState('load');
  await retainDelete();await page.screenshot({path:path.join(out,'predict_result.png')});
  const rows=await page.evaluate(()=>({blind:window.jQuery('#acctable').DataTable().rows().data().toArray(),validation:window.jQuery('#valdata').DataTable().rows().data().toArray(),survival_curves:(document.querySelector('#survplotDiv')?.data||[]).map(t=>({name:t.name,x:Array.from(t.x),y:Array.from(t.y)}))}));
  fs.writeFileSync(path.join(out,'browser_rows.json'),JSON.stringify(rows,null,2)+'\n');
  assert(rows.blind.length===10&&rows.validation.length===30,'Expected ten blind and thirty labeled predictions');
  const body=await page.locator('body').innerText();assert(body.includes('Signed-model validation metrics:'),'Validation metrics missing');
  if(task==='survival')assert(rows.survival_curves.length===20&&body.includes('censoring reference'),'Survival outputs/limitation missing');
  const expectedAbort=failed.filter(r=>r.error==='net::ERR_ABORTED'&&r.url.endsWith('/model/'+modelName)&&fs.statSync(bundle).size>0);
  const unexpected=failed.filter(r=>!expectedAbort.includes(r));
  const report={task,completed_utc:new Date().toISOString(),scope:'Actual public HTTPS production upload-to-Predict',origin:base,tls_verification_disabled:false,
   browser_version:browser.version(),source:result.software,analysis_project_id:result.project_id,prediction_project_id:predictionId,
   model_file_name:modelName,model_bundle_sha256:sha(fs.readFileSync(bundle)),selection_lock_sha256:result.selection_lock_sha256,
   successful_posts:posts,page_errors:errors,unexpected_failed_requests:unexpected,expected_download_aborts:expectedAbort,
   prediction_rows:40,blind_rows:10,validation_rows:30,actual_budget:result.selection.validation_budget,email_address_supplied:false};
  fs.writeFileSync(path.join(out,'browser_report.json'),JSON.stringify(report,null,2)+'\n');
  assert(!errors.length&&!unexpected.length,'Browser JavaScript/resource failures');
  assert(posts.length===4&&posts.every(r=>r.status===200&&r.origin_only_referer),'Unexpected form result or Referer');
  progress('complete');console.log(JSON.stringify(report,null,2));
 }catch(e){const p=browser.contexts().flatMap(c=>c.pages()).pop();if(p)await p.screenshot({path:path.join(privateDir,'failure_'+task+'.png')}).catch(()=>{});throw e}
 finally{await browser.close()}
})().catch(e=>{console.error(e.stack);process.exit(1)});
