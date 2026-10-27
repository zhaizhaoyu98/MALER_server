const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const mode=process.argv[2]||'custom';
if(!['custom','oneclick'].includes(mode))throw Error('Use custom or oneclick');
const plan=JSON.parse(fs.readFileSync(path.join(__dirname,'release_plan.json'),'utf8'));
const inputs=String.raw`E:\CodeProject\Server\maler_revision2\r2_followup_implementation\deep_audit_20261001\evidence\product_missingness\rate_10`;
const out=path.join(__dirname,'browser_'+mode);fs.mkdirSync(out,{recursive:true});
for(const name of ['analysis.csv','predict_reordered.csv'])fs.copyFileSync(path.join(inputs,name),path.join(out,name));
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const assert=(x,msg)=>{if(!x)throw Error(msg)};
const base='https://www.inbirg.com';
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://']});
  try{
    const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true,ignoreHTTPSErrors:false});
    context.setDefaultTimeout(30000);
    context.setDefaultNavigationTimeout(950000);
    const page=await context.newPage(),errors=[],failed=[],posts=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('requestfailed',r=>failed.push({url:r.url().split('?')[0],error:r.failure()?.errorText}));
    page.on('response',r=>{if(r.request().method()==='POST')posts.push({path:new URL(r.url()).pathname,status:r.status(),referer_origin_only:r.request().headers()['referer']===base+'/'})});
    page.on('dialog',d=>d.accept());
    const progress=stage=>{fs.writeFileSync(path.join(__dirname,'progress_'+mode+'.json'),JSON.stringify({stage,client_utc:new Date().toISOString(),path:new URL(page.url()==='about:blank'?base:page.url()).pathname}));console.log('Public '+mode+': '+stage)};
    progress('opening analysis');
    await page.goto(base+'/maler/analysis',{waitUntil:'load',timeout:180000});
    const plotly=await context.request.get(base+'/mlserver_static/cdn/js/plotly-2.8.3.min.js',{timeout:90000});
    const bundledPlotly=fs.readFileSync(String.raw`E:\CodeProject\Server\maler_revision2\r2_followup_implementation\figure_style_revision_20261024\web_source\mlserver\static\cdn\js\plotly-2.8.3.min.js`,'utf8').replace(/\r\n/g,'\n');
    assert(plotly.status()===200&&digest(await plotly.body())===digest(bundledPlotly),'Public Plotly differs from the bundled 2.8.3 asset (canonical LF)');
    assert((await page.locator('body').innerText()).includes('Public research uploads are available.'),'Public availability notice missing');
    assert(await page.locator('#button_submit').isEnabled(),'Public Submit remains disabled');
    assert(await page.evaluate(()=>typeof window.jQuery)==='function','jQuery did not load');
    const csrf=await context.request.post(base+'/maler/preview',{form:{projectid:'invalid-csrf-check'}});
    assert(csrf.status()===403,'Missing CSRF was not rejected');
    await page.locator('#model_bclass').click();
    await page.locator('#upload_file').setInputFiles(path.join(out,'analysis.csv'));
    await page.locator('#data_consent').check();
    progress('data selected');
    if(mode==='custom'){
      await page.locator('#button_cp').click();
      await page.locator('#task_logistic').click();
      progress('custom logistic selected');
      const compatible={l1:['liblinear','saga'],l2:['lbfgs','liblinear','newton-cg','sag','saga'],none:['lbfgs','newton-cg','sag','saga']};
      const untouched=await page.locator('#logistic_C').inputValue();
      for(const [penalty,solvers] of Object.entries(compatible)){
        await page.locator('#logistic_penalty').selectOption(penalty);
        const actual=await page.locator('#logistic_solver option').evaluateAll(options=>options.map(o=>o.value));
        assert(JSON.stringify(actual)===JSON.stringify(solvers),'Incompatible logistic solver choices');
        assert(await page.locator('#logistic_C').inputValue()===untouched,'Penalty change cleared another control');
      }
      await page.locator('#logistic_penalty').selectOption('l2');
      await page.locator('#logistic_solver').selectOption('liblinear');
      await page.locator('#logistic_C').fill('1.0');
    }else await page.locator('#button_oc').click();
    assert(await page.locator('#to_mail').inputValue()==='','Email must remain empty');
    await page.waitForFunction(()=>document.querySelector('#file_md5').value.length>0);
    progress('submitting preview');
    await Promise.all([page.waitForURL('**/maler/preview'),page.locator('#button_submit').click()]);
    await page.waitForLoadState('load');
    progress('preview received');
    assert(await page.evaluate(()=>typeof Plotly)==='object','Bundled preview Plotly did not load');
    await page.locator('#positive_class').selectOption('case');
    await page.locator('#threshold_policy').selectOption('explicit_probability_threshold');
    await page.locator('#threshold_value').fill('0');
    assert(await page.locator('#threshold_value').evaluate(e=>e.checkValidity()),'Cutoff zero rejected');
    await page.locator('#threshold_value').fill('.65');
    await page.screenshot({path:path.join(out,'binary_preview_cutoff.png')});
    const resultPath=mode==='custom'?'classification_cp_result':'classification_oc_result';
    const sent=page.waitForRequest(r=>r.method()==='POST'&&new URL(r.url()).pathname.includes('/'+resultPath+'/'));
    const arrived=page.waitForURL('**/maler/'+resultPath+'/**');
    const click=page.locator('#button_submit').click({noWaitAfter:true});
    await sent;
    let busyStatus=null;
    if(mode==='oneclick'){
      await new Promise(resolve=>setTimeout(resolve,2000));
      const busy=await context.request.get(base+'/maler/classification_oc_result/PUBLIC_BUSY_PROBE',{timeout:30000});
      busyStatus=busy.status();
    }
    await Promise.all([click,arrived]);
    await page.waitForLoadState('load');
    progress('model result received');
    assert((await page.locator('body').innerText()).includes('Selected family:'),'Development result missing');
    await page.screenshot({path:path.join(out,'binary_result.png')});
    const resultHref=await page.getByRole('link',{name:'Download complete JSON result'}).getAttribute('href');
    const response=await context.request.get(new URL(resultHref,base).href);
    assert(response.status()===200,'Result JSON inaccessible');
    const result=await response.json();fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(result,null,2)+'\n');
    assert(result.software.commit===plan.source_commit&&result.software.verified&&!result.software.dirty,'Wrong/dirty production source');
    assert(result.validation.threshold_value===.65,'Cutoff changed');
    const lockHash=result.selection_lock_sha256;assert(/^[a-f0-9]{64}$/.test(lockHash),'Lock hash missing');
    const modelLink=page.locator('a[href*="/model/"]');
    const modelHref=await modelLink.getAttribute('href');
    const cleanupTargets=[await page.locator('form[action$="/delete"]').evaluate(el=>({action:el.getAttribute('action'),form:Object.fromEntries(new FormData(el).entries())}))];
    const modelName=path.posix.basename(new URL(modelHref,base).pathname);
    const downloadPromise=page.waitForEvent('download');await modelLink.click();
    const modelPath=path.join(out,'browser_model.maler');await(await downloadPromise).saveAs(modelPath);
    await page.goto(base+'/maler/predict',{waitUntil:'load'});
    assert(await page.locator('#button_submit').isEnabled(),'Predict remains disabled');
    await page.locator('#model_bclass').click();
    await page.locator('#upload_model').setInputFiles(modelPath);
    await page.locator('#upload_file').setInputFiles(path.join(out,'predict_reordered.csv'));
    await page.locator('#data_consent').check();
    await Promise.all([page.waitForURL('**/maler/predict_preview'),page.locator('#button_submit').click()]);
    await page.waitForLoadState('load');await page.screenshot({path:path.join(out,'predict_preview.png')});
    const predictionId=await page.locator('#projectid').inputValue();
    await Promise.all([page.waitForURL('**/maler/predict_result/**'),page.locator('#button_submit').click()]);
    await page.waitForLoadState('load');await page.screenshot({path:path.join(out,'predict_result.png')});
    const rows=await page.evaluate(()=>window.jQuery('#acctable').DataTable().rows().data().toArray());
    assert(rows.length===40,'Prediction row count mismatch');
    fs.writeFileSync(path.join(out,'browser_prediction_rows.json'),JSON.stringify(rows,null,2)+'\n');
    cleanupTargets.push(await page.locator('form[action$="/delete"]').evaluate(el=>({action:el.getAttribute('action'),form:Object.fromEntries(new FormData(el).entries())})));
    await context.storageState({path:path.join(__dirname,'private_access','browser_state_'+mode+'.json')});
    fs.writeFileSync(path.join(__dirname,'private_access','cleanup_'+mode+'.json'),JSON.stringify(cleanupTargets,null,2));
    const expectedDownloadAborts=failed.filter(r=>r.error==='net::ERR_ABORTED'&&r.url.endsWith('/model/'+modelName)&&fs.statSync(modelPath).size>0);
    const unexpected=failed.filter(r=>!expectedDownloadAborts.includes(r));
    const report={scope:'Actual public HTTPS production upload-to-Predict workflow; no SSH forwarding for browser requests',
      completed_utc:new Date().toISOString(),mode,source:result.software,browser_version:browser.version(),origin:base,
      tls_verification_disabled:false,email_address_supplied:false,analysis_project_id:result.project_id,
      prediction_project_id:predictionId,model_file_name:modelName,
      analysis_input_sha256:digest(fs.readFileSync(path.join(out,'analysis.csv'))),
      prediction_input_sha256:digest(fs.readFileSync(path.join(out,'predict_reordered.csv'))),
      model_bundle_sha256:digest(fs.readFileSync(modelPath)),selection_lock_sha256:lockHash,
      positive_class:result.positive_class,threshold_policy:result.validation.threshold_policy,threshold_value:result.validation.threshold_value,
      actual_budget:result.selection.validation_budget,csrf_missing_token_status:csrf.status(),
      concurrent_compute_probe_status:busyStatus,successful_form_posts:posts,page_errors:errors,
      expected_download_aborts:expectedDownloadAborts,unexpected_failed_requests:unexpected,browser_prediction_count:rows.length,
      steps:['public HTTPS upload','parse preview','explicit positive class/cutoff','development/result','signed model download','Predict upload','Predict preview','Predict reuse']};
    fs.writeFileSync(path.join(out,'browser_e2e_report.json'),JSON.stringify(report,null,2)+'\n');
    assert(errors.length===0,'JavaScript errors');assert(unexpected.length===0,'Resource failures');
    assert(posts.length===4&&posts.every(r=>r.status===200&&r.referer_origin_only),'Unexpected form result statuses or Referer policy');
    console.log(JSON.stringify(report,null,2));
  }catch(error){
    const pages=browser.contexts().flatMap(c=>c.pages());
    if(pages.length)await pages[pages.length-1].screenshot({path:path.join(__dirname,'private_access','failure_'+mode+'.png')}).catch(()=>{});
    throw error;
  }finally{await browser.close()}
})().catch(e=>{console.error(e.stack);process.exit(1)});
