const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const out=path.join(__dirname,'interface_capture');fs.mkdirSync(out,{recursive:true});
const assert=(x,m)=>{if(!x)throw Error(m)};
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1100},deviceScaleFactor:2,ignoreHTTPSErrors:false});
  const records=[],pages={},errors=[],failures=[];
  for(const name of ['home','analysis','predict']){
   const page=await context.newPage();page.on('pageerror',e=>errors.push({page:name,error:e.message}));page.on('requestfailed',r=>failures.push({page:name,url:r.url().split('?')[0],error:r.failure()?.errorText}));
   const response=await page.goto('https://www.inbirg.com/maler/'+name,{waitUntil:'load',timeout:60000});
   const text=await page.locator('body').innerText();
   assert(response.status()===200&&text.includes('Public research uploads are available.'),'Public page unavailable');
   assert(!/no larger than 200\s?M[Bb]/.test(text),'Stale file-size description');
   if(name!=='home')assert(text.includes('no larger than 50 MiB'),'Configured data limit missing');
   assert(response.headers()['referrer-policy']==='strict-origin','Referrer policy differs');
   records.push({page:name,status:200,body_sha256:crypto.createHash('sha256').update(await response.body()).digest('hex'),
    tls_verification_disabled:false,public_uploads_enabled_notice:true,old_200MB_description_absent:true});
   pages[name]=page;
  }
  const panels=[];
  async function capture(letter,page,locator,title,maxHeight){
   await locator.scrollIntoViewIfNeeded();const box=await locator.boundingBox();assert(box,'Panel missing');
   if(maxHeight){await page.screenshot({path:path.join(out,letter+'.png'),clip:{x:box.x,y:box.y,width:box.width,height:Math.min(box.height,maxHeight)}});}
   else await locator.screenshot({path:path.join(out,letter+'.png')});
   const data=fs.readFileSync(path.join(out,letter+'.png'));
   panels.push({letter,title,url:page.url(),selector_scope:letter==='A'?'Home content crop':'Visible original card',
    bounds_css:box,crop_height_css:maxHeight||null,bytes:data.length,sha256:crypto.createHash('sha256').update(data).digest('hex')});
  }
  const card=(page,prefix)=>page.locator('h4').filter({hasText:prefix}).first().locator('xpath=ancestor::div[contains(@class,"is-never-shadow")][1]');
  const home=pages.home.locator('body > div.container').filter({has:pages.home.locator('h1')}).first();
  await capture('A',pages.home,home,'Home page and reviewed workflow',1180);
  await capture('B',pages.analysis,card(pages.analysis,'Step 1:'),'Analysis task selection');
  await capture('C',pages.analysis,card(pages.analysis,'Step 2:'),'Upload and authorization');
  await capture('D',pages.analysis,pages.analysis.locator('#scaling-guidance-panel'),'Feature scaling');
  await capture('E',pages.analysis,card(pages.analysis,'Step 4:'),'Feature ranking and subset search');
  await capture('F',pages.analysis,card(pages.analysis,'Step 5:'),'One-click and customized analysis');
  await capture('G',pages.predict,card(pages.predict,'Step2:'),'Signed-model reuse and data handling');
  assert(!errors.length&&!failures.length,'Browser errors or unexpected resources');
  const result={captured_utc:new Date().toISOString(),capture_date:'2026-10-24',origin:'https://www.inbirg.com',
   source_commit:JSON.parse(fs.readFileSync(path.join(__dirname,'release_plan.json'),'utf8')).source_commit,
   browser:browser.version(),viewport_css:{width:1440,height:1100},device_scale_factor:2,
   page_content_or_style_modified:false,form_submitted:false,panels,records,page_errors:errors,unexpected_failed_requests:failures,
   scope:'Seven authentic public HTTPS interface crops. Original page text, styles and controls are unchanged; no upload or model run is performed by this capture.'};
  fs.writeFileSync(path.join(out,'capture_manifest.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result,null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e.stack);process.exit(1)});
