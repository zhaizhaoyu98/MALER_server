const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path');
const out=path.join(__dirname,'public_pages');fs.mkdirSync(out,{recursive:true});
const assert=(x,m)=>{if(!x)throw Error(m)};
(async()=>{
 const b=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://']});
 try{
  const c=await b.newContext({viewport:{width:1440,height:1000},ignoreHTTPSErrors:false}),records=[];
  for(const name of ['home','analysis','predict','help']){
   const p=await c.newPage(),errors=[],failed=[];
   p.on('pageerror',e=>errors.push(e.message));p.on('requestfailed',r=>failed.push({url:r.url().split('?')[0],error:r.failure()?.errorText}));
   const response=await p.goto('https://www.inbirg.com/maler/'+name,{waitUntil:'load',timeout:60000});
   const text=await p.locator('body').innerText();
   assert(response.status()===200&&text.includes('Public research uploads are available.'),'Availability check failed');
   assert(response.headers()['referrer-policy']==='strict-origin','Referrer policy differs');
   if(['analysis','predict'].includes(name))assert(await p.locator('#button_submit').isEnabled(),'Submit disabled');
   if(['analysis','help'].includes(name))assert(text.includes('5 outer folds × 2 repeats')&&text.includes('5 outer folds × 10 repeats'),'Budget notice differs');
   assert(!errors.length&&!failed.length,'Browser page errors/resources failed');
   await p.screenshot({path:path.join(out,name+'.png'),fullPage:true});
   records.push({page:name,status:response.status(),public_notice_visible:true,referrer_policy:'strict-origin',page_errors:errors,failed_requests:failed});
   await p.close();
  }
  const result={client_utc:new Date().toISOString(),browser:b.version(),origin:'https://www.inbirg.com',tls_verification_disabled:false,passed:true,records};
  fs.writeFileSync(path.join(out,'public_pages_report.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
 }finally{await b.close()}
})().catch(e=>{console.error(e.stack);process.exit(1)});
