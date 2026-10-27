const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path');
const base='https://www.inbirg.com',assert=(ok,message)=>{if(!ok)throw Error(message)};
const preliminary=process.argv[2]==='preliminary';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://']});
 const records=[];
 try{
  const all=JSON.parse(fs.readFileSync(path.join(__dirname,'browser_layout.json'),'utf8'));
  const layout=preliminary?all.filter(row=>['binary_custom','oneclick','xgb_binary','xgb_multiclass'].includes(row.name)):all;
  for(const row of layout){
   const folder=path.join(__dirname,'browser',row.root,'private_access');
   const context=await browser.newContext({storageState:path.join(folder,'browser_state_'+row.mode+'.json'),ignoreHTTPSErrors:false});
   const targets=JSON.parse(fs.readFileSync(path.join(folder,'cleanup_'+row.mode+'.json'),'utf8'));
   assert(targets.length===2,'Expected only the two identified synthetic tasks');
   for(const target of targets){
    const matched=/^\/maler\/task\/([A-Za-z0-9_-]+)\/delete$/.exec(target.action);assert(matched,'Invalid target');
    const denied=await context.request.post(base+target.action,{form:{csrfmiddlewaretoken:target.form.csrfmiddlewaretoken},headers:{Referer:base+'/maler/analysis'}});
    assert(denied.status()===403,'Missing capability did not prevent deletion');
    const removed=await context.request.post(base+target.action,{form:target.form,headers:{Referer:base+'/maler/analysis'}});
    assert(removed.status()===200,'Protected Delete failed');
    const after=await context.request.get(base+'/maler/task/'+matched[1]+'/result.json?token='+encodeURIComponent(target.form.access_token));
    assert(after.status()===403,'Deleted task remains accessible');
    records.push({workflow:row.name,project_id:matched[1],missing_token_status:403,delete_final_status:200,after_delete_status:403});
   }
   await context.close();
  }
  assert(records.length===(preliminary?8:16),'Wrong synthetic task count');
  fs.writeFileSync(path.join(__dirname,preliminary?'preliminary_delete_report.json':'public_delete_report.json'),JSON.stringify({completed_utc:new Date().toISOString(),scope:'Explicit protected deletion of only identified synthetic tasks',records},null,2)+'\n');
  console.log('Protected Delete verified for '+records.length+' identified synthetic tasks.');
 }finally{await browser.close()}
})().catch(error=>{console.error(error.stack);process.exit(1)});
