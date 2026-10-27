import datetime,hashlib,http.client,json,re,ssl
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import build_opener,ProxyHandler,HTTPSHandler,HTTPRedirectHandler,Request
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*a,**k):return None
HERE=Path(__file__).resolve().parent
context=ssl.create_default_context();opener=build_opener(ProxyHandler({}),HTTPSHandler(context=context),NoRedirect())
def fetch(url,data=None):
    try:r=opener.open(Request(url,data=data),timeout=35)
    except HTTPError as e:r=e
    with r:return r.status,dict(r.headers),r.read()
base='https://www.inbirg.com';rows=[]
for page in ['home','analysis','predict','help']:
    status,headers,body=fetch(base+'/maler/'+page);text=body.decode()
    assert status==200 and headers.get('Cache-Control')=='no-store' and headers.get('Referrer-Policy')=='strict-origin'
    assert 'Public research uploads are available.' in text
    if page=='help':assert 'Registry version 4' in text and 'mapping is included' in text
    if page in ['analysis','predict']:assert 'axios' not in text
    if page in ['analysis','predict']:
        assert 'no larger than 50 MiB' in text and not re.search(r'no larger than 200\s?M[Bb]',text)
    if page in ['analysis','predict']:
        button=re.search(r'<button[^>]*id="button_submit"[^>]*>',text);assert button and 'disabled' not in button[0]
    rows.append({'page':page,'status':status,'no_store':True,'uploads_available_notice':True,'referrer_policy':headers['Referrer-Policy'],'sha256':hashlib.sha256(body).hexdigest()})
status,h,_=fetch('http://www.inbirg.com/maler/home');assert status==308 and h['Location']==base+'/maler/home'
rows.append({'route':'http_home','status':status,'location':h['Location']})
for route in ['preview','predict_preview']:
    for method,data,expected in [('GET',None,400),('POST',b'',403)]:
        status,_,_=fetch(base+'/maler/'+route,data);assert status==expected
        rows.append({'method':method,'route':route,'status':status})
conn=http.client.HTTPSConnection('www.inbirg.com',context=context,timeout=20)
conn.request('POST','/maler/preview',body=b'',headers={'Content-Length':str(161*1024*1024),'Content-Type':'application/octet-stream','Connection':'close'})
r=conn.getresponse();assert r.status==413;r.read();conn.close();rows.append({'route':'declared_oversize_only_no_dataset','status':413})
record={'client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Final certificate-verifying public information/control checks; no biomedical upload','tls_verification_disabled':False,'configured_50MiB_descriptions_verified':True,'passed':True,'records':rows}
(HERE/'final_public_checks.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))
