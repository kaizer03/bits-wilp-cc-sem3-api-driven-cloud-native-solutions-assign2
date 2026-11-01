"""Integration verification against a dedicated local application instance."""
import argparse
import json
import time
from pathlib import Path

import httpx


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8093');p.add_argument('--output',default='.runtime/runtime-checks.json');a=p.parse_args()
    rows=[];output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps({'url':a.url,'scope':'Functional checks against a dedicated local instance. No production, cloud or load-test claim.','checks':rows},indent=2))
    with httpx.Client(base_url=a.url,timeout=180) as c:
        def request(method,path,payload=None):
            started=time.perf_counter()
            try:
                r=c.request(method,path,json=payload);r.raise_for_status();value=r.json()
                rows.append({'operation':path,'status':r.status_code,'seconds':time.perf_counter()-started,'result':value});save();return value
            except Exception as e:
                rows.append({'operation':path,'error':type(e).__name__,'seconds':time.perf_counter()-started});save();raise
        request('GET','/api/health')
        request('GET','/api/models')
        ticket=request('POST','/api/tickets',{'subject':'Runtime verification','customer':'Sample Customer','message':'Please cancel my order before it is dispatched.'})
        i=ticket['id'];print('Analysing dedicated runtime case',flush=True)
        result=request('POST',f'/api/tickets/{i}/analyse')
        assert result['analysis']['intent']['label']=='cancel_order'
        assert result['analysis']['summary'] and result['analysis']['reply']
        answer=request('POST','/api/knowledge/ask',{'question':'Can I cancel an order after dispatch?'})
        assert answer['evidence'] and not answer['abstained']
        answer=request('POST','/api/knowledge/ask',{'question':'What is the warranty period?'})
        assert answer['abstained'] and not answer['evidence']
        path=Path('data/evaluation/audio/cancel.wav');started=time.perf_counter()
        with path.open('rb') as f:r=c.post('/api/transcribe',files={'file':(path.name,f,'audio/wav')})
        r.raise_for_status();rows.append({'operation':'/api/transcribe','status':r.status_code,'seconds':time.perf_counter()-started,'result':r.json()});save()
        assert 'cancel' in r.json()['text'].lower()
        request('POST',f'/api/tickets/{i}/approve',{'reply':'Cancellation can be requested before dispatch. Please verify the order dispatch status before confirming it.'})
        assert request('PATCH',f'/api/tickets/{i}/status?status=resolved')['status']=='resolved'
        print('All runtime checks passed.',flush=True)
