"""Evaluate authored diagnostics, policy references and synthetic audio fixtures."""
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone

import httpx
import jiwer
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

from backend.app.config import ROOT,RESULTS,MODELS,PROMPT_VERSION
from backend.app.models import service
from backend.app.knowledge import retrieve,POLICIES


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8092');a=p.parse_args()
    fixture=ROOT/'data/evaluation/reference_cases.json';cases=json.loads(fixture.read_text())
    started=time.perf_counter();rows={'intent':[],'sentiment':[],'policy':[],'speech':[]}
    for row in cases['intent']:
        prediction=service.intent(row['text'])
        rows['intent'].append(row|{'prediction':prediction['label'],'correct':row['label']==prediction['label']})
    for row in cases['sentiment']:
        prediction=service.sentiment(row['text'])
        rows['sentiment'].append(row|{'prediction':prediction['label'],'correct':row['label']==prediction['label']})
    latency=[]
    with httpx.Client(base_url=a.url,timeout=180) as client:
        for row in cases['policy']:
            start=time.perf_counter()
            retrieved=retrieve(row['question'])
            r=client.post('/api/knowledge/ask',json={'question':row['question']});r.raise_for_status();answer=r.json()
            latency.append(time.perf_counter()-start)
            rows['policy'].append(row|{'retrieved':[r['id'] for r in retrieved],'answer':answer,'retrieval_correct':row['policy'] in [r['id'] for r in retrieved] if row['policy'] else not retrieved,'abstention_correct':answer['abstained']==(row['policy'] is None),'reference_terms_present':all(term.lower() in answer['answer'].lower() for term in row.get('must_contain',[])) if row['policy'] else None,'seconds':latency[-1]})
            print(row['question'],answer['answer'],flush=True)
        for row in cases['audio']:
            path=ROOT/'data/evaluation/audio'/f"{row['id']}.wav"
            if not path.exists():continue
            start=time.perf_counter()
            with open(path,'rb') as f:r=client.post('/api/transcribe',files={'file':(path.name,f,'audio/wav')})
            r.raise_for_status();result=r.json()
            normalise=jiwer.Compose([jiwer.ToLowerCase(),jiwer.RemovePunctuation(),jiwer.RemoveMultipleSpaces(),jiwer.Strip()])
            ref=normalise(row['text']);hyp=normalise(result['text'])
            rows['speech'].append(row|{'transcript':result['text'],'wer':jiwer.wer(ref,hyp),'reference_normalised':ref,'hypothesis_normalised':hyp,'seconds':time.perf_counter()-start,'duration_seconds':result['duration_seconds'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    mean=lambda values:sum(values)/len(values) if values else None
    supported=[r for r in rows['policy'] if r['policy']]
    metrics={'authored_intent_accuracy':mean([r['correct'] for r in rows['intent']]),'authored_intent_macro_f1':f1_score([r['label'] for r in rows['intent']],[r['prediction'] for r in rows['intent']],labels=sorted({r['label'] for r in rows['intent']}),average='macro',zero_division=0),'sentiment_accuracy':mean([r['correct'] for r in rows['sentiment']]),'retrieval_recall_at_2':mean([r['retrieval_correct'] for r in supported]),'abstention_accuracy':mean([r['abstention_correct'] for r in rows['policy']]),'policy_reference_term_coverage':mean([r['reference_terms_present'] for r in supported]),'policy_end_to_end_p95_seconds':float(np.percentile(latency,95)),'synthetic_speech_wer':jiwer.wer([r['reference_normalised'] for r in rows['speech']],[r['hypothesis_normalised'] for r in rows['speech']]) if rows['speech'] else None}
    report={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Development diagnostics used for model improvement: 27 intent, 12 sentiment, 23 policy and 4 synthetic speech examples. Reference-term coverage is an automated proxy, not a human factuality score.','fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'prompt_version':PROMPT_VERSION,'policy_version':POLICIES['version'],'metrics':metrics,'rows':rows,'total_seconds':time.perf_counter()-started,'limitations':['Authored examples informed model development and are not independent validation.','Audio fixtures use Apple speech synthesis, not human customer recordings; WER does not measure accent or noise robustness.','Policy reference-term checks can miss semantic mistakes and penalise correct paraphrases.','Policy p95 combines retrieval in the diagnostic process with the API call and is not concurrent throughput.','Abstention includes explicit topic rules as well as retrieval thresholds.']}
    RESULTS.mkdir(exist_ok=True);(RESULTS/'quality_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
