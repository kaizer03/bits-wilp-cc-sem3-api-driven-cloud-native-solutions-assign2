"""Evaluate legal-label likelihood decoding consistently across neural models."""
import json
import time
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer

from backend.app.config import MODELS,RESULTS,DATASETS
from backend.app.intent import classify_candidates
from training.train_intent import pick,score


if __name__=='__main__':
    torch.set_num_threads(6)
    report=json.loads((RESULTS/'intent_experiment.json').read_text())
    rows=pick(json.loads((DATASETS/'test.json').read_text()),20)
    labels=report['labels'];device='mps' if torch.backends.mps.is_available() else 'cpu'
    all_predictions={}
    for key,name,index in [('intent_base','Original FLAN-T5-small',0),('intent','Fine-tuned FLAN-T5-small',2)]:
        tokenizer=AutoTokenizer.from_pretrained(MODELS/key,local_files_only=True)
        model=AutoModelForSeq2SeqLM.from_pretrained(MODELS/key,local_files_only=True).to(device).eval()
        predictions=[];start=time.perf_counter()
        for i in range(0,len(rows),8):
            batch=classify_candidates(model,tokenizer,[r['text'] for r in rows[i:i+8]],labels,device)
            predictions.extend(r['label'] for r in batch)
        report['models'][index]=score(name,rows,predictions,labels,time.perf_counter()-start)
        all_predictions[key]=predictions
        del model
        if device=='mps':torch.mps.empty_cache()
        print(name,report['models'][index]['accuracy'],flush=True)
    report['decoding']='Both neural models rank the same 27 legal labels by mean conditional token log likelihood. The encoder is evaluated once per input. No confidence calibration is claimed.'
    report['limitations']=[x for x in report['limitations'] if 'original language-model baseline' not in x]
    report['limitations'].append('The first experiment used unconstrained generation. Cross-version changes include the authored supplement and label-scoring inference; that comparison is not a single-factor ablation.')
    (RESULTS/'intent_experiment.json').write_text(json.dumps(report,indent=2))
    old=json.loads((DATASETS/'test_predictions.json').read_text())
    for row,b,t in zip(old,all_predictions['intent_base'],all_predictions['intent']):row.update(original=b,fine_tuned=t)
    (DATASETS/'test_predictions.json').write_text(json.dumps(old,indent=2))
