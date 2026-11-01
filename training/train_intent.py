"""Fine-tune a text-to-text intent model and retain comparable evaluations."""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import datetime,timezone

import numpy as np
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from backend.app.config import DATASETS,MODELS,RESULTS,ROOT


def prompt(text):
    return f'Identify the customer support intent. Message: {text}\nIntent:'


def label_text(label):
    return label.replace('_',' ')


def pick(rows,n):
    grouped=defaultdict(list)
    for r in rows:grouped[r['label']].append(r)
    rng=random.Random(506)
    result=[]
    for label, group in sorted(grouped.items()):
        rng.shuffle(group);result.extend(group[:n])
    rng.shuffle(result)
    return result


def generate(model,tokenizer,rows,device):
    predictions=[]
    started=time.perf_counter()
    model.eval()
    for i in range(0,len(rows),32):
        batch=tokenizer([prompt(r['text']) for r in rows[i:i+32]],padding=True,truncation=True,max_length=96,return_tensors='pt').to(device)
        with torch.inference_mode():out=model.generate(**batch,max_new_tokens=12,do_sample=False,num_beams=1)
        predictions.extend(t.strip().lower().replace(' ','_') for t in tokenizer.batch_decode(out,skip_special_tokens=True))
        if i%160==0:print(f'Evaluation {i}/{len(rows)}',flush=True)
    return predictions,time.perf_counter()-started


def score(name,rows,predictions,labels,seconds):
    true=[r['label'] for r in rows]
    return {'name':name,'accuracy':accuracy_score(true,predictions),'macro_f1':f1_score(true,predictions,labels=labels,average='macro',zero_division=0),'weighted_f1':f1_score(true,predictions,labels=labels,average='weighted',zero_division=0),'invalid_label_rate':sum(p not in labels for p in predictions)/len(rows),'examples':len(rows),'evaluation_seconds':seconds,'report':classification_report(true,predictions,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(true,predictions,labels=labels).tolist()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train-per-class',type=int,default=160);p.add_argument('--test-per-class',type=int,default=20);p.add_argument('--epochs',type=int,default=3);p.add_argument('--supplement',action='store_true');p.add_argument('--version',default='northstar-intent-1');a=p.parse_args()
    torch.manual_seed(506);np.random.seed(506);random.seed(506);torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    train=pick(json.loads((DATASETS/'train.json').read_text()),a.train_per_class)
    validation=pick(json.loads((DATASETS/'validation.json').read_text()),15)
    test=pick(json.loads((DATASETS/'test.json').read_text()),a.test_per_class)
    supplement_meta=None
    if a.supplement:
        path=ROOT/'data/training/support_scenarios.json';supplement=json.loads(path.read_text())
        test_keys={r['text'].strip().lower() for r in test+validation}
        assert not any(r['text'].strip().lower() in test_keys for r in supplement['rows'])
        train.extend([dict(r,normalised=r['text'].lower()) for r in supplement['rows']]*supplement['repeat_in_training'])
        random.Random(506).shuffle(train)
        supplement_meta={'source':'data/training/support_scenarios.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'unique_scenarios':len(supplement['rows']),'repeat':supplement['repeat_in_training'],'scope':supplement['scope']}
    labels=sorted({r['label'] for r in train})
    tokenizer=AutoTokenizer.from_pretrained(MODELS/'intent_base',local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(MODELS/'intent_base',local_files_only=True).to(device)
    print(f'Training {len(train)} examples / {len(labels)} intents on {device}; validation {len(validation)}; test {len(test)}',flush=True)
    def tokenise(batch):
        x=tokenizer([prompt(t) for t in batch['text']],max_length=96,truncation=True)
        x['labels']=tokenizer(text_target=[label_text(t) for t in batch['label']],max_length=12,truncation=True)['input_ids']
        return x
    train_ds=Dataset.from_list(train).map(tokenise,batched=True,remove_columns=list(train[0]))
    val_ds=Dataset.from_list(validation).map(tokenise,batched=True,remove_columns=list(validation[0]))
    args=Seq2SeqTrainingArguments(output_dir=str(ROOT/'training/checkpoints'),num_train_epochs=a.epochs,per_device_train_batch_size=16,per_device_eval_batch_size=32,learning_rate=3e-4,weight_decay=.01,eval_strategy='epoch',save_strategy='epoch',save_total_limit=1,load_best_model_at_end=True,metric_for_best_model='eval_loss',logging_steps=25,report_to=[],seed=506,dataloader_pin_memory=False,optim='adamw_torch',warmup_ratio=.05)
    trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,processing_class=tokenizer,data_collator=DataCollatorForSeq2Seq(tokenizer,model=model))
    started=time.perf_counter();trainer.train();training_seconds=time.perf_counter()-started
    target=MODELS/'intent';trainer.save_model(str(target));tokenizer.save_pretrained(target)
    (target/'labels.json').write_text(json.dumps(labels))
    (target/'support_metadata.json').write_text(json.dumps({'version':a.version,'supplement':supplement_meta},indent=2))
    RESULTS.mkdir(exist_ok=True)
    tuned_pred,tuned_sec=generate(model,tokenizer,test,device)
    tuned=score('Fine-tuned FLAN-T5-small',test,tuned_pred,labels,tuned_sec)
    training_logs=trainer.state.log_history
    del trainer,model
    if device=='mps':torch.mps.empty_cache()
    baseline=AutoModelForSeq2SeqLM.from_pretrained(MODELS/'intent_base',local_files_only=True).to(device)
    base_pred,base_sec=generate(baseline,tokenizer,test,device)
    base=score('Original FLAN-T5-small',test,base_pred,labels,base_sec)
    classic=make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),LogisticRegression(max_iter=1000,random_state=506))
    started=time.perf_counter();classic.fit([r['text'] for r in train],[r['label'] for r in train]);classic_pred=classic.predict([r['text'] for r in test]).tolist();classic_sec=time.perf_counter()-started
    conventional=score('TF-IDF + Logistic Regression',test,classic_pred,labels,classic_sec)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*') if p.is_file()}
    provenance=json.loads((DATASETS/'provenance.json').read_text())
    report={'experiment_id':datetime.now(timezone.utc).strftime('intent-%Y%m%dT%H%M%SZ'),'created_at':datetime.now(timezone.utc).isoformat(),'task':'27-class retail support intent generation','device':device,'base_model':json.loads((MODELS/'registry.json').read_text())['intent_base'],'training':{'examples':len(train),'validation_examples':len(validation),'test_examples':len(test),'epochs':a.epochs,'learning_rate':3e-4,'seed':506,'seconds':training_seconds,'selection':'lowest validation loss; held-out test evaluated after training','logs':training_logs},'data':provenance,'labels':labels,'models':[base,conventional,tuned],'weights_sha256':hashes,'limitations':provenance['limitations']+['Evaluation is an offline balanced sample, not production traffic.','The original language-model baseline uses the same instruction but is not task-adapted.','TF-IDF timing includes fitting, unlike neural prediction timing.']}
    (RESULTS/'intent_experiment.json').write_text(json.dumps(report,indent=2))
    report['version']=a.version;report['training']['supplement']=supplement_meta
    (RESULTS/'intent_experiment.json').write_text(json.dumps(report,indent=2))
    (DATASETS/'test_predictions.json').write_text(json.dumps([r|{'original':b,'conventional':c,'fine_tuned':t} for r,b,c,t in zip(test,base_pred,classic_pred,tuned_pred)],indent=2))
    print(json.dumps({'training_seconds':training_seconds,'results':[{k:r[k] for k in ('name','accuracy','macro_f1','invalid_label_rate')} for r in report['models']]},indent=2),flush=True)
