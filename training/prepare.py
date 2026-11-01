"""Download versioned public model weights and prepare disjoint support data."""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from backend.app.config import DATASETS, MODELS, MODEL_IDS, ROOT

DATASET_ID = 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
MANIFEST_PATH=ROOT/'data/model_manifest.json'
MANIFEST=json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def normalise(text):
    return re.sub(r'[^a-z0-9 ]', '', re.sub(r'\{\{.*?\}\}', 'reference', text.lower())).strip()


def prepare_data():
    DATASETS.mkdir(exist_ok=True)
    api = HfApi()
    info = api.dataset_info(DATASET_ID,revision=MANIFEST.get('dataset',{}).get('revision'))
    files = api.list_repo_files(DATASET_ID, repo_type='dataset', revision=info.sha)
    filename = next(p for p in files if p.endswith('.csv'))
    source = hf_hub_download(DATASET_ID, filename, repo_type='dataset', revision=info.sha)
    expected=MANIFEST.get('dataset',{}).get('sha256')
    if expected and hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise RuntimeError('Dataset checksum does not match the published manifest.')
    raw = list(csv.DictReader(open(source, encoding='utf-8-sig')))
    seen = set()
    grouped = defaultdict(list)
    for row in raw:
        key = normalise(row['instruction'])
        if not key or key in seen:
            continue
        seen.add(key)
        grouped[row['intent']].append({'text':row['instruction'], 'label':row['intent'], 'normalised':key})
    splits = {'train':[], 'validation':[], 'test':[]}
    rng = random.Random(506)
    for label, rows in sorted(grouped.items()):
        rng.shuffle(rows)
        n = len(rows)
        a,b = int(n*.70),int(n*.85)
        splits['train'].extend(rows[:a])
        splits['validation'].extend(rows[a:b])
        splits['test'].extend(rows[b:])
    keys = {name:{r['normalised'] for r in rows} for name,rows in splits.items()}
    assert not keys['train'] & keys['validation'] and not keys['train'] & keys['test'] and not keys['validation'] & keys['test']
    for name, rows in splits.items():
        rng.shuffle(rows)
        (DATASETS/f'{name}.json').write_text(json.dumps(rows,indent=2))
    metadata = {'dataset_id':DATASET_ID,'revision':info.sha,'source_file':filename,'sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'raw_rows':len(raw),'unique_rows':len(seen),'removed_duplicates':len(raw)-len(seen),'seed':506,'split_counts':{k:len(v) for k,v in splits.items()},'class_counts':{k:dict(Counter(r['label'] for r in v)) for k,v in splits.items()},'normalisation':'lowercase; replace reference placeholders; strip punctuation; exact normalised grouping','limitations':['Near paraphrases can remain across splits.','The Bitext corpus is templated domain data; held-out scores do not establish performance on natural customer traffic.']}
    (DATASETS/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:metadata[k] for k in ('raw_rows','unique_rows','split_counts')},indent=2),flush=True)


def prepare_models(names):
    MODELS.mkdir(exist_ok=True)
    registry_path = MODELS/'registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for name in names:
        model_id = MODEL_IDS[name]
        revision = MANIFEST.get('models',{}).get(name,{}).get('revision') or HfApi().model_info(model_id).sha
        print(f'Downloading {name}: {model_id} @ {revision}',flush=True)
        snapshot_download(model_id,revision=revision,local_dir=MODELS/name,allow_patterns=['*.json','*.safetensors','tokenizer*','vocab.*','merges.txt','*.model','pytorch_model.bin'],ignore_patterns=['onnx/*','openvino/*','tf_model*','flax_model*'])
        registry[name]={'model_id':model_id,'revision':revision,'directory':name}
        registry_path.write_text(json.dumps(registry,indent=2))
        print(f'Ready: {name}',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-only',action='store_true')
    p.add_argument('--models',nargs='*',default=list(MODEL_IDS))
    a=p.parse_args()
    prepare_data()
    if not a.data_only:
        prepare_models(a.models)
