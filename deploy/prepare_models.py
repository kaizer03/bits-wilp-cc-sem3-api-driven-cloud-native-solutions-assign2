"""Download only the pinned CPU inference snapshots; custom intent weights are separate."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'data/model_manifest.json').read_text())
registry_path = ROOT / '.models' / 'registry.json'
registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
for name in ('intent_base', 'generation_cpu', 'sentiment', 'speech', 'embedding'):
    item = manifest['models'][name]
    model_id = item.get('model_id') or item.get('id')
    if not model_id:
        raise ValueError(f'Missing model ID for {name}')
    print(f'Downloading {name} at {item["revision"]}', flush=True)
    snapshot_download(repo_id=model_id, revision=item['revision'], local_dir=ROOT / '.models' / name,
        allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'vocab.*', 'merges.txt', '*.model', 'pytorch_model.bin'],
        ignore_patterns=['onnx/*', 'openvino/*', 'tf_model*', 'flax_model*'])
    registry[name] = {'model_id': model_id, 'revision': item['revision'], 'directory': name}
    registry_path.write_text(json.dumps(registry, indent=2))
