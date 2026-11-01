import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.getenv('RUNTIME_DIR', ROOT / '.runtime')).resolve()
MODELS = Path(os.getenv('MODEL_DIR', ROOT / '.models')).resolve()
DATASETS = Path(os.getenv('DATASET_DIR', ROOT / '.datasets')).resolve()
RESULTS = ROOT / 'results'
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

MODEL_IDS = {
    'intent_base': 'google/flan-t5-small',
    'generation': 'Qwen/Qwen2.5-1.5B-Instruct',
    'generation_cpu': 'Qwen/Qwen2.5-0.5B-Instruct',
    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'speech': 'openai/whisper-base.en',
    'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
}
PROMPT_VERSION = 'support-1.1'
