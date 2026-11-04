# Northstar Retail Support

A retail service workspace with voice intake, intent routing, sentiment analysis, case summaries, policy questions and reviewed response drafts. Built for CCZG506 Assignment II, Group 15.

Northstar Retail and the example customer cases are fictional. The application records support work; it has no connection to a live order, payment or messaging system.

## Application

- Case intake accepts typed messages or English audio. Transcripts can be reviewed before saving.
- A fine-tuned FLAN-T5-small language model routes requests into 27 support intents.
- A sentiment model supplies a contextual signal; it does not make a customer decision.
- Published policies are retrieved with sentence embeddings and lexical overlap.
- A local instruction model generates summaries, policy answers and reply drafts.
- Answers with detected unsupported numbers or completed-action claims are replaced with the verbatim policy. Unsupported topics are referred for human review. These checks do not establish complete factual correctness.
- An agent edits and approves a draft before resolving a case. Approval does not send a message.
- Model versions, prompt/policy versions, request timings, failures and offline evaluation results are visible in the workspace.

## Local execution

Python 3.12 and Node.js 24 are the reference development runtimes. The first model download needs an internet connection and several gigabytes of free storage. Runtime model inference is local; no provider key is required.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m training.prepare
.venv/bin/python -m training.train_intent --supplement --version northstar-intent-2
.venv/bin/python -m training.rescore_intent
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python -m backend
```

The application and API documentation are available at `http://127.0.0.1:8092` and `http://127.0.0.1:8092/docs`. An existing prepared checkout only needs the last command. Optional settings are listed in `.env.example`; credentials and runtime data are excluded from Git.

For frontend development, `npm run dev` uses port 5173 and proxies `/api` to the backend. The built frontend is served directly by FastAPI.

## Training and evaluation

`training.prepare` downloads the model and dataset revisions in `data/model_manifest.json`. It normalises reference placeholders and punctuation, removes duplicate normalised utterances, and creates disjoint 70/15/15 train/validation/test partitions with seed 506. Near paraphrases can remain across partitions.

The reference fine-tuning run uses 160 public training examples per intent, 91 project-authored scenarios repeated four times, three epochs, a learning rate of 0.0003 and validation-loss checkpoint selection. The 540-example public test sample has 20 examples per intent. Both original and fine-tuned neural models rank the same legal labels using mean conditional token log likelihood. The score is not calibrated confidence. TF-IDF with logistic regression supplies a conventional baseline.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m training.sample_cases --analyse
.venv/bin/python -m training.evaluate
```

The sample-case command adds clearly labeled fictional enquiries and processes them through the APIs. It is intended for a local demonstration database, not an existing customer database.

Experiment results are in `results/intent_experiment.json` and `results/quality_evaluation.json`. The earlier experiment is retained under `results/experiments`. Authored diagnostic cases informed development and are not independent validation. Four speech fixtures use Apple speech synthesis; their word error rate does not establish performance on human accents, background noise or telephone audio. Reference-term checks are automated proxies rather than human factuality scores.

## Linux container execution

Prepare the model directory first, then run:

```bash
docker compose up --build -d
docker compose logs -f app
```

The container uses the compact Qwen2.5-0.5B generation profile in full precision. Native Apple GPU inference uses Qwen2.5-1.5B. `GENERATION_PROFILE=compact` or `standard` selects a profile without source changes. Optional `CPU_QUANTIZE=1` enables dynamic int8 conversion, which can have substantial initialisation memory cost; it is disabled by default. Native Apple GPU training and container CPU serving are separate execution paths. Profile-specific response quality and resource use must be measured separately. The model directory is mounted read-only; runtime data uses a named volume.

The port is bound to loopback by default. Access from another computer requires a separately configured deployment with appropriate authentication and transport protection. The application does not claim multi-user production readiness.

Local Linux ARM64 execution and the BITS AWS Ubuntu AMD64 deployment have both completed functional verification. The cloud run used t3a.medium with 4 GiB RAM and 8 GiB swap. Nine real API checks passed, and a resolved voice case with its approval history survived an EC2 stop and start. A cold case analysis took 96 seconds in this constrained lab run. Results are recorded in results/cloud_runtime_checks.json and results/cloud_storage_verification.json. Windows execution remains unverified. SQLite requires a filesystem supporting its locking semantics.

## Cloud runtime

`deploy/bootstrap_ubuntu.sh` prepares an Ubuntu host with Docker and persistent disk swap. `deploy/prepare_models.py` downloads only the CPU inference snapshots pinned in the model manifest. The trained intent checkpoint is transferred separately. `deploy/start.sh` builds and starts the same Compose service used locally. Local containers bind to loopback by default. APP_BIND_HOST=0.0.0.0 enables the explicitly configured lab endpoint. The BITS demo uses the EC2 public IPv4 address on port 8092, with security-group rules restricting that port and SSH to the approved client IP. It carries fictional cases over HTTP. An authenticated SSH tunnel remains available for encrypted access. Public IPv4 addresses change after stop/start, and browser microphone recording requires HTTPS or localhost; audio-file uploads work on the HTTP lab endpoint. Set APP_ENVIRONMENT=cloud in the host environment file. The CPU preparation command records installed model revisions in the registry, and the startup script makes transferred model files readable by the container user. Docker starts on boot; its named data volume retains case history.

## Project structure

```text
backend/app/         Application API, persistence, retrieval and model inference
frontend/src/        React and Bootstrap workspace
training/            Dataset preparation, fine-tuning and evaluation
data/policies/       Versioned fictional store policies
data/training/       Authored training scenarios
data/evaluation/     Diagnostic references and synthetic speech fixtures
results/             Measured experiment results and limitations
```

Large downloaded models, raw data, database files, caches and local environment files are not source-control artifacts. Trained weights can be reproduced or distributed separately with their SHA-256 manifest.

## Sources

- [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — CDLA Sharing 1.0. Raw data is downloaded from the pinned source and is not repackaged here.
- [FLAN-T5-small](https://huggingface.co/google/flan-t5-small), [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Whisper base English](https://huggingface.co/openai/whisper-base.en) and [MiniLM embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache 2.0 model cards.
- [Cardiff sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — CC BY 4.0. Its social-media training domain differs from retail support.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/), [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/), [Bootstrap](https://getbootstrap.com/) and [SQLAlchemy](https://www.sqlalchemy.org/).
- DM Sans and Manrope are bundled through Fontsource; font licences are SIL Open Font License. Bootstrap Icons is MIT licensed.

Model cards and dataset terms govern their respective third-party materials. Licence identifiers are recorded for attribution and do not replace those terms.
