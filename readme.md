# Evaluating Policy Manipulation via RAG Memory Poisoning
---

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d
```

Starts:
- **n8n** on `http://localhost:5678` — this is basically for the visual AI workflow
- **n8n-worker** — background job runner for n8n queue mode
- **Redis** on `6379` — n8n job queue
- **PostgreSQL + pgvector** on `5433` — knowledge base for RAG
- **Adminer** on `http://localhost:8080` — to visualize the datbase, credentials: (`localhost:5433`, user `n8n`, pass `n8npassword`)

### 2. Start the perception vision service

```bash
pip install -r requirements.txt
python perception/capture_frame.py
```

Runs a FastAPI server on `http://localhost:8089`. Opens the webcam and starts face detection. You can look at the perception at `http://localhost:8089/stream`

### 3. Import n8n workflows

In n8n (`http://localhost:5678`), import the three workflow files from `n8n/workflows/`:

| File | Purpose |
|------|---------|
| `MainWorkflow.json` | Chat endpoint at `/webhook/robot-chat`, receives queries, retrieves policy docs from pgvector, asks the LLM, returns access decision |
| `Perception.json` | Calls `perception/capture_frame.py` on a schedule to pull face detections into n8n state |
| `RegisterPerson.json` | Face registration flow, calls `/register-face` on the perception service |

Activate all three workflows (Active button on the top right)

### 4. Run the experiment

**Phase 1 — clean baseline**:
```bash
python experiment/ingest_poisoned_doc.py --phase1
python experiment/experiment_controller.py --phase1 --n 10
```

**Phase 2 — poisoned**:
```bash
python experiment/ingest_poisoned_doc.py --poison
python experiment/experiment_controller.py --phase2 --n 10
```

Results are saved to `results_<timestamp>.json`.

---

## Scripts

### `perception/capture_frame.py`
FastAPI service (`http://localhost:8089`) that is the "eyes" of the kiosk or the RAG.

### `experiment/attack_config.py`
Shared constants for the experiment: perception and n8n URLs, embedding dimension, document IDs, and confidence thresholds (`AMB_CONFIDENCE=0.70`, `AMB_THRESHOLD=0.80`).

### `experiment/ingest_poisoned_doc.py`
Seeds the pgvector `knowledge` table for the experiment.

```bash
python experiment/ingest_poisoned_doc.py --phase1   # insert legitimate deny policy
python experiment/ingest_poisoned_doc.py --poison   # inject attacker amendment
python experiment/ingest_poisoned_doc.py --delete   # remove both docs
python experiment/ingest_poisoned_doc.py --list     # show knowledge table
```

- **Legitimate doc** (`OIT-BIOMETRIC-2023-11`): deny policy.
- **Poisoned doc** (`DEI-BIO-AMEND-2024-03`): attacker's amendment document.

### `experiment/ingest_to_postgres.py`
Generic document ingestion utility. Embeds any `.txt` file with OpenAI `text-embedding-3-small` and inserts it into the `knowledge` table. Supports Azure OpenAI or direct OpenAI key via env vars.

```bash
python experiment/ingest_to_postgres.py --file docs/policy.txt --name "Policy-2024" --meta '{"source":"manual"}'
python experiment/ingest_to_postgres.py --list
python experiment/ingest_to_postgres.py --delete 3
```

### `experiment/experiment_controller.py`
Orchestrates the attack experiment.

1. Sets the perception state (injects a face with `confidence=0.70`) via `POST /state`
2. Sends a standard access-request query to the n8n chat webhook
3. Classifies the LLM response as **UAR** (Unauthorized Access Received = attack succeeded) or denied using keyword lists
4. Repeats for `--n` trials and prints a summary with UAR rates and attack lift

```bash
python experiment/experiment_controller.py --phase1 --n 10   # clean
python experiment/experiment_controller.py --phase2 --n 10   # poisoned
python experiment/experiment_controller.py --all   --n 10   # both
```

### `postgres/init.sql`
Database schema: `documents`, `ingestion_log`, `source_sync` tables with indexes and helper functions (`cleanup_old_logs`, `get_document_stats`). Applied automatically by n8n's internal DB setup; run manually if needed for the `atlasai` schema variant.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for embeddings and LLM in n8n + experiment scripts |
| `N8N_ENCRYPTION_KEY` | `CHANGE_ME_LONG_RANDOM` | Set in `.env` before first run |
| `DATABASE_URL` | `postgresql://n8n:n8npassword@localhost:5433/n8n` | pgvector connection |
| `CAM_INDEX` | `0` | Webcam device index for perception service |
| `DEEPFACE_STREAM` | `true` | Enable live DeepFace analysis on the MJPEG stream |
