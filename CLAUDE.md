# Lexaior legal RAG (kafka workers + microservices API)

Self-hosted RAG for Indian legal documents, following the Evaior design doc (folder → file → per-stage
pipeline). No AWS. Two Python projects, each with its own `.venv`:

- `lawyer/microservices` — FastAPI upload/status API, SQLAlchemy models, Alembic migrations.
- `lawyer/kafka` — Kafka stage workers (`pyconsumers/`), Docker stack, tests.

## Rules
- Never commit or push unless asked in that same message.
- Never edit `kafka/.env` (secrets). New settings go in `helpers/secrets_control.py` + `.env.example`.
- Ask before any download or install (models, brew, pip packages).
- No tiny new files: extend an existing module unless the new code is substantial.
- Async Postgres only: SQLAlchemy async + asyncpg. Schema changes = Alembic migration in microservices.
- Keep replies short; run real-model end-to-end checks only at the end of a phase.

## Stack
Postgres+pgvector, Kafka (KRaft), Valkey, SeaweedFS (S3 API) in Docker (`kafka/docker-compose.yml`).
Ollama runs **natively** on the Mac (`brew services start ollama`, GPU); the compose `ollama` service is
behind the `ollama` profile for Linux. Models: `qwen3-vl:4b-instruct` (vision + metaclassifier),
BGE-M3 in `kafka/.cache/huggingface` (embeddings, MPS). Docling parses PDF/DOCX.

## Commands
- Stack: `cd kafka && docker compose up -d`
- Migrations: `cd microservices && make migrate` / `make migration-check`
- API: `cd microservices && make serve` (port 8000, header `X-User-Id`)
- Worker: `cd kafka && make worker`
- Tests: `make unit`, `make integration` (both projects), `cd kafka && make parser-tests` (real Docling)

## Pipeline (folder model)
`unstructured_thread` (folder) → `unstructured_data` (file) with per-stage
`{stage}_status/updated_at/attempts/error/owner/lease_expires_at`. Stages:
standardization → markdown + image → metaclassifier; CSV files use only `parquet`.
- Message per folder `{"unstructured_thread_id"}` via the transactional outbox; Kafka only wakes a
  `StageWorker`, which claims files from the Postgres fair queue (`userallocation/UserAllocation.py`).
- Handlers return `StageResult`; output writes are fenced with `lock_claim` (lease ownership).
- Sweeper (`stall_sweeper.py`) recovers expired leases and stuck WAITING stages. Retrigger API in microservices.
- Handlers are registered in `pyconsumers/start.py` `STAGE_HANDLERS`.

Key modules (`kafka/pyconsumers/core/rag_pipeline/`): `execute_standardization.py` (FolderStandardization),
`execute_indexing.py` (MarkdownRag), `multimodal_processors/image_rag.py` (ImageRag),
`execute_metaclassifier.py` (MetaClassifier: cited facts, verified against printed text),
`chunking/text_chunker.py` (legal-unit chunking `legal-v1`).

## Status (2026-10-27)
Done: A data model · B upload API · C workers/fair queue/sweeper · D standardization, markdown RAG,
image RAG · E legal-unit chunking · F metaclassifier (migration 0006 `details`) · Ollama moved to GPU ·
G formats (DOC via LibreOffice→PDF, PPTX/XLSX native Docling with PDF fallback, TXT, CSV→Parquet stage).
AR_ID citations: deferred until the user says.

Next: **H retrieval as agent tools.** The RAG layer feeds a later multi-agent tool (project → attached
documents → several agents → reports; discussed later, not part of H). A project = `data_fabric`. Build in
microservices, scoped to the project and to the user (`X-User-Id`), JSON in/out:
- `search(project_id, query, filters, top_k)`: BGE-M3 query vector + pgvector HNSW (`hnsw.iterative_scan`)
  over `text_nodes` and `image_rag_vector_table`, `tsv` full-text (english + simple), exact legal-reference
  boost (Section/Clause/case number via `legal_path`/content), RRF fusion to ~50, `bge-reranker-v2-m3`
  rerank to top k (ask before downloading, ~1.1 GB).
- `list_documents(project_id, filters)` over `unstructured_metadata`; `get_document_facts(doc_id)`;
  `expand_passage(node_id)` via `node_relationships` / same clause.
- Every result carries file, page, clause path, node id, score, `model_derived`. No answer generation.
Then I: cleanup of old `documents` tables/consumers (confirm before deleting). Commit only when asked.
Known: `kafka/.env` has an unused stale `LLM_MODEL=qwen3:4b`; `qwen3:8b` exists in native Ollama.
