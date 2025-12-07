# AGENTS DATA GENERATION GUIDE

Purpose: produce ~400 new role records to expand `data/backend_roles.json` from 100 → ~500 entries. Stay 100% schema-compatible; no training or merge steps here—just data creation quality rules.

## 1) Exact schema (must not deviate)
Each role is a JSON object with keys in this order:
- `id` (string) — unique slug, pattern `be-XXX` with zero-padded numbers starting at **101** (e.g., `be-101`, `be-499`). Do not reuse existing IDs.
- `title` (string)
- `level` (enum): `junior` | `mid` | `senior` | `staff` | `principal`
- `focus` (enum): `api-development`, `distributed-systems`, `databases`, `security`, `devops`, `ml`, `frontend`, `mobile`, `search`, `payments`, `reliability`, `observability`, `platform`, `data-pipelines`, `real-time`
- `stack` (array of 3–6 tech strings)
- `description` (string, ≤220 chars, 1–2 sentences, specific)
- `key_skills` (array of 3–6 concise skill strings)

Example (good):
```json
{
  "id": "be-131",
  "title": "Senior Backend Engineer - Observability",
  "level": "senior",
  "focus": "observability",
  "stack": ["Go", "Kubernetes", "OpenTelemetry", "Kafka", "ClickHouse"],
  "description": "Build and scale telemetry pipelines for metrics, traces, and logs across microservices on Kubernetes.",
  "key_skills": ["Go", "distributed systems", "observability", "Kafka", "OTel"]
}
```

## 2) Quality checklist (reject if any fail)
- Unique `id`; non-blank `title`.
- `description` is role-specific, includes domain/scale/constraints, ≤220 chars.
- `stack` aligns with `focus`; 3–6 items, no grab-bags.
- `key_skills` are concrete (tech/competency), not soft skills; 3–6 items.
- Level fits responsibilities (e.g., `junior` avoids “own architecture”; `principal` mentions strategy/mentorship).
- No duplicates or near-duplicates (compare title + focus + stack).

## 3) Coverage targets (approximate)
- Levels: ~20% junior, 30% mid, 30% senior, 15% staff, 5% principal.
- Focus balance: include all focus enums; prioritize underrepresented areas (security, observability, payments, search, real-time, data-pipelines).
- Tech diversity: mix languages (Go, Java, Python, Node, Rust), datastores (PostgreSQL, MySQL, MongoDB, Cassandra, Redis), messaging (Kafka, RabbitMQ, NATS), and cloud/K8s when appropriate.

## 4) Focus-specific hints
- `payments`: mention PCI/idempotency/ledgers/queues; stacks like Java/Spring, Go, Postgres, Kafka/RabbitMQ.
- `real-time`: WebSocket/GRPC, Redis, Kafka/NATS; latency and connection scale.
- `observability`: OTel, Prometheus/Grafana, log pipelines (ClickHouse/ELK), sampling/retention.
- `search`: Elasticsearch/OpenSearch/Lucene; relevance tuning, indexing throughput.
- `data-pipelines`: Beam/Spark/Flink, warehouses (BigQuery/Snowflake/Redshift), schema evolution, backfill.
- `security`: OAuth2/OIDC, RBAC/ABAC, secrets management, threat modeling.
- `distributed-systems`: consistency/partitioning, RPC (gRPC), service discovery, SLOs.
- `api-development`: REST/GraphQL, pagination, versioning, rate limits; Python/Node/Java/Go.

## 5) Generation procedure (for agents)
1. Work in batches of 50–100 roles to keep context small.
2. Start `id` at the next available (assume `be-101` if only 100 exist) and increment sequentially.
3. For each role, pick level → focus → stack → description → key_skills, ensuring consistency.
4. Keep descriptions concise and specific (scale, domain, constraints).
5. Vary stacks and domains across the batch to avoid near-duplicates.
6. After each batch, run validation (below) before appending to the master JSON.

## 6) Validation commands
- JSON valid:  
  `python3 - <<'PY'\nimport json; json.load(open('data/backend_roles.json')); print('json ok')\nPY`
- Unique IDs:  
  `python3 - <<'PY'\nimport json; r=json.load(open('data/backend_roles.json')); ids=[x['id'] for x in r]; assert len(ids)==len(set(ids)), 'duplicate ids'; print('ids unique')\nPY`
- Level/focus enums only:  
  `python3 - <<'PY'\nimport json\nlevels={'junior','mid','senior','staff','principal'}\nfocuses={'api-development','distributed-systems','databases','security','devops','ml','frontend','mobile','search','payments','reliability','observability','platform','data-pipelines','real-time'}\nroles=json.load(open('data/backend_roles.json'))\nfor i,r in enumerate(roles):\n    assert r['level'] in levels, f\"bad level {r['level']} at {i}\"\n    assert r['focus'] in focuses, f\"bad focus {r['focus']} at {i}\"\nprint('enums ok')\nPY`

## 7) Output expectations
- Deliver a JSON array of role objects that can be appended directly to `data/backend_roles.json` (no trailing commas, valid UTF-8/ASCII-friendly).
- Ensure no line exceeds normal JSON width for readability, but content length is secondary to validity and specificity.
