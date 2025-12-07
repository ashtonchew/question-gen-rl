# Human Feedback UI

A web interface for collecting human ratings on generated technical screening questions. The feedback data can be exported and used to enhance the RL training reward signal.

## Quick Start

```bash
cd feedback-ui
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000

## Configuration

The app automatically loads `XAI_API_KEY` from the repo root `.env` file. No additional configuration needed if you've already set up the main project.

If needed, you can also set it manually:
```bash
export XAI_API_KEY="your-grok-api-key"
```

## Features

### Rate Questions (`/`)
- Displays a random role from `data/backend_roles.json`
- Generates a screening question using Grok API
- Rate on three criteria (1-5 scale):
  - **Relevance**: Does this test skills needed for the role?
  - **Clarity**: Is the question unambiguous?
  - **Discriminative**: Would this distinguish good from weak candidates?
- Add optional comments
- Submit and get a new question

### Statistics (`/stats`)
- View aggregated metrics (total ratings, averages)
- See recent feedback entries
- Export data for training

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main rating page |
| `/submit` | POST | Submit rating |
| `/stats` | GET | Statistics page |
| `/api/stats` | GET | JSON statistics |
| `/api/export` | GET | Export as JSON |
| `/api/export?format=parquet` | GET | Export as parquet |
| `/api/health` | GET | Health check |

## Training Integration

Export feedback data and use it to enhance training rewards:

```python
import pandas as pd

# Load human feedback
feedback = pd.read_parquet("feedback-ui/data/feedback.parquet")

# Reward is pre-computed: (relevance + clarity + discriminative) / 15
# Range: 0.0 to 1.0 (same as LLM judge)
human_reward = feedback["reward"]

# Option: Mix with LLM judge scores
mixed_reward = 0.7 * judge_reward + 0.3 * human_reward
```

## File Structure

```
feedback-ui/
├── app.py           # FastAPI application
├── generator.py     # Grok API question generation
├── database.py      # SQLAlchemy models
├── requirements.txt # Dependencies
├── README.md        # This file
├── data/
│   └── feedback.db  # SQLite database (auto-created)
├── static/
│   └── styles.css   # CSS styling
└── templates/
    ├── base.html    # Base layout
    ├── index.html   # Rating form
    └── stats.html   # Statistics page
```

## Database Schema

| Field | Type | Description |
|-------|------|-------------|
| `question` | Text | Generated question |
| `role_context` | Text | Role JSON |
| `role_id` | String | Role identifier |
| `relevance` | Int (1-5) | Human rating |
| `clarity` | Int (1-5) | Human rating |
| `discriminative` | Int (1-5) | Human rating |
| `comments` | Text | Optional feedback |
| `source` | String | "generated" or "user_provided" |
| `created_at` | DateTime | Timestamp |

## Testing

All endpoints have been verified:

| Test | Status | Result |
|------|--------|--------|
| Health check (`/api/health`) | PASS | `{"status":"healthy","service":"feedback-ui"}` |
| Main page (`/`) | PASS | Renders role info + Grok-generated question |
| Stats API (`/api/stats`) | PASS | Returns aggregated metrics |
| Submit rating (`/submit`) | PASS | 303 redirect, data stored in SQLite |
| Export JSON (`/api/export`) | PASS | Returns feedback array with computed reward |
| Export parquet (`/api/export?format=parquet`) | PASS | Valid parquet file |

### Verified Behavior

- Loads 500 roles from `data/backend_roles.json`
- Database auto-created at `feedback-ui/data/feedback.db`
- `XAI_API_KEY` loaded from repo root `.env`
- Reward calculation: `(relevance + clarity + discriminative) / 15`

### Manual Testing

```bash
# Health check
curl http://localhost:8000/api/health

# Stats
curl http://localhost:8000/api/stats

# Submit a test rating
curl -X POST http://localhost:8000/submit \
  -d "question=Test question" \
  -d "role_context={}" \
  -d "role_id=test-001" \
  -d "relevance=4" \
  -d "clarity=5" \
  -d "discriminative=3"

# Export as JSON
curl http://localhost:8000/api/export

# Export as parquet
curl "http://localhost:8000/api/export?format=parquet" -o feedback.parquet
```
