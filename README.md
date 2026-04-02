# PII Compliance Auditor — OpenEnv Environment

A real-world OpenEnv environment where an AI agent audits documents for
Personally Identifiable Information (PII) compliance — simulating the actual
work done by data privacy teams at companies operating under GDPR and CCPA.

## Motivation

Data privacy compliance is a genuine, high-stakes problem. Thousands of
companies must ensure PII is identified and protected across their documents.
This environment trains and evaluates agents on exactly this task.

## Environment Overview

The agent receives documents and must identify and classify all PII present.
Three tasks of increasing difficulty test the agent's ability to handle
progressively harder detection challenges.

### Action Space

The environment exposes four MCP tools:

| Tool | Description |
|---|---|
| `list_tasks()` | List all available tasks with difficulty levels |
| `get_task(task_id)` | Retrieve document + instructions for a task |
| `submit_findings(task_id, findings_json)` | Submit detected PII for grading |
| `get_current_state()` | Get current episode state |

**findings_json format:**
```json
[
  {"text": "John Smith", "pii_type": "NAME"},
  {"text": "john@example.com", "pii_type": "EMAIL"}
]
```

### Observation Space

Each tool call returns a JSON string with structured data including:
- Task document and instructions
- Grading results (score, precision, recall, F1, feedback)
- Episode state (step count, submissions, cumulative reward)

### PII Types

`NAME` `EMAIL` `PHONE` `CREDIT_CARD` `SSN` `ADDRESS` `DOB` `PASSWORD`

## Tasks

### Task 1 — Easy: Basic PII Detection
- **Document:** Customer support email
- **PII in scope:** NAME, EMAIL, PHONE
- **Challenge:** 3 clearly labeled PII items
- **Grader:** Recall-based — rewards finding all items
- **Expected score (strong LLM):** 0.9–1.0

### Task 2 — Medium: Employee Onboarding Audit
- **Document:** Employee onboarding form
- **PII in scope:** NAME, ADDRESS, DOB, SSN, PHONE, EMAIL
- **Challenge:** 6 PII items, some embedded in sentences not labeled fields
- **Grader:** Partial credit for right text / wrong type. FP penalty 0.1
- **Expected score (strong LLM):** 0.6–0.8

### Task 3 — Hard: Data Breach Incident Report
- **Document:** Restricted incident report
- **PII in scope:** All 8 types
- **Challenge:** 13 PII items across two individuals, unconventional formats
- **Grader:** Strict F1 — exact text AND type match required. FP penalty 0.15
- **Expected score (strong LLM):** 0.3–0.6

## Reward Function

Rewards are partial and meaningful — not binary:

- **Easy:** Score = recall (each found item = credit)
- **Medium:** Partial credit (0.5) for right text, wrong type; FP penalty
- **Hard:** Strict F1 minus FP penalty — precision and recall both matter

This ensures the agent always receives gradient signal, even when partially correct.

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -e .

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t pii-compliance-auditor .
docker run -p 8000:8000 pii-compliance-auditor
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:8000"

python inference.py
```

### Using the Client

```python
from client import PIIEnv

with PIIEnv(base_url="http://localhost:8000") as env:
    env.reset()

    # List tasks
    tasks = env.call_tool("list_tasks")

    # Get a task document
    task = env.call_tool("get_task", task_id="task_1_easy")

    # Submit findings
    result = env.call_tool(
        "submit_findings",
        task_id="task_1_easy",
        findings_json='[{"text": "Sarah Mitchell", "pii_type": "NAME"}]'
    )
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Reset episode |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/ws` | WebSocket | Persistent session |
| `/docs` | GET | OpenAPI docs |

## Baseline Scores

Scores achieved by `meta-llama/Llama-3.1-8B-Instruct` via HF Inference API:

| Task | Difficulty | Score |
|---|---|---|
| Basic PII Detection | Easy | ~0.85 |
| Onboarding Form Audit | Medium | ~0.60 |
| Breach Report Audit | Hard | ~0.35 |

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API key |
| `ENV_URL` | Environment server URL (default: http://localhost:8000) |