---
title: PII Compliance Auditor
emoji: 🔒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - pii
  - compliance
  - agent
---

# PII Compliance Auditor

A real-world OpenEnv environment where an AI agent audits documents for Personally Identifiable Information (PII) — simulating the compliance work done by data privacy teams operating under GDPR and CCPA regulations.

---

## Overview

Data privacy compliance is a genuine, high-stakes problem. Thousands of companies must identify and protect PII across their documents every day. This environment trains and evaluates AI agents on exactly this task — across three documents of increasing difficulty.

The agent receives a document, identifies all PII present, classifies each item by type, and submits its findings for grading. Rewards are partial and meaningful, ensuring gradient signal throughout the episode.

---

## Environment Design

### Action Space

The environment exposes four MCP tools:

| Tool | Description |
|---|---|
| `list_tasks()` | List all available tasks with difficulty levels |
| `get_task(task_id)` | Retrieve document and instructions for a task |
| `submit_findings(task_id, findings_json)` | Submit detected PII items for grading |
| `get_current_state()` | Get current episode state and progress |

**Submission format:**
```json
[
  {"text": "Sarah Mitchell", "pii_type": "NAME"},
  {"text": "sarah@example.com", "pii_type": "EMAIL"}
]
```

### Observation Space

Each tool call returns a structured JSON response containing:
- Task document and instructions
- Grading result: score, precision, recall, F1, false positives, feedback
- Episode state: step count, submissions so far, cumulative reward, done flag

### PII Types

| Type | Example |
|---|---|
| NAME | Dr. Amelia Voss |
| EMAIL | amelia@gmail.com |
| PHONE | +1-415-992-3847 |
| CREDIT_CARD | 4539-1488-0343-6467 |
| SSN | 309-52-7781 |
| ADDRESS | 12 Birchwood Lane, Chicago, IL 60614 |
| DOB | March 3rd, 1979 |
| PASSWORD | Sunrise@2024! |

---

## Tasks

### Task 1 - Easy: Basic PII Detection
- **Document:** Customer support email
- **PII in scope:** NAME, EMAIL, PHONE
- **Challenge:** 3 clearly labeled PII items in natural language
- **Grader:** Recall-based — each correctly found item earns credit
- **Baseline score:** 1.00

### Task 2 - Medium: Employee Onboarding Audit
- **Document:** Employee onboarding form
- **PII in scope:** NAME, ADDRESS, DOB, SSN, PHONE, EMAIL
- **Challenge:** 6 PII items, some embedded in sentences rather than labeled fields
- **Grader:** Partial credit (0.5) for correct text with wrong type. FP penalty 0.1
- **Baseline score:** 0.90

### Task 3 - Hard: Data Breach Incident Report
- **Document:** Restricted incident report covering two individuals
- **PII in scope:** All 8 types
- **Challenge:** 13 PII items, unconventional formats, embedded in prose, multiple people
- **Grader:** Strict F1 — exact text AND type required. FP penalty 0.15
- **Baseline score:** 1.00

---

## Reward Function

Rewards are partial and trajectory-level, not binary end-of-episode signals.

- **Task 1:** Score = recall. Every correctly identified item earns proportional credit.
- **Task 2:** Partial credit for right text / wrong type. False positives penalized.
- **Task 3:** Strict F1 minus false positive penalty. Both precision and recall matter.

---

## Baseline Results

Model: meta-llama/Llama-3.1-8B-Instruct via HuggingFace Inference API

| Task | Difficulty | Score |
|---|---|---|
| Basic PII Detection | Easy | 1.00 |
| Employee Onboarding Audit | Medium | 0.90 |
| Data Breach Incident Report | Hard | 1.00 |
| Average | | 0.97 |

---

## Setup and Usage

### Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t pii-compliance-auditor .
docker run -p 7860:7860 pii-compliance-auditor
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="https://ajaybandiwaddar01-pii-compliance-auditor.hf.space"

python inference.py
```

### Use the Client

```python
from client import PIIEnv

with PIIEnv(base_url="http://localhost:7860").sync() as env:
    env.reset()

    task = env.call_tool("get_task", task_id="task_1_easy")

    result = env.call_tool(
        "submit_findings",
        task_id="task_1_easy",
        findings_json='[{"text": "Sarah Mitchell", "pii_type": "NAME"}]'
    )
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Health check |
| /reset | POST | Reset episode |
| /step | POST | Execute action |
| /state | GET | Get current state |
| /docs | GET | Interactive Swagger UI |

---

## Environment Variables

| Variable | Description |
|---|---|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier for inference |
| HF_TOKEN | HuggingFace API key |
| ENV_URL | Environment server URL |

---

## Project Structure

```
```
pii-compliance-auditor/
├── server/
│   ├── pii_environment.py   # Core environment + graders + task registry
│   └── app.py               # FastAPI server via create_app()
├── client.py                # PIIEnv(MCPToolClient)
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv spec manifest
├── Dockerfile               # Container definition
├── requirements.txt         # Dependencies
├── pyproject.toml           # Project metadata
├── uv.lock                  # Locked dependencies for reproducibility
└── README.md
```