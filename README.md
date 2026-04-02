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

A real-world OpenEnv environment for PII detection and classification.

An AI agent audits documents for Personally Identifiable Information (PII)
compliance, simulating work done by data privacy teams under GDPR and CCPA.

## Tasks

- Task 1 (Easy): Customer support email, detect NAME, EMAIL, PHONE
- Task 2 (Medium): Employee onboarding form, 6 embedded PII items
- Task 3 (Hard): Data breach report, 13 PII items across all 8 types

## API Endpoints

- POST /reset - Start new episode
- POST /step - Execute action
- GET /state - Get current state
- GET /health - Health check

## Environment Variables

- API_BASE_URL - LLM API endpoint
- MODEL_NAME - Model identifier
- HF_TOKEN - HuggingFace API key
