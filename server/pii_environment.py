# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
PII Compliance Auditor — Environment Implementation.

A real-world OpenEnv environment where an AI agent must identify and
classify Personally Identifiable Information (PII) in documents.

Three tasks of increasing difficulty:
  Task 1 (Easy)   — Customer support email, 3 obvious PII items
  Task 2 (Medium) — Employee onboarding form, 6 embedded PII items
  Task 3 (Hard)   — Data breach report, 13 PII items across all 8 types

Graders are fully deterministic: given the same submission, always
the same score.
"""

import json
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

# ── Ground Truth Registry ──────────────────────────────────────────────────

TASKS = {
    "task_1_easy": {
        "task_name": "Basic PII Detection",
        "difficulty": "easy",
        "description": (
            "You are a PII compliance auditor. Analyze the customer support email below "
            "and identify ALL personally identifiable information (PII). "
            "Look for: NAME, EMAIL, PHONE. "
            "Note: not every number or name-like word is PII — use judgment. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "EMAIL", "PHONE"],
        "document": """Subject: Account Access Issue — Ticket #4159-2847

Hi Support Team,

My name is Sarah Mitchell and I've been unable to log into my account
since yesterday. The issue started after I updated my billing details on
the 3rd. Please reach me at sarah.mitchell@outlook.com or call me on
+1-415-992-3847 at any time — my alternate is also reachable but
sarah.mitchell@outlook.com is preferred.

Could you also check with your colleague James if ticket 4159 is linked?

Looking forward to your help.
Best,
Sarah""",
        "ground_truth": [
            {"text": "Sarah Mitchell", "pii_type": "NAME"},
            {"text": "sarah.mitchell@outlook.com", "pii_type": "EMAIL"},
            {"text": "+1-415-992-3847", "pii_type": "PHONE"},
        ],
        "grader": "easy",
    },

    "task_2_medium": {
        "task_name": "Employee Onboarding Form Audit",
        "difficulty": "medium",
        "description": (
            "You are a PII compliance auditor. Analyze the employee onboarding form below "
            "and identify ALL personally identifiable information (PII). "
            "Look for: NAME, ADDRESS, DOB, SSN, PHONE, EMAIL. "
            "Note: some PII is embedded in sentences rather than labeled fields. "
            "Not all numbers are PII — department codes and employee IDs are not. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "ADDRESS", "DOB", "SSN", "PHONE", "EMAIL"],
        "document": """EMPLOYEE ONBOARDING FORM — CONFIDENTIAL
Department Code: 472-80 | Employee ID: TXN-9901

Full Name: James Okafor
The employee was born on the 4th of July, 1988, in Lagos, Nigeria.
Current residential address: 47 Maple Drive, Austin, Texas 78701.
For payroll, the SSN on file reads 472-80-1937.
James can be reached at +1-737-204-9901.
Work email has been provisioned: j.okafor@techcorp.io

IT notes: laptop shipped to the residential address above.
The account was activated by HR rep Linda — see internal ID TXN-9901.

Emergency contact: Linda Okafor, reachable at l.okafor.hr@techcorp.io""",
        "ground_truth": [
            {"text": "James Okafor", "pii_type": "NAME"},
            {"text": "4th of July, 1988", "pii_type": "DOB"},
            {"text": "47 Maple Drive, Austin, Texas 78701", "pii_type": "ADDRESS"},
            {"text": "472-80-1937", "pii_type": "SSN"},
            {"text": "+1-737-204-9901", "pii_type": "PHONE"},
            {"text": "j.okafor@techcorp.io", "pii_type": "EMAIL"},
            {"text": "Linda Okafor", "pii_type": "NAME"},
            {"text": "l.okafor.hr@techcorp.io", "pii_type": "EMAIL"},
        ],
        "grader": "medium",
    },

    "task_3_hard": {
        "task_name": "Technical Artifact Forensic Audit",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic review of a leaked technical artifact "
            "containing git diffs, SQL statements, API logs, and support chat transcripts. "
            "Identify ALL personally identifiable information (PII). "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS, DOB, PASSWORD. "
            "CRITICAL WARNINGS: "
            "(1) The document contains convincing red herrings — partial card numbers, usernames without domains, "
            "version strings, order IDs, product codes, and bcrypt hashes. Do NOT flag these as PII. "
            "(2) PII appears inside code strings, SQL values, JSON fields, and natural conversation — "
            "there are NO labeled fields like 'email:' or 'phone:' to guide you. "
            "(3) Some formats are non-standard: AMEX cards have 15 digits, "
            "international phones have country codes, addresses may be non-Western. "
            "Extract EXACT text as it appears. Both text AND pii_type must be correct for full credit. "
            "Submit as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN",
            "ADDRESS", "DOB", "PASSWORD"
        ],
        "document": """FORENSIC EXPORT — INTERNAL USE ONLY
Ticket: SEC-2024-1847 | Analyst: redacted | Date: 2024-10-02

=== SECTION A: GIT DIFF (accidental credential commit) ===

diff --git a/config/prod.yaml b/config/prod.yaml
--- a/config/prod.yaml
+++ b/config/prod.yaml
@@ -12,7 +12,7 @@
 payment:
   processor: stripe
-  api_key: "sk_live_placeholder"
+  primary_card: "3782 822463 10005"
+  # fallback — Discover: 6011 1111 1111 1117
+  card_holder: "Kenji Watanabe"
+  notify: k.watanabe@devmail.jp

 auth:
-  admin_pass: "changeme"
+  admin_pass: "W@tana8e#Secure!"
+  # DO NOT COMMIT — remove before PR
+  # bcrypt ref: $2b$12$W@tana8eHash... (not the real password)

diff --git a/tests/fixtures/user_seed.sql b/tests/fixtures/user_seed.sql
--- a/tests/fixtures/user_seed.sql
+++ b/tests/fixtures/user_seed.sql
@@ -0,0 +1,6 @@
+-- Seed data accidentally included real records (revert immediately)
+INSERT INTO users (name, dob, ssn, address) VALUES (
+  'Kenji Watanabe',
+  '1995-02-23',
+  '523-45-7890',
+  '2-14-5 Shibuya, Tokyo 150-0002, Japan'
+);
+-- Product code for reference: 523-45 (unrelated to above)
+-- Order ID: 6011-1117 (do not confuse with payment methods)

=== SECTION B: API RESPONSE LOG (production traffic sample) ===

2024-10-01 18:42:03 POST /api/v2/checkout 200
Request-ID: 3782-8224-ABCD
Payload (truncated):
{
  "customer": {
    "full_name": "Fatima Al-Rashidi",
    "contact": "fatima.ar.1988@outlook.com",
    "mobile": "971-4-123-9876",
    "billing_address": "PO Box 4422, Dubai Marina, Dubai, UAE",
    "card": "6011 1111 1111 1117",
    "dob": "1988-11-03"
  },
  "session": "a8f3c9d",
  "version": "1988-11",
  "ip": "192.168.0.14"
}

=== SECTION C: SUPPORT CHAT TRANSCRIPT ===

[10:14] Agent_07: Hi, can I get your name please?
[10:14] Customer: yeah its fatima, Fatima Al-Rashidi
[10:15] Agent_07: Thanks Fatima. Phone number on the account?
[10:15] Customer: its nine-seven-one, four, one-two-three, nine-eight-seven-six
[10:16] Agent_07: Got it. And the reset token for your account was sent — did you set
         the new password to what you told our bot earlier?
[10:16] Customer: yes its #fa_r@sh!d1_2024 i set it this morning
[10:17] Agent_07: Perfect. Your card ending 1117 has been flagged. Full number
         on file is 6011 1111 1111 1117. Shipping to PO Box 4422, Dubai Marina, Dubai, UAE?
[10:17] Customer: yes thats right. also can you update my phone to +81-90-3344-5567
[10:18] Agent_07: Done. Anything else?""",
        "ground_truth": [
            {"text": "3782 822463 10005", "pii_type": "CREDIT_CARD"},
            {"text": "6011 1111 1111 1117", "pii_type": "CREDIT_CARD"},
            {"text": "Kenji Watanabe", "pii_type": "NAME"},
            {"text": "k.watanabe@devmail.jp", "pii_type": "EMAIL"},
            {"text": "W@tana8e#Secure!", "pii_type": "PASSWORD"},
            {"text": "1995-02-23", "pii_type": "DOB"},
            {"text": "523-45-7890", "pii_type": "SSN"},
            {"text": "2-14-5 Shibuya, Tokyo 150-0002, Japan", "pii_type": "ADDRESS"},
            {"text": "Fatima Al-Rashidi", "pii_type": "NAME"},
            {"text": "fatima.ar.1988@outlook.com", "pii_type": "EMAIL"},
            {"text": "971-4-123-9876", "pii_type": "PHONE"},
            {"text": "PO Box 4422, Dubai Marina, Dubai, UAE", "pii_type": "ADDRESS"},
            {"text": "1988-11-03", "pii_type": "DOB"},
            {"text": "#fa_r@sh!d1_2024", "pii_type": "PASSWORD"},
            {"text": "+81-90-3344-5567", "pii_type": "PHONE"},
        ],
        "grader": "hard",
    },
}
# ── Grading Logic ──────────────────────────────────────────────────────────

# Common LLM aliases for PII types — maps to our canonical names
PII_TYPE_ALIASES = {
    "DATE_OF_BIRTH": "DOB",
    "BIRTH_DATE": "DOB",
    "BIRTHDATE": "DOB",
    "DATE": "DOB",
    "CARD_NUMBER": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CARD": "CREDIT_CARD",
    "DEBIT_CARD": "CREDIT_CARD",
    "PAYMENT_CARD": "CREDIT_CARD",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "SOCIAL_SECURITY": "SSN",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "MOBILE": "PHONE",
    "MOBILE_NUMBER": "PHONE",
    "FULL_NAME": "NAME",
    "PERSON_NAME": "NAME",
    "STREET_ADDRESS": "ADDRESS",
    "HOME_ADDRESS": "ADDRESS",
    "PASS": "PASSWORD",
    "CREDENTIAL": "PASSWORD",
    "EMAIL_ADDRESS": "EMAIL",
}

def _normalize_pii_type(pii_type: str) -> str:
    """Normalize LLM pii_type to our canonical names."""
    canonical = pii_type.upper().strip()
    return PII_TYPE_ALIASES.get(canonical, canonical)

def _normalize(text: str) -> str:
    return text.lower().strip()


def _normalize_numeric(text: str) -> str:
    """For numeric PII (CC, SSN, PHONE), strip separators before comparing.
    This ensures format variations (spaces/dots/dashes) don't cause false mismatches."""
    import re
    return re.sub(r"[\s\-\.]", "", text.lower().strip())


NUMERIC_PII_TYPES = {"CREDIT_CARD", "SSN", "PHONE"}


def _grade_easy(predicted: list, ground_truth: list) -> dict:
    """Recall-based. Full credit per correct text+type match. No FP penalty."""
    correct = 0
    matched = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and _normalize_pii_type(pred.get("pii_type", "")) == gt["pii_type"]):
                    correct += 1
                    matched.add(i)
                    break
    total = len(ground_truth)
    recall = correct / total if total else 0.0
    fp = max(len(predicted) - correct, 0)
    precision = correct / len(predicted) if predicted else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "score": round(recall, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct": correct,
        "total": total,
        "false_positives": fp,
        "partial_credit": 0.0,
        "feedback": f"Found {correct}/{total} PII items correctly. Score based on recall.",
    }


def _grade_medium(predicted: list, ground_truth: list) -> dict:
    """Partial credit: right text wrong type = 0.5. FP penalty = 0.1 each."""
    full_credit = 0
    partial = 0.0
    matched = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                pred_text = _normalize(pred.get("text", ""))
                gt_text = _normalize(gt["text"])
                text_match = (pred_text in gt_text or gt_text in pred_text)
                type_match = pred.get("pii_type", "").upper() == gt["pii_type"]
                if text_match and type_match:
                    full_credit += 1
                    matched.add(i)
                    break
                elif text_match and not type_match:
                    partial += 0.5
                    matched.add(i)
                    break
    total = len(ground_truth)
    fp = max(len(predicted) - len(matched), 0)
    raw = (full_credit + partial) / total if total else 0.0
    penalty = fp * 0.1
    score = max(0.0, min(1.0, raw - penalty))
    precision = full_credit / len(predicted) if predicted else 0.0
    recall = full_credit / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "score": round(score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct": full_credit,
        "total": total,
        "false_positives": fp,
        "partial_credit": round(partial, 4),
        "feedback": (
            f"Full matches: {full_credit}, Partial (wrong type): {partial}, "
            f"FP penalty: -{round(penalty, 2)}. Final score: {round(score, 4)}"
        ),
    }


def _grade_hard(predicted: list, ground_truth: list) -> dict:
    """Strict F1. Both text AND type must match exactly. FP penalty = 0.15.
    For numeric PII types (CREDIT_CARD, SSN, PHONE), separators are normalized
    so format variations (spaces/dots/dashes) do not penalize correct detections."""
    correct = 0
    matched = set()
    for pred in predicted:
        pred_type = _normalize_pii_type(pred.get("pii_type", ""))
        pred_text = pred.get("text", "")
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                gt_type = gt["pii_type"]
                if pred_type == gt_type:
                    if gt_type in NUMERIC_PII_TYPES:
                        text_match = _normalize_numeric(pred_text) == _normalize_numeric(gt["text"])
                    else:
                        text_match = _normalize(pred_text) == _normalize(gt["text"])
                    if text_match:
                        correct += 1
                        matched.add(i)
                        break
    total = len(ground_truth)
    fp = max(len(predicted) - correct, 0)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    penalty = fp * 0.08
    score = max(0.0, min(1.0, f1 - penalty))
    return {
        "score": round(score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct": correct,
        "total": total,
        "false_positives": fp,
        "partial_credit": 0.0,
        "feedback": (
            f"Strict F1: {round(f1, 4)}, FP penalty: -{round(penalty, 2)}. "
            f"Final score: {round(score, 4)}"
        ),
    }


GRADERS = {
    "easy": _grade_easy,
    "medium": _grade_medium,
    "hard": _grade_hard,
}

# ── Environment ────────────────────────────────────────────────────────────

class PIIEnvironment(MCPEnvironment):
    """
    PII Compliance Auditor — OpenEnv Environment.

    The agent is presented with documents containing PII and must identify
    and classify each PII item correctly. Three tasks of increasing difficulty.

    Tools exposed via MCP:
        list_tasks()              → lists all available tasks
        get_task(task_id)         → returns task document + instructions
        submit_findings(task_id, findings_json) → grades and returns score
        get_current_state()       → returns episode metadata
    """

    def __init__(self):
        mcp = FastMCP("pii_compliance_auditor")

        @mcp.tool
        def list_tasks() -> str:
            """
            List all available PII auditing tasks.

            Returns:
                JSON string with task ids, names, and difficulty levels.
            """
            tasks_summary = [
                {
                    "task_id": tid,
                    "task_name": t["task_name"],
                    "difficulty": t["difficulty"],
                    "pii_types_in_scope": t["pii_types_in_scope"],
                }
                for tid, t in TASKS.items()
            ]
            return json.dumps(tasks_summary, indent=2)

        @mcp.tool
        def get_task(task_id: str) -> str:
            """
            Get the document and instructions for a specific task.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'

            Returns:
                JSON string with task details and document to analyze.
            """
            if task_id not in TASKS:
                return json.dumps({
                    "error": f"Unknown task_id '{task_id}'. "
                             f"Valid options: {list(TASKS.keys())}"
                })
            task = TASKS[task_id]
            self._current_task_id = task_id
            self._state.step_count += 1
            return json.dumps({
                "task_id": task_id,
                "task_name": task["task_name"],
                "difficulty": task["difficulty"],
                "description": task["description"],
                "pii_types_in_scope": task["pii_types_in_scope"],
                "document": task["document"],
            }, indent=2)

        @mcp.tool
        def submit_findings(task_id: str, findings_json: str) -> str:
            """
            Submit detected PII findings for grading.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
                findings_json: JSON string — list of {text, pii_type} objects.
                    Example: '[{"text": "John Smith", "pii_type": "NAME"}]'

            Returns:
                JSON string with score (0.0-1.0), precision, recall, f1,
                false_positives, partial_credit, and feedback.
            """
            if task_id not in TASKS:
                return json.dumps({
                    "error": f"Unknown task_id '{task_id}'.",
                    "score": 0.0,
                })
            try:
                predicted = json.loads(findings_json)
                if not isinstance(predicted, list):
                    raise ValueError("findings_json must be a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                return json.dumps({
                    "error": f"Invalid findings_json: {e}",
                    "score": 0.0,
                })

            task = TASKS[task_id]
            grader_fn = GRADERS[task["grader"]]
            result = grader_fn(predicted, task["ground_truth"])

            # Track cumulative reward
            self._cumulative_reward += result["score"]
            self._state.step_count += 1
            self._submissions[task_id] = result["score"]

            # Mark done if all 3 tasks submitted
            done = len(self._submissions) >= len(TASKS)
            self._done = done

            return json.dumps({
                "task_id": task_id,
                "task_name": task["task_name"],
                "difficulty": task["difficulty"],
                **result,
                "done": done,
                "tasks_completed": len(self._submissions),
                "tasks_total": len(TASKS),
            }, indent=2)

        @mcp.tool
        def get_current_state() -> str:
            """
            Get the current state of the episode.

            Returns:
                JSON string with episode_id, step_count, submissions so far,
                cumulative_reward, and done flag.
            """
            return json.dumps({
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "submissions": self._submissions,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "done": self._done,
                "tasks_available": list(TASKS.keys()),
            }, indent=2)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward = 0.0
        self._submissions: dict = {}
        self._done = False
        self._current_task_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode."""
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._cumulative_reward = 0.0
        self._submissions = {}
        self._done = False
        self._current_task_id = None

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": (
                    "PII Compliance Auditor ready. "
                    "Use list_tasks() to see available tasks, "
                    "get_task(task_id) to retrieve a document, "
                    "and submit_findings(task_id, findings_json) to get your score."
                ),
                "tasks_available": list(TASKS.keys()),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions gracefully."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use MCP tools: list_tasks(), get_task(), "
                    "submit_findings(), get_current_state()."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step — delegates MCP actions to base class."""
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state