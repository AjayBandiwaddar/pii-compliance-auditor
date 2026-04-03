# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
PII Compliance Auditor — Environment Implementation.

Three tasks of increasing difficulty:
  Task 1 (Easy)   — Support email, 5 items, recall grader    → ~0.80
  Task 2 (Medium) — Onboarding form, 10 items, partial grader → ~0.60
  Task 3 (Hard)   — Forensic audit, 15 items, strict F1       → ~0.30

Graders are fully deterministic.
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
            "Note: not every person mentioned is a data subject — "
            "only flag PII belonging to the customer contacting support. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "EMAIL", "PHONE"],
        "document": """Subject: Billing Problem — Urgent

Hi there,

My name is Rachel Donovan and I've been charged twice for my subscription
this month. Please contact me at rachel.d@fastmail.com or on my mobile
+44-7911-123456 as soon as possible.

I also have a backup address if the first doesn't work:
r.donovan.backup@gmail.com — please try both.

My account was handled by your agent Tom Nguyen last time, if that helps.

Thanks,
Rachel""",
        "ground_truth": [
            {"text": "Rachel Donovan", "pii_type": "NAME"},
            {"text": "rachel.d@fastmail.com", "pii_type": "EMAIL"},
            {"text": "+44-7911-123456", "pii_type": "PHONE"},
            {"text": "r.donovan.backup@gmail.com", "pii_type": "EMAIL"},
            {"text": "Tom Nguyen", "pii_type": "NAME"},
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
            "Department codes and employee IDs are NOT PII — do not flag them. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": ["NAME", "ADDRESS", "DOB", "SSN", "PHONE", "EMAIL"],
        "document": """EMPLOYEE ONBOARDING FORM — CONFIDENTIAL
Department Code: DEPT-7721 | Employee Ref: EMP-43210

Primary Employee
Full Name: Ananya Krishnan
Date of birth: 12 September 1991
Residential address: Koramangala 5th Block, Bangalore 560095
Mobile: +91-98765-43210
Work email provisioned: ananya.k@techventures.in
Government ID (SSN equivalent on file): 384-62-1947

Notes: Laptop to be shipped to the residential address on file.
Onboarding buddy assigned: David Osei (senior engineer).
David joined the company on 14/03/1988 — please coordinate schedules.
His contact for onboarding queries: d.osei.ref@consultco.com
David is reachable on his Ghana line at +233-20-756-1122 during IST hours.

HR rep: Linda — see internal ref TXN-43210.""",
        "ground_truth": [
            {"text": "Ananya Krishnan", "pii_type": "NAME"},
            {"text": "12 September 1991", "pii_type": "DOB"},
            {"text": "Koramangala 5th Block, Bangalore 560095", "pii_type": "ADDRESS"},
            {"text": "+91-98765-43210", "pii_type": "PHONE"},
            {"text": "ananya.k@techventures.in", "pii_type": "EMAIL"},
            {"text": "384-62-1947", "pii_type": "SSN"},
            {"text": "David Osei", "pii_type": "NAME"},
            {"text": "14/03/1988", "pii_type": "DOB"},
            {"text": "d.osei.ref@consultco.com", "pii_type": "EMAIL"},
            {"text": "+233-20-756-1122", "pii_type": "PHONE"},
        ],
        "grader": "medium",
    },

    "task_3_hard": {
        "task_name": "Multi-Party Compliance Forensic Audit",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic audit on a leaked "
            "internal document containing form data, transaction logs, and support notes. "
            "Identify ALL personally identifiable information (PII). "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, "
            "ADDRESS, DOB, PASSWORD. "
            "CRITICAL WARNINGS: "
            "(1) The document contains red herrings — partial card numbers, IP addresses, "
            "version strings, reference codes, and SSN fragments. Do NOT flag these. "
            "(2) Two different individuals appear in the document — audit both. "
            "(3) Some formats are non-standard: credit cards use spaces, Japanese phone "
            "numbers have no country code, addresses may be non-Western. "
            "(4) Passwords appear inside technical notes without obvious labels. "
            "Extract EXACT text as it appears. Both text AND pii_type must be correct."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN",
            "ADDRESS", "DOB", "PASSWORD"
        ],
        "document": """INTERNAL AUDIT EXPORT — RESTRICTED
Reference: AUD-2024-3391 | Date: 2024-11-05 | Analyst: compliance-bot

=== RECORD 1: CUSTOMER ACCOUNT (EU) ===

Account holder: Isabel Ferreira
Registered email: i.ferreira@netcorp.pt
Date of birth on file: 1987-06-14
Primary card: 4532 0151 1283 0366
Billing address: Rua das Flores 22, 1200-195 Lisboa, Portugal
Support line: +351-91-234-5678
Govt ID cross-ref: 221-54-8832
Last known credential (plaintext capture from legacy system):
  stored_pwd = "Tr0ub4dor&3!"

Transaction log excerpt:
  TXN-4532-0151 | EUR 142.00 | APPROVED | ref: 221-54 | ip: 192.168.10.5
  Card BIN: 4532 | Scheme: VISA | Version: 1987-06

=== RECORD 2: PARTNER ACCOUNT (APAC) ===

The second account is registered to one Hiroshi Tanaka, a corporate
partner based in Tokyo. His registered contact is
h.tanaka.work@jp-systems.co.jp and billing is routed to card
4916 1234 5678 9012 (exp 03/27).

Registered address: 3-7-1 Shinjuku, Tokyo 160-0022
Date of birth as per KYC submission: 1979-11-30
Local contact number: 090-4432-8871
System credential recovered from config backup:
  # tanaka admin token: K@ts0_2024#jp

Support notes:
  - Opened ticket ref 4916-1234 re: billing dispute (not a card number)
  - DOB field showed 1979-11 initially (data entry error, now corrected)
  - IP at time of access: 10.0.4.88""",
        "ground_truth": [
            {"text": "Isabel Ferreira", "pii_type": "NAME"},
            {"text": "i.ferreira@netcorp.pt", "pii_type": "EMAIL"},
            {"text": "1987-06-14", "pii_type": "DOB"},
            {"text": "4532 0151 1283 0366", "pii_type": "CREDIT_CARD"},
            {"text": "Rua das Flores 22, 1200-195 Lisboa, Portugal", "pii_type": "ADDRESS"},
            {"text": "+351-91-234-5678", "pii_type": "PHONE"},
            {"text": "221-54-8832", "pii_type": "SSN"},
            {"text": "Tr0ub4dor&3!", "pii_type": "PASSWORD"},
            {"text": "Hiroshi Tanaka", "pii_type": "NAME"},
            {"text": "h.tanaka.work@jp-systems.co.jp", "pii_type": "EMAIL"},
            {"text": "4916 1234 5678 9012", "pii_type": "CREDIT_CARD"},
            {"text": "3-7-1 Shinjuku, Tokyo 160-0022", "pii_type": "ADDRESS"},
            {"text": "1979-11-30", "pii_type": "DOB"},
            {"text": "090-4432-8871", "pii_type": "PHONE"},
            {"text": "K@ts0_2024#jp", "pii_type": "PASSWORD"},
        ],
        "grader": "hard",
    },
}

# ── Grading Logic ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.lower().strip()


def _grade_easy(predicted: list, ground_truth: list) -> dict:
    """Recall-based. Full credit per correct text+type match. No FP penalty."""
    correct = 0
    matched = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and pred.get("pii_type", "").upper() == gt["pii_type"]):
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
            f"FP penalty: -{round(penalty, 2)}. Final: {round(score, 4)}"
        ),
    }


def _grade_hard(predicted: list, ground_truth: list) -> dict:
    """Strict F1. Exact text AND type required. FP penalty = 0.08 each."""
    correct = 0
    matched = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and pred.get("pii_type", "").upper() == gt["pii_type"]):
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
            f"Final: {round(score, 4)}"
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

    Tools:
        list_tasks()                          → all available tasks
        get_task(task_id)                     → document + instructions
        submit_findings(task_id, findings_json) → grade and score
        get_current_state()                   → episode metadata
    """

    def __init__(self):
        mcp = FastMCP("pii_compliance_auditor")

        @mcp.tool
        def list_tasks() -> str:
            """List all available PII auditing tasks."""
            return json.dumps([
                {
                    "task_id": tid,
                    "task_name": t["task_name"],
                    "difficulty": t["difficulty"],
                    "pii_types_in_scope": t["pii_types_in_scope"],
                }
                for tid, t in TASKS.items()
            ], indent=2)

        @mcp.tool
        def get_task(task_id: str) -> str:
            """
            Get document and instructions for a task.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
            """
            if task_id not in TASKS:
                return json.dumps({"error": f"Unknown task_id '{task_id}'."})
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
                findings_json: JSON array of {text, pii_type} objects.
            """
            if task_id not in TASKS:
                return json.dumps({"error": f"Unknown task_id.", "score": 0.0})
            try:
                predicted = json.loads(findings_json)
                if not isinstance(predicted, list):
                    raise ValueError("Must be a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                return json.dumps({"error": str(e), "score": 0.0})

            task = TASKS[task_id]
            result = GRADERS[task["grader"]](predicted, task["ground_truth"])
            self._cumulative_reward += result["score"]
            self._state.step_count += 1
            self._submissions[task_id] = result["score"]
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
            """Get current episode state."""
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

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
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
                    "Use list_tasks() → get_task(task_id) → submit_findings()."
                ),
                "tasks_available": list(TASKS.keys()),
            },
        )

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False, reward=0.0,
            metadata={"error": f"Unknown action: {type(action).__name__}. Use MCP tools."},
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state