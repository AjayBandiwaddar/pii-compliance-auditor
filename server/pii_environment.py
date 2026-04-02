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
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "EMAIL", "PHONE"],
        "document": """Subject: Account Access Issue

Hi Support Team,

My name is Sarah Mitchell and I've been unable to log into my account
since yesterday. Please reach me at sarah.mitchell@outlook.com or call
me on +1-415-992-3847 at any time.

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
            "Note: some PII may be embedded in sentences rather than labeled fields. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "ADDRESS", "DOB", "SSN", "PHONE", "EMAIL"],
        "document": """EMPLOYEE ONBOARDING FORM — CONFIDENTIAL

Full Name: James Okafor
The employee was born on the 4th of July, 1988, in Lagos, Nigeria.
Current residential address: 47 Maple Drive, Austin, Texas 78701.
For payroll, the SSN on file reads 472-80-1937.
James can be reached at +1-737-204-9901.
Work email has been provisioned: j.okafor@techcorp.io

Emergency contact: Linda Okafor (spouse)""",
        "ground_truth": [
            {"text": "James Okafor", "pii_type": "NAME"},
            {"text": "4th of July, 1988", "pii_type": "DOB"},
            {"text": "47 Maple Drive, Austin, Texas 78701", "pii_type": "ADDRESS"},
            {"text": "472-80-1937", "pii_type": "SSN"},
            {"text": "+1-737-204-9901", "pii_type": "PHONE"},
            {"text": "j.okafor@techcorp.io", "pii_type": "EMAIL"},
        ],
        "grader": "medium",
    },

    "task_3_hard": {
        "task_name": "System Audit Log — Advanced PII Extraction",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic audit on a leaked system log file. "
            "Identify ALL personally identifiable information (PII) present. "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS, DOB, PASSWORD. "
            "WARNING: The document contains red herrings — values that resemble PII but are not "
            "(e.g. error codes, IP addresses, version numbers, case IDs). Do NOT flag these. "
            "PII may appear inside code snippets, log lines, database dumps, or config files. "
            "Some formats are unconventional: dots instead of dashes, spaces instead of dashes, "
            "international phone formats, addresses embedded in strings. "
            "Extract the EXACT text as it appears. Both text AND pii_type must be correct for full credit. "
            "Submit as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN",
            "ADDRESS", "DOB", "PASSWORD"
        ],
        "document": """SYSTEM AUDIT EXPORT — CLASSIFICATION: RESTRICTED
Export ID: SYS-2024-09-AUDIT | Generated: 2024-09-18T03:22:11Z
Reviewed by: Compliance Team | Case #291-44-8830

--- SECTION 1: APPLICATION CONFIG SNAPSHOT ---

# app_config.yaml (committed to internal repo by mistake)
database:
  host: 192.168.1.47
  port: 5432
  user: svc_account
  password: "p@ssw0rd#Webb2024"

billing:
  processor: stripe_v3
  fallback_card: "4916 3301 0198 2104"
  backup_card: "5412 7534 9823 0011"

notifications:
  # Primary contact for billing alerts — M.T. Webb
  email: m.webb.92@protonmail.com
  phone: +1.206.887.4401

--- SECTION 2: DATABASE ROW DUMP (users table) ---

| id  | full_name     | dob        | ssn         | address                                          |
|-----|---------------|------------|-------------|--------------------------------------------------|
| 104 | Marcus T. Webb| 08/15/1996 | 291.44.8830 | 740 Evergreen Terrace, Springfield, OR 97477     |
| 105 | Priya Nair    | 1990-03-22 | 408.55.1923 | Flat 9, Tower B, Prestige Elysian, Bangalore 560068 |

NOTE: SSN format uses dots — confirm with HR. Error code 4916-3301 flagged during export.
Version 2.104 of the schema was active at time of breach.

--- SECTION 3: ACCESS LOGS (raw) ---

2024-09-17 22:14:03 | AUTH_FAIL | user=priya.nair | ip=10.0.0.22 | attempt=3
2024-09-17 22:14:09 | AUTH_OK   | user=priya.nair | email=p.nair@encrypted.io | session=a8f3c
2024-09-17 22:15:44 | PROFILE_VIEW | accessed_by=priya.nair | target_user=Marcus T. Webb
2024-09-17 22:16:01 | EXPORT_REQ  | fields=[name,dob,ssn,address,card] | requester=p.nair@encrypted.io

--- SECTION 4: SUPPORT TICKET #98-4501 ---

Customer called from 0091-98-4501-3376 (India) requesting card dispute.
Agent notes: caller verified identity, DOB confirmed as 08/15/1996.
Card disputed: last four digits 2104. Full number retrieved from vault: 4916 3301 0198 2104.
Billing address on file: 740 Evergreen Terrace, Springfield, OR 97477.
Escalated to senior agent. ID: 98-4501 logged for tracking.""",
        "ground_truth": [
            {"text": "p@ssw0rd#Webb2024", "pii_type": "PASSWORD"},
            {"text": "4916 3301 0198 2104", "pii_type": "CREDIT_CARD"},
            {"text": "5412 7534 9823 0011", "pii_type": "CREDIT_CARD"},
            {"text": "m.webb.92@protonmail.com", "pii_type": "EMAIL"},
            {"text": "+1.206.887.4401", "pii_type": "PHONE"},
            {"text": "Marcus T. Webb", "pii_type": "NAME"},
            {"text": "08/15/1996", "pii_type": "DOB"},
            {"text": "291.44.8830", "pii_type": "SSN"},
            {"text": "740 Evergreen Terrace, Springfield, OR 97477", "pii_type": "ADDRESS"},
            {"text": "Priya Nair", "pii_type": "NAME"},
            {"text": "1990-03-22", "pii_type": "DOB"},
            {"text": "408.55.1923", "pii_type": "SSN"},
            {"text": "Flat 9, Tower B, Prestige Elysian, Bangalore 560068", "pii_type": "ADDRESS"},
            {"text": "p.nair@encrypted.io", "pii_type": "EMAIL"},
            {"text": "0091-98-4501-3376", "pii_type": "PHONE"},
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
            f"FP penalty: -{round(penalty, 2)}. Final score: {round(score, 4)}"
        ),
    }


def _grade_hard(predicted: list, ground_truth: list) -> dict:
    """Strict F1. Both text AND type must match exactly. FP penalty = 0.15."""
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
    penalty = fp * 0.15
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