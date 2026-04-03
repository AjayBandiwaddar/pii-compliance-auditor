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
You can also try my office line during business hours: +44-20-7946-0123.

Thanks,
Rachel""",
        "ground_truth": [
            {"text": "Rachel Donovan", "pii_type": "NAME"},
            {"text": "rachel.d@fastmail.com", "pii_type": "EMAIL"},
            {"text": "+44-7911-123456", "pii_type": "PHONE"},
            {"text": "r.donovan.backup@gmail.com", "pii_type": "EMAIL"},
            {"text": "Tom Nguyen", "pii_type": "NAME"},
            {"text": "+44-20-7946-0123", "pii_type": "PHONE"},
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

HR rep: Linda Mensah — see internal ref TXN-43210.
Linda's employee number for payroll cross-reference: SSN 619-33-4402.""",
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
            {"text": "Linda Mensah", "pii_type": "NAME"},
            {"text": "619-33-4402", "pii_type": "SSN"},
        ],
        "grader": "medium",
    },

    "task_3_hard": {
        "task_name": "Multi-Party Compliance Forensic Audit",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic audit on a leaked "
            "internal compliance dossier covering two international accounts. "
            "Identify ALL personally identifiable information (PII). "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, "
            "ADDRESS, DOB, PASSWORD. "
            "IMPORTANT RULES: "
            "(1) Extract text EXACTLY as it appears — do not reformat dates, "
            "normalize phone numbers, remove spaces from card numbers, or change separators. "
            "(2) SSNs may use dots instead of dashes — extract with dots as shown. "
            "(3) Credit card numbers may contain spaces — extract with spaces as shown. "
            "(4) The document contains supplemental record entries such as emergency contacts, "
            "former addresses, voided cards, and authorized delegates — audit all sections "
            "thoroughly and flag every person, address, contact, and credential you find. "
            "(5) Two primary account holders appear — audit both completely. "
            "Submit as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN",
            "ADDRESS", "DOB", "PASSWORD"
        ],
        "document": """COMPLIANCE DOSSIER — DUAL ACCOUNT AUDIT
Case: COMP-2024-7731 | Classification: RESTRICTED | Analyst: auto-review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCOUNT A — EASTERN EUROPE SEGMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Account holder: Dmitri Volkov
Primary contact: d.volkov@securenet.ru
Secondary/backup address: d.volkov.backup@protonmail.com
Mobile (RU): +7-916-234-5678
Date registered: 1983-09-27
Residential: ul. Tverskaya 14, kv. 7, Moscow 125009
Card on file [spaces preserved per vault format]: 5425 2334 3010 9903
Tax ID cross-reference (dot-format as received): 531.20.7732

Credential audit note — legacy plaintext capture:
  CONFIG_KEY="V0lk0v#Secure_83!"

--- Supplemental Record: Account A ---
Emergency contact name: Elena Volkova (spouse)
Emergency contact email: elena.volkova@mail.ru
Emergency contact phone: +7-495-111-2233
Former residential address (pre-2021, now voided): ul. Arbat 33, kv. 2, Moscow 119002
Voided card (cancelled 2022): 4111 1111 1111 1111

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCOUNT B — APAC SEGMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The second account belongs to Aiko Shimizu, a corporate client
whose registered email is aiko.s@nantech.co.jp.
Office landline on record: +81-3-5412-8876
Personal mobile (local JP format): 080-3344-9921
KYC date of birth field: 1991-03-15
Registered billing address: 2-8-12 Minami-Aoyama, Minato-ku, Tokyo 107-0062
Card vault entry [spaces preserved]: 3566 0020 2006 0505
Government ID (dot-separated format): 412.75.9301

System notes — credential recovery log:
  # token_value = "Sh!mizu_2024#"

--- Supplemental Record: Account B ---
Authorized delegate: Kenji Shimizu (brother)
Delegate contact: kenji.shimizu@nantech.co.jp
Previous billing address (superseded 2023): 1-14-2 Shibuya, Tokyo 150-0002
Shared office switchboard: +81-3-0000-1234""",
        "ground_truth": [
            {"text": "Dmitri Volkov", "pii_type": "NAME"},
            {"text": "d.volkov@securenet.ru", "pii_type": "EMAIL"},
            {"text": "d.volkov.backup@protonmail.com", "pii_type": "EMAIL"},
            {"text": "+7-916-234-5678", "pii_type": "PHONE"},
            {"text": "1983-09-27", "pii_type": "DOB"},
            {"text": "ul. Tverskaya 14, kv. 7, Moscow 125009", "pii_type": "ADDRESS"},
            {"text": "5425 2334 3010 9903", "pii_type": "CREDIT_CARD"},
            {"text": "531.20.7732", "pii_type": "SSN"},
            {"text": "V0lk0v#Secure_83!", "pii_type": "PASSWORD"},
            {"text": "Aiko Shimizu", "pii_type": "NAME"},
            {"text": "aiko.s@nantech.co.jp", "pii_type": "EMAIL"},
            {"text": "+81-3-5412-8876", "pii_type": "PHONE"},
            {"text": "080-3344-9921", "pii_type": "PHONE"},
            {"text": "1991-03-15", "pii_type": "DOB"},
            {"text": "2-8-12 Minami-Aoyama, Minato-ku, Tokyo 107-0062", "pii_type": "ADDRESS"},
            {"text": "3566 0020 2006 0505", "pii_type": "CREDIT_CARD"},
            {"text": "412.75.9301", "pii_type": "SSN"},
            {"text": "Sh!mizu_2024#", "pii_type": "PASSWORD"},
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
    """Partial credit: right text wrong type = 0.5. Final medium score is calibrated downward."""
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
    penalty = fp * 0.04
    calibrated = (raw - penalty) * 0.7
    score = max(0.0, min(1.0, calibrated))
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
            f"FP penalty: -{round(penalty, 2)}, medium calibration x0.7. "
            f"Final: {round(score, 4)}"
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
    penalty = fp * 0.065
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
