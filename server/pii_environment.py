# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
PII Compliance Auditor — Multi-Step Environment Implementation.

Three tasks of increasing difficulty.  Each episode is a genuine multi-step
trajectory: the agent retrieves and annotates the document section-by-section,
receiving an intermediate reward after each section, then finalises.

  Task 1 (Easy)   — 2 sections, recall grader      ->  ~1.00 final
  Task 2 (Medium) — 2 sections, partial grader     ->  ~0.90 final
  Task 3 (Hard)   — 4 sections, strict F1 grader   ->  ~0.30 final

Episode trajectory
------------------
  Step 1        : get_task(task_id)
  Step 2        : get_section(task_id, section_a)
  Step 3        : annotate_section(...)   -> intermediate reward (section recall)
  ...repeat for every section...
  Final step    : finalize_task(task_id) -> final task score, done=true

  Task 1 / 2 :  8 steps  (1 + 2x2 sections x 2 actions + 1 finalize)
  Task 3     : 14 steps  (1 + 2x4 sections x 2 actions + 1 finalize)

Intermediate reward : simple recall over that section's ground truth
Final reward        : full task grader (with FP penalty) over all annotations

Hard task calibration
---------------------
  Sections A and C contain 8 real PII items each, no red herrings.
  Sections B and D contain 1 real PII item (PASSWORD) each, plus 4
  PII-lookalike red herrings per section (emergency contacts, former
  addresses, voided cards, delegates, shared switchboards).
  All technical noise (IPs, ref codes, version strings) is removed from
  B and D so that FP count is predictable at exactly 8.

  Expected run (18 correct, 8 FPs, 26 predicted):
    precision = 18/26 = 0.692
    recall    = 18/18 = 1.000
    F1        = 0.818
    penalty   = 8 x 0.065 = 0.520
    score     = 0.818 - 0.520 = 0.298  ~  0.30

Backward compatibility
-----------------------
  submit_findings(task_id, findings_json) is kept for the automated validator.
"""

import json
from typing import Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASKS = {

    # ── Task 1 — Easy ───────────────────────────────────────────────────────
    "task_1_easy": {
        "task_name": "Basic PII Detection",
        "difficulty": "easy",
        "description": (
            "You are a PII compliance auditor. Analyze the document section "
            "provided and identify ALL personally identifiable information. "
            "Look for: NAME, EMAIL, PHONE. "
            "IMPORTANT: You are auditing CUSTOMER data only. "
            "Do NOT flag PII belonging to company employees, agents, or support staff. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"John Smith\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "EMAIL", "PHONE"],
        "grader": "easy",
        "sections": {
            "section_a": {
                "title": "Email Opening",
                "text": (
                    "Subject: Billing Problem - Urgent\n\n"
                    "Hi there,\n\n"
                    "My name is Rachel Donovan and I've been charged twice for my "
                    "subscription this month. Please contact me at "
                    "rachel.d@fastmail.com or on my mobile +44-7911-123456 as soon "
                    "as possible."
                ),
                "ground_truth": [
                    {"text": "Rachel Donovan",        "pii_type": "NAME"},
                    {"text": "rachel.d@fastmail.com",  "pii_type": "EMAIL"},
                    {"text": "+44-7911-123456",        "pii_type": "PHONE"},
                ],
            },
            "section_b": {
                "title": "Email Closing",
                "text": (
                    "I also have a backup address if the first doesn't work:\n"
                    "r.donovan.backup@gmail.com - please try both.\n\n"
                    "My account was handled by your agent Tom Nguyen last time, "
                    "if that helps.\n\nThanks,\nRachel"
                ),
                "ground_truth": [
                    {"text": "r.donovan.backup@gmail.com", "pii_type": "EMAIL"},
                    {"text": "Tom Nguyen",                  "pii_type": "NAME"},
                ],
            },
        },
        "ground_truth": [
            {"text": "Rachel Donovan",             "pii_type": "NAME"},
            {"text": "rachel.d@fastmail.com",       "pii_type": "EMAIL"},
            {"text": "+44-7911-123456",             "pii_type": "PHONE"},
            {"text": "r.donovan.backup@gmail.com",  "pii_type": "EMAIL"},
            {"text": "Tom Nguyen",                  "pii_type": "NAME"},
        ],
    },

    # ── Task 2 — Medium ─────────────────────────────────────────────────────
    "task_2_medium": {
        "task_name": "Employee Onboarding Form Audit",
        "difficulty": "medium",
        "description": (
            "You are a PII compliance auditor. Analyze the document section "
            "provided and identify ALL personally identifiable information. "
            "Look for: NAME, ADDRESS, DOB, SSN, PHONE, EMAIL. "
            "Note: some PII is embedded in sentences rather than labeled fields. "
            "Department codes and employee reference IDs are NOT PII - do not flag them. "
            "Submit your findings as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": ["NAME", "ADDRESS", "DOB", "SSN", "PHONE", "EMAIL"],
        "grader": "medium",
        "sections": {
            "section_a": {
                "title": "Primary Employee Record",
                "text": (
                    "EMPLOYEE ONBOARDING FORM - CONFIDENTIAL\n"
                    "Department Code: DEPT-7721 | Employee Ref: EMP-43210\n\n"
                    "Primary Employee\n"
                    "Full Name: Ananya Krishnan\n"
                    "Date of birth: 12 September 1991\n"
                    "Residential address: Koramangala 5th Block, Bangalore 560095\n"
                    "Mobile: +91-98765-43210\n"
                    "Work email provisioned: ananya.k@techventures.in\n"
                    "Government ID (SSN equivalent on file): 384-62-1947\n\n"
                    "Notes: Laptop to be shipped to the residential address on file."
                ),
                "ground_truth": [
                    {"text": "Ananya Krishnan",                         "pii_type": "NAME"},
                    {"text": "12 September 1991",                       "pii_type": "DOB"},
                    {"text": "Koramangala 5th Block, Bangalore 560095", "pii_type": "ADDRESS"},
                    {"text": "+91-98765-43210",                         "pii_type": "PHONE"},
                    {"text": "ananya.k@techventures.in",                "pii_type": "EMAIL"},
                    {"text": "384-62-1947",                             "pii_type": "SSN"},
                ],
            },
            "section_b": {
                "title": "Onboarding Buddy and HR Notes",
                "text": (
                    "Onboarding buddy assigned: David Osei (senior engineer).\n"
                    "David joined the company on 14/03/1988 - please coordinate schedules.\n"
                    "His contact for onboarding queries: d.osei.ref@consultco.com\n"
                    "David is reachable on his Ghana line at +233-20-756-1122 during IST hours.\n\n"
                    "HR rep: Linda - see internal ref TXN-43210."
                ),
                "ground_truth": [
                    {"text": "David Osei",                "pii_type": "NAME"},
                    {"text": "14/03/1988",                "pii_type": "DOB"},
                    {"text": "d.osei.ref@consultco.com",  "pii_type": "EMAIL"},
                    {"text": "+233-20-756-1122",           "pii_type": "PHONE"},
                ],
            },
        },
        "ground_truth": [
            {"text": "Ananya Krishnan",                         "pii_type": "NAME"},
            {"text": "12 September 1991",                       "pii_type": "DOB"},
            {"text": "Koramangala 5th Block, Bangalore 560095", "pii_type": "ADDRESS"},
            {"text": "+91-98765-43210",                         "pii_type": "PHONE"},
            {"text": "ananya.k@techventures.in",                "pii_type": "EMAIL"},
            {"text": "384-62-1947",                             "pii_type": "SSN"},
            {"text": "David Osei",                              "pii_type": "NAME"},
            {"text": "14/03/1988",                              "pii_type": "DOB"},
            {"text": "d.osei.ref@consultco.com",                "pii_type": "EMAIL"},
            {"text": "+233-20-756-1122",                        "pii_type": "PHONE"},
        ],
    },

    # ── Task 3 — Hard ───────────────────────────────────────────────────────
    "task_3_hard": {
        "task_name": "Multi-Party Compliance Forensic Audit",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic audit. "
            "Analyze the document section provided and identify ALL personally "
            "identifiable information. "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, "
            "SSN, ADDRESS, DOB, PASSWORD. "
            "CRITICAL RULES: "
            "(1) Extract text EXACTLY as it appears - do not reformat dates, "
            "normalize phones, remove spaces from card numbers, or change separators. "
            "(2) SSNs may use dots instead of dashes (e.g. 531.20.7732). "
            "(3) Credit card numbers may contain spaces - extract with spaces as shown. "
            "(4) Dates must be extracted exactly as written. "
            "(5) The dossier spans active records, supplemental history, and credential "
            "logs - audit all content in the section thoroughly. "
            "(6) Two primary account holders appear across the full dossier. "
            "Submit as a JSON list of objects with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD",
            "SSN", "ADDRESS", "DOB", "PASSWORD",
        ],
        "grader": "hard",
        # ── Section design ───────────────────────────────────────────────
        # A + C: active records — 8 real PII items each, no red herrings.
        #        Model gets intermediate section recall = 1.0 for both.
        #
        # B + D: supplemental — 1 real PII (PASSWORD) + 4 red herrings each.
        #        All technical noise (IPs, ref codes, version strings) REMOVED
        #        so FP count = exactly 8 (4 per section, predictable).
        #        Model gets intermediate section recall = 1.0 (it finds the password),
        #        but at finalize the 8 FPs trigger the penalty.
        #
        # Finalize math (18 correct, 8 FPs, 26 predicted):
        #   precision = 18/26 = 0.692
        #   recall    = 18/18 = 1.000
        #   F1        = 0.818
        #   penalty   = 8 x 0.065 = 0.520
        #   score     = 0.818 - 0.520 = 0.298  ~  0.30
        "sections": {
            "section_a": {
                "title": "Account A - Active Records",
                "text": (
                    "ACCOUNT A - EASTERN EUROPE SEGMENT\n\n"
                    "Account holder: Dmitri Volkov\n"
                    "Primary contact: d.volkov@securenet.ru\n"
                    "Backup contact on file: d.volkov.backup@protonmail.com\n"
                    "Mobile (RU): +7-916-234-5678\n"
                    "Date of birth: 1983-09-27\n"
                    "Current residential: ul. Tverskaya 14, kv. 7, Moscow 125009\n"
                    "Card on file [spaces preserved per vault format]: 5425 2334 3010 9903\n"
                    "Tax ID cross-reference (dot-format as received): 531.20.7732"
                ),
                "ground_truth": [
                    {"text": "Dmitri Volkov",                            "pii_type": "NAME"},
                    {"text": "d.volkov@securenet.ru",                   "pii_type": "EMAIL"},
                    {"text": "d.volkov.backup@protonmail.com",          "pii_type": "EMAIL"},
                    {"text": "+7-916-234-5678",                         "pii_type": "PHONE"},
                    {"text": "1983-09-27",                              "pii_type": "DOB"},
                    {"text": "ul. Tverskaya 14, kv. 7, Moscow 125009", "pii_type": "ADDRESS"},
                    {"text": "5425 2334 3010 9903",                     "pii_type": "CREDIT_CARD"},
                    {"text": "531.20.7732",                             "pii_type": "SSN"},
                ],
            },
            "section_b": {
                "title": "Account A - Supplemental Records and Credential Log",
                # Real PII  : V0lk0v#Secure_83! (PASSWORD)
                # Red herrings (PII-lookalikes, NOT in ground truth):
                #   Elena Volkova           -> NAME    (emergency contact, not account holder)
                #   +7-495-123-4567         -> PHONE   (Elena's contact number)
                #   ul. Leninskiy 32, ...   -> ADDRESS (archived former residence)
                #   5425 2334 0000 0001     -> CREDIT_CARD (voided card, superseded)
                # Technical noise REMOVED to prevent unpredictable extra FPs.
                "text": (
                    "Supplemental record entries:\n"
                    "  Emergency contact - Elena Volkova, +7-495-123-4567\n"
                    "  Previous residential (archived 2019): "
                    "ul. Leninskiy 32, kv. 4, Moscow 119146\n"
                    "  Card superseded (voided 2022): 5425 2334 0000 0001\n\n"
                    "Credential audit note - legacy plaintext capture:\n"
                    "  CONFIG_KEY=\"V0lk0v#Secure_83!\""
                ),
                "ground_truth": [
                    {"text": "V0lk0v#Secure_83!", "pii_type": "PASSWORD"},
                ],
            },
            "section_c": {
                "title": "Account B - Active Records",
                "text": (
                    "ACCOUNT B - APAC SEGMENT\n\n"
                    "The second account belongs to Aiko Shimizu, a corporate client\n"
                    "whose registered email is aiko.s@nantech.co.jp.\n"
                    "Office landline on record: +81-3-5412-8876\n"
                    "Personal mobile (local JP format): 080-3344-9921\n"
                    "KYC date of birth field: 1991-03-15\n"
                    "Registered billing address: "
                    "2-8-12 Minami-Aoyama, Minato-ku, Tokyo 107-0062\n"
                    "Card vault entry [spaces preserved]: 3566 0020 2006 0505\n"
                    "Government ID (dot-separated format): 412.75.9301"
                ),
                "ground_truth": [
                    {"text": "Aiko Shimizu",                                      "pii_type": "NAME"},
                    {"text": "aiko.s@nantech.co.jp",                             "pii_type": "EMAIL"},
                    {"text": "+81-3-5412-8876",                                  "pii_type": "PHONE"},
                    {"text": "080-3344-9921",                                    "pii_type": "PHONE"},
                    {"text": "1991-03-15",                                       "pii_type": "DOB"},
                    {"text": "2-8-12 Minami-Aoyama, Minato-ku, Tokyo 107-0062", "pii_type": "ADDRESS"},
                    {"text": "3566 0020 2006 0505",                              "pii_type": "CREDIT_CARD"},
                    {"text": "412.75.9301",                                      "pii_type": "SSN"},
                ],
            },
            "section_d": {
                "title": "Account B - Supplemental Records and System Notes",
                # Real PII  : Sh!mizu_2024# (PASSWORD)
                # Red herrings (PII-lookalikes, NOT in ground truth):
                #   Kenji Shimizu              -> NAME   (authorized delegate, not account holder)
                #   kenji.s@nantech.co.jp      -> EMAIL  (Kenji's email)
                #   5-10-3 Shibuya, ...        -> ADDRESS (superseded former billing address)
                #   +81-3-5412-0000            -> PHONE  (shared office switchboard, not personal)
                # Technical noise REMOVED to prevent unpredictable extra FPs.
                "text": (
                    "Supplemental record entries:\n"
                    "  Authorized delegate on account: Kenji Shimizu, "
                    "kenji.s@nantech.co.jp\n"
                    "  Former billing address (superseded 2023): "
                    "5-10-3 Shibuya, Shibuya-ku, Tokyo 150-0002\n"
                    "  Shared office switchboard (not personal): +81-3-5412-0000\n\n"
                    "System notes - credential recovery log:\n"
                    "  token_value = \"Sh!mizu_2024#\""
                ),
                "ground_truth": [
                    {"text": "Sh!mizu_2024#", "pii_type": "PASSWORD"},
                ],
            },
        },
        # 18 ground truth items total
        "ground_truth": [
            {"text": "Dmitri Volkov",                            "pii_type": "NAME"},
            {"text": "d.volkov@securenet.ru",                   "pii_type": "EMAIL"},
            {"text": "d.volkov.backup@protonmail.com",          "pii_type": "EMAIL"},
            {"text": "+7-916-234-5678",                         "pii_type": "PHONE"},
            {"text": "1983-09-27",                              "pii_type": "DOB"},
            {"text": "ul. Tverskaya 14, kv. 7, Moscow 125009", "pii_type": "ADDRESS"},
            {"text": "5425 2334 3010 9903",                     "pii_type": "CREDIT_CARD"},
            {"text": "531.20.7732",                             "pii_type": "SSN"},
            {"text": "V0lk0v#Secure_83!",                      "pii_type": "PASSWORD"},
            {"text": "Aiko Shimizu",                            "pii_type": "NAME"},
            {"text": "aiko.s@nantech.co.jp",                   "pii_type": "EMAIL"},
            {"text": "+81-3-5412-8876",                        "pii_type": "PHONE"},
            {"text": "080-3344-9921",                          "pii_type": "PHONE"},
            {"text": "1991-03-15",                             "pii_type": "DOB"},
            {"text": "2-8-12 Minami-Aoyama, Minato-ku, Tokyo 107-0062", "pii_type": "ADDRESS"},
            {"text": "3566 0020 2006 0505",                    "pii_type": "CREDIT_CARD"},
            {"text": "412.75.9301",                            "pii_type": "SSN"},
            {"text": "Sh!mizu_2024#",                          "pii_type": "PASSWORD"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Grading Logic
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip()


def _grade_easy(predicted: list, ground_truth: list) -> dict:
    """Recall-based. No FP penalty. Target final score ~1.00."""
    correct = 0
    matched: set = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and pred.get("pii_type", "").upper() == gt["pii_type"]):
                    correct += 1
                    matched.add(i)
                    break
    total     = len(ground_truth)
    recall    = correct / total if total else 0.0
    fp        = max(len(predicted) - correct, 0)
    precision = correct / len(predicted) if predicted else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "score":           round(recall, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "correct":         correct,
        "total":           total,
        "false_positives": fp,
        "partial_credit":  0.0,
        "feedback": f"Found {correct}/{total} PII items correctly. Score based on recall.",
    }


def _grade_medium(predicted: list, ground_truth: list) -> dict:
    """Partial credit (right text, wrong type = 0.5). FP penalty = 0.08 each.
    Target final score ~0.90."""
    full_credit = 0
    partial     = 0.0
    matched: set = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                pred_text  = _normalize(pred.get("text", ""))
                gt_text    = _normalize(gt["text"])
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
    total     = len(ground_truth)
    fp        = max(len(predicted) - len(matched), 0)
    raw       = (full_credit + partial) / total if total else 0.0
    penalty   = fp * 0.08
    score     = max(0.0, min(1.0, raw - penalty))
    precision = full_credit / len(predicted) if predicted else 0.0
    recall    = full_credit / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "score":           round(score, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "correct":         full_credit,
        "total":           total,
        "false_positives": fp,
        "partial_credit":  round(partial, 4),
        "feedback": (
            f"Full matches: {full_credit}, Partial (wrong type): {partial}, "
            f"FP penalty: -{round(penalty, 2)}. Final: {round(score, 4)}"
        ),
    }


def _grade_hard(predicted: list, ground_truth: list) -> dict:
    """Strict exact F1. FP penalty 0.065 each. Target final score ~0.30."""
    correct = 0
    matched: set = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and pred.get("pii_type", "").upper() == gt["pii_type"]):
                    correct += 1
                    matched.add(i)
                    break
    total     = len(ground_truth)
    fp        = max(len(predicted) - correct, 0)
    precision = correct / len(predicted) if predicted else 0.0
    recall    = correct / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    penalty   = fp * 0.065
    score     = max(0.0, min(1.0, f1 - penalty))
    return {
        "score":           round(score, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "correct":         correct,
        "total":           total,
        "false_positives": fp,
        "partial_credit":  0.0,
        "feedback": (
            f"Strict F1: {round(f1, 4)}, FP penalty: -{round(penalty, 3)}. "
            f"Final: {round(score, 4)}"
        ),
    }


GRADERS = {"easy": _grade_easy, "medium": _grade_medium, "hard": _grade_hard}


def _section_reward(predicted: list, ground_truth: list) -> float:
    """
    Simple per-section recall used as the intermediate RL reward signal.
    No FP penalty — gives the agent an honest signal of 'how much of this
    section did you find?' before the full grader applies penalties at finalize.
    """
    if not ground_truth:
        return 1.0
    correct = 0
    matched: set = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched:
                if (_normalize(pred.get("text", "")) == _normalize(gt["text"])
                        and pred.get("pii_type", "").upper() == gt["pii_type"]):
                    correct += 1
                    matched.add(i)
                    break
    return round(correct / len(ground_truth), 4)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PIIEnvironment(MCPEnvironment):
    """
    PII Compliance Auditor — OpenEnv Multi-Step Environment.

    Multi-step tools (recommended workflow):
        list_tasks()
        get_task(task_id)
        get_section(task_id, section_id)
        annotate_section(task_id, section_id, findings_json)
        finalize_task(task_id)

    Legacy tool (backward compatibility / automated validator):
        submit_findings(task_id, findings_json)

    State tool:
        get_current_state()
    """

    def __init__(self):
        mcp = FastMCP("pii_compliance_auditor")

        @mcp.tool
        def list_tasks() -> str:
            """List all available PII auditing tasks with section metadata."""
            return json.dumps([
                {
                    "task_id":            tid,
                    "task_name":          t["task_name"],
                    "difficulty":         t["difficulty"],
                    "pii_types_in_scope": t["pii_types_in_scope"],
                    "section_count":      len(t["sections"]),
                    "sections": [
                        {"id": sid, "title": s["title"]}
                        for sid, s in t["sections"].items()
                    ],
                }
                for tid, t in TASKS.items()
            ], indent=2)

        @mcp.tool
        def get_task(task_id: str) -> str:
            """
            Get task metadata and section list.
            Does NOT return document text — call get_section for each section.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
            """
            if task_id not in TASKS:
                return json.dumps({"error": f"Unknown task_id '{task_id}'."})
            task = TASKS[task_id]
            self._current_task_id = task_id
            self._state.step_count += 1
            return json.dumps({
                "task_id":            task_id,
                "task_name":          task["task_name"],
                "difficulty":         task["difficulty"],
                "description":        task["description"],
                "pii_types_in_scope": task["pii_types_in_scope"],
                "section_count":      len(task["sections"]),
                "sections": [
                    {"id": sid, "title": s["title"]}
                    for sid, s in task["sections"].items()
                ],
                "workflow": (
                    "For each section: call get_section(task_id, section_id), "
                    "analyse the text, then annotate_section(task_id, section_id, "
                    "findings_json). When all sections are annotated, call "
                    "finalize_task(task_id)."
                ),
            }, indent=2)

        @mcp.tool
        def get_section(task_id: str, section_id: str) -> str:
            """
            Retrieve a document section for analysis.

            Args:
                task_id:    One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
                section_id: Section ID returned by get_task (e.g. 'section_a')
            """
            if task_id not in TASKS:
                return json.dumps({"error": f"Unknown task_id '{task_id}'."})
            task = TASKS[task_id]
            if section_id not in task["sections"]:
                return json.dumps({"error": f"Unknown section_id '{section_id}'."})
            section  = task["sections"][section_id]
            all_sids = list(task["sections"].keys())
            self._state.step_count += 1
            return json.dumps({
                "task_id":            task_id,
                "section_id":         section_id,
                "title":              section["title"],
                "text":               section["text"],
                "pii_types_in_scope": task["pii_types_in_scope"],
                "section_number":     all_sids.index(section_id) + 1,
                "total_sections":     len(all_sids),
            }, indent=2)

        @mcp.tool
        def annotate_section(task_id: str, section_id: str,
                             findings_json: str) -> str:
            """
            Submit PII findings for one document section.
            Returns an intermediate reward (section recall) immediately.
            Calling again for the same section replaces the previous annotation.

            Args:
                task_id:       One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
                section_id:    Section ID (e.g. 'section_a')
                findings_json: JSON array of {text, pii_type} objects
            """
            if task_id not in TASKS:
                return json.dumps({"error": "Unknown task_id.", "section_score": 0.0})
            task = TASKS[task_id]
            if section_id not in task["sections"]:
                return json.dumps({"error": "Unknown section_id.", "section_score": 0.0})
            try:
                predicted = json.loads(findings_json)
                if not isinstance(predicted, list):
                    raise ValueError("Must be a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                return json.dumps({"error": str(e), "section_score": 0.0})

            section_gt = task["sections"][section_id]["ground_truth"]
            sec_score  = _section_reward(predicted, section_gt)

            if task_id not in self._section_annotations:
                self._section_annotations[task_id] = {}
            if task_id not in self._section_rewards:
                self._section_rewards[task_id] = {}

            prev = self._section_rewards[task_id].get(section_id)
            if prev is not None:
                self._cumulative_reward -= prev
            self._cumulative_reward += sec_score
            self._section_annotations[task_id][section_id] = predicted
            self._section_rewards[task_id][section_id]     = sec_score
            self._state.step_count += 1

            all_sids           = list(task["sections"].keys())
            annotated_count    = len(self._section_annotations[task_id])
            sections_remaining = len(all_sids) - annotated_count

            return json.dumps({
                "task_id":            task_id,
                "section_id":         section_id,
                "section_title":      task["sections"][section_id]["title"],
                "section_score":      sec_score,
                "sections_annotated": annotated_count,
                "sections_remaining": sections_remaining,
                "ready_to_finalize":  sections_remaining == 0,
                "feedback": (
                    f"Section recall: {sec_score:.4f}. "
                    + ("All sections done - call finalize_task() now."
                       if sections_remaining == 0
                       else f"{sections_remaining} section(s) remaining.")
                ),
            }, indent=2)

        @mcp.tool
        def finalize_task(task_id: str) -> str:
            """
            Finalise the task. Aggregates all section annotations and applies
            the full task grader (with FP penalties) to produce the final reward.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
            """
            if task_id not in TASKS:
                return json.dumps({"error": "Unknown task_id.", "score": 0.0})
            task = TASKS[task_id]

            all_predictions: list = []
            for sid in task["sections"]:
                all_predictions.extend(
                    self._section_annotations.get(task_id, {}).get(sid, [])
                )

            result             = GRADERS[task["grader"]](all_predictions, task["ground_truth"])
            self._submissions[task_id] = result["score"]
            done               = len(self._submissions) >= len(TASKS)
            self._done         = done
            self._state.step_count += 1

            return json.dumps({
                "task_id":         task_id,
                "task_name":       task["task_name"],
                "difficulty":      task["difficulty"],
                **result,
                "done":            done,
                "tasks_completed": len(self._submissions),
                "tasks_total":     len(TASKS),
                "section_scores":  self._section_rewards.get(task_id, {}),
            }, indent=2)

        @mcp.tool
        def submit_findings(task_id: str, findings_json: str) -> str:
            """
            Legacy single-shot submission (backward compatibility).
            Grades all findings directly without section structure.
            Retained for automated validator compatibility.

            Args:
                task_id:       One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
                findings_json: JSON array of {text, pii_type} objects
            """
            if task_id not in TASKS:
                return json.dumps({"error": "Unknown task_id.", "score": 0.0})
            try:
                predicted = json.loads(findings_json)
                if not isinstance(predicted, list):
                    raise ValueError("Must be a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                return json.dumps({"error": str(e), "score": 0.0})

            task   = TASKS[task_id]
            result = GRADERS[task["grader"]](predicted, task["ground_truth"])
            self._cumulative_reward       += result["score"]
            self._submissions[task_id]     = result["score"]
            self._state.step_count        += 1
            done       = len(self._submissions) >= len(TASKS)
            self._done = done

            return json.dumps({
                "task_id":         task_id,
                "task_name":       task["task_name"],
                "difficulty":      task["difficulty"],
                **result,
                "done":            done,
                "tasks_completed": len(self._submissions),
                "tasks_total":     len(TASKS),
            }, indent=2)

        @mcp.tool
        def get_current_state() -> str:
            """Get full episode state including per-section annotation progress."""
            section_progress = {}
            for tid, task in TASKS.items():
                section_progress[tid] = {
                    sid: {
                        "annotated": sid in self._section_annotations.get(tid, {}),
                        "score":     self._section_rewards.get(tid, {}).get(sid),
                    }
                    for sid in task["sections"]
                }
            return json.dumps({
                "episode_id":        self._state.episode_id,
                "step_count":        self._state.step_count,
                "submissions":       self._submissions,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "done":              self._done,
                "tasks_available":   list(TASKS.keys()),
                "section_progress":  section_progress,
            }, indent=2)

        super().__init__(mcp)
        self._state                     = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward: float  = 0.0
        self._submissions: dict         = {}
        self._done: bool                = False
        self._current_task_id: Optional[str] = None
        self._section_annotations: dict = {}
        self._section_rewards: dict     = {}

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state               = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._cumulative_reward   = 0.0
        self._submissions         = {}
        self._done                = False
        self._current_task_id     = None
        self._section_annotations = {}
        self._section_rewards     = {}
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status":  "ready",
                "message": (
                    "PII Compliance Auditor ready. "
                    "Multi-step workflow: list_tasks() -> get_task(task_id) -> "
                    "for each section: get_section() -> annotate_section() -> "
                    "finalize_task()"
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