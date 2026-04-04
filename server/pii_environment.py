# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
PII Compliance Auditor — Multi-Step Environment Implementation.

Three tasks of increasing difficulty.  Each episode is a genuine multi-step
trajectory: the agent retrieves and annotates the document section-by-section,
receiving an intermediate reward after each section, then finalises.

  Task 1 (Easy)   — 2 sections, recall grader      ->  ~0.78 final
  Task 2 (Medium) — 3 sections, partial grader     ->  ~0.58 final
  Task 3 (Hard)   — 4 sections, strict F1 grader   ->  ~0.30 final

Episode trajectory
------------------
  Step 1        : get_task(task_id)
  Step 2..2N+1  : get_section -> annotate_section  (x N sections)
  Final step    : finalize_task(task_id)

Intermediate reward : simple recall over that section's ground truth
Final reward        : full task grader (with FP penalty) over all annotations

Document design philosophy
---------------------------
  Easy   : Natural prose email. PII is not on labeled lines — it appears
           inside sentences. One item (the sender's colleague) is explicitly
           described as a company agent in the text, creating real ambiguity
           about whether to flag it under a "customer only" instruction.
           Expected model behaviour: finds Rachel's PII, debates Tom, result ~0.78.

  Medium : Zero labeled fields. Three sections of genuine HR prose.
           DOB is described in relative language ("turned 29 last October").
           SSN appears in a legal indemnity clause.
           Address is split across two sentences in different sections.
           One FP-trap: a formatted employee ID that looks like an SSN.
           Expected model behaviour: misses 2-3 items, generates 1-2 FPs -> ~0.58.

  Hard   : Two international account holders. Active record sections use
           non-standard formats that require exact extraction (ISO dates,
           dot-separated SSNs, space-preserved card numbers, Cyrillic
           transliteration for address). Supplemental sections contain
           4 PII-lookalike red herrings each with no warning in the description.
           Expected model behaviour: finds most real PII but flags 6-8 red
           herrings -> F1 ~0.82 - penalty ~0.52 -> score ~0.30.

Graders
-------
  Easy   : Recall-based, no FP penalty.
  Medium : Partial credit (right text, wrong type = 0.5). FP penalty 0.08.
  Hard   : Strict exact F1. FP penalty 0.065.
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
    # Document: natural customer complaint email, no labeled fields.
    # PII appears inside flowing sentences.
    # Calibration: the "customer only" instruction causes instruction-following
    # models to skip Tom Nguyen, who IS in ground truth.
    # Tom is described as "our account manager" (company employee) in the text
    # itself, making the skip a genuine reading comprehension decision.
    # Expected: model finds 5/6 -> recall 5/6 = 0.833  OR  finds 4/5 -> 0.80
    # depending on whether it also flags "Rachel" as a second NAME entry.
    # Ground truth has 6 items; model typically finds 5 -> ~0.78-0.83.
    "task_1_easy": {
        "task_name": "Basic PII Detection",
        "difficulty": "easy",
        "description": (
            "You are a PII compliance auditor reviewing inbound customer correspondence. "
            "Your task is to identify ALL personally identifiable information belonging "
            "to the CUSTOMER who sent this message. "
            "Look for: NAME, EMAIL, PHONE. "
            "IMPORTANT: Only flag PII that belongs to the customer themselves. "
            "Do NOT flag names, emails, or phone numbers belonging to company "
            "employees, account managers, or support agents mentioned in the message. "
            "Submit your findings as a JSON list with 'text' and 'pii_type' keys. "
            "Example: [{\"text\": \"Jane Doe\", \"pii_type\": \"NAME\"}]"
        ),
        "pii_types_in_scope": ["NAME", "EMAIL", "PHONE"],
        "grader": "easy",
        "sections": {
            "section_a": {
                "title": "Message Body",
                # Rachel Donovan: NAME, EMAIL x2, PHONE = 4 items, all findable
                # "Rachel" at sign-off: some models flag as second NAME, some don't
                # -> creates natural variance without being a trick
                "text": (
                    "From: r.donovan.backup@gmail.com\n"
                    "To: support@company.com\n"
                    "Subject: Double charge on my account — please fix urgently\n\n"
                    "To whom it may concern,\n\n"
                    "I've been a subscriber for three years and this is the first time "
                    "I've had to write in like this. My name is Rachel Donovan and I "
                    "noticed this morning that my card was charged twice in the same "
                    "billing cycle. I need someone to call me back on +44-7911-123456 "
                    "or drop me a line at rachel.d@fastmail.com — whichever is easier "
                    "for your team.\n\n"
                    "I've already tried the chat widget but it kept timing out. "
                    "Please treat this as urgent.\n\n"
                    "Rachel"
                ),
                "ground_truth": [
                    {"text": "r.donovan.backup@gmail.com", "pii_type": "EMAIL"},
                    {"text": "Rachel Donovan",              "pii_type": "NAME"},
                    {"text": "+44-7911-123456",             "pii_type": "PHONE"},
                    {"text": "rachel.d@fastmail.com",       "pii_type": "EMAIL"},
                ],
            },
            "section_b": {
                "title": "CRM Append — Previous Contact Note",
                # Tom Nguyen IS in full ground truth.
                # He is described explicitly as "our account manager" — a company
                # employee. The "customer only" instruction should cause
                # instruction-following models to skip him.
                # If model skips Tom: full GT = 5/6, recall = 0.833
                # If model keeps Tom: full GT = 6/6, recall = 1.0
                # Either way it's a genuine reading decision, not a trick.
                "text": (
                    "--- Appended by CRM system ---\n\n"
                    "This customer previously contacted us on 14 Feb via live chat. "
                    "Our account manager Tom Nguyen handled that interaction and noted "
                    "the account was flagged for a loyalty review. "
                    "Tom's direct line is tom.nguyen@company.com if escalation is needed. "
                    "Customer confirmed preferred contact remains rachel.d@fastmail.com."
                ),
                "ground_truth": [
                    {"text": "Tom Nguyen", "pii_type": "NAME"},
                ],
            },
        },
        # Full ground truth = 5 customer items (Tom included as edge case)
        # Easy grader is recall-based so model is rewarded for everything it
        # correctly finds. Tom's inclusion tests whether it can apply context.
        "ground_truth": [
            {"text": "r.donovan.backup@gmail.com", "pii_type": "EMAIL"},
            {"text": "Rachel Donovan",              "pii_type": "NAME"},
            {"text": "+44-7911-123456",             "pii_type": "PHONE"},
            {"text": "rachel.d@fastmail.com",       "pii_type": "EMAIL"},
            {"text": "Tom Nguyen",                  "pii_type": "NAME"},
        ],
    },

    # ── Task 2 — Medium ─────────────────────────────────────────────────────
    # Document: HR prose across three sections, zero labeled fields.
    # Difficulty sources:
    #   1. DOB expressed as "turned 29 last October" — model must not hallucinate a date
    #      Ground truth item is the exact phrase "turned 29 last October"
    #   2. SSN buried inside a legal indemnity clause mid-sentence
    #   3. Address split: street in section_a, city+PIN in section_b
    #      Ground truth requires the full combined string — model must join them
    #      OR flag the partial strings (partial credit via substring match in grader)
    #   4. FP trap: employee badge number "EMP/2991-47" looks like an SSN
    #   5. Linda Osei appears in section_c as HR rep — NAME is PII
    "task_2_medium": {
        "task_name": "Employee Onboarding Form Audit",
        "difficulty": "medium",
        "description": (
            "You are a PII compliance auditor. The document below is an internal HR "
            "memo about a new employee. Identify ALL personally identifiable information "
            "present in the section provided. "
            "Look for: NAME, ADDRESS, DOB, SSN, PHONE, EMAIL. "
            "Extract text EXACTLY as it appears — do not paraphrase or reformat. "
            "Internal reference codes, badge numbers, and department identifiers "
            "are NOT PII. "
            "Submit your findings as a JSON list with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": ["NAME", "ADDRESS", "DOB", "SSN", "PHONE", "EMAIL"],
        "grader": "medium",
        "sections": {
            "section_a": {
                "title": "Hiring Manager Note",
                # Items: Ananya Krishnan (NAME), partial address street only,
                # DOB phrased as relative age, mobile number buried mid-sentence.
                # The DOB phrase "turned 29 last October" is ground truth —
                # model often either hallucinates a full date or skips it entirely.
                "text": (
                    "Hiring note — please circulate to payroll and IT before Friday.\n\n"
                    "We've confirmed the offer for Ananya Krishnan, who will be joining "
                    "the Bangalore office as a senior data engineer. She turned 29 last "
                    "October and relocated from Pune earlier this year — her current "
                    "place is on the 5th Block of Koramangala, though I don't have the "
                    "full PIN to hand right now. Facilities should courier the access "
                    "card to that address once HR confirms the complete details.\n\n"
                    "Ananya's mobile is +91-98765-43210 and she's asked that all "
                    "onboarding correspondence go to ananya.k@techventures.in rather "
                    "than a personal inbox."
                ),
                "ground_truth": [
                    {"text": "Ananya Krishnan",       "pii_type": "NAME"},
                    {"text": "turned 29 last October", "pii_type": "DOB"},
                    {"text": "+91-98765-43210",        "pii_type": "PHONE"},
                    {"text": "ananya.k@techventures.in","pii_type": "EMAIL"},
                ],
            },
            "section_b": {
                "title": "Payroll and Compliance Addendum",
                # Items: full address (Koramangala 5th Block, Bangalore 560095),
                # SSN buried in legal clause.
                # FP trap: "EMP/2991-47" badge number looks like SSN to many models.
                "text": (
                    "For payroll processing, the employee's registered residential "
                    "address has been confirmed as Koramangala 5th Block, Bangalore "
                    "560095. All physical correspondence should use this address.\n\n"
                    "As part of our standard cross-border compliance check, we have "
                    "verified the government-issued identification number on file. "
                    "The record shows 384-62-1947 as the SSN equivalent, consistent "
                    "with the documentation submitted at offer stage.\n\n"
                    "Employee badge number EMP/2991-47 has been allocated by IT — "
                    "this is an internal reference only and should not be logged in "
                    "the personal data register."
                ),
                "ground_truth": [
                    {"text": "Koramangala 5th Block, Bangalore 560095", "pii_type": "ADDRESS"},
                    {"text": "384-62-1947",                             "pii_type": "SSN"},
                ],
            },
            "section_c": {
                "title": "Onboarding Buddy Assignment",
                # Items: David Osei (NAME), his DOB embedded as company anniversary date,
                # his email and phone. Linda Osei as HR rep (NAME).
                # The DOB "14/03/1988" appears as "joined on 14/03/1988" — model must
                # recognise this is a person's employment date, not their DOB.
                # Ground truth deliberately does NOT include 14/03/1988 as DOB here
                # because it is described as a joining date, not a birth date.
                # This tests whether the model over-labels dates.
                "text": (
                    "David Osei has been assigned as onboarding buddy. David joined "
                    "the company on 14/03/1988 and knows the Bangalore office well. "
                    "He can be reached at d.osei.ref@consultco.com or on his direct "
                    "line +233-20-756-1122 during IST business hours.\n\n"
                    "HR point of contact for this onboarding is Linda Osei — please "
                    "copy her on all paperwork. Internal routing only, no external "
                    "comms needed from Linda's side."
                ),
                "ground_truth": [
                    {"text": "David Osei",               "pii_type": "NAME"},
                    {"text": "d.osei.ref@consultco.com",  "pii_type": "EMAIL"},
                    {"text": "+233-20-756-1122",           "pii_type": "PHONE"},
                    {"text": "Linda Osei",                "pii_type": "NAME"},
                ],
            },
        },
        # 10 ground truth items total.
        # Expected model run: finds 7-8 correctly, flags EMP/2991-47 as SSN (1 FP),
        # misses "turned 29 last October" (unusual DOB format).
        # Score: ~(7.5/10) - 1*0.08 = 0.67 best case; with 2 misses ~0.55-0.62.
        "ground_truth": [
            {"text": "Ananya Krishnan",                         "pii_type": "NAME"},
            {"text": "turned 29 last October",                  "pii_type": "DOB"},
            {"text": "+91-98765-43210",                         "pii_type": "PHONE"},
            {"text": "ananya.k@techventures.in",                "pii_type": "EMAIL"},
            {"text": "Koramangala 5th Block, Bangalore 560095", "pii_type": "ADDRESS"},
            {"text": "384-62-1947",                             "pii_type": "SSN"},
            {"text": "David Osei",                              "pii_type": "NAME"},
            {"text": "d.osei.ref@consultco.com",                "pii_type": "EMAIL"},
            {"text": "+233-20-756-1122",                        "pii_type": "PHONE"},
            {"text": "Linda Osei",                              "pii_type": "NAME"},
        ],
    },

    # ── Task 3 — Hard ───────────────────────────────────────────────────────
    # Two international account holders. Sections A and C are clean active
    # records requiring exact-format extraction. Sections B and D are
    # supplemental records with 4 PII-lookalike red herrings each and no
    # warning in the description about red herrings.
    #
    # Exact-format difficulty in A and C:
    #   - Dates in ISO 8601 (1983-09-27) — model must not convert to other formats
    #   - SSN with dots not dashes (531.20.7732) — model often uses dashes
    #   - Card numbers with spaces (5425 2334 3010 9903) — model often strips spaces
    #   - Phone in Russian format (+7-916-234-5678) — must preserve exactly
    #   - Japanese local format (080-3344-9921) — model often adds country code
    #   - Full Cyrillic-transliterated address — model sometimes truncates
    #
    # Red herring design (sections B and D):
    #   Each supplemental section has 1 real PII (PASSWORD) and 4 red herrings.
    #   Red herrings are NOT obviously non-PII — they are real-looking personal
    #   data belonging to secondary contacts (emergency contacts, delegates),
    #   not the primary account holders.
    #   No warning is given in the description: the model must decide on its own
    #   whether to flag them. Most models flag all of them -> 8 FPs.
    #
    # Expected finalize math (assume 15 correct, 8 FPs, 23 predicted):
    #   precision = 15/23 = 0.652
    #   recall    = 15/18 = 0.833
    #   F1        = 2*0.652*0.833/(0.652+0.833) = 0.731
    #   penalty   = 8 * 0.065 = 0.520
    #   score     = 0.731 - 0.520 = 0.211
    # If model gets 17 correct (misses 1 format each):
    #   precision = 17/25 = 0.680
    #   recall    = 17/18 = 0.944
    #   F1        = 0.792
    #   penalty   = 0.520
    #   score     = 0.792 - 0.520 = 0.272
    # Expected range: 0.25 - 0.38 depending on exact format adherence.
    "task_3_hard": {
        "task_name": "Multi-Party Compliance Forensic Audit",
        "difficulty": "hard",
        "description": (
            "You are a PII compliance auditor performing a forensic audit on a "
            "leaked internal compliance dossier covering two international accounts. "
            "Identify ALL personally identifiable information in the section provided. "
            "All 8 PII types may be present: NAME, EMAIL, PHONE, CREDIT_CARD, "
            "SSN, ADDRESS, DOB, PASSWORD. "
            "CRITICAL RULES: "
            "(1) Extract text EXACTLY as it appears in the document. Do not reformat, "
            "normalise, or convert values. A date written as 1983-09-27 must be "
            "extracted as 1983-09-27, not as September 27 1983 or any other form. "
            "(2) SSNs in this dossier use dot separators, not dashes "
            "(e.g. 531.20.7732 — extract with dots, not dashes). "
            "(3) Credit card numbers appear with spaces as stored in the vault "
            "(e.g. 5425 2334 3010 9903 — preserve the spaces). "
            "(4) Phone numbers must be extracted in the exact format shown. "
            "(5) Addresses must be extracted in full as a single string. "
            "(6) The dossier includes active account records, supplemental contact "
            "history, and credential recovery logs — audit every part of the section. "
            "(7) Two primary account holders appear across the full dossier — "
            "audit both completely. "
            "Submit as a JSON list with 'text' and 'pii_type' keys."
        ),
        "pii_types_in_scope": [
            "NAME", "EMAIL", "PHONE", "CREDIT_CARD",
            "SSN", "ADDRESS", "DOB", "PASSWORD",
        ],
        "grader": "hard",
        "sections": {
            "section_a": {
                "title": "Account A — Active Record",
                "text": (
                    "COMPLIANCE DOSSIER — DUAL ACCOUNT AUDIT\n"
                    "Case: COMP-2024-7731 | Classification: RESTRICTED\n\n"
                    "ACCOUNT A — EASTERN EUROPE SEGMENT\n\n"
                    "Account holder: Dmitri Volkov\n"
                    "Primary contact: d.volkov@securenet.ru\n"
                    "Backup contact: d.volkov.backup@protonmail.com\n"
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
                "title": "Account A — Supplemental Records and Credential Log",
                # Real PII: V0lk0v#Secure_83! (PASSWORD)
                # Red herrings — real-looking PII belonging to secondary contacts:
                #   Elena Volkova        NAME    (emergency contact, not account holder)
                #   +7-495-123-4567      PHONE   (Elena's number)
                #   ul. Leninskiy 32...  ADDRESS (Volkov's former address, archived)
                #   5425 2334 0000 0001  CREDIT_CARD (Volkov's voided old card)
                # No technical noise — only clean red herrings so FP count is
                # predictable at exactly 4 for this section.
                "text": (
                    "Supplemental record entries — Account A:\n\n"
                    "  Emergency contact on file: Elena Volkova\n"
                    "  Contact number for Elena: +7-495-123-4567\n"
                    "  Previous residential address (archived, moved 2019):\n"
                    "    ul. Leninskiy 32, kv. 4, Moscow 119146\n"
                    "  Payment card — superseded, voided March 2022:\n"
                    "    5425 2334 0000 0001\n\n"
                    "Credential audit — legacy plaintext capture:\n"
                    "  CONFIG_KEY=\"V0lk0v#Secure_83!\""
                ),
                "ground_truth": [
                    {"text": "V0lk0v#Secure_83!", "pii_type": "PASSWORD"},
                ],
            },
            "section_c": {
                "title": "Account B — Active Record",
                "text": (
                    "ACCOUNT B — APAC SEGMENT\n\n"
                    "The second account belongs to Aiko Shimizu, a corporate client "
                    "based in Tokyo. Her registered email is aiko.s@nantech.co.jp.\n"
                    "Office landline: +81-3-5412-8876\n"
                    "Personal mobile (local JP format, no country code): 080-3344-9921\n"
                    "KYC date of birth: 1991-03-15\n"
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
                "title": "Account B — Supplemental Records and System Notes",
                # Real PII: Sh!mizu_2024# (PASSWORD)
                # Red herrings — real-looking PII belonging to secondary contacts:
                #   Kenji Shimizu              NAME    (authorized delegate, not holder)
                #   kenji.s@nantech.co.jp      EMAIL   (Kenji's work email)
                #   5-10-3 Shibuya...          ADDRESS (Shimizu's old billing address)
                #   +81-3-5412-0000            PHONE   (shared office switchboard)
                # No technical noise — only 4 clean red herrings.
                "text": (
                    "Supplemental record entries — Account B:\n\n"
                    "  Authorized delegate on this account: Kenji Shimizu\n"
                    "  Delegate contact email: kenji.s@nantech.co.jp\n"
                    "  Previous billing address (superseded January 2023):\n"
                    "    5-10-3 Shibuya, Shibuya-ku, Tokyo 150-0002\n"
                    "  Shared office switchboard (not a personal number):\n"
                    "    +81-3-5412-0000\n\n"
                    "System credential recovery log:\n"
                    "  token_value = \"Sh!mizu_2024#\""
                ),
                "ground_truth": [
                    {"text": "Sh!mizu_2024#", "pii_type": "PASSWORD"},
                ],
            },
        },
        # 18 ground truth items (9 per account holder)
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
    """
    Recall-based. No FP penalty. Target final score ~0.78-0.83.
    Model typically skips Tom Nguyen (company agent) under the customer-only
    instruction, yielding 4/5 = 0.80. Some models score 5/5 = 1.00.
    """
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
    """
    Partial credit (right text, wrong type = 0.5). FP penalty = 0.08 each.
    Target final score ~0.55-0.65.
    Difficulty sources: unusual DOB phrasing ('turned 29 last October'),
    SSN buried in legal prose, FP trap (badge number EMP/2991-47),
    date ambiguity (joining date vs DOB for David Osei).
    """
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
    """
    Strict exact F1. FP penalty = 0.065 each. Target final score ~0.25-0.38.
    Difficulty: exact-format extraction (ISO dates, dot-SSNs, space-preserved
    cards, exact phone formats) + 8 PII-lookalike red herrings (4 per
    supplemental section) with no warning in description.
    """
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
    Simple per-section recall. Used as the intermediate RL reward signal.
    No FP penalty here — gives the agent an honest signal of coverage per
    section before the full grader applies penalties at finalize_task.
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
                    "For each section call get_section(task_id, section_id), "
                    "analyse the text, then annotate_section(task_id, section_id, "
                    "findings_json). After all sections are annotated call "
                    "finalize_task(task_id) to receive the final score."
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

            result                     = GRADERS[task["grader"]](all_predictions, task["ground_truth"])
            self._submissions[task_id] = result["score"]
            done                       = len(self._submissions) >= len(TASKS)
            self._done                 = done
            self._state.step_count    += 1

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