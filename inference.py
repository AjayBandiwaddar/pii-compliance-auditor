"""
Inference Script — PII Compliance Auditor
==========================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (exactly):
    [START] task=<task_name> env=pii_compliance_auditor model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys

from openai import OpenAI

from client import PIIEnv

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS    = 10
ENV_NAME     = "pii_compliance_auditor"

TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── LLM Helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert PII (Personally Identifiable Information) compliance auditor.

Your job is to analyze documents and identify ALL personally identifiable information.

You MUST use ONLY these exact pii_type values — no variations allowed:
- NAME
- EMAIL
- PHONE
- CREDIT_CARD
- SSN
- ADDRESS
- DOB
- PASSWORD

CRITICAL RULES:
1. Only use the exact pii_type values listed above. Never use DATE_OF_BIRTH, CARD_NUMBER, or any other variation.
2. Extract the EXACT text as it appears in the document.
3. Do NOT paraphrase or modify the text.
4. Return ONLY a valid JSON array — no markdown, no explanation.

Output format:
[
  {"text": "<exact text from document>", "pii_type": "<TYPE>"},
  ...
]"""


def call_llm(document: str, description: str, pii_types: list) -> str:
    """Call the LLM and return its raw response."""
    user_msg = (
        f"Task instructions:\n{description}\n\n"
        f"PII types to detect: {', '.join(pii_types)}\n\n"
        f"Document to analyze:\n{document}\n\n"
        "Return ONLY a JSON array of detected PII items."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_llm_output(raw: str) -> list:
    """Parse LLM output to list of {text, pii_type} dicts."""
    # Strip markdown fences if present
    if "```" in raw:
        lines = raw.split("\n")
        raw = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


# ── Main Inference Loop ────────────────────────────────────────────────────

def run_task(env: PIIEnv, task_id: str) -> dict:
    """Run one full task episode. Returns result dict."""
    step_num   = 0
    rewards    = []
    success    = False
    error_msg  = None
    task_name  = task_id  # fallback

    try:
        # ── Get task ──────────────────────────────────────────────────────
        step_num += 1
        raw_task = env.call_tool("get_task", task_id=task_id)
        task_data = json.loads(raw_task) if isinstance(raw_task, str) else raw_task

        task_name   = task_data.get("task_name", task_id)
        document    = task_data.get("document", "")
        description = task_data.get("description", "")
        pii_types   = task_data.get("pii_types_in_scope", [])

        print(
            f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}",
            flush=True,
        )
        print(
            f"[STEP] step={step_num} action=get_task({task_id!r}) "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        # ── Call LLM ──────────────────────────────────────────────────────
        step_num += 1
        llm_raw     = call_llm(document, description, pii_types)
        findings    = parse_llm_output(llm_raw)
        findings_json = json.dumps(findings)

        print(
            f"[STEP] step={step_num} action=llm_inference "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        # ── Submit findings ───────────────────────────────────────────────
        step_num += 1
        raw_result = env.call_tool(
            "submit_findings",
            task_id=task_id,
            findings_json=findings_json,
        )
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result

        score   = float(result.get("score", 0.0))
        done    = result.get("done", False)
        rewards.append(score)
        success = score >= 0.5
        error_msg = result.get("error", None)

        done_str = "true" if done else "false"
        error_str = error_msg if error_msg else "null"

        print(
            f"[STEP] step={step_num} action=submit_findings({task_id!r}) "
            f"reward={score:.2f} done={done_str} error={error_str}",
            flush=True,
        )

    except Exception as e:
        error_msg = str(e)
        rewards.append(0.0)
        print(
            f"[STEP] step={step_num + 1} action=error "
            f"reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={step_num} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": rewards[-1] if rewards else 0.0,
        "steps": step_num,
        "success": success,
    }


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"PII Compliance Auditor — Inference", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"Server: {ENV_URL}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_results = []

    with PIIEnv(base_url=ENV_URL).sync() as env:
        env.reset()

        for task_id in TASK_IDS:
            print(f"\n--- Running {task_id} ---", flush=True)
            result = run_task(env, task_id)
            all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    total = 0.0
    for r in all_results:
        print(
            f"  {r['task_name']:<45} {r['score']:.4f}",
            flush=True,
        )
        total += r["score"]
    avg = total / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()