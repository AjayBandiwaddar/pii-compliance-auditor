"""
Inference Script - PII Compliance Auditor (Multi-Step)
=======================================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Multi-step episode trajectory per task:
    get_task  ->  (get_section -> llm_inference -> annotate_section) x N
              ->  finalize_task

STDOUT FORMAT (exactly):
    [START] task=<task_name> env=pii_compliance_auditor model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Notes:
    - Reward is 0.00 for get_task and get_section steps (no score yet).
    - Reward is the section recall for each annotate_section step.
    - Reward is the final task score for finalize_task.
    - [END] rewards lists every non-zero reward in order.
"""

import json
import os

from openai import OpenAI
from client import PIIEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME     = "pii_compliance_auditor"
TASK_IDS     = ["task_1_easy", "task_2_medium", "task_3_hard"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert PII (Personally Identifiable Information) compliance auditor.

Your job is to analyze a document section and identify ALL personally identifiable information present.

You MUST use ONLY these exact pii_type values:
- NAME
- EMAIL
- PHONE
- CREDIT_CARD
- SSN
- ADDRESS
- DOB
- PASSWORD

CRITICAL RULES:
1. Use only the exact pii_type values listed above. Never use DATE_OF_BIRTH, CARD_NUMBER, or variants.
2. Extract the EXACT text as it appears in the document. Do NOT paraphrase or reformat.
3. For credit cards: preserve spaces (e.g. "5425 2334 3010 9903").
4. For SSNs: preserve the separator used (dots or dashes) exactly as shown.
5. For dates: extract exactly as written, never convert to another format.
6. Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Output format:
[
  {"text": "<exact text from document>", "pii_type": "<TYPE>"},
  ...
]

If no PII is found in the section, return an empty array: []"""


def call_llm(section_text: str, description: str, pii_types: list) -> str:
    """Send one document section to the LLM and return raw response."""
    user_msg = (
        f"Task instructions:\n{description}\n\n"
        f"PII types to detect: {', '.join(pii_types)}\n\n"
        f"Document section to analyze:\n{section_text}\n\n"
        "Return ONLY a JSON array of detected PII items."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_llm_output(raw: str) -> list:
    """Parse LLM output to list of {text, pii_type} dicts."""
    if "```" in raw:
        lines = raw.split("\n")
        raw   = "\n".join(l for l in lines if not l.strip().startswith("```"))
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []

# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_task(env: PIIEnv, task_id: str) -> dict:
    """
    Run one full multi-step task episode.
    Returns a result dict with final score and step count.
    """
    step_num      = 0
    all_rewards   = []   # rewards from annotate_section and finalize_task
    task_name     = task_id
    final_score   = 0.0
    error_msg     = None

    try:
        # ── Step 1: get task metadata ──────────────────────────────────────
        step_num += 1
        raw_task  = env.call_tool("get_task", task_id=task_id)
        task_data = json.loads(raw_task) if isinstance(raw_task, str) else raw_task

        task_name   = task_data.get("task_name", task_id)
        sections    = task_data.get("sections", [])   # [{id, title}, ...]
        description = task_data.get("description", "")
        pii_types   = task_data.get("pii_types_in_scope", [])

        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        print(
            f"[STEP] step={step_num} action=get_task('{task_id}') "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        # ── For each section: get -> LLM -> annotate ───────────────────────
        for section_info in sections:
            section_id = section_info["id"]

            # get_section
            step_num += 1
            raw_sec   = env.call_tool("get_section", task_id=task_id, section_id=section_id)
            sec_data  = json.loads(raw_sec) if isinstance(raw_sec, str) else raw_sec
            sec_text  = sec_data.get("text", "")
            sec_title = sec_data.get("title", section_id)
            print(
                f"[STEP] step={step_num} action=get_section('{section_id}') "
                f"reward=0.00 done=false error=null",
                flush=True,
            )

            # llm_inference for this section
            step_num += 1
            llm_raw      = call_llm(sec_text, description, pii_types)
            findings     = parse_llm_output(llm_raw)
            findings_json = json.dumps(findings)
            print(
                f"[STEP] step={step_num} action=llm_inference('{section_id}') "
                f"reward=0.00 done=false error=null",
                flush=True,
            )

            # annotate_section
            step_num   += 1
            raw_annot   = env.call_tool(
                "annotate_section",
                task_id=task_id,
                section_id=section_id,
                findings_json=findings_json,
            )
            annot_result = json.loads(raw_annot) if isinstance(raw_annot, str) else raw_annot
            sec_score    = float(annot_result.get("section_score", 0.0))
            all_rewards.append(sec_score)
            print(
                f"[STEP] step={step_num} action=annotate_section('{section_id}') "
                f"reward={sec_score:.2f} done=false error=null",
                flush=True,
            )

        # ── Final step: finalize_task ─────────────────────────────────────
        step_num  += 1
        raw_final  = env.call_tool("finalize_task", task_id=task_id)
        final      = json.loads(raw_final) if isinstance(raw_final, str) else raw_final
        final_score = float(final.get("score", 0.0))
        done        = final.get("done", False)
        all_rewards.append(final_score)
        done_str    = "true" if done else "false"
        print(
            f"[STEP] step={step_num} action=finalize_task('{task_id}') "
            f"reward={final_score:.2f} done={done_str} error=null",
            flush=True,
        )

    except Exception as exc:
        error_msg = str(exc)
        all_rewards.append(0.0)
        print(
            f"[STEP] step={step_num + 1} action=error "
            f"reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    success_str  = "true" if final_score >= 0.5 else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in all_rewards)
    print(
        f"[END] success={success_str} steps={step_num} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "score":     final_score,
        "steps":     step_num,
        "success":   final_score >= 0.5,
    }


def main() -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"PII Compliance Auditor - Inference (Multi-Step)", flush=True)
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

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    total = 0.0
    for r in all_results:
        print(f"  {r['task_name']:<45} {r['score']:.4f}", flush=True)
        total += r["score"]
    avg = total / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()