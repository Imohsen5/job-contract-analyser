# utils/agent.py
import json
import time
from openai import OpenAI
from typing import Any, Dict, List, Optional

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_finding",
            "description": (
                "Record a single finding from the contract. "
                "Call this once for every benefit, obligation, risk, or unclear clause you identify. "
                "You MUST copy the evidence character-for-character from the contract text above."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["benefit", "obligation", "needs_attention", "high_risk", "negotiation_point"],
                        "description": (
                            "benefit — something positive for the employee. "
                            "obligation — something the employee must do. "
                            "needs_attention — vague, unclear, or one-sided clause. "
                            "high_risk — clause that could seriously harm the employee. "
                            "negotiation_point — specific thing the employee should push back on."
                        ),
                    },
                    "point": {
                        "type": "string",
                        "description": "Plain explanation in the language of the contract of what this means for the employee.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "The EXACT phrase or sentence copied verbatim from the contract text. "
                            "Do not translate, paraphrase, or rewrite it. "
                            "Keep it to the shortest excerpt that proves your point."
                        ),
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number where this text appears in the contract.",
                    },
                },
                "required": ["category", "point", "evidence", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_review",
            "description": (
                "Call this once you have recorded all findings to submit your final review. "
                "Also extract all party information (employee and employer) from the contract."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "e.g. 'Full-time Employment Contract' or 'Job Offer Letter'",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "2-3 sentences in the language of the contract: what kind of contract this is "
                            "and its overall risk level for the employee."
                        ),
                    },
                    # ── Employee fields ──────────────────────────────────────
                    "employee_name": {
                        "type": "string",
                        "description": "Full name of the employee (second party) as it appears in the contract.",
                    },
                    "employee_job_title": {
                        "type": "string",
                        "description": "Job title of the employee as stated in the contract.",
                    },
                    "employee_nationality": {
                        "type": "string",
                        "description": "Nationality of the employee if mentioned in the contract.",
                    },
                    "employee_id_number": {
                        "type": "string",
                        "description": "National ID, passport, or Iqama number of the employee if present.",
                    },
                    # ── Employer fields ──────────────────────────────────────
                    "employer_company_name": {
                        "type": "string",
                        "description": "Full legal name of the employer company (first party).",
                    },
                    "employer_company_address": {
                        "type": "string",
                        "description": "Address of the employer company if stated in the contract.",
                    },
                    "employer_representative_name": {
                        "type": "string",
                        "description": "Full name of the person signing on behalf of the employer.",
                    },
                    "employer_representative_title": {
                        "type": "string",
                        "description": "Title or role of the employer's signatory (e.g. HR Manager, CEO).",
                    },
                    # ── Contract detail fields ───────────────────────────────
                    "contract_start_date": {
                        "type": "string",
                        "description": "Contract or employment start date as written in the contract.",
                    },
                    "contract_duration": {
                        "type": "string",
                        "description": "Duration or end date of the contract if specified.",
                    },
                    "work_location": {
                        "type": "string",
                        "description": "City, office, or location where the employee will work.",
                    },
                },
                "required": ["document_type", "summary"],
            },
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an employment contract review agent. You help job seekers understand contracts before signing.

You will be given the full text of a contract. Read it completely, then use the tools to record every finding.

════════════════════════════════
HOW TO USE THE TOOLS
════════════════════════════════

Step 1 — call add_finding() for every clause you identify. Cover ALL of these:
  • Salary, bonuses, allowances
  • Working hours and overtime
  • Probation period and its conditions
  • Termination and notice period
  • Annual leave and public holidays
  • Health insurance and other benefits
  • Confidentiality and non-disclosure
  • Non-compete restrictions
  • Salary deductions and financial penalties
  • Resignation conditions and penalties
  • Binding to unseen policies or future changes except for clearly defined categories like "workplace conduct" or "governance policies"
  • Work location and job title flexibility

Step 2 — call finish_review() once with the document type, summary, AND all party details
         (employee name, job title, nationality, ID; employer company name, address,
          representative name and title; contract start date, duration, work location).
         Extract these directly from the contract text — do not leave them blank if present.

════════════════════════════════
EVIDENCE RULES
════════════════════════════════

The "evidence" field must be copied EXACTLY from the contract text — character for character, in the original language.
- Arabic contract → Arabic evidence
- English contract → English evidence
- Never translate, summarise, or rephrase the evidence
- Pick the shortest excerpt that proves your point

════════════════════════════════
RISK CLASSIFICATION
════════════════════════════════

Use high_risk for:
- Salary undefined or changeable by employer alone
- Probation extendable without limit
- Overtime with no separate compensation
- Unlimited deductions from salary
- Employee must follow unseen policies
- Employee pays fees for work-related equipment or expenses
- Financial penalty for resigning
- Immediate termination with no notice or compensation

Use needs_attention for clauses that are vague, one-sided, or missing key details but not immediately dangerous.

Use negotiation_point for specific actionable asks the employee can make before signing.
"""


def _format_contract(pages: List[Dict[str, Any]]) -> str:
    """Format extracted pages into a single string with clear page markers."""
    sections = []
    for page in pages:
        text = page["text"].strip()
        if text:
            sections.append(f"[PAGE {page['page_number']}]\n{text}")
    return "\n\n".join(sections)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    pages: List[Dict[str, Any]],
    user_question: str,
    client: OpenAI,
    trace_log: Optional[List[Dict[str, Any]]] = None,
    max_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Send the full contract to the LLM. The agent calls add_finding() for every
    clause it identifies, then calls finish_review() when done.

    trace_log — optional list; a dict is appended for every tool call.
    """
    if trace_log is None:
        trace_log = []

    contract_text = _format_contract(pages)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_question}\n\n"
                f"Here is the full contract text:\n\n{contract_text}"
            ),
        },
    ]

    # Accumulate findings as the agent calls add_finding()
    findings: Dict[str, List[Dict[str, Any]]] = {
        "benefit": [],
        "obligation": [],
        "needs_attention": [],
        "high_risk": [],
        "negotiation_point": [],
    }
    final_meta: Dict[str, str] = {}

    agent_start = time.perf_counter()
    finding_count = 0

    for iteration in range(max_iterations):
        iter_start = time.perf_counter()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
        )

        iter_elapsed = time.perf_counter() - iter_start
        choice = response.choices[0]
        usage = response.usage
        messages.append(choice.message)

        # ── Agent is calling tools ────────────────────────────────────────────
        if choice.finish_reason == "tool_calls":
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # ── add_finding ───────────────────────────────────────────────
                if name == "add_finding":
                    finding_count += 1
                    category = args.get("category", "needs_attention")
                    finding = {
                        "point":    args.get("point", ""),
                        "evidence": args.get("evidence", ""),
                        "page":     args.get("page", 0),
                    }
                    findings[category].append(finding)

                    trace_log.append({
                        "type":        "finding",
                        "finding_num": finding_count,
                        "category":    category,
                        "point":       finding["point"],
                        "evidence":    finding["evidence"],
                        "page":        finding["page"],
                        "llm_time_ms": round(iter_elapsed * 1000),
                        "tokens":      usage.total_tokens,
                    })

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      "Finding recorded.",
                    })

                # ── finish_review ─────────────────────────────────────────────
                elif name == "finish_review":
                    final_meta = {
                        "document_type": args.get("document_type", "Employment Document"),
                        "summary":       args.get("summary", ""),
                        # Employee
                        "employee_name":         args.get("employee_name", ""),
                        "employee_job_title":    args.get("employee_job_title", ""),
                        "employee_nationality":  args.get("employee_nationality", ""),
                        "employee_id_number":    args.get("employee_id_number", ""),
                        # Employer
                        "employer_company_name":         args.get("employer_company_name", ""),
                        "employer_company_address":      args.get("employer_company_address", ""),
                        "employer_representative_name":  args.get("employer_representative_name", ""),
                        "employer_representative_title": args.get("employer_representative_title", ""),
                        # Contract details
                        "contract_start_date": args.get("contract_start_date", ""),
                        "contract_duration":   args.get("contract_duration", ""),
                        "work_location":       args.get("work_location", ""),
                    }

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      "Review complete.",
                    })

                    total_elapsed = time.perf_counter() - agent_start
                    trace_log.append({
                        "type":             "done",
                        "total_findings":   finding_count,
                        "total_iterations": iteration + 1,
                        "total_time_s":     round(total_elapsed, 1),
                        "final_tokens": {
                            "prompt":     usage.prompt_tokens,
                            "completion": usage.completion_tokens,
                            "total":      usage.total_tokens,
                        },
                    })

                    return {
                        "document_type": final_meta.get("document_type", "Employment Document"),
                        "summary":       final_meta.get("summary", ""),
                        # ── Flat party fields returned at top level ──────────
                        "employee_name":                  final_meta.get("employee_name", ""),
                        "employee_job_title":             final_meta.get("employee_job_title", ""),
                        "employee_nationality":           final_meta.get("employee_nationality", ""),
                        "employee_id_number":             final_meta.get("employee_id_number", ""),
                        "employer_company_name":          final_meta.get("employer_company_name", ""),
                        "employer_company_address":       final_meta.get("employer_company_address", ""),
                        "employer_representative_name":   final_meta.get("employer_representative_name", ""),
                        "employer_representative_title":  final_meta.get("employer_representative_title", ""),
                        "contract_start_date":            final_meta.get("contract_start_date", ""),
                        "contract_duration":              final_meta.get("contract_duration", ""),
                        "work_location":                  final_meta.get("work_location", ""),
                        # ── Findings ─────────────────────────────────────────
                        "benefits":           findings["benefit"],
                        "obligations":        findings["obligation"],
                        "needs_attention":    findings["needs_attention"],
                        "high_risk":          findings["high_risk"],
                        "negotiation_points": [f["point"] for f in findings["negotiation_point"]],
                        "legal_note": (
                            "This analysis is informational only. "
                            "Please consult a qualified lawyer before signing any contract."
                        ),
                        "_trace": trace_log,
                    }

        # ── Model stopped without calling finish_review ───────────────────────
        else:
            messages.append({
                "role":    "user",
                "content": "You have not called finish_review yet. Please call it now to complete the review.",
            })

    raise RuntimeError("Agent did not complete within the maximum number of iterations.")