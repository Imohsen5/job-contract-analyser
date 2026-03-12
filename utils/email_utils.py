# utils/email_utils.py
from openai import OpenAI
from typing import Any, Dict, List


def draft_inquiry_email(
    result: Dict[str, Any],
    contract_text: str,
    client: OpenAI,
    sender_name: str = "",
    recipient_title: str = "",
) -> str:
    """
    Draft a professional inquiry email based on contract findings.

    - Detects the contract language and writes the email in that same language.
    - Extracts real names and company from the contract text automatically.
    - Writes as this specific employee, not as a generic policy commentator.
    - Only raises points that are genuinely unclear or need negotiation.
    """

    # Collect only genuinely actionable points
    points_to_raise: List[str] = []

    for item in result.get("needs_attention", []):
        point    = item.get("point", "")
        evidence = item.get("evidence", "")
        page     = item.get("page", "")
        ref      = f' (Clause: "{evidence}", Page {page})' if evidence else ""
        points_to_raise.append(f"[Needs clarification] {point}{ref}")

    for item in result.get("high_risk", []):
        point    = item.get("point", "")
        evidence = item.get("evidence", "")
        page     = item.get("page", "")
        ref      = f' (Clause: "{evidence}", Page {page})' if evidence else ""
        points_to_raise.append(f"[Needs review] {point}{ref}")

    for pt in result.get("negotiation_points", []):
        points_to_raise.append(f"[Negotiation] {pt}")

    if not points_to_raise:
        return "No unclear or high-risk clauses were found that require raising with the employer."

    points_block = "\n".join(f"- {p}" for p in points_to_raise)

    # First 3000 chars is enough to find all party names and contract language
    contract_excerpt = contract_text[:3000]

    sender_line    = f"Sender name: {sender_name.strip()}" if sender_name.strip() else "Use the employee name extracted from the contract."
    recipient_line = f"Recipient: {recipient_title.strip()}" if recipient_title.strip() else "Address the email to the company representative who signed the contract."

    prompt = f"""You are helping a new employee draft a professional email to their prospective employer.

The goal is to politely ask for clarification or negotiation on specific clauses in their employment contract.

══════════════════════════════
STRICT RULES
══════════════════════════════

1. LANGUAGE: Detect the language the contract is written in (e.g. Arabic, English) from the
   contract excerpt below, and write the entire email in that same language.
   Do not translate. If the contract is in Arabic, the email must be in Arabic.
   If the contract is in English, the email must be in English.

2. NAMES: Extract from the contract excerpt:
   - Employee name (second party)
   - Company name (first party)
   - Company representative who signed the contract
   - Employee job title
   Use these directly in the email. If any detail is missing, use a placeholder like
   [Company Name] or [Manager Name].

3. VOICE: Write in the employee's own voice — as one specific person asking about
   their own contract. Never write as a collective or as someone critiquing general policy.
   Correct: "I noticed in my contract that..."
   Wrong:   "Employees may not understand this clause..."

4. SCOPE: Raise only the points listed below. Do not add or invent any additional concerns.

5. FOR EACH POINT: Reference the exact clause wording from the contract where provided,
   then ask one specific and practical question about it.

6. Do NOT question standard labour law requirements or general legal boilerplate —
   these are non-negotiable and raising them weakens the employee's position.

7. TONE: Professional, respectful, and constructive. Not confrontational or complainy.

8. Close with a clear thank-you and a request to discuss at a convenient time.

══════════════════════════════
CONTRACT EXCERPT (for name and language detection)
══════════════════════════════
{contract_excerpt}

══════════════════════════════
POINTS TO RAISE IN THE EMAIL
══════════════════════════════
{points_block}

{sender_line}
{recipient_line}

Write the complete email only — subject line, body, and sign-off.
Do not write anything before or after the email.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()