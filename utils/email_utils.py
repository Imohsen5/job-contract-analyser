# utils/email_utils.py
from openai import OpenAI
from typing import Any, Dict, List


def draft_inquiry_email(
    result: Dict[str, Any],
    client: OpenAI,
    sender_name: str = "",
    recipient_title: str = "HR / Hiring Manager",
) -> str:
    """
    Draft a professional inquiry email based on the contract findings.
    Uses needs_attention, high_risk, and negotiation_points from the agent result.
    """

    # Build a concise brief of what needs clarification
    points_to_raise: List[str] = []

    for item in result.get("needs_attention", []):
        quote = item.get("evidence", "")
        point = item.get("point", "")
        page  = item.get("page", "")
        ref   = f' (Page {page}, clause: "{quote}")' if quote else ""
        points_to_raise.append(f"[Unclear clause] {point}{ref}")

    for item in result.get("high_risk", []):
        quote = item.get("evidence", "")
        point = item.get("point", "")
        page  = item.get("page", "")
        ref   = f' (Page {page}, clause: "{quote}")' if quote else ""
        points_to_raise.append(f"[Concerning clause] {point}{ref}")

    for pt in result.get("negotiation_points", []):
        points_to_raise.append(f"[Negotiation] {pt}")

    if not points_to_raise:
        return "No unclear or high-risk points were found to raise in an email."

    points_block = "\n".join(f"- {p}" for p in points_to_raise)
    sender_line  = sender_name.strip() if sender_name.strip() else "[Your Name]"
    doc_type     = result.get("document_type", "the contract")

    prompt = f"""You are helping a job seeker write a professional, polite email to their prospective employer.
The email should ask for clarification or negotiation on specific points in {doc_type}.

Tone: professional, respectful, and constructive — not confrontational.

Points to raise (include all of them, grouped logically):
{points_block}

Sender name: {sender_line}
Recipient: {recipient_title}

Write a complete, ready-to-send email with:
- A clear subject line
- A brief opening (1-2 sentences thanking them and expressing continued interest)
- One short paragraph per issue, referencing the exact clause wording where provided
- A positive closing that keeps the conversation open
- A professional sign-off

Return only the email text. No commentary before or after it.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()
