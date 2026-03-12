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

    - Detects the contract language and writes the email in that language.
    - Extracts real names and company from the contract text automatically.
    - Writes as this specific employee, not as a generic policy commentator.
    - Only raises points that are genuinely unclear or need negotiation —
      not standard legal boilerplate.
    """

    # Collect only genuinely actionable points
    points_to_raise: List[str] = []

    for item in result.get("needs_attention", []):
        point    = item.get("point", "")
        evidence = item.get("evidence", "")
        page     = item.get("page", "")
        ref      = f' — النص الوارد: "{evidence}" (صفحة {page})' if evidence else ""
        points_to_raise.append(f"[يحتاج توضيح] {point}{ref}")

    for item in result.get("high_risk", []):
        point    = item.get("point", "")
        evidence = item.get("evidence", "")
        page     = item.get("page", "")
        ref      = f' — النص الوارد: "{evidence}" (صفحة {page})' if evidence else ""
        points_to_raise.append(f"[بند يحتاج مراجعة] {point}{ref}")

    for pt in result.get("negotiation_points", []):
        points_to_raise.append(f"[نقطة تفاوض] {pt}")

    if not points_to_raise:
        return "لم يتم تحديد أي نقاط غير واضحة أو عالية الخطورة تستوجب التواصل مع صاحب العمل."

    points_block = "\n".join(f"- {p}" for p in points_to_raise)

    # Provide only the first 3000 chars for name extraction — enough to find all parties
    contract_excerpt = contract_text[:3000]

    prompt = f"""أنت تساعد موظفًا جديدًا على صياغة رسالة بريد إلكتروني احترافية إلى صاحب العمل.

الهدف: الاستفسار بأدب عن بنود معينة في عقد العمل الخاص به قبل التوقيع، أو بعد مراجعته.

══════════════════════════════
تعليمات صارمة
══════════════════════════════

1. اكتب الرسالة باللغة العربية الفصحى، لأن العقد محرر بالعربية.

2. استخرج من النص التالي للعقد:
   - اسم الموظف (الطرف الثاني)
   - اسم الشركة (الطرف الأول)
   - اسم ممثل الشركة الموقع على العقد
   - المسمى الوظيفي للموظف
   واستخدمها في الرسالة مباشرة. إذا لم تجد معلومة معينة، استخدم [اسم الشركة] أو [اسم المسؤول] كبديل.

3. اكتب الرسالة بصوت الموظف نفسه — وليس بصوت ممثل جماعي أو ناقد للسياسات العامة.
   مثال صحيح: "لاحظت في عقدي أن..."
   مثال خاطئ: "قد لا يفهم الموظفون هذا البند..."

4. اذكر فقط النقاط الواردة أدناه — لا تُضف نقاطًا من عندك.

5. لكل نقطة: اذكر النص الحرفي من العقد إن وُجد، ثم اطرح سؤالًا محددًا وعمليًا.

6. لا تنتقد أنظمة العمل السعودية أو الاشتراطات القانونية العامة — هذه ليست قابلة للتفاوض وإثارتها تُضعف موقف الموظف.

7. الأسلوب: مهني، محترم، ودود — لا متذمر ولا متصادم.

8. اختتم بشكر واضح وطلب موعد للحوار.

══════════════════════════════
مقتطف من العقد (لاستخراج الأسماء)
══════════════════════════════
{contract_excerpt}

══════════════════════════════
النقاط المراد إثارتها
══════════════════════════════
{points_block}

{"اسم المُرسِل: " + sender_name.strip() if sender_name.strip() else "استخدم اسم الموظف المستخرج من العقد."}
{"المُرسَل إليه: " + recipient_title.strip() if recipient_title.strip() else "وجّه الرسالة لممثل الشركة الموقّع على العقد."}

اكتب الرسالة كاملة فقط — بسطر الموضوع، ثم المتن، ثم التحية الختامية. لا تكتب أي شيء قبلها أو بعدها.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()