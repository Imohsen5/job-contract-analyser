# app.py
import os
import time
import requests
import streamlit as st
from openai import OpenAI

from utils.pdf_utils import extract_text_from_pdf
from utils.agent import run_agent
from utils.email_utils import draft_inquiry_email

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Contract Review Agent", page_icon="📄", layout="wide")

st.title("📄 Job Contract Review Agent")
st.caption(
    "Upload your job offer or employment contract and get a plain-language review. "
    "Every finding is backed by the exact wording from your document."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your key is used only for this session and never stored.",
    )
    st.markdown("---")
    show_trace = st.toggle(
        "Show pipeline trace", value=True,
        help="See every finding the agent records.",
    )
    st.markdown("---")
    st.info(
        "**This tool is informational only.**\n\n"
        "It does not replace professional legal advice. "
        "Always consult a lawyer before signing important contracts."
    )

# ── Webhook URL from secrets ──────────────────────────────────────────────────

webhook_url = st.secrets.get("N8N_WEBHOOK_URL", "")

# ── Upload & question ─────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload your contract (PDF)", type=["pdf"])

user_question = st.text_area(
    "What should the agent focus on?",
    value=(
        "Review this employment contract. "
        "Identify all benefits, employee obligations, unclear or one-sided clauses, "
        "high-risk terms, and suggest negotiation points."
    ),
    height=90,
)

run_btn = st.button("🔍 Analyze Contract", type="primary", disabled=not uploaded_file)

# ── Helpers ───────────────────────────────────────────────────────────────────

CATEGORY_STYLE = {
    "benefit":           ("✅", "Benefits"),
    "obligation":        ("📋", "Employee Obligations"),
    "needs_attention":   ("⚠️", "Needs Attention"),
    "high_risk":         ("🚨", "High-Risk Clauses"),
    "negotiation_point": ("💡", "Negotiation Points"),
}


def render_findings(title: str, icon: str, items: list, empty_msg: str):
    st.subheader(f"{icon} {title}")
    if not items:
        st.caption(empty_msg)
        return
    for item in items:
        if isinstance(item, dict):
            point    = item.get("point", "")
            evidence = item.get("evidence", "")
            page     = item.get("page", "")
        else:
            st.markdown(f"- {item}")
            continue

        label = f"Page {page} — {point}" if page else point
        if len(label) > 110:
            label = label[:107] + "..."

        with st.expander(label):
            st.markdown(f"**{point}**")
            if evidence:
                st.markdown("📌 **Evidence from contract:**")
                page_note = f" *(Page {page})*" if page else ""
                st.info(f'"{evidence}"{page_note}')


def render_trace(trace_log: list):
    findings = [e for e in trace_log if e["type"] == "finding"]
    done     = next((e for e in trace_log if e["type"] == "done"), None)

    if not findings and not done:
        st.caption("No trace data recorded.")
        return

    if done:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total findings",  done["total_findings"])
        c2.metric("LLM iterations",  done["total_iterations"])
        c3.metric("Total time",      f"{done['total_time_s']}s")
        c4.metric("Total tokens",    done["final_tokens"]["total"])

    st.markdown("---")
    st.markdown("#### 📋 Findings recorded by the agent")

    category_colors = {
        "benefit":           "🟢",
        "obligation":        "🔵",
        "needs_attention":   "🟡",
        "high_risk":         "🔴",
        "negotiation_point": "🟣",
    }

    for event in findings:
        color = category_colors.get(event["category"], "⚪")
        icon, label = CATEGORY_STYLE.get(event["category"], ("📌", event["category"]))
        with st.expander(
            f"{color} **Finding {event['finding_num']}** | {icon} {label} | "
            f"Page {event['page']} | {event['llm_time_ms']} ms | "
            f"{event['tokens']} tokens so far"
        ):
            st.markdown(f"**Point:** {event['point']}")
            st.markdown("**Evidence (verbatim from contract):**")
            st.info(f'"{event["evidence"]}" *(Page {event["page"]})*')

    if done:
        st.markdown("---")
        st.markdown("#### 📈 Token usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens",     done["final_tokens"]["prompt"])
        col2.metric("Completion tokens", done["final_tokens"]["completion"])
        col3.metric("Total tokens",      done["final_tokens"]["total"])


def build_webhook_payload(result: dict, pages: list, char_count: int) -> dict:
    return {
        "meta": {
            "document_type": result.get("document_type", ""),
            "pages":         len(pages),
            "characters":    char_count,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary":            result.get("summary", ""),
        "benefits":           result.get("benefits", []),
        "obligations":        result.get("obligations", []),
        "needs_attention":    result.get("needs_attention", []),
        "high_risk":          result.get("high_risk", []),
        "negotiation_points": result.get("negotiation_points", []),
        "legal_note":         result.get("legal_note", ""),
    }


# ── Main analysis flow ────────────────────────────────────────────────────────

if run_btn:
    api_key = api_key_input or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Step 1 — Extract PDF
    t0 = time.perf_counter()
    with st.status("📖 Reading PDF...", expanded=False) as status:
        try:
            pages = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

        total_text = " ".join(p["text"] for p in pages)
        char_count = len(total_text)
        token_est  = char_count // 4

        if char_count < 200:
            st.error(
                "The extracted text is too short. "
                "The PDF may be scanned or image-only. Please use a text-based PDF."
            )
            st.stop()

        pdf_time = round((time.perf_counter() - t0) * 1000)
        status.update(
            label=(
                f"📖 PDF read — {len(pages)} pages, "
                f"{char_count:,} chars (~{token_est:,} tokens) | {pdf_time} ms"
            ),
            state="complete",
        )

    # Step 2 — Run agent
    trace_log: list = []
    with st.status("🤖 Agent is reading the contract...", expanded=False) as status:
        try:
            result = run_agent(pages, user_question, client, trace_log=trace_log)
        except Exception as e:
            st.error(f"Agent analysis failed: {e}")
            st.stop()

        done_event = next((e for e in trace_log if e["type"] == "done"), {})
        status.update(
            label=(
                f"🤖 Agent done — "
                f"{done_event.get('total_findings', '?')} findings, "
                f"{done_event.get('total_time_s', '?')}s"
            ),
            state="complete",
        )

    # Persist to session state so interactions below don't wipe the results
    st.session_state["result"]     = result
    st.session_state["pages"]      = pages
    st.session_state["char_count"] = char_count
    st.session_state["trace_log"]  = trace_log
    st.session_state["client"]     = client
    # Clear any previous email draft when a new analysis runs
    st.session_state.pop("email_draft", None)

# ── Render results ────────────────────────────────────────────────────────────

if "result" in st.session_state:
    result     = st.session_state["result"]
    pages      = st.session_state["pages"]
    char_count = st.session_state["char_count"]
    trace_log  = st.session_state["trace_log"]
    client     = st.session_state["client"]

    # FIX: proper if-block instead of ternary that returned a DeltaGenerator object
    st.success("✅ Analysis complete!")
    st.divider()

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("📝 Summary")
        st.write(result.get("summary", ""))

        render_findings("Benefits", "✅",
                        result.get("benefits", []),
                        "No clear benefits were found.")
        render_findings("Employee Obligations", "📋",
                        result.get("obligations", []),
                        "No explicit obligations were found.")
        render_findings("Needs Attention", "⚠️",
                        result.get("needs_attention", []),
                        "No unclear or one-sided clauses were flagged.")
        render_findings("High-Risk Clauses", "🚨",
                        result.get("high_risk", []),
                        "No major high-risk clauses were found.")

        # Negotiation Points
        st.subheader("💡 Negotiation Points")
        neg_points = result.get("negotiation_points", [])
        if neg_points:
            for pt in neg_points:
                st.markdown(f"- {pt}")
        else:
            st.caption("No negotiation points were suggested.")

        # ── Feature 1: Inquiry Email ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("✉️ Draft Inquiry Email")
        st.caption(
            "Generate a professional email to send to your employer asking for "
            "clarification on unclear or risky clauses — with the exact contract "
            "wording referenced in each question."
        )

        with st.form("email_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                sender_name = st.text_input(
                    "Your name (optional)",
                    placeholder="e.g. Ahmed Al-Zahrani",
                )
            with col_b:
                recipient_title = st.text_input(
                    "Recipient title (optional)",
                    placeholder="e.g. HR Manager",
                    value="HR / Hiring Manager",
                )
            draft_btn = st.form_submit_button("✉️ Generate Email Draft", type="primary")

        # Placeholder anchored directly below the form — spinner renders HERE
        email_placeholder = st.empty()

        if draft_btn:
            has_issues = (
                result.get("needs_attention") or
                result.get("high_risk") or
                result.get("negotiation_points")
            )
            if not has_issues:
                email_placeholder.info(
                    "No unclear or risky clauses were found — nothing to raise in an email."
                )
            else:
                # Spinner renders inside the placeholder, right below the button
                with email_placeholder.container():
                    with st.spinner("Drafting your email..."):
                        try:
                            email_draft = draft_inquiry_email(
                                result, client,
                                sender_name=sender_name,
                                recipient_title=recipient_title,
                            )
                            st.session_state["email_draft"] = email_draft
                        except Exception as e:
                            st.error(f"Failed to draft email: {e}")

        # Display the draft if it exists — st.code has a built-in copy button
        if "email_draft" in st.session_state:
            st.markdown("**📋 Your draft — copy and send:**")
            st.caption("Click the copy icon (⧉) in the top-right corner of the box.")
            st.code(st.session_state["email_draft"], language=None)

    with col_side:
        st.subheader("📊 Document Info")
        st.metric("Pages",       len(pages))
        st.metric("Characters",  f"{char_count:,}")
        st.metric("Est. tokens", f"{char_count // 4:,}")
        st.markdown(f"**Type:** {result.get('document_type', 'Unknown')}")
        st.divider()
        st.subheader("⚖️ Legal Note")
        st.warning(result.get("legal_note", "This output is informational only."))

    # ── Feature 2: n8n Webhook ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Send to n8n")

    payload = build_webhook_payload(result, pages, char_count)

    wh_col1, wh_col2 = st.columns([3, 1])
    with wh_col1:
        with st.expander("Preview payload (JSON)", expanded=False):
            st.json(payload)
    with wh_col2:
        send_btn = st.button(
            "📤 Send to n8n",
            type="primary",
            disabled=not webhook_url,
            help=(
                "Add N8N_WEBHOOK_URL to .streamlit/secrets.toml to enable this."
                if not webhook_url else "Send the full analysis payload to your n8n workflow."
            ),
        )

    if not webhook_url:
        st.caption("Add `N8N_WEBHOOK_URL` to `.streamlit/secrets.toml` to enable this.")

    if send_btn and webhook_url:
        with st.spinner("Sending to n8n..."):
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
                if response.status_code in (200, 201, 202):
                    st.success(
                        f"✅ Sent successfully — n8n responded with status {response.status_code}."
                    )
                    with st.expander("n8n response"):
                        try:
                            st.json(response.json())
                        except Exception:
                            st.text(response.text)
                else:
                    st.error(
                        f"n8n returned status {response.status_code}. "
                        f"Response: {response.text[:300]}"
                    )
            except requests.exceptions.Timeout:
                st.error("Request timed out. Check your n8n instance is running.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect. Check the webhook URL in secrets.toml.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # ── Pipeline trace ────────────────────────────────────────────────────────
    if show_trace:
        st.divider()
        with st.expander("🔬 Pipeline Trace — what happened under the hood", expanded=False):
            render_trace(trace_log)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("📄 View extracted pages"):
        for page in pages:
            st.markdown(f"**Page {page['page_number']}**")
            st.write(page["text"][:3000] or "(No text extracted)")
            st.markdown("---")

    with st.expander("🔩 View raw JSON result"):
        display = {k: v for k, v in result.items() if k != "_trace"}
        st.json(display)