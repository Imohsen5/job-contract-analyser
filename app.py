# app.py
import os
import time
import streamlit as st
from openai import OpenAI

from utils.pdf_utils import extract_text_from_pdf
from utils.agent import run_agent

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
    show_trace = st.toggle("Show pipeline trace", value=True,
                           help="See every finding the agent records in real time.")
    st.markdown("---")
    st.info(
        "**This tool is informational only.**\n\n"
        "It does not replace professional legal advice. "
        "Always consult a lawyer before signing important contracts."
    )

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
    "benefit":            ("✅", "Benefits"),
    "obligation":         ("📋", "Employee Obligations"),
    "needs_attention":    ("⚠️", "Needs Attention"),
    "high_risk":          ("🚨", "High-Risk Clauses"),
    "negotiation_point":  ("💡", "Negotiation Points"),
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
            # negotiation_points are plain strings
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

    # ── Summary bar ───────────────────────────────────────────────────────────
    if done:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total findings", done["total_findings"])
        c2.metric("LLM iterations", done["total_iterations"])
        c3.metric("Total time", f"{done['total_time_s']}s")
        c4.metric("Total tokens", done["final_tokens"]["total"])

    st.markdown("---")

    # ── Findings breakdown ────────────────────────────────────────────────────
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

    # ── Token usage ───────────────────────────────────────────────────────────
    if done:
        st.markdown("---")
        st.markdown("#### 📈 Token usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens",     done["final_tokens"]["prompt"])
        col2.metric("Completion tokens", done["final_tokens"]["completion"])
        col3.metric("Total tokens",      done["final_tokens"]["total"])
        st.caption(
            "With full-contract-in-context there is only ONE LLM call cycle. "
            "Token cost is fixed by contract length, not by number of searches."
        )


# ── Main flow ─────────────────────────────────────────────────────────────────

if run_btn:
    api_key = api_key_input or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # ── Step 1: Extract PDF ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    with st.status("📖 Reading PDF...", expanded=False) as status:
        try:
            pages = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

        total_text = " ".join(p["text"] for p in pages)
        if len(total_text) < 200:
            st.error(
                "The extracted text is too short. "
                "The PDF may be scanned or image-only. Please use a text-based PDF."
            )
            st.stop()

        pdf_time = round((time.perf_counter() - t0) * 1000)
        char_count = len(total_text)
        token_estimate = char_count // 4   # rough estimate: 1 token ≈ 4 chars

        status.update(
            label=(
                f"📖 PDF read — {len(pages)} pages, "
                f"{char_count:,} chars (~{token_estimate:,} tokens estimated) | "
                f"{pdf_time} ms"
            ),
            state="complete",
        )

    # ── Step 2: Run agent (full contract in context) ──────────────────────────
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
                f"{done_event.get('total_findings', '?')} findings recorded, "
                f"{done_event.get('total_time_s', '?')}s"
            ),
            state="complete",
        )

    st.success("✅ Analysis complete!")
    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
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

        st.subheader("💡 Negotiation Points")
        neg_points = result.get("negotiation_points", [])
        if neg_points:
            for pt in neg_points:
                st.markdown(f"- {pt}")
        else:
            st.caption("No negotiation points were suggested.")

    with col_side:
        st.subheader("📊 Document Info")
        st.metric("Pages", len(pages))
        st.metric("Characters", f"{char_count:,}")
        st.metric("Est. tokens", f"{token_estimate:,}")
        st.markdown(f"**Type:** {result.get('document_type', 'Unknown')}")
        st.divider()
        st.subheader("⚖️ Legal Note")
        st.warning(result.get("legal_note", "This output is informational only."))

    # ── Pipeline trace ────────────────────────────────────────────────────────
    if show_trace:
        st.divider()
        with st.expander("🔬 Pipeline Trace — what happened under the hood", expanded=True):
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