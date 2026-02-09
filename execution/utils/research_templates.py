"""
Shared research constraint templates used by both Gemini and Perplexity researchers.

These functions generate writer constraints, fallback constraints, and revision
instructions from fact sheets and verification results. Extracted to avoid
duplication between gemini_researcher.py and perplexity_researcher.py.
"""

from typing import Dict


# Constant fallback text when research fails
FALLBACK_CONSTRAINTS_TEXT = """
======================================================================
\u26a0\ufe0f  RESEARCH FAILED - STRICT CONSTRAINTS IN EFFECT
======================================================================

Because fact verification failed, you MUST:
1. Avoid ALL specific numbers, percentages, and metrics
2. Write in general terms with opinion hedging
3. Use phrases like "teams report...", "some engineers find..."
4. DO NOT claim specific hardware specs, costs, or performance numbers

======================================================================
"""


def generate_writer_constraints(fact_sheet: Dict, provider_label: str = "") -> str:
    """
    Generate natural language constraints for the Writer from a fact sheet.

    Args:
        fact_sheet: Dict with verified_facts, unverified_claims, general_knowledge keys.
                    Items in lists can be either dicts or plain strings.
        provider_label: Optional label to append to the header (e.g. "via Perplexity").

    Returns:
        Formatted constraint text for the writer.
    """
    fact_sheet = fact_sheet or {}
    lines = []
    header_suffix = f" ({provider_label})" if provider_label else ""
    lines.append("=" * 70)
    lines.append(f"FACT SHEET - YOUR ONLY SOURCE OF TRUTH{header_suffix}")
    lines.append("=" * 70)
    lines.append("")

    verified = fact_sheet.get("verified_facts", [])
    if verified:
        lines.append("\u2705 VERIFIED FACTS (You MAY use these):")
        for f in verified:
            if isinstance(f, dict):
                lines.append(f"   \u2022 {f.get('fact', f)}")
                lines.append(f"     Source: {f.get('source_url', f.get('source', 'N/A'))}")
            else:
                lines.append(f"   \u2022 {f}")
        lines.append("")
    else:
        lines.append("\u26a0\ufe0f  NO VERIFIED FACTS")
        lines.append("   Write with conviction but WITHOUT specific numbers.")
        lines.append("")

    unverified = fact_sheet.get("unverified_claims", [])
    if unverified:
        lines.append("\u274c DO NOT USE THESE CLAIMS:")
        for u in unverified:
            if isinstance(u, dict):
                lines.append(f"   \u2022 {u.get('claim', u)}")
                lines.append(f"     Reason: {u.get('reason', 'Could not verify')}")
            else:
                lines.append(f"   \u2022 {u}")
        lines.append("")

    general = fact_sheet.get("general_knowledge", [])
    if general:
        lines.append("\U0001f4da GENERAL KNOWLEDGE (Safe without citation):")
        for g in general:
            lines.append(f"   \u2022 {g}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("RULES: Only use verified facts. No fake metrics. Cite sources.")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_revision_instructions(verification: Dict) -> str:
    """
    Generate specific revision instructions based on what failed verification.

    Args:
        verification: Dict with false_claims, unverifiable_claims, suspicious_claims,
                      verified_claims keys. Items can be dicts or plain strings.

    Returns:
        Formatted revision instruction text.
    """
    verification = verification or {}
    lines = []
    lines.append("REVISION REQUIRED - Fix these specific issues:")
    lines.append("")

    false_claims = verification.get("false_claims", [])
    if false_claims:
        lines.append("\u274c FALSE CLAIMS (Must correct or remove):")
        for c in false_claims:
            if isinstance(c, dict):
                lines.append(f"   \u2022 {c.get('claim', c)}")
                lines.append(f"     Why false: {c.get('why_false', 'N/A')}")
                if c.get('correction'):
                    lines.append(f"     Correction: {c.get('correction')}")
            else:
                lines.append(f"   \u2022 {c}")
        lines.append("")

    unverifiable = verification.get("unverifiable_claims", [])
    if unverifiable:
        lines.append("\u26a0\ufe0f  UNVERIFIABLE CLAIMS (Remove or hedge):")
        for c in unverifiable:
            if isinstance(c, dict):
                lines.append(f"   \u2022 {c.get('claim', c)}")
                lines.append(f"     Why: {c.get('why_unverifiable', 'Could not verify')}")
            else:
                lines.append(f"   \u2022 {c}")
        lines.append("")

    suspicious = verification.get("suspicious_claims", [])
    if suspicious:
        lines.append("\U0001f6a9 SUSPICIOUS CLAIMS (Review carefully):")
        for c in suspicious:
            if isinstance(c, dict):
                lines.append(f"   \u2022 {c.get('claim', c)}")
                lines.append(f"     Red flag: {c.get('red_flag', 'Needs verification')}")
            else:
                lines.append(f"   \u2022 {c}")
        lines.append("")

    verified = verification.get("verified_claims", [])
    if verified:
        lines.append("\u2705 VERIFIED (Keep these):")
        for c in verified:
            if isinstance(c, dict):
                lines.append(f"   \u2022 {c.get('claim', c)}")
            else:
                lines.append(f"   \u2022 {c}")
        lines.append("")

    return "\n".join(lines)
