"""
Fact Research Agent - LLM-powered research with web search.

This agent is the FIRST line of defense against hallucination.
It searches the web for REAL facts before the Writer touches anything.

Philosophy:
- No regex, no hardcoded patterns - pure LLM intelligence
- Web search is the source of truth
- Better to have fewer verified facts than many fake ones
- The fact sheet becomes the Writer's ONLY allowed source for specifics
"""

import json
from typing import Dict, Optional, Callable
from execution.utils.json_parser import extract_json_from_llm
from .base_agent import BaseAgent, LLMError


class FactResearchAgent(BaseAgent):
    """
    Investigative researcher that gathers verified facts via web search.

    The output of this agent is the GROUND TRUTH for the rest of the pipeline.
    If it's not in the fact sheet, the Writer cannot claim it.
    """

    def __init__(self):
        super().__init__(
            role="Investigative Technical Researcher",
            persona="""You are an investigative journalist specializing in technology.

YOUR MISSION: Find REAL, VERIFIABLE facts before any content is written.

PRINCIPLES:
1. SEARCH FIRST, WRITE NEVER - Your job is research, not writing
2. SOURCES OR IT DIDN'T HAPPEN - Every fact needs a URL
3. NUMBERS ARE SACRED - Triple-check any metrics, specs, or statistics
4. UNKNOWN IS VALUABLE - "I couldn't verify this" is useful information
5. SKEPTICISM IS YOUR FRIEND - If something sounds too good/specific, verify it

WHAT YOU RESEARCH:
- Hardware specs (GPU TFLOPS, memory, bandwidth) - from official docs
- Model specifications (parameters, context windows) - from papers/official releases
- Performance benchmarks - from published benchmarks, not random claims
- Pricing and costs - from official pricing pages
- Company/people claims - verify who actually built what

OUTPUT FORMAT:
You return a structured fact sheet that becomes the Writer's bible.
If a fact isn't in your sheet, the Writer CANNOT use it.

Remember: You are the guardrail against hallucination. Be thorough.""",
            model="llama-3.3-70b-versatile"
        )

    def research(self, topic: str, source_content: str = "", web_search_func: Optional[Callable] = None) -> Dict:
        """
        Research a topic using web search and return verified facts.

        Args:
            topic: The topic to research
            source_content: Original source content (Reddit post, etc.)
            web_search_func: Function to perform web searches.
                            Signature: search(query: str) -> str (search results)

        Returns:
            Fact sheet dict with verified facts, unknowns, and writer constraints
        """
        # Step 1: Analyze topic to determine what needs verification
        research_plan = self._plan_research(topic, source_content)

        # Step 2: Execute searches if we have search capability
        search_results = {}
        if web_search_func:
            search_results = self._execute_searches(research_plan, web_search_func)

        # Step 3: Synthesize into fact sheet
        fact_sheet = self._synthesize_facts(topic, source_content, search_results)

        # Step 4: Generate writer constraints
        fact_sheet["writer_constraints"] = self._generate_constraints(fact_sheet)

        return fact_sheet

    def _plan_research(self, topic: str, source_content: str) -> Dict:
        """Use LLM to plan what needs to be researched."""
        prompt = f"""Analyze this topic and determine what facts need verification.

TOPIC: {topic}

SOURCE CONTENT (if any):
{source_content[:2000] if source_content else "No source content provided"}

Your job: Identify SPECIFIC claims that need verification before writing.

Return JSON:
{{
    "key_entities": [
        {{"name": "...", "type": "hardware|model|company|metric|claim", "why_verify": "..."}}
    ],
    "search_queries": [
        {{"query": "...", "purpose": "what we're trying to verify"}}
    ],
    "red_flags": [
        "..." // Claims in source that seem suspicious/unverifiable
    ],
    "can_write_without_search": ["..."]  // General knowledge that doesn't need verification
}}

Be SPECIFIC. If the topic mentions "MI50 GPU", we need to search for actual MI50 specs.
If it mentions "10 trillion parameters", that's a red flag - verify or flag as suspicious."""

        try:
            response = self.call_llm(prompt, temperature=0.2)
        except LLMError as e:
            print(f"  [!] LLM call failed during research planning: {e}")
            return {
                "key_entities": [],
                "search_queries": [{"query": topic, "purpose": "general research"}],
                "red_flags": [],
                "can_write_without_search": []
            }

        result = extract_json_from_llm(response)
        if result is not None:
            return result
        return {
            "key_entities": [],
            "search_queries": [{"query": topic, "purpose": "general research"}],
            "red_flags": [],
            "can_write_without_search": []
        }

    def _execute_searches(self, research_plan: Dict, search_func: Callable) -> Dict:
        """Execute web searches based on the research plan."""
        results = {}

        queries = research_plan.get("search_queries", [])
        for item in queries[:5]:  # Limit to 5 searches
            query = item.get("query", "")
            purpose = item.get("purpose", "")

            if not query:
                continue

            try:
                search_result = search_func(query)
                results[query] = {
                    "purpose": purpose,
                    "results": search_result
                }
            except Exception as e:
                results[query] = {
                    "purpose": purpose,
                    "error": str(e)
                }

        return results

    def _synthesize_facts(self, topic: str, source_content: str, search_results: Dict) -> Dict:
        """Use LLM to synthesize search results into verified facts."""

        # Format search results for the prompt
        search_context = ""
        if search_results:
            for query, data in search_results.items():
                if "error" in data:
                    search_context += f"\n\nSEARCH: {query}\nERROR: {data['error']}"
                else:
                    search_context += f"\n\nSEARCH: {query}\nPURPOSE: {data['purpose']}\nRESULTS:\n{data['results'][:3000]}"
        else:
            search_context = "\n\nNO WEB SEARCH PERFORMED - Working only with source content."

        prompt = f"""Synthesize verified facts from the research.

TOPIC: {topic}

SOURCE CONTENT:
{source_content[:1500] if source_content else "None provided"}

SEARCH RESULTS:{search_context}

Your job: Extract ONLY facts that are VERIFIED by the search results or official sources.

Return JSON:
{{
    "verified_facts": [
        {{
            "fact": "The AMD MI50 has 13.4 TFLOPS FP32 compute",
            "source": "AMD official specifications page",
            "source_url": "https://...",
            "confidence": "high"
        }}
    ],
    "unverified_claims": [
        {{
            "claim": "10 trillion parameters per second",
            "reason": "This metric doesn't exist - parameters are static, not processed per second",
            "recommendation": "DO NOT USE - fabricated metric"
        }}
    ],
    "general_knowledge": [
        "GPUs are used for parallel processing"  // Safe to state without citation
    ],
    "unknowns": [
        "Exact cost of the setup - couldn't find pricing"
    ]
}}

RULES:
1. If search results don't confirm a specific number, it goes in unverified_claims
2. "Parameters per second" is NOT a real metric - flag it
3. Trillion-scale claims need SPECIFIC sources (GPT-4, Claude, etc. announcements)
4. Hardware specs MUST match official documentation
5. When in doubt, mark as unverified"""

        try:
            response = self.call_llm(prompt, temperature=0.1)
        except LLMError as e:
            print(f"  [!] LLM call failed during fact synthesis: {e}")
            return {
                "verified_facts": [],
                "unverified_claims": [],
                "general_knowledge": [],
                "unknowns": ["LLM call failed during synthesis"],
                "warning": "Research synthesis failed - Writer should avoid ALL specific claims"
            }

        result = extract_json_from_llm(response)
        if result is not None:
            return result
        return {
            "verified_facts": [],
            "unverified_claims": [],
            "general_knowledge": [],
            "unknowns": ["Failed to parse research results"],
            "warning": "Research synthesis failed - Writer should avoid ALL specific claims"
        }

    def _generate_constraints(self, fact_sheet: Dict) -> str:
        """Generate natural language constraints for the Writer."""

        lines = []
        lines.append("=" * 70)
        lines.append("FACT SHEET - YOUR ONLY SOURCE OF TRUTH")
        lines.append("=" * 70)
        lines.append("")

        # Verified facts
        verified = fact_sheet.get("verified_facts", [])
        if verified:
            lines.append("✅ VERIFIED FACTS - You MAY use these with confidence:")
            for f in verified:
                lines.append(f"   • {f['fact']}")
                lines.append(f"     Source: {f.get('source', 'N/A')} | Confidence: {f.get('confidence', 'N/A')}")
            lines.append("")
        else:
            lines.append("⚠️  NO VERIFIED FACTS AVAILABLE")
            lines.append("   Write with conviction but WITHOUT specific numbers or claims.")
            lines.append("")

        # Unverified claims - DO NOT USE
        unverified = fact_sheet.get("unverified_claims", [])
        if unverified:
            lines.append("❌ UNVERIFIED CLAIMS - DO NOT USE THESE:")
            for u in unverified:
                lines.append(f"   • {u['claim']}")
                lines.append(f"     Why: {u.get('reason', 'Could not verify')}")
            lines.append("")

        # General knowledge - safe to use
        general = fact_sheet.get("general_knowledge", [])
        if general:
            lines.append("📚 GENERAL KNOWLEDGE - Safe to state without citation:")
            for g in general:
                lines.append(f"   • {g}")
            lines.append("")

        # Unknowns
        unknowns = fact_sheet.get("unknowns", [])
        if unknowns:
            lines.append("❓ UNKNOWNS - We couldn't find this information:")
            for u in unknowns:
                lines.append(f"   • {u}")
            lines.append("")

        # Final instructions
        lines.append("=" * 70)
        lines.append("WRITING RULES:")
        lines.append("1. Specific numbers/specs → ONLY from verified facts above")
        lines.append("2. Claims not in this sheet → Write around them or use hedging")
        lines.append("3. If you need a fact we don't have → State opinion, not fake data")
        lines.append("4. When uncertain → 'Teams report...' not 'Studies show 40%...'")
        lines.append("=" * 70)

        return "\n".join(lines)

    def format_for_revision(self, fact_sheet: Dict, violations: list) -> str:
        """
        Generate dynamic revision prompt based on what violations were found.

        This is the KEY to making the pipeline dynamic - the revision prompt
        changes based on WHAT failed, not generic instructions.
        """
        lines = []
        lines.append("REVISION REQUIRED - Here's exactly what to fix:")
        lines.append("")

        # Group violations by type
        for v in violations:
            vtype = v.get("type", "UNKNOWN")

            if vtype == "FAKE_METRIC":
                lines.append(f"❌ FAKE METRIC DETECTED: '{v.get('match', '')}'")
                lines.append(f"   This metric doesn't exist. {v.get('reason', '')}")
                lines.append(f"   FIX: Remove entirely OR replace with real metric from fact sheet")
                lines.append("")

            elif vtype == "SUSPICIOUS_SPEC":
                lines.append(f"❌ SUSPICIOUS SPECIFICATION: '{v.get('match', '')}'")
                lines.append(f"   {v.get('reason', 'Could not verify this claim')}")
                lines.append(f"   FIX: Check fact sheet for verified specs. If not there, remove.")
                lines.append("")

            elif vtype == "FABRICATED_STATS":
                lines.append(f"❌ FABRICATED STATISTIC: '{v.get('match', '')}'")
                lines.append(f"   No source for this number.")
                lines.append(f"   FIX: Remove the number. Write with conviction without stats.")
                lines.append("")

            elif vtype == "PHANTOM_EVIDENCE":
                lines.append(f"❌ PHANTOM EVIDENCE: '{v.get('phrase', '')}'")
                lines.append(f"   Vague appeal to authority without source.")
                lines.append(f"   FIX: Either cite specific source OR rephrase as opinion/observation")
                lines.append("")

        # Add the fact sheet as reference
        lines.append("")
        lines.append("REFERENCE - Use ONLY these verified facts:")
        lines.append(fact_sheet.get("writer_constraints", "No fact sheet available"))

        return "\n".join(lines)
