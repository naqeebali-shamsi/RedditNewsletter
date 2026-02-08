#!/usr/bin/env python3
"""
Orchestrator for the Medium Article Generation Pipeline.
Coordinates the multi-agent workflow: Editor -> Critic -> Writer -> Visuals.
"""

import sys
import os
import argparse
import datetime
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Centralized configuration
from execution.config import config, OUTPUT_DIR

from execution.agents.editor import EditorAgent
from execution.agents.critic import CriticAgent
from execution.agents.writer import WriterAgent
from execution.agents.base_agent import LLMError
from execution.agents.visuals import VisualsAgent
from execution.agents.gemini_researcher import GeminiResearchAgent
from execution.agents.perplexity_researcher import PerplexityResearchAgent
from execution.prompts.voice_templates import get_voice_prompt, EXTERNAL_VOICE_PROMPT, INTERNAL_VOICE_PROMPT

def main():
    parser = argparse.ArgumentParser(description="Generate a high-quality Medium article.")
    parser.add_argument("--topic", help="Topic or Signal to write about", required=True)
    parser.add_argument("--source-content", help="Original source content (Reddit post, etc.) for fact verification", default="")
    parser.add_argument("--source-type", choices=["external", "internal"], default="external",
                       help="Source type for voice selection: 'external' (observer) or 'internal' (owner)")
    args = parser.parse_args()

    # Get voice context based on source type
    voice_type = "Journalist Observer" if args.source_type == "external" else "Practitioner Owner"

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║            Medium Article Factory (Hybrid Agents)            ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Voice Mode: {voice_type:<47} ║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    # 1. Initialize Agents
    print("🤖 Initializing Agents...")
    editor = EditorAgent()       # Groq - fast inference
    critic = CriticAgent()       # Groq - fast inference
    writer = WriterAgent()       # Groq - fast inference
    visuals = VisualsAgent()     # Vertex AI - image generation

    # Grounded Research Agents - for REAL fact verification with web search
    # Priority: Gemini 3 Flash (Google Search) -> Perplexity Sonar (built-in citations) -> Pattern fallback
    research_agent = None
    research_agent_name = None

    # Try Gemini 3 Flash first (best Google Search integration)
    try:
        research_agent = GeminiResearchAgent(model="gemini-3-flash-preview")
        research_agent_name = "Gemini 3 Flash"
        print("   ✓ Research Agent: Gemini 3 Flash (Google Search Grounding)")
    except Exception as e:
        print(f"   ⚠️  Gemini unavailable: {e}")

        # Fallback to Perplexity Sonar Pro (excellent factuality, built-in citations)
        try:
            research_agent = PerplexityResearchAgent(model="sonar-pro")
            research_agent_name = "Perplexity Sonar Pro"
            print("   ✓ Research Agent: Perplexity Sonar Pro (Grounded Search)")
        except Exception as e2:
            print(f"   ⚠️  Perplexity unavailable: {e2}")
            print("   ⚠️  No grounded research available - will use pattern-based validation")

    print(f"   ✓ Agents Ready: Editor, Critic, Writer, Visuals" + (f", {research_agent_name}" if research_agent else ""))

    # 1.5 FACT RESEARCH PHASE (Grounded Web Search)
    fact_sheet = {}
    fact_constraints = ""

    if research_agent:
        print(f"\n🔍 [{research_agent_name}] Searching web for facts on: '{args.topic}'...")

        try:
            # Use grounded search for REAL fact verification
            fact_sheet = research_agent.research_topic(
                topic=args.topic,
                source_content=args.source_content
            )

            # Get constraints for the Writer
            fact_constraints = fact_sheet.get("writer_constraints", "")

            verified_count = len(fact_sheet.get("verified_facts", []))
            unverified_count = len(fact_sheet.get("unverified_claims", []))
            unknown_count = len(fact_sheet.get("unknowns", []))

            print(f"   ✓ Research Complete (via {research_agent_name}):")
            print(f"      Verified Facts: {verified_count}")
            print(f"      Unverified (DO NOT USE): {unverified_count}")
            print(f"      Unknowns: {unknown_count}")

            # Show search queries used (if available in metadata)
            grounding_meta = fact_sheet.get("grounding_metadata", {})
            perplexity_citations = fact_sheet.get("perplexity_citations", [])
            if grounding_meta.get("search_queries"):
                print(f"      Searches: {grounding_meta['search_queries'][:3]}")
            elif perplexity_citations:
                print(f"      Citations: {len(perplexity_citations)} sources")

            if verified_count == 0:
                print(f"   ⚠️  NO VERIFIED FACTS - Writer will avoid specific claims")

        except Exception as e:
            print(f"   ⚠️  Research failed: {e}")
            print(f"   ⚠️  Writer will proceed without fact sheet")
    else:
        print(f"\n⚠️  Skipping research phase (no research agent configured)")

    # 2. Strategy Phase
    print(f"\n🧠 [Editor] Planning article on: '{args.topic}'...")
    outline = editor.create_outline(args.topic)
    print(f"   ✓ Outline Created")

    print(f"\n🧐 [Critic] Reviewing outline...")
    critique = critic.critique_outline(outline)
    print(f"   ✓ Critique Received: {critique[:50]}...")

    print(f"\n🧠 [Editor] Refining outline based on critique...")
    try:
        refined_outline = editor.call_llm(f"Refine this outline based on the critique:\nTarget: {outline}\nCritique: {critique}")
    except LLMError as e:
        print(f"   ⚠️  Editor LLM failed: {e}. Using original outline.")
        refined_outline = outline
    print(f"   ✓ Outline Refined")

    # 3. Drafting Phase (Simplified for this script: Full Draft)
    # In a full production run, we would loop through sections.
    # Here we ask the writer to draft the full piece based on the refined outline.

    # Voice-aware drafting instructions
    if args.source_type == "external":
        voice_instruction = """
CRITICAL VOICE REQUIREMENT - JOURNALIST OBSERVER:
This content is sourced from external community discussions (Reddit/HN/Twitter).
You are an OBSERVER sharing learnings, NOT the person who did this work.

FORBIDDEN: "I", "we", "our", "my", "I built", "we discovered", "our team"
USE INSTEAD: "teams found", "engineers discovered", "the approach", "one developer's experience"

Write as a well-connected technology journalist sharing insights from the engineering community.
Use "you" to address the reader, but NEVER claim ownership of the work described.

Example: Instead of "I spent 3 weeks debugging" write "The team spent 3 weeks debugging"
Example: Instead of "Our approach reduced latency" write "This approach reduced latency"
"""
    else:
        voice_instruction = """
VOICE: PRACTITIONER OWNER
This is YOUR experience. Use ownership voice authentically.
Use "I", "we", "our" to share your personal engineering journey.
"""

    # Combine fact constraints with voice instruction for the Writer
    writer_context = f"""
{fact_constraints}

{voice_instruction}

REMEMBER: You can ONLY use specific numbers/specs from the FACT SHEET above.
If you need a fact not in the sheet, write with conviction but WITHOUT fabricating data.
"""

    print(f"\n✍️  [Writer] Generating First Draft ({voice_type})...")
    draft = writer.write_section(refined_outline, critique=f"Prepare the full draft.\n{writer_context}")
    print(f"   ✓ First Draft Complete ({len(draft)} chars)")

    # 3.5. VERIFICATION PHASE - Use grounded research to verify draft against real sources
    print(f"\n🔬 [Verification] Checking draft accuracy...")

    MAX_REVISIONS = 2
    for attempt in range(MAX_REVISIONS + 1):
        # Use grounded research agent to verify draft against web sources
        if research_agent:
            print(f"   🔍 [{research_agent_name}] Verifying claims via web search...")
            verification = research_agent.verify_draft(draft, topic=args.topic)

            accuracy_score = verification.get("overall_accuracy_score", 0)
            recommendation = verification.get("recommendation", "REJECT")
            false_claims = verification.get("false_claims", [])
            unverifiable = verification.get("unverifiable_claims", [])

            print(f"   📊 Accuracy Score: {accuracy_score}/100")
            print(f"   📋 Recommendation: {recommendation}")
            print(f"      False claims: {len(false_claims)}")
            print(f"      Unverifiable: {len(unverifiable)}")

            if recommendation == "PASS":
                print(f"   ✓ Verification PASSED")
                break

            if attempt < MAX_REVISIONS:
                print(f"   🔄 Revision {attempt + 1}/{MAX_REVISIONS}...")

                # Get specific revision instructions from Gemini
                revision_instructions = verification.get("revision_instructions", "")

                revision_prompt = f"""REVISE THIS DRAFT - Verification failed.

CURRENT DRAFT:
{draft}

{revision_instructions}

{fact_constraints}

CRITICAL RULES:
1. REMOVE or CORRECT all false claims listed above
2. REMOVE or HEDGE all unverifiable claims
3. DO NOT invent new numbers to replace removed ones
4. Use the FACT SHEET as your source of truth
5. Write with conviction but WITHOUT fabrication

Keep the same structure. Make it TECHNICALLY HONEST."""

                try:
                    draft = writer.call_llm(revision_prompt)
                    print(f"   ✓ Revision complete ({len(draft)} chars)")
                except LLMError as e:
                    print(f"   ⚠️  Writer LLM failed during revision: {e}. Keeping current draft.")
            else:
                print(f"   ⚠️  Max revisions reached. Proceeding with current draft.")

        else:
            # Fallback to Technical Supervisor (pattern-based) if Gemini unavailable
            from execution.agents.technical_supervisor import TechnicalSupervisorAgent
            tech_supervisor = TechnicalSupervisorAgent()

            tech_result = tech_supervisor.validate(draft, topic=args.topic)

            if tech_result["passed"]:
                print(f"   ✓ Technical validation PASSED (Score: {tech_result['score']}/100)")
                break

            print(f"   ⚠️  Issues found (Score: {tech_result['score']}/100)")

            if attempt < MAX_REVISIONS:
                print(f"   🔄 Revision {attempt + 1}/{MAX_REVISIONS}...")
                revision_prompt = f"""REVISE THIS DRAFT - Technical issues detected.

CURRENT DRAFT:
{draft}

ISSUES:
{tech_result['summary']}

{fact_constraints}

Remove fabricated stats. Fix or remove suspicious claims. Be honest."""

                try:
                    draft = writer.call_llm(revision_prompt)
                    print(f"   ✓ Revision complete ({len(draft)} chars)")
                except LLMError as e:
                    print(f"   ⚠️  Writer LLM failed during revision: {e}. Keeping current draft.")
            else:
                print(f"   ⚠️  Max revisions reached.")

    # 4. Specialist Refinement (Natural Voice Pipeline)
    from execution.agents.specialist import SpecialistAgent

    print(f"\n🔧 [Specialist] Refining Hook & Title...")
    hook_agent = SpecialistAgent(
        constraint_name="Hook Specialist",
        constraint_instruction="""Your job is to make readers STOP scrolling.

1. Rewrite the title to create curiosity or promise transformation.
   Good patterns: "Why I stopped X and started Y", "The X that nobody talks about", "What 3 years of X taught me about Y"
2. The opening line should feel like the start of a conversation, not a thesis statement.
3. The hook should make the reader think "this person gets it" within 10 seconds.

Keep the author's voice - don't make it sound like marketing copy."""
    )
    draft = hook_agent.refine(draft)
    print(f"   ✓ Hook & Title Optimized")

    print(f"\n🔧 [Specialist] Weaving Authentic Narrative...")

    # Voice-aware storytelling instructions based on source type
    if args.source_type == "external":
        story_instruction = """Your job is to make this feel like it's written by a REAL technology journalist sharing insights from the community, not an AI.

CRITICAL VOICE RULE: You are an OBSERVER, not the person who did this work.
- NEVER use: "we", "our", "my", "I built", "I created", "we discovered", "we found"
- USE INSTEAD: "teams found", "engineers discovered", "the approach", "the implementation", "one developer's experience"

1. Add 1-2 brief moments that show you've OBSERVED the community: "One team's frustration", "A pattern that keeps emerging", "What struck me about this discussion"
   Keep these SHORT (2-3 sentences max) and weave them naturally into transitions.
2. Use "you" to create conversation with the reader - "You've probably hit this wall" - but NOT "I" for ownership
3. The personality should feel like a well-connected journalist explaining what they've learned from watching the best engineers work.

Do NOT claim ownership of work you observed. If it sounds like you built it, rewrite it."""
    else:
        story_instruction = """Your job is to make this feel like it's written by a REAL engineer, not an AI.

1. Add 1-2 brief personal moments: a frustration, a realization, a late-night debugging session.
   Keep these SHORT (2-3 sentences max) and weave them naturally into transitions.
2. Use "I" and "you" to create conversation - "I've seen this pattern fail" or "You've probably hit this wall"
3. The personality should feel like a smart colleague explaining something over coffee, not a textbook.

Do NOT add fake-sounding stories. If it feels forced, cut it."""

    story_agent = SpecialistAgent(
        constraint_name="Storytelling Architect",
        constraint_instruction=story_instruction
    )
    draft = story_agent.refine(draft)
    print(f"   ✓ Authentic Narrative Added")

    print(f"\n🔧 [Specialist] Refining Voice & Tone...")

    # Voice-aware tone instructions based on source type
    if args.source_type == "external":
        voice_instruction = """Your job is to make this sound like ONE consistent, authentic technology journalist.

CRITICAL VOICE CHECK: Scan for and REMOVE any ownership language:
- "we", "our", "my" → Replace with "the team", "this approach", "the engineers"
- "I built", "I created" → Replace with "was built", "the implementation"
- "we discovered", "we found" → Replace with "teams discovered", "engineers found"

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
3. Vary sentence rhythm: mix punchy short sentences with longer explanations.
4. The tone should be: informed observer, confident analyst, technical but accessible.

Read it aloud - if it sounds like YOU built it (and you didn't), rewrite those parts."""
    else:
        voice_instruction = """Your job is to make this sound like ONE consistent, authentic person.

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
3. Vary sentence rhythm: mix punchy short sentences with longer explanations.
4. The tone should be: confident but not arrogant, technical but accessible, opinionated but fair.

Read it aloud - if it sounds like a robot wrote it, rewrite those parts."""

    voice_agent = SpecialistAgent(
        constraint_name="Voice & Tone Specialist",
        constraint_instruction=voice_instruction
    )
    draft = voice_agent.refine(draft)
    print(f"   ✓ Voice Refined")

    print(f"\n🔧 [Specialist] Ensuring Value Density...")
    density_agent = SpecialistAgent(
        constraint_name="Value Density Specialist",
        constraint_instruction="""Your job is to make every paragraph EARN its place.

1. Every section should leave the reader with something actionable or a new perspective.
2. Cut fluff, throat-clearing, and obvious statements. Get to the point faster.
3. Where appropriate, add concrete specifics: tools, numbers, code snippets, real examples.
4. The reader should finish thinking "I learned something useful" not "I read a lot of words"

Do NOT add bullet lists for the sake of it. Natural prose with clear takeaways > forced structure."""
    )
    draft = density_agent.refine(draft)
    print(f"   ✓ Value Density Maximized")

    # 4.5. Final Formatting & Polish
    print(f"\n✨ [Specialist] Final Polish...")

    # Voice-aware final polish instructions
    if args.source_type == "external":
        polish_instruction = """Final pass before publication. Your job is COHESION, CLEAN OUTPUT, and VOICE COMPLIANCE.

FINAL VOICE CHECK (CRITICAL for external sources):
- Scan for ANY remaining: "we", "our", "my", "I built", "I created", "we discovered"
- If found, replace with observer alternatives: "teams", "the approach", "engineers discovered"
- The voice should be JOURNALIST OBSERVER throughout - sharing learnings, not claiming ownership

1. Strip ALL internal labels, metadata markers, section tags, and any "Value-Bait:", "Hook:", etc. prefixes.
2. Format first line as # H1 Title (clean, no labels).
3. Ensure smooth transitions between sections - no jarring jumps.
4. Remove any repetitive phrases or ideas that got duplicated across specialist passes.
5. If there are forced bullet lists that interrupt flow, convert them to natural prose.
6. The final piece should read like ONE technology journalist wrote it - informed observer, not practitioner.

Output ONLY the polished markdown. No explanations, no meta-commentary."""
    else:
        polish_instruction = """Final pass before publication. Your job is COHESION and CLEAN OUTPUT.

1. Strip ALL internal labels, metadata markers, section tags, and any "Value-Bait:", "Hook:", etc. prefixes.
2. Format first line as # H1 Title (clean, no labels).
3. Ensure smooth transitions between sections - no jarring jumps.
4. Remove any repetitive phrases or ideas that got duplicated across specialist passes.
5. If there are forced bullet lists that interrupt flow, convert them to natural prose.
6. The final piece should read like ONE person wrote it in ONE sitting - not a committee.

Output ONLY the polished markdown. No explanations, no meta-commentary."""

    polisher = SpecialistAgent(
        constraint_name="Final Editor",
        constraint_instruction=polish_instruction
    )
    draft = polisher.refine(draft)
    print(f"   ✓ Formatting Cleaned")

    # 5. Final Editor Review (Ensuring cohesive flow)
    print(f"\n🧠 [Editor] Final Quality Gate...")
    review = editor.review_draft(draft, "Full Article Post-Specialists")
    
    if "REVISE" in review:
        print(f"   ⚠️  Revision Needed: {review}")
        try:
            draft = writer.call_llm(f"Apply these final editor corrections while keeping the specialists' work intact:\n{review}\n\nDraft:\n{draft}")
            print(f"   ✓ Final Revision Complete")
        except LLMError as e:
            print(f"   ⚠️  Final revision LLM failed: {e}. Keeping current draft.")
    else:
        print(f"   ✓ Draft Approved")

    # 6. Visuals Phase (Nano Banana Pro - Server-Side Generation)
    print(f"\n🎨 [Visuals] Designing Infographics...")
    visual_plan = visuals.suggest_visuals(draft)

    generated_image_paths = []

    if visual_plan and isinstance(visual_plan, list):
        print(f"   ✓ Planned {len(visual_plan)} visuals")

        # Option A: Server-side generation with Nano Banana Pro (recommended)
        # Uses GOOGLE_API_KEY from .env for direct image generation
        images_dir = OUTPUT_DIR / "images"
        generated_image_paths, generation_results = visuals.generate_all_visuals(
            visual_plan,
            output_dir=str(images_dir),
            return_details=True  # Get full error details
        )

        if generated_image_paths:
            print(f"   🍌 Generated {len(generated_image_paths)} images")
            for path in generated_image_paths:
                print(f"      📸 {path}")

        # Report failures with detailed error info
        failures = [r for r in generation_results if not r.success]
        if failures:
            print(f"\n   ⚠️  {len(failures)} image(s) failed:")
            for r in failures:
                print(f"      ❌ {r.concept_name}")
                print(f"         Type: {r.error_type}")
                print(f"         Error: {r.error}")
                if r.error_code:
                    print(f"         Code: {r.error_code}")
                if r.raw_error:
                    print(f"         Raw: {r.raw_error[:200]}...")

        # Option B: Fallback to Puter.js HTML dashboard (client-side, free)
        # Uncomment below if you prefer browser-based generation
        # from execution.puter_bridge import generate_puter_html
        # html_path = generate_puter_html(visual_plan, OUTPUT_DIR)
        # print(f"   🍌 Nano Banana Dashboard: {html_path}")
        # print(f"   👉 Open in browser for free client-side generation")

    else:
        print("   ⚠️  No visuals suggested.")

    # 7. Compilation
    print(f"\n📄 [Editor] Compiling final artifact...")
    
    lines = draft.split('\n')
    title = lines[0] if lines else "Untitled"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medium_full_{timestamp}.md"
    filepath = OUTPUT_DIR / filename

    # Build image references section
    images_section = ""
    if generated_image_paths:
        images_section = "\n## Generated Images\n"
        for i, path in enumerate(generated_image_paths, 1):
            # Create relative path for markdown
            rel_path = Path(path).relative_to(OUTPUT_DIR) if Path(path).is_relative_to(OUTPUT_DIR) else Path(path).name
            images_section += f"![Visual {i}](./{rel_path})\n\n"

    final_content = f"""
# {title}

**Generated by Multi-Agent Hybrid Pipeline**
*Signal: {args.topic}*
*Voice: {voice_type} (source: {args.source_type})*

---

{draft}

---
{images_section}
## Visuals Plan
{json.dumps(visual_plan, indent=2) if visual_plan else "None"}
"""
    
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(final_content)

    print(f"   📂 Output: {filepath}")

if __name__ == "__main__":
    main()
