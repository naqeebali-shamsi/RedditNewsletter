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

# Add project root to path (n:/RedditNews)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from execution.agents.editor import EditorAgent
from execution.agents.critic import CriticAgent
from execution.agents.writer import WriterAgent
from execution.agents.visuals import VisualsAgent

# Ensure output directory
OUTPUT_DIR = Path("n:/RedditNews/drafts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Generate a high-quality Medium article.")
    parser.add_argument("--topic", help="Topic or Signal to write about", required=True)
    args = parser.parse_args()

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║            Medium Article Factory (Hybrid Agents)            ║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    # 1. Initialize Agents
    print("🤖 Initializing Agents...")
    editor = EditorAgent()
    critic = CriticAgent()
    writer = WriterAgent()
    visuals = VisualsAgent()
    print("   ✓ Agents Ready: Editor, Critic, Writer, Visuals")

    # 2. Strategy Phase
    print(f"\n🧠 [Editor] Planning article on: '{args.topic}'...")
    outline = editor.create_outline(args.topic)
    print(f"   ✓ Outline Created")

    print(f"\n🧐 [Critic] Reviewing outline...")
    critique = critic.critique_outline(outline)
    print(f"   ✓ Critique Received: {critique[:50]}...")

    print(f"\n🧠 [Editor] Refining outline based on critique...")
    refined_outline = editor.call_llm(f"Refine this outline based on the critique:\nTarget: {outline}\nCritique: {critique}")
    print(f"   ✓ Outline Refined")

    # 3. Drafting Phase (Simplified for this script: Full Draft)
    # In a full production run, we would loop through sections. 
    # Here we ask the writer to draft the full piece based on the refined outline.
    
    print(f"\n✍️  [Writer] Generating First Draft...")
    draft = writer.write_section(refined_outline, critique="Prepare the full draft.")
    print(f"   ✓ First Draft Complete ({len(draft)} chars)")

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
    story_agent = SpecialistAgent(
        constraint_name="Storytelling Architect",
        constraint_instruction="""Your job is to make this feel like it's written by a REAL engineer, not an AI.

1. Add 1-2 brief personal moments: a frustration, a realization, a late-night debugging session.
   Keep these SHORT (2-3 sentences max) and weave them naturally into transitions.
2. Use "I" and "you" to create conversation - "I've seen this pattern fail" or "You've probably hit this wall"
3. The personality should feel like a smart colleague explaining something over coffee, not a textbook.

Do NOT add fake-sounding stories. If it feels forced, cut it."""
    )
    draft = story_agent.refine(draft)
    print(f"   ✓ Authentic Narrative Added")

    print(f"\n🔧 [Specialist] Refining Voice & Tone...")
    voice_agent = SpecialistAgent(
        constraint_name="Voice & Tone Specialist",
        constraint_instruction="""Your job is to make this sound like ONE consistent, authentic person.

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
3. Vary sentence rhythm: mix punchy short sentences with longer explanations.
4. The tone should be: confident but not arrogant, technical but accessible, opinionated but fair.

Read it aloud - if it sounds like a robot wrote it, rewrite those parts."""
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
    polisher = SpecialistAgent(
        constraint_name="Final Editor",
        constraint_instruction="""Final pass before publication. Your job is COHESION and CLEAN OUTPUT.

1. Strip ALL internal labels, metadata markers, section tags, and any "Value-Bait:", "Hook:", etc. prefixes.
2. Format first line as # H1 Title (clean, no labels).
3. Ensure smooth transitions between sections - no jarring jumps.
4. Remove any repetitive phrases or ideas that got duplicated across specialist passes.
5. If there are forced bullet lists that interrupt flow, convert them to natural prose.
6. The final piece should read like ONE person wrote it in ONE sitting - not a committee.

Output ONLY the polished markdown. No explanations, no meta-commentary."""
    )
    draft = polisher.refine(draft)
    print(f"   ✓ Formatting Cleaned")

    # 5. Final Editor Review (Ensuring cohesive flow)
    print(f"\n🧠 [Editor] Final Quality Gate...")
    review = editor.review_draft(draft, "Full Article Post-Specialists")
    
    if "REVISE" in review:
        print(f"   ⚠️  Revision Needed: {review}")
        # Call the model via one of the agents to merge feedback
        draft = writer.call_llm(f"Apply these final editor corrections while keeping the specialists' work intact:\n{review}\n\nDraft:\n{draft}")
        print(f"   ✓ Final Revision Complete")
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
        generated_image_paths = visuals.generate_all_visuals(
            visual_plan,
            output_dir=str(images_dir)
        )

        if generated_image_paths:
            print(f"   🍌 Generated {len(generated_image_paths)} images")
            for path in generated_image_paths:
                print(f"      📸 {path}")

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
