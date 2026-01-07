from .base_agent import BaseAgent
import json
from typing import List, Dict, Optional

# Import Imagen client for server-side generation
from .nano_banana_client import NanoBananaClient


class VisualsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Infographic Director",
            persona="""You are the Creative Director for visuals.
Your goal is to design 'Nano Banana' style infographics (High fidelity, modern).
You do NOT create charts in code. You create *descriptions* for a human or high-end AI artist.

Your output must be:
1. **Concept Name**: Catchy title for the graphic.
2. **Description**: Clear instruction for the artist.
3. **Prompt**: The exact prompt for Nano Banana/Flux (e.g., 'A minimalist flow diagram... vector style...').""",
            model="llama-3.1-8b-instant"
        )

    def suggest_visuals(self, article_text):
        """Suggest infographics based on the content density."""
        prompt = f"""
Analyze this article text:
{article_text[:3000]}... (truncated)

Find 2-3 complex concepts that need a 'Nano Banana' Infographic.
Focus on:
- Architectures
- decision trees ('Should I use X?')
- 'Before vs After' flows

For each, provide a JSON object with EXACTLY these keys:
- "concept_name": Catchy title for the graphic
- "description": Brief description of what the visual shows
- "prompt": The FULL image generation prompt for Imagen (high fidelity, modern infographic style, clean vector design, professional color palette)

Output format: JSON array. Example:
[
  {{
    "concept_name": "Data Flow Architecture",
    "description": "Shows how data moves through the system",
    "prompt": "Professional infographic showing data flow architecture, clean vector style, arrows connecting boxes labeled API, Database, Cache, modern blue and white color scheme, high fidelity, 4K quality"
  }}
]

IMPORTANT: The "prompt" field must contain a detailed image generation prompt, not just a description.
"""
        response = self.call_llm(prompt, temperature=0.5)

        # Handle empty response
        if not response or not response.strip():
            print(f"  [Visuals] LLM returned empty response")
            return []

        original_response = response  # Keep for debug

        # Strip markdown if present
        if "```json" in response:
            try:
                response = response.split("```json")[1].split("```")[0]
            except IndexError:
                print(f"  [Visuals] Failed to extract JSON from ```json block")
                print(f"  [Visuals] Raw response: {original_response[:500]}")
                return []
        elif "```" in response:
            try:
                response = response.split("```")[1].split("```")[0]
            except IndexError:
                print(f"  [Visuals] Failed to extract JSON from ``` block")
                print(f"  [Visuals] Raw response: {original_response[:500]}")
                return []

        # Try to find JSON array in response if not already clean
        response = response.strip()
        if not response.startswith("["):
            # Try to find JSON array anywhere in response
            import re
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                response = match.group(0)
            else:
                print(f"  [Visuals] No JSON array found in response")
                print(f"  [Visuals] Raw response: {original_response[:500]}")
                return []

        try:
            result = json.loads(response)
            if not isinstance(result, list):
                print(f"  [Visuals] JSON is not a list: {type(result)}")
                return []
            return result
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [Visuals] Failed to parse visual plan JSON: {e}")
            print(f"  [Visuals] Attempted to parse: {response[:300]}")
            print(f"  [Visuals] Original response: {original_response[:300]}")
            return []

    def generate_image(self, prompt, output_path):
        """
        Generate a single image using Nano Banana Pro.

        Args:
            prompt: Image generation prompt
            output_path: Path to save the image

        Returns:
            bool: True if successful
        """
        from pathlib import Path

        client = NanoBananaClient(simulate_if_no_key=True)
        result = client.generate_image(
            prompt=prompt,
            filename=Path(output_path).name,
            concept_name="Single Image"
        )
        return result.success

    def generate_all_visuals(
        self,
        visual_plan: List[Dict],
        output_dir: Optional[str] = None,
        return_details: bool = False
    ) -> List[str]:
        """
        Generate all images from a visual plan using Nano Banana Pro.

        This is the primary method for server-side batch image generation,
        replacing the Puter.js client-side approach.

        Args:
            visual_plan: List of visual dicts from suggest_visuals()
            output_dir: Optional custom output directory
            return_details: If True, returns tuple (paths, results) with full error details

        Returns:
            List of file paths for generated images
            Or tuple (paths, results) if return_details=True

        Example:
            visuals = VisualsAgent()
            plan = visuals.suggest_visuals(article_text)
            image_paths = visuals.generate_all_visuals(plan)

            # With error details:
            paths, results = visuals.generate_all_visuals(plan, return_details=True)
            for r in results:
                if not r.success:
                    print(f"Failed: {r.concept_name} - {r.error_type}: {r.error}")
        """
        if not visual_plan:
            print("  ⚠️ No visual plan provided")
            return ([], []) if return_details else []

        print(f"\n🍌 [Nano Banana Pro] Generating {len(visual_plan)} images...")

        # Use the nano_banana_client module
        client = NanoBananaClient(
            output_dir=output_dir,
            simulate_if_no_key=True
        )

        paths, results = client.generate_from_visual_plan(visual_plan)

        if client.is_simulation_mode:
            print("  ⚠️ Running in simulation mode (no API key)")
            print("     Set GOOGLE_API_KEY in .env for real generation")

        # Log any failures with detailed error info
        failures = [r for r in results if not r.success]
        if failures:
            print(f"\n  ⚠️ {len(failures)} image(s) failed to generate:")
            for r in failures:
                print(f"     - {r.concept_name}: [{r.error_type}] {r.error}")
                if r.error_code:
                    print(f"       Code: {r.error_code}")

        return (paths, results) if return_details else paths
