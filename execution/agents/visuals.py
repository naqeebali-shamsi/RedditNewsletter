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

For each, provide:
1. Concept Name
2. Visual Description
3. Image Generation Prompt (High fidelity, styling for 'Nano Banana')

Output format: JSON list.
"""
        response = self.call_llm(prompt, temperature=0.5)
        
        # Strip markdown if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [Visuals] Failed to parse visual plan JSON: {e}")
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
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate all images from a visual plan using Nano Banana Pro.

        This is the primary method for server-side batch image generation,
        replacing the Puter.js client-side approach.

        Args:
            visual_plan: List of visual dicts from suggest_visuals()
            output_dir: Optional custom output directory

        Returns:
            List of file paths for generated images

        Example:
            visuals = VisualsAgent()
            plan = visuals.suggest_visuals(article_text)
            image_paths = visuals.generate_all_visuals(plan)
        """
        if not visual_plan:
            print("  ⚠️ No visual plan provided")
            return []

        print(f"\n🍌 [Nano Banana Pro] Generating {len(visual_plan)} images...")

        # Use the nano_banana_client module
        client = NanoBananaClient(
            output_dir=output_dir,
            simulate_if_no_key=True
        )

        paths = client.generate_from_visual_plan(visual_plan)

        if client.is_simulation_mode:
            print("  ⚠️ Running in simulation mode (no API key)")
            print("     Set GOOGLE_API_KEY in .env for real generation")

        return paths
