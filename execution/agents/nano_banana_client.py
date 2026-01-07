"""
Imagen Client - Google Imagen 3/4 Image Generation

This module provides a clean interface to Google's Imagen models
for server-side image generation. Replaces the client-side Puter.js bridge.

Requirements:
    pip install google-genai>=1.52.0 Pillow

Usage:
    from execution.agents.nano_banana_client import NanoBananaClient

    client = NanoBananaClient()
    paths = client.generate_from_visual_plan(visual_plan)

Reference:
    https://ai.google.dev/gemini-api/docs/image-generation

Available Models (as of Jan 2026):
    - imagen-4.0-generate-001 (standard, recommended)
    - imagen-4.0-fast-generate-001 (faster, cheaper)
    - imagen-4.0-ultra-generate-001 (highest quality)
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of a single image generation attempt."""
    success: bool
    file_path: Optional[Path] = None
    concept_name: str = ""
    error: Optional[str] = None
    simulated: bool = False


class NanoBananaClient:
    """
    Client for Google's Imagen image generation API.

    Supports:
        - Batch image generation from visual plans
        - Automatic retry with exponential backoff
        - Simulation mode when API key is missing
        - Windows-compatible pathlib operations
    """

    # Available Imagen models (Jan 2026)
    # Note: Imagen 3 models appear deprecated - only Imagen 4 is available
    MODEL_IMAGEN_4 = "imagen-4.0-generate-001"            # Standard quality
    MODEL_IMAGEN_4_FAST = "imagen-4.0-fast-generate-001"  # Faster/cheaper
    MODEL_IMAGEN_4_ULTRA = "imagen-4.0-ultra-generate-001"  # Highest quality

    # Default to standard Imagen 4
    DEFAULT_MODEL = MODEL_IMAGEN_4

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        simulate_if_no_key: bool = True
    ):
        """
        Initialize the Imagen client.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            model: Model to use. Defaults to imagen-4.0-generate-001.
            output_dir: Directory to save images. Defaults to ./drafts/images/
            simulate_if_no_key: If True, return placeholder paths when no API key.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.simulate_if_no_key = simulate_if_no_key
        self._client = None

        # Use relative path by default, allow override
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Relative to project root
            self.output_dir = Path(__file__).parent.parent.parent / "drafts" / "images"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize client if API key available
        if self.api_key:
            self._init_client()
        elif not simulate_if_no_key:
            raise ValueError(
                "No API key provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _init_client(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            print(f"  [Imagen] Client initialized")
            print(f"           Model: {self.model}")
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai>=1.52.0"
            )

    @property
    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode (no real API calls)."""
        return self._client is None

    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        concept_name: str = "image",
        max_retries: int = 5,
        retry_delay: float = 10.0,
        aspect_ratio: str = "1:1"
    ) -> GenerationResult:
        """
        Generate a single image from a text prompt.

        Args:
            prompt: The image generation prompt.
            filename: Output filename (auto-generated if None).
            concept_name: Human-readable name for logging.
            max_retries: Number of retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
            aspect_ratio: Image aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4")

        Returns:
            GenerationResult with success status and file path.
        """
        # Generate safe filename
        if not filename:
            safe_name = self._sanitize_filename(concept_name)
            timestamp = int(time.time())
            filename = f"{safe_name}_{timestamp}.png"

        output_path = self.output_dir / filename

        # Simulation mode - return early without importing types
        if self.is_simulation_mode:
            print(f"  [Imagen] SIMULATED: {concept_name}")
            print(f"           Prompt: {prompt[:60]}...")
            return GenerationResult(
                success=True,
                file_path=output_path,
                concept_name=concept_name,
                simulated=True,
                error="Simulation mode - no API key configured"
            )

        # Import types only when actually needed (not in simulation mode)
        try:
            from google.genai import types
        except ImportError:
            return GenerationResult(
                success=False,
                concept_name=concept_name,
                error="google-genai package not installed"
            )

        # Real generation with retries
        for attempt in range(max_retries):
            try:
                print(f"  [Imagen] Generating: {concept_name} (attempt {attempt + 1}/{max_retries})")

                # Configure image generation
                # Note: Only "block_low_and_above" is supported for safety_filter_level
                image_config = types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_low_and_above",
                )

                response = self._client.models.generate_images(
                    model=self.model,
                    prompt=prompt,
                    config=image_config
                )

                # Check for generated images
                if response.generated_images and len(response.generated_images) > 0:
                    image_obj = response.generated_images[0].image
                    image_obj.save(str(output_path))
                    print(f"           Saved: {output_path}")
                    return GenerationResult(
                        success=True,
                        file_path=output_path,
                        concept_name=concept_name
                    )

                # No image in response
                print(f"           No image returned in response")
                return GenerationResult(
                    success=False,
                    concept_name=concept_name,
                    error="No image in API response"
                )

            except Exception as e:
                error_msg = str(e)
                print(f"           Error: {error_msg[:100]}")

                # Rate limiting - exponential backoff
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg.upper():
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"           Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Safety filter - don't retry
                if "SAFETY" in error_msg.upper() or "blocked" in error_msg.lower():
                    return GenerationResult(
                        success=False,
                        concept_name=concept_name,
                        error=f"Content blocked by safety filter"
                    )

                # Model not found - critical error, don't retry
                if "404" in error_msg or "NOT_FOUND" in error_msg.upper():
                    return GenerationResult(
                        success=False,
                        concept_name=concept_name,
                        error=f"Model not found: {self.model}. Try imagen-3.0-generate-002"
                    )

                # Other errors - retry with short delay
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                    continue

                return GenerationResult(
                    success=False,
                    concept_name=concept_name,
                    error=error_msg
                )

        return GenerationResult(
            success=False,
            concept_name=concept_name,
            error="Max retries exceeded"
        )

    def generate_from_visual_plan(
        self,
        visual_plan: List[Dict],
        delay_between: float = 2.0
    ) -> List[str]:
        """
        Generate images for all items in a visual plan.

        Args:
            visual_plan: List of dicts with keys like:
                - 'Concept Name' or 'concept_name'
                - 'Description' or 'description'
                - 'Prompt' or 'image_generation_prompt'
            delay_between: Seconds to wait between generations.

        Returns:
            List of file paths for successfully generated images.
        """
        if not visual_plan:
            print("  [Imagen] No visuals in plan, skipping.")
            return []

        print(f"\n  [Imagen] Processing {len(visual_plan)} visuals...")
        print(f"           Model: {self.model}")

        if self.is_simulation_mode:
            print("  [Imagen] SIMULATION MODE (no API key)")

        results: List[str] = []

        for idx, item in enumerate(visual_plan):
            # Extract fields (handle various key formats)
            concept_name = (
                item.get("Concept Name") or
                item.get("concept_name") or
                item.get("ConceptName") or
                f"visual_{idx + 1}"
            )

            prompt = (
                item.get("Prompt") or
                item.get("prompt") or
                item.get("image_generation_prompt") or
                item.get("description") or
                item.get("Description") or
                ""
            )

            if not prompt:
                print(f"           Skipping {concept_name}: No prompt found")
                continue

            # Generate image
            result = self.generate_image(
                prompt=prompt,
                concept_name=concept_name
            )

            if result.success and result.file_path:
                results.append(str(result.file_path))
            elif result.error:
                print(f"           Failed: {result.error}")

            # Delay between requests (rate limiting protection)
            if idx < len(visual_plan) - 1 and not self.is_simulation_mode:
                time.sleep(delay_between)

        print(f"  [Imagen] Generated {len(results)}/{len(visual_plan)} images")
        return results

    def _sanitize_filename(self, name: str) -> str:
        """Convert a concept name to a safe filename."""
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
        safe = safe.replace(" ", "_").lower()
        return safe[:50]


# Convenience function for quick usage
def generate_visuals(
    visual_plan: List[Dict],
    output_dir: Optional[str] = None,
    model: Optional[str] = None
) -> List[str]:
    """
    Quick function to generate images from a visual plan.

    Args:
        visual_plan: List of visual dictionaries from VisualsAgent.
        output_dir: Optional custom output directory.
        model: Optional model override.

    Returns:
        List of generated image file paths.
    """
    client = NanoBananaClient(output_dir=output_dir, model=model)
    return client.generate_from_visual_plan(visual_plan)


if __name__ == "__main__":
    # Quick test
    print("Testing Imagen Client...")

    client = NanoBananaClient(simulate_if_no_key=True)
    print(f"Simulation mode: {client.is_simulation_mode}")
    print(f"Model: {client.model}")
    print(f"Output dir: {client.output_dir}")

    test_plan = [
        {
            "Concept Name": "Test Image",
            "Prompt": "A simple red circle on white background, minimalist style"
        }
    ]

    paths = client.generate_from_visual_plan(test_plan)
    print(f"Results: {paths}")
