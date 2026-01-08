"""
Imagen Client - Google Imagen Image Generation via Vertex AI

This module provides a clean interface to Google's Imagen models
for server-side image generation using Vertex AI SDK.

Requirements:
    pip install google-cloud-aiplatform Pillow

Usage:
    from execution.agents.nano_banana_client import NanoBananaClient

    client = NanoBananaClient()
    paths = client.generate_from_visual_plan(visual_plan)

Authentication:
    Uses Google Cloud Application Default Credentials (ADC).
    Run: gcloud auth application-default login
    Or set GOOGLE_APPLICATION_CREDENTIALS env var.

Available Models:
    - imagen-3.0-generate-001 (recommended)
    - imagen-3.0-fast-generate-001 (faster)
"""

import os
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field


class ImagenAPIError(Exception):
    """Base exception for Imagen API errors with full context."""
    def __init__(self, message: str, error_code: Optional[int] = None,
                 error_status: Optional[str] = None, raw_response: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        self.error_status = error_status
        self.raw_response = raw_response
        super().__init__(self.full_message)

    @property
    def full_message(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.error_status:
            parts.append(f"Status: {self.error_status}")
        if self.raw_response:
            parts.append(f"Raw: {self.raw_response[:500]}")
        return " | ".join(parts)


class BillingError(ImagenAPIError):
    """Raised when billing/quota issues occur."""
    pass


class RateLimitError(ImagenAPIError):
    """Raised when rate limits are hit."""
    pass


class SafetyFilterError(ImagenAPIError):
    """Raised when content is blocked by safety filters."""
    pass


class ModelNotFoundError(ImagenAPIError):
    """Raised when the requested model doesn't exist."""
    pass


@dataclass
class GenerationResult:
    """Result of a single image generation attempt."""
    success: bool
    file_path: Optional[Path] = None
    concept_name: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None  # billing, rate_limit, safety, model_not_found, unknown
    error_code: Optional[int] = None
    raw_error: Optional[str] = None
    simulated: bool = False
    attempts: int = 0


class NanoBananaClient:
    """
    Client for Google's Imagen image generation via Vertex AI.

    Supports:
        - Batch image generation from visual plans
        - Automatic retry with exponential backoff
        - Simulation mode when not authenticated
        - Windows-compatible pathlib operations
    """

    # Available Imagen models via Vertex AI (Jan 2026)
    MODEL_IMAGEN_4 = "imagen-4.0-generate-001"            # Standard quality (recommended)
    MODEL_IMAGEN_4_FAST = "imagen-4.0-fast-generate-001"  # Faster, lower latency
    MODEL_IMAGEN_4_ULTRA = "imagen-4.0-ultra-generate-001"  # Best quality (1 image at a time)
    MODEL_IMAGEN_3 = "imagen-3.0-generate-001"            # Legacy

    # Default to Imagen 4 standard
    DEFAULT_MODEL = MODEL_IMAGEN_4

    # GCP Project and Location
    DEFAULT_PROJECT = "ghostwriter-483610"
    DEFAULT_LOCATION = "us-central1"

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        simulate_if_no_key: bool = True
    ):
        """
        Initialize the Imagen client using Vertex AI.

        Args:
            project: GCP project ID. Defaults to ghostwriter-483610.
            location: GCP region. Defaults to us-central1.
            model: Model to use. Defaults to imagen-3.0-generate-001.
            output_dir: Directory to save images. Defaults to ./drafts/images/
            simulate_if_no_key: If True, return placeholder paths when not authenticated.
        """
        self.project = project or os.getenv("GCP_PROJECT", self.DEFAULT_PROJECT)
        self.location = location or os.getenv("GCP_LOCATION", self.DEFAULT_LOCATION)
        self.model = model or self.DEFAULT_MODEL
        self.simulate_if_no_key = simulate_if_no_key
        self._imagen_model = None

        # Use relative path by default, allow override
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Relative to project root
            self.output_dir = Path(__file__).parent.parent.parent / "drafts" / "images"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vertex AI client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Vertex AI client."""
        try:
            import vertexai
            from vertexai.preview.vision_models import ImageGenerationModel

            # Initialize Vertex AI
            vertexai.init(project=self.project, location=self.location)

            # Load the Imagen model
            self._imagen_model = ImageGenerationModel.from_pretrained(self.model)

            print(f"  [Imagen] Vertex AI initialized")
            print(f"           Project: {self.project}")
            print(f"           Location: {self.location}")
            print(f"           Model: {self.model}")

        except ImportError:
            if not self.simulate_if_no_key:
                raise ImportError(
                    "google-cloud-aiplatform not installed. "
                    "Run: pip install google-cloud-aiplatform"
                )
            print(f"  [Imagen] vertexai not installed, simulation mode")
            self._imagen_model = None

        except Exception as e:
            if not self.simulate_if_no_key:
                raise
            print(f"  [Imagen] Failed to init Vertex AI: {e}")
            print(f"  [Imagen] Running in simulation mode")
            self._imagen_model = None

    @property
    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode (no real API calls)."""
        return self._imagen_model is None

    def _parse_error(self, exception: Exception) -> Dict:
        """
        Parse an API exception and extract detailed error information.

        Returns dict with: error_type, error_code, error_status, message, raw
        """
        error_str = str(exception)
        error_repr = repr(exception)

        result = {
            "error_type": "unknown",
            "error_code": None,
            "error_status": None,
            "message": error_str,
            "raw": error_repr,
            "retryable": False
        }

        # Extract error code if present
        import re
        code_match = re.search(r'(\d{3})\s', error_str)
        if code_match:
            result["error_code"] = int(code_match.group(1))

        # Extract status if present
        status_match = re.search(r"status['\"]?\s*:\s*['\"]?(\w+)", error_str, re.IGNORECASE)
        if status_match:
            result["error_status"] = status_match.group(1)

        # Categorize error type
        error_upper = error_str.upper()

        if "billed users" in error_str.lower() or "billing" in error_str.lower():
            result["error_type"] = "billing"
            result["message"] = "Billing/quota issue - API requires billed account or quota exceeded"
            result["retryable"] = True  # Sometimes transient

        elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_upper or "quota" in error_str.lower():
            result["error_type"] = "rate_limit"
            result["message"] = "Rate limit or quota exceeded"
            result["retryable"] = True

        elif "SAFETY" in error_upper or "blocked" in error_str.lower():
            result["error_type"] = "safety"
            result["message"] = "Content blocked by safety filter"
            result["retryable"] = False

        elif "404" in error_str or "NOT_FOUND" in error_upper:
            result["error_type"] = "model_not_found"
            result["message"] = f"Model not found: {self.model}"
            result["retryable"] = False

        elif "401" in error_str or "UNAUTHENTICATED" in error_upper:
            result["error_type"] = "auth"
            result["message"] = "Authentication failed - check API key"
            result["retryable"] = False

        elif "403" in error_str or "PERMISSION_DENIED" in error_upper:
            result["error_type"] = "permission"
            result["message"] = "Permission denied - check API key permissions"
            result["retryable"] = False

        elif "500" in error_str or "INTERNAL" in error_upper:
            result["error_type"] = "server"
            result["message"] = "Google server error"
            result["retryable"] = True

        return result

    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        concept_name: str = "image",
        max_retries: int = 3,
        retry_delay: float = 5.0,
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
            GenerationResult with success status, file path, and detailed error info.
        """
        # Generate safe filename
        if not filename:
            safe_name = self._sanitize_filename(concept_name)
            timestamp = int(time.time())
            filename = f"{safe_name}_{timestamp}.png"

        output_path = self.output_dir / filename

        # Simulation mode - return early
        if self.is_simulation_mode:
            print(f"  [Imagen] SIMULATED: {concept_name}")
            print(f"           Prompt: {prompt[:60]}...")
            return GenerationResult(
                success=True,
                file_path=output_path,
                concept_name=concept_name,
                simulated=True,
                error="Simulation mode - Vertex AI not initialized",
                attempts=0
            )

        last_error_info = None

        # Real generation with retries using Vertex AI
        for attempt in range(max_retries):
            try:
                print(f"  [Imagen] Generating: {concept_name} (attempt {attempt + 1}/{max_retries})")

                # Generate image using Vertex AI Imagen model
                response = self._imagen_model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    add_watermark=False,
                )

                # Check for generated images
                if response.images and len(response.images) > 0:
                    # Save the image
                    response.images[0].save(str(output_path))
                    print(f"           SUCCESS: Saved to {output_path}")
                    return GenerationResult(
                        success=True,
                        file_path=output_path,
                        concept_name=concept_name,
                        attempts=attempt + 1
                    )

                # No image in response - log full response for debugging
                print(f"           WARNING: No image in response")
                print(f"           Response type: {type(response)}")
                return GenerationResult(
                    success=False,
                    concept_name=concept_name,
                    error="No image in API response",
                    error_type="empty_response",
                    raw_error=str(response),
                    attempts=attempt + 1
                )

            except Exception as e:
                # Parse error for detailed information
                error_info = self._parse_error(e)
                last_error_info = error_info

                # Log full error details
                print(f"           ERROR [{error_info['error_type']}]: {error_info['message']}")
                print(f"           Code: {error_info['error_code']} | Status: {error_info['error_status']}")
                print(f"           Raw: {str(e)[:200]}")

                # Don't retry non-retryable errors
                if not error_info["retryable"]:
                    print(f"           Non-retryable error, stopping.")
                    return GenerationResult(
                        success=False,
                        concept_name=concept_name,
                        error=error_info["message"],
                        error_type=error_info["error_type"],
                        error_code=error_info["error_code"],
                        raw_error=error_info["raw"],
                        attempts=attempt + 1
                    )

                # Retry with exponential backoff + jitter
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"           Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

        # All retries exhausted
        print(f"           FAILED: Max retries ({max_retries}) exceeded")
        return GenerationResult(
            success=False,
            concept_name=concept_name,
            error=last_error_info["message"] if last_error_info else "Max retries exceeded",
            error_type=last_error_info["error_type"] if last_error_info else "max_retries",
            error_code=last_error_info["error_code"] if last_error_info else None,
            raw_error=last_error_info["raw"] if last_error_info else None,
            attempts=max_retries
        )

    def generate_from_visual_plan(
        self,
        visual_plan: List[Dict],
        delay_between: float = 3.0,
        stop_on_error: bool = False
    ) -> tuple[List[str], List[GenerationResult]]:
        """
        Generate images for all items in a visual plan.

        Args:
            visual_plan: List of dicts with keys like:
                - 'Concept Name' or 'concept_name'
                - 'Description' or 'description'
                - 'Prompt' or 'image_generation_prompt'
            delay_between: Seconds to wait between generations.
            stop_on_error: If True, stop on first error.

        Returns:
            Tuple of (successful_paths, all_results)
            - successful_paths: List of file paths for successfully generated images
            - all_results: List of GenerationResult objects with full error details
        """
        if not visual_plan:
            print("  [Imagen] No visuals in plan, skipping.")
            return [], []

        print(f"\n  [Imagen] Processing {len(visual_plan)} visuals...")
        print(f"           Model: {self.model}")
        print(f"           Delay between: {delay_between}s")

        if self.is_simulation_mode:
            print("  [Imagen] SIMULATION MODE (no API key)")

        successful_paths: List[str] = []
        all_results: List[GenerationResult] = []

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
                item.get("Image Generation Prompt") or  # LLM often uses this format
                item.get("Visual Description") or        # Alternative from prompt
                item.get("visual_description") or
                item.get("description") or
                item.get("Description") or
                ""
            )

            if not prompt:
                # Debug: show what keys were in the item
                print(f"           Skipping {concept_name}: No prompt found")
                print(f"           Available keys: {list(item.keys())}")
                all_results.append(GenerationResult(
                    success=False,
                    concept_name=concept_name,
                    error="No prompt found",
                    error_type="missing_prompt"
                ))
                continue

            print(f"\n  [{idx + 1}/{len(visual_plan)}] {concept_name}")

            # Generate image
            result = self.generate_image(
                prompt=prompt,
                concept_name=concept_name
            )
            all_results.append(result)

            if result.success and result.file_path:
                successful_paths.append(str(result.file_path))
            else:
                # Log detailed error info
                print(f"           FAILED: {result.error}")
                print(f"           Type: {result.error_type}")
                print(f"           Code: {result.error_code}")
                if result.raw_error:
                    print(f"           Raw: {result.raw_error[:300]}")

                if stop_on_error:
                    print(f"           Stopping due to stop_on_error=True")
                    break

            # Delay between requests (rate limiting protection)
            if idx < len(visual_plan) - 1 and not self.is_simulation_mode:
                print(f"           Waiting {delay_between}s before next request...")
                time.sleep(delay_between)

        # Summary
        print(f"\n  [Imagen] === SUMMARY ===")
        print(f"           Total: {len(visual_plan)}")
        print(f"           Success: {len(successful_paths)}")
        print(f"           Failed: {len(visual_plan) - len(successful_paths)}")

        # Log all failures with details
        failures = [r for r in all_results if not r.success]
        if failures:
            print(f"\n  [Imagen] === FAILURES ===")
            for f in failures:
                print(f"           - {f.concept_name}: [{f.error_type}] {f.error}")

        return successful_paths, all_results

    def _sanitize_filename(self, name: str) -> str:
        """Convert a concept name to a safe filename."""
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
        safe = safe.replace(" ", "_").lower()
        return safe[:50]


# Convenience function for quick usage
def generate_visuals(
    visual_plan: List[Dict],
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    return_details: bool = False
) -> List[str]:
    """
    Quick function to generate images from a visual plan.

    Args:
        visual_plan: List of visual dictionaries from VisualsAgent.
        output_dir: Optional custom output directory.
        model: Optional model override.
        return_details: If True, return tuple (paths, results) with full error info.

    Returns:
        List of generated image file paths.
        Or tuple (paths, results) if return_details=True.
    """
    client = NanoBananaClient(output_dir=output_dir, model=model)
    paths, results = client.generate_from_visual_plan(visual_plan)
    return (paths, results) if return_details else paths


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
