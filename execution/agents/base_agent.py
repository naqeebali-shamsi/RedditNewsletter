"""
Shared base class for all agents.
Handles LLM interaction and common utilities.
"""

import os
import time

try:
    from google import genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class BaseAgent:
    def __init__(self, role, persona, model="gemini-2.0-flash-exp"):
        self.role = role
        self.persona = persona
        self.model_name = model
        self.provider = "unknown"
        self._get_api_key()
        self._setup_client()

    def _get_api_key(self):
        # Priority: Groq -> Vertex (Project) -> Gemini (Key) -> OpenAI
        if os.getenv("GROQ_API_KEY"):
            self.provider = "groq"
            self.api_key = os.getenv("GROQ_API_KEY")
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            self.provider = "gemini"
            self.api_key = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
            self.vertexai = True
        elif os.getenv("GOOGLE_API_KEY"):
            self.provider = "gemini"
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.vertexai = False
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = None

    def _setup_client(self):
        if self.provider == "groq" and OpenAI:
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )
            self.client_type = "groq"
        elif self.provider == "gemini" and genai:
            if getattr(self, 'vertexai', False) and not self.api_key:
                # Use ADC for Vertex if no key provided
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location
                )
            else:
                # Use API Key (optionally with Vertex)
                self.client = genai.Client(
                    api_key=self.api_key,
                    vertexai=getattr(self, 'vertexai', False)
                )
            self.client_type = "gemini"
        elif self.provider == "openai" and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
            self.client_type = "openai"
        else:
            self.client_type = "unknown"

    def call_llm(self, prompt, system_instruction=None, temperature=0.7):
        """Standardized call to the underlying LLM."""
        if self.client_type == "unknown":
            return "Error: LLM client not configured."

        full_system_prompt = f"You are the {self.role}.\nPersona: {self.persona}\n"
        if system_instruction:
            full_system_prompt += f"\nSpecific Instructions:\n{system_instruction}"

        if self.client_type == "groq":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Groq Error: {str(e)}"

        elif self.client_type == "gemini":
            try:
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=full_system_prompt,
                        temperature=temperature
                    )
                )
                return response.text
            except Exception as e:
                return f"Gemini Error: {str(e)}"

        elif self.client_type == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"

        return "Error: Unsupported client type."
