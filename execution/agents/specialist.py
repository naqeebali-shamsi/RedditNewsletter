from .base_agent import BaseAgent

class SpecialistAgent(BaseAgent):
    """
    A lightweight agent designed to enforce a SINGLE specific constraint.
    """
    def __init__(self, constraint_name, constraint_instruction, model="llama-3.1-8b-instant"):
        super().__init__(
            role=f"Specialist: {constraint_name}",
            persona=f"You are a specialized editor focused ONLY on: {constraint_name}.",
            model=model
        )
        self.constraint = constraint_instruction

    def refine(self, text):
        prompt = f"""
Input Text:
{text}

Your Task:
Refine the text above to strictly satisfy this constraint:
"{self.constraint}"

Rules:
- Do NOT rewrite the whole thing if it's not needed.
- Keep the original meaning.
- Only make changes relevant to the constraint.
- Return the full refined text.
"""
        return self.call_llm(prompt)
