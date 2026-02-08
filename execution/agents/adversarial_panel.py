"""
Adversarial Expert Panel Agent.

Simulates a panel of world-class copywriting experts who ruthlessly
critique content before it's allowed to publish. Each expert brings
a different lens (conversion, brand, SEO, creative) to ensure
content meets elite standards.

The panel operates in rounds, with each expert providing:
- A score (1-10)
- Specific failures found
- Concrete fix instructions

Content must achieve a minimum average score to pass the quality gate.
"""

from .base_agent import BaseAgent, LLMError
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import re
import os
from abc import ABC, abstractmethod


@dataclass
class ExpertCritique:
    """Single expert's critique of content."""
    expert_name: str
    agency: str
    score: int  # 1-10
    verdict: str  # Brief verdict
    failures: List[str]  # Specific failures found
    fixes: List[str]  # Concrete fix instructions


@dataclass
class PanelVerdict:
    """Aggregated verdict from all experts."""
    average_score: float
    passed: bool
    expert_critiques: List[ExpertCritique]
    critical_failures: List[str]  # Failures mentioned by 2+ experts
    priority_fixes: List[str]  # Top 5 fixes to implement
    iteration: int
    verdict: str = "pending"  # "approved", "revision_needed", "rejected"
    wsj_checklist: Dict = None  # WSJ Four Showstoppers results
    expert_scores: Dict = None  # Model-keyed scores for pipeline

    def __post_init__(self):
        """Initialize computed fields."""
        if self.expert_scores is None:
            self.expert_scores = {}
            for c in self.expert_critiques:
                self.expert_scores[c.expert_name] = c.score

        if self.verdict == "pending":
            if self.passed:
                self.verdict = "approved"
            elif self.average_score < 4.0:
                self.verdict = "rejected"
            else:
                self.verdict = "revision_needed"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "average_score": self.average_score,
            "passed": self.passed,
            "verdict": self.verdict,
            "expert_scores": self.expert_scores,
            "critical_failures": self.critical_failures,
            "priority_fixes": self.priority_fixes,
            "iteration": self.iteration,
            "wsj_checklist": self.wsj_checklist or {},
            "expert_critiques": [
                {
                    "expert": c.expert_name,
                    "agency": c.agency,
                    "score": c.score,
                    "verdict": c.verdict,
                    "failures": c.failures,
                    "fixes": c.fixes
                }
                for c in self.expert_critiques
            ]
        }


class AdversarialPanelAgent(BaseAgent):
    """
    Multi-persona adversarial critique agent.

    Embodies world-class copywriting experts from:
    - Top agencies (Digital Commerce Partners, AWISEE, Feldman Creative)
    - Brand giants (Google, Apple, Zomato, Swiggy)
    - SEO specialists (Victorious, Compose.ly, Content Whale)
    - Creative houses (Serviceplan, Emirates Graphic)

    Each expert has distinct evaluation criteria and pet peeves.
    """

    # Minimum score to pass (out of 10)
    PASS_THRESHOLD = 7.0

    # Maximum review iterations before escalating
    MAX_ITERATIONS = 3

    # Expert panel definitions
    EXPERT_PANELS = {
        "agency": [
            {
                "name": "Digital Commerce Partners",
                "role": "Conversion Strategist",
                "focus": "Content that CONVERTS. Every word must earn its place.",
                "pet_peeves": [
                    "Weak CTAs ('What do you think?')",
                    "No clear value proposition",
                    "Generic openings",
                    "Template-driven sameness"
                ],
                "evaluation_criteria": [
                    "Hook strength (first 10 words)",
                    "CTA specificity and urgency",
                    "Value density per paragraph",
                    "Differentiation from competitors"
                ]
            },
            {
                "name": "AWISEE",
                "role": "B2B SaaS Conversion Expert",
                "focus": "Data-driven storytelling. Show me the PROOF.",
                "pet_peeves": [
                    "Claims without evidence",
                    "Placeholder text ('...')",
                    "Missing metrics/numbers",
                    "Vague 'practical implications'"
                ],
                "evaluation_criteria": [
                    "Data/metrics presence",
                    "Specificity of examples",
                    "Logical flow of argument",
                    "Actionable takeaways"
                ]
            },
            {
                "name": "Feldman Creative",
                "role": "B2B Clarity Specialist",
                "focus": "Extreme clarity. If a 12-year-old can't understand it, rewrite it.",
                "pet_peeves": [
                    "Jargon without explanation",
                    "Convoluted sentences",
                    "Passive voice overuse",
                    "Melodramatic openings"
                ],
                "evaluation_criteria": [
                    "Sentence clarity",
                    "Jargon-to-explanation ratio",
                    "Active voice usage",
                    "Reader empathy"
                ]
            }
        ],
        "brand": [
            {
                "name": "Google (Think with Google)",
                "role": "Search & UX Strategist",
                "focus": "User intent alignment. Does this answer what people actually search for?",
                "pet_peeves": [
                    "No keyword strategy visible",
                    "Headers that don't scan",
                    "Driving traffic to competitors",
                    "Missing meta/structure"
                ],
                "evaluation_criteria": [
                    "Search intent alignment",
                    "Scannability of headers",
                    "Internal value (not just links out)",
                    "Structured information hierarchy"
                ]
            },
            {
                "name": "Apple",
                "role": "Brand & Editorial Purist",
                "focus": "Simplicity is sophistication. One voice, one tone, one purpose.",
                "pet_peeves": [
                    "Tonal inconsistency",
                    "Visual chaos in layout",
                    "Multiple competing messages",
                    "Forced frameworks (DO/STOP/CHECK spam)"
                ],
                "evaluation_criteria": [
                    "Voice consistency throughout",
                    "Visual hierarchy clarity",
                    "Single clear message",
                    "Elegant simplicity"
                ]
            },
            {
                "name": "Zomato/Swiggy",
                "role": "Engagement & Virality Expert",
                "focus": "Personality and shareability. Would anyone screenshot this?",
                "pet_peeves": [
                    "Zero personality/wit",
                    "Nothing quotable",
                    "Same hashtags every post",
                    "No cultural relevance"
                ],
                "evaluation_criteria": [
                    "Memorable moments",
                    "Shareability factor",
                    "Brand personality presence",
                    "Engagement hooks"
                ]
            }
        ],
        "seo": [
            {
                "name": "Victorious",
                "role": "Technical SEO Authority",
                "focus": "Algorithmic visibility. If Google can't understand it, readers won't find it.",
                "pet_peeves": [
                    "No target keyword",
                    "Missing meta description",
                    "Broken H1/H2 hierarchy",
                    "No E-E-A-T signals"
                ],
                "evaluation_criteria": [
                    "Keyword presence and placement",
                    "Header hierarchy",
                    "Content depth for ranking",
                    "Authority signals"
                ]
            },
            {
                "name": "Compose.ly",
                "role": "Enterprise Quality Gate",
                "focus": "Production-ready quality. Would we deliver this to a Fortune 500 client?",
                "pet_peeves": [
                    "HTML/markdown artifacts",
                    "Incomplete sections",
                    "Duplicate boilerplate",
                    "No clear audience targeting"
                ],
                "evaluation_criteria": [
                    "Technical cleanliness",
                    "Section completeness",
                    "Content originality",
                    "Audience specificity"
                ]
            }
        ],
        "creative": [
            {
                "name": "Serviceplan",
                "role": "Award-Winning Creative Director",
                "focus": "Creative excellence. Would this win at Cannes? If not, why publish it?",
                "pet_peeves": [
                    "Machine-generated feel",
                    "No narrative arc",
                    "Algorithm metadata in content",
                    "Zero creative risk"
                ],
                "evaluation_criteria": [
                    "Creative originality",
                    "Narrative structure",
                    "Human editorial judgment",
                    "Memorable craft"
                ]
            },
            {
                "name": "Emirates Graphic",
                "role": "Global Brand Excellence",
                "focus": "Would this embarrass us in front of a discerning client?",
                "pet_peeves": [
                    "Scroll-past openings",
                    "Nothing to share/quote",
                    "No distinct voice",
                    "Generic across all markets"
                ],
                "evaluation_criteria": [
                    "Opening hook test",
                    "Share/quote worthiness",
                    "Voice distinctiveness",
                    "Global quality standard"
                ]
            }
        ]
    }

    # Kill phrases - instant failures
    KILL_PHRASES = [
        ("...", "PLACEHOLDER_TEXT", "Placeholder text found - incomplete section"),
        ("What's been your experience", "WEAK_CTA", "Generic weak CTA - no urgency or specificity"),
        ("This aligns with what I'm seeing", "TEMPLATE_PHRASE", "Template boilerplate detected"),
        ("In this article", "BORING_OPENER", "Boring opener - no hook"),
        ("<!-- SC_OFF", "HTML_ARTIFACT", "Raw HTML artifacts in content"),
        ("<div class=", "HTML_ARTIFACT", "Raw HTML artifacts in content"),
        ("Contains some technical content", "METADATA_LEAK", "Internal metadata leaked into content"),
        ("contains 2 technical keywords", "METADATA_LEAK", "Internal scoring leaked into content"),
        ("As AI engineering matures", "CLICHE_CLOSER", "Generic cliche closer"),
        ("Drop a comment below", "WEAK_CTA", "Weak engagement CTA"),
        ("#AIEngineering #MachineLearning #LLMOps #ProductionAI", "HASHTAG_SPAM", "Same hashtags on every post"),
    ]

    # WSJ Four Showstoppers - Critical editorial failures from Wall Street Journal standards
    # Reference: https://www.wsj.com/news/how-we-report-the-news
    WSJ_FOUR_SHOWSTOPPERS = {
        "inaccuracy": {
            "name": "Factual Accuracy",
            "description": "Every fact must be verified. Errors destroy credibility.",
            "checks": [
                "All statistics have cited sources",
                "Technical claims are verifiable",
                "Dates, names, and figures are correct",
                "No unverified claims presented as fact"
            ],
            "weight": 1.5  # Highest weight - inaccuracy is fatal
        },
        "unfairness": {
            "name": "Fairness & Balance",
            "description": "Present multiple perspectives. Avoid one-sided coverage.",
            "checks": [
                "Counterarguments acknowledged",
                "No straw-manning of opposing views",
                "Attribution is clear and fair",
                "Tone is professional, not partisan"
            ],
            "weight": 1.3
        },
        "disorganization": {
            "name": "Structure & Flow",
            "description": "Clear logical structure. Easy to follow.",
            "checks": [
                "Strong opening hook",
                "Logical section progression",
                "Clear transitions between ideas",
                "Satisfying conclusion/CTA"
            ],
            "weight": 1.0
        },
        "poor_writing": {
            "name": "Writing Quality",
            "description": "Clear, concise prose. No jargon without explanation.",
            "checks": [
                "Sentences are clear and direct",
                "No unnecessary jargon",
                "Active voice predominates",
                "Appropriate reading level for audience"
            ],
            "weight": 1.0
        }
    }

    # Multi-model routing configuration
    # Different LLMs excel at different review dimensions
    MODEL_ROUTING = {
        "ethics_fairness": {
            "model": "claude-sonnet-4-20250514",  # Claude excels at nuanced ethical reasoning
            "provider": "anthropic",
            "panels": ["brand"],  # Brand consistency, voice appropriateness
            "weight": 1.3
        },
        "accuracy_facts": {
            "model": "gemini-2.0-flash-exp",  # Gemini with grounding for fact-checking
            "provider": "google",
            "panels": ["seo"],  # Technical accuracy, SEO compliance
            "weight": 1.5
        },
        "structure_style": {
            "model": "gpt-4o",  # GPT-4o for structure and style
            "provider": "openai",
            "panels": ["agency", "creative"],  # Conversion, creativity
            "weight": 1.0
        }
    }

    def __init__(self, model="gemini-2.0-flash-exp", multi_model: bool = True):
        super().__init__(
            role="Adversarial Expert Panel Coordinator",
            persona="""You coordinate a panel of world-class copywriting experts.
Your job is to ensure NO substandard content ever gets published.
You are ruthless, specific, and constructive.
You identify exact failures and prescribe exact fixes.""",
            model=model
        )
        self.multi_model = multi_model
        self._model_clients = {}
        if multi_model:
            self._setup_model_clients()

    def _setup_model_clients(self):
        """Initialize clients for multi-model review."""
        # Anthropic (Claude) for ethics/fairness
        try:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                from anthropic import Anthropic
                self._model_clients["anthropic"] = Anthropic(api_key=anthropic_key)
        except Exception as e:
            pass  # Will fall back to default model

        # OpenAI (GPT-4o) for structure/style
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self._model_clients["openai"] = OpenAI(api_key=openai_key)
        except Exception as e:
            pass

        # Google (Gemini) already handled by base class

    def _call_model(self, prompt: str, provider: str, model: str, temperature: float = 0.3) -> str:
        """Call a specific model for review. Raises LLMError on total failure."""
        try:
            if provider == "anthropic" and "anthropic" in self._model_clients:
                response = self._model_clients["anthropic"].messages.create(
                    model=model,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            elif provider == "openai" and "openai" in self._model_clients:
                response = self._model_clients["openai"].chat.completions.create(
                    model=model,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content

            else:
                # Fall back to default Gemini model via base class
                return self.call_llm(prompt, temperature=temperature)

        except LLMError:
            raise  # Don't swallow LLM exceptions from the fallback
        except Exception:
            # Non-LLM error from anthropic/openai direct call; try default model
            return self.call_llm(prompt, temperature=temperature)

    def _get_model_for_panel(self, panel_name: str) -> tuple:
        """Get the appropriate model/provider for a panel."""
        if not self.multi_model:
            return ("gemini-2.0-flash-exp", "google", 1.0)

        for route_name, config in self.MODEL_ROUTING.items():
            if panel_name in config["panels"]:
                return (config["model"], config["provider"], config["weight"])

        # Default fallback
        return ("gemini-2.0-flash-exp", "google", 1.0)

    def _check_kill_phrases(self, content: str) -> List[tuple]:
        """Check for instant-failure phrases."""
        found = []
        content_lower = content.lower()
        for phrase, code, message in self.KILL_PHRASES:
            if phrase.lower() in content_lower:
                found.append((code, message, phrase))
        return found

    def check_wsj_showstoppers(self, content: str, topic: str = "") -> Dict:
        """
        Check content against WSJ Four Showstoppers framework.

        Returns a dict with pass/fail for each showstopper and overall status.
        """
        prompt = f"""You are a WSJ-caliber editorial quality reviewer.

Evaluate this content against the WSJ "Four Showstoppers" - the critical failures
that would prevent publication in a world-class publication.

CONTENT:
---
{content}
---

TOPIC CONTEXT: {topic or "General tech/AI content"}

Evaluate EACH showstopper category:

1. INACCURACY: Are there any factual errors, unverified claims, or misleading statistics?
2. UNFAIRNESS: Is the coverage one-sided? Are counterarguments acknowledged?
3. DISORGANIZATION: Is the structure confusing? Are transitions clear?
4. POOR WRITING: Is there jargon without explanation? Unclear sentences?

Respond in this EXACT JSON format:
{{
    "inaccuracy": {{
        "passed": true/false,
        "score": 1-10,
        "issues": ["issue1", "issue2"]
    }},
    "unfairness": {{
        "passed": true/false,
        "score": 1-10,
        "issues": ["issue1", "issue2"]
    }},
    "disorganization": {{
        "passed": true/false,
        "score": 1-10,
        "issues": ["issue1", "issue2"]
    }},
    "poor_writing": {{
        "passed": true/false,
        "score": 1-10,
        "issues": ["issue1", "issue2"]
    }},
    "overall_passed": true/false,
    "blocking_issues": ["critical issue that blocks publication"]
}}

A score of 7+ means passed. Any score below 7 is a fail for that category.
overall_passed is only true if ALL four categories pass.
Output ONLY valid JSON."""

        try:
            response = self.call_llm(prompt, temperature=0.2)
        except LLMError:
            # LLM unavailable — return conservative default
            return {
                "inaccuracy": {"passed": False, "score": 5, "issues": ["Could not evaluate (LLM error)"]},
                "unfairness": {"passed": False, "score": 5, "issues": ["Could not evaluate (LLM error)"]},
                "disorganization": {"passed": False, "score": 5, "issues": ["Could not evaluate (LLM error)"]},
                "poor_writing": {"passed": False, "score": 5, "issues": ["Could not evaluate (LLM error)"]},
                "overall_passed": False,
                "weighted_score": 5.0,
                "blocking_issues": ["WSJ evaluation failed (LLM error) - manual review required"]
            }

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())

                # Ensure all expected keys exist
                for key in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]:
                    if key not in result:
                        result[key] = {"passed": False, "score": 5, "issues": ["Not evaluated"]}

                # Calculate weighted overall score
                weighted_score = 0
                total_weight = 0
                for key, config in self.WSJ_FOUR_SHOWSTOPPERS.items():
                    if key in result:
                        weighted_score += result[key].get("score", 5) * config["weight"]
                        total_weight += config["weight"]

                result["weighted_score"] = round(weighted_score / total_weight, 1) if total_weight > 0 else 0
                result["overall_passed"] = result.get("overall_passed", result["weighted_score"] >= 7.0)

                return result
        except (json.JSONDecodeError, KeyError) as e:
            pass

        # Default failure response
        return {
            "inaccuracy": {"passed": False, "score": 5, "issues": ["Could not evaluate"]},
            "unfairness": {"passed": False, "score": 5, "issues": ["Could not evaluate"]},
            "disorganization": {"passed": False, "score": 5, "issues": ["Could not evaluate"]},
            "poor_writing": {"passed": False, "score": 5, "issues": ["Could not evaluate"]},
            "overall_passed": False,
            "weighted_score": 5.0,
            "blocking_issues": ["WSJ evaluation failed - manual review required"]
        }

    def _run_single_expert(
        self,
        content: str,
        expert: dict,
        platform: str,
        source_type: str = "external",
        model: str = None,
        provider: str = None
    ) -> ExpertCritique:
        """Run content through a single expert's evaluation using specified model."""

        # Voice context based on source type
        if source_type == "external":
            voice_context = """
CRITICAL VOICE REQUIREMENT (External Source - Observer Voice):
This content is sourced from Reddit/external community. The author did NOT build/create this work.
- FORBIDDEN: "I built", "we created", "our team", "my approach", "we discovered"
- MUST USE: "teams found", "engineers discovered", "this approach", "the implementation"
- "I" is ONLY allowed for observations: "I noticed", "I've been tracking"
- Any ownership claims ("I built", "we created") are INSTANT FAILURES
"""
        else:
            voice_context = """
VOICE NOTE (Internal Source - Practitioner Voice):
This content is from the author's own work. Ownership voice ("I", "we", "our") is appropriate.
"""

        prompt = f"""You are {expert['name']} - {expert['role']}.

YOUR FOCUS: {expert['focus']}
{voice_context}

YOUR PET PEEVES (instant point deductions):
{chr(10).join(f"- {p}" for p in expert['pet_peeves'])}

YOUR EVALUATION CRITERIA:
{chr(10).join(f"- {c}" for c in expert['evaluation_criteria'])}

---

CONTENT TO REVIEW ({platform}):

{content}

---

Evaluate this content HARSHLY. Be specific about failures.

Respond in this EXACT JSON format:
{{
    "score": <1-10>,
    "verdict": "<2-5 word verdict>",
    "failures": [
        "<specific failure 1>",
        "<specific failure 2>"
    ],
    "fixes": [
        "<concrete fix instruction 1>",
        "<concrete fix instruction 2>"
    ]
}}

SCORING GUIDE:
- 1-3: Unpublishable garbage
- 4-5: Major rewrites needed
- 6: Mediocre, needs work
- 7: Acceptable minimum
- 8: Good quality
- 9-10: Exceptional (rare)

Be brutally honest. Output ONLY valid JSON."""

        # Use multi-model routing if specified
        try:
            if model and provider:
                response = self._call_model(prompt, provider, model, temperature=0.3)
            else:
                response = self.call_llm(prompt, temperature=0.3)
        except LLMError:
            # LLM failed — return degraded critique instead of crashing
            return ExpertCritique(
                expert_name=expert['name'],
                agency=expert.get('role', 'Expert'),
                score=4,
                verdict="LLM error - assume mediocre",
                failures=["Could not reach LLM for review - content likely needs manual check"],
                fixes=["Review content manually for quality issues"]
            )

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return ExpertCritique(
                    expert_name=expert['name'],
                    agency=expert.get('role', 'Expert'),
                    score=min(10, max(1, int(data.get('score', 5)))),
                    verdict=data.get('verdict', 'No verdict'),
                    failures=data.get('failures', [])[:5],  # Max 5 failures
                    fixes=data.get('fixes', [])[:5]  # Max 5 fixes
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback critique if parsing fails
            return ExpertCritique(
                expert_name=expert['name'],
                agency=expert.get('role', 'Expert'),
                score=4,
                verdict="Parse error - assume mediocre",
                failures=["Could not parse expert response - content likely problematic"],
                fixes=["Review content manually for quality issues"]
            )

    def review_content(
        self,
        content: str,
        platform: str = "medium",
        panels: List[str] = None,
        iteration: int = 1,
        source_type: str = "external",
        include_wsj_check: bool = True,
        topic: str = ""
    ) -> PanelVerdict:
        """
        Run content through the adversarial expert panel with multi-model review.

        Args:
            content: The draft content to review
            platform: 'linkedin' or 'medium'
            panels: Which expert panels to use (default: all)
            iteration: Current iteration number (for tracking)
            source_type: 'external' (observer voice) or 'internal' (owner voice)
            include_wsj_check: Run WSJ Four Showstoppers check (default: True)
            topic: Topic context for WSJ check

        Returns:
            PanelVerdict with scores, failures, fixes, and WSJ checklist
        """

        panels = panels or ["agency", "brand", "seo", "creative"]

        # Step 1: Check for instant kill phrases
        kill_phrase_hits = self._check_kill_phrases(content)

        # Step 2: Run WSJ Four Showstoppers check
        wsj_result = None
        if include_wsj_check:
            wsj_result = self.check_wsj_showstoppers(content, topic)

        # Step 3: Run through expert panels with multi-model routing
        all_critiques: List[ExpertCritique] = []
        panel_weights: Dict[str, float] = {}

        for panel_name in panels:
            # Get model routing for this panel
            model, provider, weight = self._get_model_for_panel(panel_name)
            panel_weights[panel_name] = weight

            panel_experts = self.EXPERT_PANELS.get(panel_name, [])
            for expert in panel_experts:
                # Pass source_type and model routing
                critique = self._run_single_expert(
                    content, expert, platform, source_type,
                    model=model, provider=provider
                )
                # Tag critique with panel weight for aggregation
                critique.agency = f"{critique.agency} (weight: {weight})"
                all_critiques.append(critique)

        # Step 4: Weighted score aggregation
        if all_critiques:
            # Calculate weighted average based on panel weights
            total_weighted_score = 0
            total_weight = 0
            for c in all_critiques:
                # Extract weight from agency string or default to 1.0
                weight = 1.0
                for panel_name, w in panel_weights.items():
                    if panel_name in str(c.agency).lower():
                        weight = w
                        break
                total_weighted_score += c.score * weight
                total_weight += weight

            avg_score = total_weighted_score / total_weight if total_weight > 0 else 0
        else:
            avg_score = 0

        # Apply kill phrase penalties
        if kill_phrase_hits:
            penalty = len(kill_phrase_hits) * 1.5  # 1.5 point penalty per kill phrase
            avg_score = max(1, avg_score - penalty)

        # Apply WSJ showstopper penalty if any category failed
        if wsj_result and not wsj_result.get("overall_passed", True):
            # Penalty of 0.5 per failed showstopper
            failed_count = sum(1 for key in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]
                             if not wsj_result.get(key, {}).get("passed", True))
            avg_score = max(1, avg_score - (failed_count * 0.5))

        # Find critical failures (mentioned by 2+ experts)
        all_failures = []
        for c in all_critiques:
            all_failures.extend(c.failures)

        # Add WSJ blocking issues to failures
        if wsj_result and wsj_result.get("blocking_issues"):
            for issue in wsj_result["blocking_issues"]:
                all_failures.insert(0, f"[WSJ] {issue}")

        # Simple frequency count for critical failures
        failure_counts = {}
        for f in all_failures:
            f_normalized = f.lower()[:50]
            failure_counts[f_normalized] = failure_counts.get(f_normalized, 0) + 1

        critical_failures = [
            f for f in all_failures
            if failure_counts.get(f.lower()[:50], 0) >= 2
        ][:5]

        # Add kill phrase failures to critical
        for code, message, phrase in kill_phrase_hits:
            critical_failures.insert(0, f"[{code}] {message}")

        # Collect priority fixes (from highest-weight experts first)
        priority_fixes = []
        for c in sorted(all_critiques, key=lambda x: x.score):  # Lowest scores first
            for fix in c.fixes:
                if fix not in priority_fixes:
                    priority_fixes.append(fix)
                    if len(priority_fixes) >= 5:
                        break
            if len(priority_fixes) >= 5:
                break

        return PanelVerdict(
            average_score=round(avg_score, 1),
            passed=avg_score >= self.PASS_THRESHOLD,
            expert_critiques=all_critiques,
            critical_failures=critical_failures[:5],
            priority_fixes=priority_fixes[:5],
            iteration=iteration,
            wsj_checklist=wsj_result
        )

    def generate_fix_instructions(self, verdict: PanelVerdict) -> str:
        """
        Generate detailed fix instructions for the writer based on panel verdict.

        Returns a prompt that can be passed to the writer agent for revision.
        """

        instruction = f"""## REVISION REQUIRED (Iteration {verdict.iteration})

**Panel Score: {verdict.average_score}/10** (Need 7.0+ to pass)

### CRITICAL FAILURES TO FIX:
{chr(10).join(f"- {f}" for f in verdict.critical_failures)}

### PRIORITY FIXES (in order):
{chr(10).join(f"{i+1}. {f}" for i, f in enumerate(verdict.priority_fixes))}

### EXPERT BREAKDOWN:
"""

        for critique in verdict.expert_critiques:
            instruction += f"""
**{critique.expert_name}** ({critique.agency}): {critique.score}/10 - "{critique.verdict}"
- Failures: {'; '.join(critique.failures[:2]) if critique.failures else 'None specified'}
"""

        instruction += """
### REVISION RULES:
1. Address ALL critical failures - these are non-negotiable
2. Work through priority fixes in order
3. Do NOT add new problems while fixing old ones
4. Maintain the core message while improving delivery
5. The next review will be HARSHER if you don't show improvement

Output the COMPLETE revised content. No explanations, just the improved content.
"""

        return instruction

    def format_verdict_summary(self, verdict: PanelVerdict) -> str:
        """Format verdict for human-readable output."""

        status = "PASSED" if verdict.passed else "FAILED"

        summary = f"""
{'='*60}
ADVERSARIAL PANEL VERDICT - Iteration {verdict.iteration}
{'='*60}

OVERALL SCORE: {verdict.average_score}/10 [{status}]
{'Threshold: 7.0' if not verdict.passed else 'Quality gate passed!'}

"""
        # WSJ Four Showstoppers summary
        if verdict.wsj_checklist:
            wsj = verdict.wsj_checklist
            wsj_status = "PASSED" if wsj.get("overall_passed") else "FAILED"
            summary += f"""WSJ FOUR SHOWSTOPPERS: {wsj_status}
"""
            for key in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]:
                if key in wsj:
                    cat = wsj[key]
                    check = "+" if cat.get("passed") else "X"
                    name = self.WSJ_FOUR_SHOWSTOPPERS[key]["name"]
                    summary += f"  [{check}] {name}: {cat.get('score', 0)}/10\n"

            if wsj.get("blocking_issues") and wsj["blocking_issues"]:
                summary += f"  BLOCKING: {wsj['blocking_issues'][0]}\n"
            summary += "\n"

        summary += "EXPERT SCORES:\n"
        for c in verdict.expert_critiques:
            bar = "=" * c.score + "-" * (10 - c.score)
            summary += f"  {c.expert_name[:25]:<25} [{bar}] {c.score}/10\n"

        if verdict.critical_failures:
            summary += f"""
CRITICAL FAILURES:
{chr(10).join(f"  ! {f}" for f in verdict.critical_failures)}
"""

        if verdict.priority_fixes and not verdict.passed:
            summary += f"""
PRIORITY FIXES:
{chr(10).join(f"  {i+1}. {f}" for i, f in enumerate(verdict.priority_fixes))}
"""

        summary += "=" * 60

        return summary
