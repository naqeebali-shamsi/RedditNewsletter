"""
Agent package for The Agency pipeline.

Includes the Adversarial Expert Panel for quality-gated content production.
"""

from .base_agent import BaseAgent
from .editor import EditorAgent
from .critic import CriticAgent
from .writer import WriterAgent
from .specialist import SpecialistAgent
from .topic_researcher import TopicResearchAgent
from .visuals import VisualsAgent
from .commit_analyzer import CommitAnalysisAgent
from .adversarial_panel import AdversarialPanelAgent, PanelVerdict, ExpertCritique
from .technical_supervisor import TechnicalSupervisorAgent
from .fact_researcher import FactResearchAgent
from .gemini_researcher import GeminiResearchAgent
from .perplexity_researcher import PerplexityResearchAgent
from .fact_verification_agent import FactVerificationAgent, VerificationStatus, FactVerificationReport, verify_article_facts
from .style_enforcer import StyleEnforcerAgent, StyleScore

__all__ = [
    'BaseAgent',
    'EditorAgent',
    'CriticAgent',
    'WriterAgent',
    'SpecialistAgent',
    'TopicResearchAgent',
    'VisualsAgent',
    'CommitAnalysisAgent',
    'AdversarialPanelAgent',
    'PanelVerdict',
    'ExpertCritique',
    'TechnicalSupervisorAgent',
    'FactResearchAgent',
    'GeminiResearchAgent',
    'PerplexityResearchAgent',
    'FactVerificationAgent',
    'VerificationStatus',
    'FactVerificationReport',
    'verify_article_facts',
    'StyleEnforcerAgent',
    'StyleScore',
]
