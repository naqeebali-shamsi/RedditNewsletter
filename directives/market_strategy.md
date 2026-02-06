# AI Engineering Subreddit Strategy & Market Context

## Persona
- **Role**: Mid-level SWE transitioning to AI Engineer (2026).
- **Background**: 3.8 YOE (Crest Data Systems), Master's (Dalhousie), Ex-DevOps Intern, Ex-Full Stack (Dubai).
- **Goal**: Build authority on LinkedIn/Medium to secure a stable AI Engineering role.
- **Voice**: Practitioner, technical, authoritative but authentic (sharing lessons learned).

## Source Priority

| Priority | Source | Cost | Rate Limits | Best For |
|----------|--------|------|-------------|----------|
| 1 | HackerNews API | Free | Unlimited | Engineering trends, 65% negativity bias aligns with Contrast Hook |
| 2 | RSS Feeds | Free | Unlimited | Broad tech coverage (Lobsters, Dev.to, Hacker Noon, InfoQ) |
| 3 | Reddit API | Paid | Rate-limited | Deep community sentiment, niche practitioner discussions |
| 4 | GitHub Trending | Free | Rate-limited | Open source momentum, technical patterns |

### Pulse-Driven Content Selection
- Topics with 3+ cross-source mentions get priority
- HN + Reddit overlap = high-signal topic (engineer consensus)
- RSS-only topics = potential early signals (ahead of community discussion)
- Use `execution/pulse_aggregator.py` for daily trend summaries

## Subreddit Tier List

### S+ Tier (Highest Priority)
1. **r/LocalLLaMA** (45-50k): Practitioners, local deployment, quantization, hardware.
2. **r/LLMDevs** (4.9k): Hyper-specialized, LLMOps, prompt versioning, fine-tuning.
3. **r/LanguageTechnology** (44.7k): Professional NLP, theory + application.

### S Tier (High Priority)
1. **r/MachineLearning** (2.8M): Research implementations, paper discussions.
2. **r/deeplearning** (50-100k): Architecture-focused, neural network design.
3. **r/mlops** (50k+): Operations/Infra, deployment strategies.
4. **r/learnmachinelearning** (355k): Educational, career transitions.

## Content Themes
1. **Production Reality vs. Hype**: Failures in production, cost realities, contrarian takes.
2. **Technical Decision Frameworks**: vLLM vs TensorRT, Fine-tuning vs Prompting.
3. **Postmortems & Lessons**: "We built X wrong, here's why."
4. **Ecosystem Deep Dives**: Comparison of tools (Vector DBs, Inference engines).
5. **Scaling & Optimization**: Infrastructure decisions at scale.

## Success Metric
- **Signal**: Posts that offer concrete details, insights, or problems relevant to AI/Automation.
- **Noise**: Generic news, surface-level trends, "Is AI taking over?" speculation.
