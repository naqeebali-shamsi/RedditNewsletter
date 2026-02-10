# Directive: Writing Rules

> Unified voice, style, and quality standards for all content generation. Merges personal style, source-aware voice transformation, and framework writing rules.

## 1. Voice & Identity

### Target Audience
- Mid-Senior Engineers, Tech Leads, AI Engineers

### Positioning
- **Expert Pragmatist** in AI engineering (production LLMs, Claude workflows, tech debt)
- Deep technical knowledge + real-world practicality
- Authentic, direct, battle-scarred engineer perspective

### Ownership
- Use "I/We" for internal work and experiences
- Avoid: Academic fluff, passive voice, generic AI-generated content

---

## 2. Source-Aware Voice Transformation

Automatically adjust narrative voice based on content source to maintain authenticity. We never claim ownership of work we didn't do, but we also don't break the narrative by citing sources explicitly.

### Source Type Classification

| Source | source_type | Voice Style |
|--------|-------------|-------------|
| Reddit, HN, Twitter | `external` | Journalist Observer |
| GitHub commits, Internal docs | `internal` | Practitioner Owner |
| Mixed/Unknown | `external` | Journalist Observer (safe default) |

### External Voice (Journalist Observer)

Use when `source_type == "external"` -- content is from Reddit, Hacker News, Twitter, or other external sources.

**Forbidden Words & Phrases** (never use in external-sourced content):

```
# Ownership pronouns
we, We, WE
our, Our, OUR
my, My, MY

# Ownership claims
I built, I created, I developed, I designed, I implemented
we built, we created, we developed, we designed, we implemented
we discovered, we found, we learned, we realized
our team, our project, our system, our approach
my team, my project, my system, my approach
```

**Allowed Alternatives:**

| Instead of... | Use... |
|---------------|--------|
| we found | teams have found |
| we discovered | engineers discovered |
| our approach | one approach / this approach |
| my method | this method |
| we reported | developers reported |
| our team | the team / a team |
| our project | the project |
| I built | was built / the implementation |
| we learned | the lesson learned |
| my experience | one engineer's experience |

**Tone Guidelines (External):**
- **Role**: Authoritative technology reporter sharing insights from the community
- **Framing**: "Here's what I've learned from observing..." not "Here's what I built..."
- **Authority**: Confident and knowledgeable, but as an informed observer
- **Examples**:
  - "A team recently shared how they tackled this problem..."
  - "One engineer's experience shows that..."
  - "The approach that emerged from this discussion..."
  - "What's striking about this solution is..."

### Internal Voice (Practitioner Owner)

Use when `source_type == "internal"` -- content is from your own GitHub commits, internal documentation, or original work.

**Encouraged Words & Phrases:**

```
# Ownership pronouns (use freely)
we, our, my, I

# Ownership claims (encouraged)
I built, I created, we developed
we discovered, we learned
our team, our project
my experience building...
```

**Tone Guidelines (Internal):**
- **Role**: Battle-scarred engineer sharing war stories
- **Framing**: "Here's what we built and what we learned the hard way..."
- **Authority**: Confident practitioner with firsthand experience
- **Examples**:
  - "I still remember the day our pipeline crashed..."
  - "We spent three weeks debugging this issue..."
  - "Our approach was unconventional, but it worked..."
  - "What I learned from building this system..."

### Quality Invariants (Both Voices)

Regardless of voice type, every article must preserve:

1. **Emotional Hooks**: Stories need tension, drama, relatable moments
2. **Specific Metrics**: Numbers add credibility ("reduced latency by 50%")
3. **Actionable Takeaways**: Reader should learn something useful
4. **Technical Accuracy**: Facts must be correct
5. **Engaging Tone**: Never dry or academic

### Voice Implementation

- **Detection**: `source_type` field is set at signal ingestion (`fetch_reddit.py`, `fetch_github.py`)
- **Missing/unknown** `source_type` defaults to `"external"` (safe choice)
- **Application**: Voice rules are applied in prompt templates during generation; `generate_drafts.py` and `generate_medium_full.py` select prompts based on `source_type`
- **Validation**: `validate_voice.py` scans output for forbidden pronouns; violations flagged with line numbers and suggestions; run after generation, before publishing

---

## 3. Authentic & Direct Style

- Include real experiences: "I built," "I broke," "Pager duty," "Context leak" (internal voice only)
- **Target: 80% Technical, 20% Narrative**
  - Narrative breakdown: 1-2 sentences of authentic struggle (war story) + technical context/setup
  - For 300-word LinkedIn post: ~60 words narrative (15-20 words war story + 40-45 words context)
  - Rest is technical content: explanations, patterns, takeaways
- Weave in short, relatable anecdotes (2-4 sentences) at starts/transitions
  - *The Hero's Journey*: "2 AM, pager explodes -- our RAG pipeline just choked on 10GB docs. Here's the fix."
  - *Emotional Pull*: Limit to 10-15% of word count. Keep it 80% technical.

---

## 4. Aggressive Hooks

- No boring intros. Never open with "This article is about..."
- Challenge common industry beliefs or failures
- **Formula**: [Common Amateur Mistake] vs. [Senior Architectural Reality]
- **Example**: "Prompting is for prototypes. Orchestration is for production."

### Hook Types
- **Curiosity Gaps**: "The LLM hack that cut our token costs 70% -- without fine-tuning."
- **Provocative Questions**: "Is your RAG setup leaking context? 3 symptoms you're ignoring."
- **Bold Claims**: Direct statements challenging conventional wisdom

### Handling Controversial Topics
- Challenge status quo only when you have evidence or experience to back it up
- If the status quo is correct, find a different angle: implementation details, edge cases, or common misunderstandings
- Avoid contrarianism for its own sake -- authenticity over controversy
- When challenging established best practices, provide strong evidence (metrics, war stories, production data)

### Ethical "Value-Bait" Titles
- **Formula**: [Number] + [Pain/Trend] + [Contrarian Promise]
- **Good**: "7 RAG Mistakes Killing Your LLM (And Fixes That Work)"
- No sensationalist clickbait ("You won't believe..."). Cut clickbait if it lacks specific value.

---

## 5. Zero Fluff Policy

If an AI could write it generically, cut it. Every sentence must add value.

### Examples of Generic Content to CUT

- "In this post, we will explore..." (generic intro)
- "Furthermore, it is important to note..." (generic transition)
- "In conclusion, we have seen that..." (generic conclusion)
- "It is worth mentioning that..." (generic filler)
- "As we can see from the above..." (generic reference)
- "This is a common problem that many developers face..." (generic statement)
- "In today's fast-paced world..." (academic intro)

### Examples of Valuable Content to KEEP

- "I spent 3 hours debugging this because..." (specific, personal)
- "The gotcha: this fails silently when..." (specific, actionable)
- "Tradeoff: Higher latency vs. zero hallucinations" (specific, technical)
- "In production, we saw 40% latency reduction after..." (specific, data-driven)
- "The edge case: this breaks when X and Y occur simultaneously" (specific, technical detail)

### Rule of Thumb

If the sentence could appear in any generic article about the topic, cut it. If it's specific to your experience, data, or edge case, keep it.

---

## 6. Structure & Formatting

### The Opening (First 10-12 lines)

1. **Hook** (Curiosity gap or bold claim)
2. **The Scene** (Short anecdote, personal struggle)
3. **The Pivot** ("You're not alone")
4. **TL;DR** (Bullet summary)

### Length Guidelines

- **LinkedIn Posts**: Max 300 words (Byte-sized)
- **Articles**: Variable, but maintain takeaway density

### Paragraph Structure

- **LinkedIn Posts (300 words)**: Max 3 lines per paragraph (~120-150 words, 2-3 sentences)
- **Articles (500-2000 words)**: Max 4-5 lines per paragraph (~150-200 words, 3-4 sentences)
- **Long-form (2000+ words)**: Max 6 lines per paragraph (~200-250 words), allow longer for flow
- Max 2 sentences per bullet
- Format: **[The Mechanic]**: [The Consequence/Value] (for takeaway bullets, not headers)

### No Meta-Labeling

- DO NOT use functional labels in headers (e.g., "### The Hook", "### The Pivot")
- Headers should be narrative or informative and contribute to knowledge
- Avoid structural labels that break the reading flow
- **Note**: The format "**[The Mechanic]**: [The Consequence/Value]" is for **takeaway bullets**, not headers

---

## 7. Technical Standards

### Takeaway Density

- Minimum 3 high-value technical takeaways per article
- Use bold lists and "The Rule:" callouts
- Every section must answer "What should I do?"
- **High-value takeaway criteria** (must meet at least 2 of 3):
  - **Actionable**: Reader can apply it immediately
  - **Insightful**: Reveals non-obvious pattern, gotcha, or edge case
  - **Specific**: Not generic advice, but concrete technique/pattern with context
- Quality over quantity: 2 excellent takeaways beats 3 mediocre ones

### Tradeoff Perspective

- **State tradeoffs when they exist and are meaningful**
- Not every decision requires a tradeoff -- some approaches are clearly superior
- When tradeoffs exist, state the opportunity cost or architectural tradeoff
- Avoid false equivalencies -- some decisions are just better in specific contexts
- Example: "Tradeoff: Higher initial design latency vs. zero production hallucinations"

### War Stories

- Include 1-2 sentences of authentic struggle per article/post
- Use keywords: "I built," "I broke," "Pager duty," "Context leak"
- **Voice by experience level**:
  - **First-hand experience**: Use "I built," "I broke" (preferred)
  - **Second-hand (team/company)**: Use "We built," "We encountered," "Our team faced"
  - **Research-based**: Use "In production systems," "Common failure mode," acknowledge sources
  - **Never**: Fake first-person experience -- authenticity over forced personalization
- **Authenticity Guidelines**:
  - Use war stories authentically, not as required tokens
  - Variety in experience types: not every story needs to be about production failures
  - Balance: some articles can be purely technical without personal struggle
  - Avoid overuse of same phrases -- vary your authentic voice

### Humor & Wordplay

- Dose sparingly: exactly 1-2 per post
- **Target**: Tech puns ("Import ant"), analogies ("APIs are like restaurant menus"), or dry self-deprecating quips
- **Tone**: Avoid sarcasm; simple humor helps fight technical monotony

---

## 8. Voice Check

- **Minimize passive voice**: Use active voice for actions with clear actors
- **Passive voice acceptable for**: System behaviors, processes, or when the actor is irrelevant (e.g., "The cache is invalidated when..." is acceptable)
- No academic fluff
- Must sound like a battle-scarred engineer, not an AI bot
- Use war stories authentically, not as required tokens -- variety in experience types

---

## 9. Forbidden Patterns

- Passive voice ("Mistakes were made")
- Academic intros ("In today's fast-paced world...")
- Walls of text (>4 lines for LinkedIn, >5 lines for articles)
- Generic advice (If ChatGPT could write it, cut it)
- Sensationalist clickbait ("You won't believe..."). Cut clickbait if it lacks specific value.
- Filler transitions: "In this post," "Furthermore," "Transitioning to"

---

## 10. Tone Customization

The rules above define the **default** voice (Expert Pragmatist). The tone system (`directives/tone_system.md`) allows switching to alternative voices while preserving core quality standards.

### How Tone Presets Interact with Writing Rules

- **Tone presets override voice-specific rules** -- Each preset carries its own forbidden phrases, war story keywords, vocabulary preferences, formality level, hook style, and CTA style. When a preset is active, its rules replace the corresponding defaults from this document.
- **Quality invariants are preserved** -- Regardless of tone, all content must maintain: emotional hooks, specific metrics, actionable takeaways, technical accuracy, and engaging tone (Section 2: Quality Invariants).
- **Source-aware voice still applies** -- The external/internal voice distinction (Section 2) operates independently of tone presets. A "Conversational Engineer" tone still uses observer voice for external-sourced content.
- **Zero fluff policy is universal** -- No preset relaxes the zero fluff policy (Section 5). Every sentence must add value regardless of tone.
- **Structure rules flex by preset** -- Paragraph length, sentence length targets, and burstiness targets come from the active ToneProfile's `sentence_style` settings rather than the fixed rules in Section 6.

### Available Presets

Six built-in presets: Expert Pragmatist (default), Thought Leader, Technical Deep Dive, Conversational Engineer, News Reporter, Contrarian Challenger. Custom profiles can be inferred from writing samples. See `directives/tone_system.md` for full details.

---

## 11. Self-Annealing Principle

The writing rules must be updated immediately whenever a new constraint or pattern is identified during development.

**Process for Rule Changes**:

1. **Document First**: Add issue to ISSUES.md with description and proposed solution
2. **Test on Sample**: Apply proposed change to sample content to verify improvement
3. **Update Rules**: Modify rule files with clear rationale
4. **Track Changes**: Document in SUMMARY.md with date, reason, and impact
5. **Update Skills**: Ensure all skills referencing the rule are updated
6. **Version Awareness**: Note significant rule changes in PROJECT.md if they affect core principles

---

## Voice Transformation Examples

### Before (Ownership Voice on External Source)
> "I still remember the day **our** RAG pipeline started hallucinating. **We** had built what **we** thought was a robust system, but **our** confidence was misplaced. **My** team and **I** spent weeks debugging..."

### After (Observer Voice on External Source)
> "The story of a RAG pipeline hallucinating is all too familiar. **A team** had built what **they** thought was a robust system, but **their** confidence was misplaced. **The engineers** spent weeks debugging..."

---

*Consolidated from: naqeebali_style.md, voice_rules.md, TWS WRITING_RULES.md*
