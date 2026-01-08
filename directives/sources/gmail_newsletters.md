# Gmail Newsletter Integration Directive

## Overview

This directive defines how GhostWriter ingests and processes newsletters from Gmail as a premium content source. Newsletters are treated as **Tier S++** (higher trust than Reddit) due to human expert curation.

## Trust Tier System

| Tier | Name | Behavior | Examples |
|------|------|----------|----------|
| **A** | Curated | Auto-signal, skip evaluation | Simon Willison, Pragmatic Engineer |
| **B** | Semi-trusted | Light evaluation | General Substack newsletters |
| **C** | Untrusted | Full Signal/Noise evaluation | Unknown senders |
| **X** | Blocked | Never fetch | Spam, off-topic |

### Tier A Seed List (Recommended Defaults)

**AI/ML Newsletters:**
- TLDR AI (`dan@tldrnewsletter.com`)
- The Batch by DeepLearning.AI (`hello@deeplearning.ai`)
- Import AI (`newsletter@importai.net`)
- The Sequence (`hello@thesequence.ai`)
- AI Supremacy (`newsletter@aisupremacy.substack.com`)

**Engineering Newsletters:**
- The Pragmatic Engineer (`newsletter@pragmaticengineer.com`)
- ByteByteGo (`alex@bytebytego.com`)
- Software Lead Weekly (`oren@softwareleadweekly.com`)
- TLDR (`dan@tldrnewsletter.com`)

**LLM/GenAI Specific:**
- Simon Willison's Weblog (`simon@simonwillison.net`)
- Latent Space (`swyx@latent.space`)
- AI Explained (`newsletter@aiexplained.substack.com`)

## OAuth Setup Requirements

### Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Gmail API**:
   - Navigate to APIs & Services > Library
   - Search for "Gmail API"
   - Click Enable

4. Configure OAuth Consent Screen:
   - Go to APIs & Services > OAuth consent screen
   - Choose "External" user type
   - Fill in app name: "GhostWriter"
   - Add your email as test user

5. Create OAuth Credentials:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "OAuth client ID"
   - Application type: "Desktop app"
   - Download the JSON file
   - Save as `credentials_gmail.json` in project root

### Required Scopes

```
https://www.googleapis.com/auth/gmail.readonly
```

This is the **minimal privilege** scope. GhostWriter only reads emails - it cannot send, delete, or modify any messages.

## Fetching Strategy

### Gmail Label Filtering

Primary method: User creates a Gmail label (e.g., "Newsletters") and applies it to newsletter emails.

```python
# Query format
query = f"label:{label_name} after:{since_date}"
```

### Fallback: Sender Whitelist

If no label is configured, fetch emails from senders in `newsletter_senders` table where `is_active = 1`.

### Time Window

- Default: Last 7 days
- Configurable via `hours_lookback` config option

## Content Extraction Pipeline

1. **Fetch message list** by label (batch API call)
2. **Get message content** in MIME format
3. **Parse HTML body** (newsletters are typically HTML)
4. **Convert to Markdown** using BeautifulSoup + html2text
5. **Extract metadata**:
   - Sender email/name
   - Subject line (becomes title)
   - Date received
   - Links contained in email
6. **Store in `content_items`** with `source_type='gmail'`

## Privacy & Security Requirements

### Mandatory User Acknowledgment

Before OAuth flow initiates, user MUST check a consent checkbox:

```
[ ] I acknowledge that I am connecting an email account used
    primarily for newsletters and I understand GhostWriter
    will read emails from the specified label.
```

### Data Handling

- **Minimal data retention**: Only store processed content, not full email
- **No PII logging**: Never log email addresses or personal content
- **Token encryption**: OAuth tokens stored with Fernet encryption
- **Audit trail**: All fetch operations logged to `audit_log` table

### Retention Policy

- **30-day soft delete**: Mark as deleted, retain for recovery
- **90-day hard delete**: Permanently remove from database

## Auto-Discovery Feature

### Purpose

Scan user's Gmail for common newsletter senders to suggest for tier assignment.

### Implementation

1. Query inbox for emails with `Unsubscribe` header (newsletter indicator)
2. Extract unique sender addresses
3. Match against known newsletter database
4. Present suggestions to user:
   - Known senders: Suggest appropriate tier
   - Unknown senders: Offer to add as Tier B or C

### Privacy Safeguard

- Only extract sender addresses, never email content
- Run on-demand (user-initiated), not automatically
- Show preview before any data is stored

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `invalid_grant` | Token expired | Re-authenticate via OAuth flow |
| `403 Forbidden` | Scope insufficient | User needs to re-consent with correct scope |
| `404 Not Found` | Label doesn't exist | Prompt user to create label or use sender whitelist |
| `429 Rate Limit` | Too many requests | Implement exponential backoff |

### Retry Strategy

```python
max_retries = 3
backoff_factor = 2  # seconds: 2, 4, 8
```

## Integration Points

### Evaluation Pipeline

- **Tier A**: Bypass `evaluate_content.py`, mark as signal automatically
- **Tier B/C**: Run through standard Signal/Noise evaluation
- **Tier X**: Never fetch, log attempt and skip

### Draft Generation

When generating drafts from newsletter content:
- Include attribution: "Inspired by [Newsletter Name]"
- Link to original if public URL available
- Maintain voice transformation (newsletter → GhostWriter style)

## Configuration Schema

```json
{
  "gmail_label": "Newsletters",
  "hours_lookback": 168,
  "max_emails_per_fetch": 50,
  "auto_discovery_enabled": true,
  "default_trust_tier": "b",
  "require_acknowledgment": true
}
```

## Execution Scripts

| Script | Purpose |
|--------|---------|
| `execution/sources/gmail_source.py` | Gmail API integration |
| `execution/fetch_all.py` | Multi-source orchestrator |
| `execution/evaluate_content.py` | Unified evaluation (Phase 3) |

## UI Components (Phase 4)

### Newsletters Tab

1. **OAuth Setup Wizard**
   - Privacy warning with acknowledgment checkbox
   - OAuth flow button
   - Connection status indicator

2. **Newsletter Queue**
   - List of fetched newsletters
   - Preview content
   - Manual tier assignment

3. **Sender Management**
   - Add/remove senders
   - Assign trust tiers
   - Toggle active status

4. **Auto-Discovery Panel**
   - Scan button
   - Suggestions list with tier recommendations
   - Bulk add functionality

## Success Criteria

- [ ] OAuth flow works from Streamlit UI
- [ ] Newsletters fetched from labeled Gmail messages
- [ ] Trust tier system correctly routes content
- [ ] Tier A newsletters bypass evaluation
- [ ] Generated articles include newsletter attribution
- [ ] No PII logged or exposed
- [ ] Token refresh works automatically
- [ ] Auto-discovery suggests relevant newsletters
