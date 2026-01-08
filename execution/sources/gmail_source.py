"""
Gmail Newsletter Content Source

Fetches newsletter content from Gmail using OAuth 2.0.
Implements label-based filtering and trust tier assignment.

Usage:
    from execution.sources import SourceFactory, SourceType

    config = {
        "label": "Newsletters",
        "hours_lookback": 168,  # 7 days
    }
    source = SourceFactory.create(SourceType.GMAIL, config)
    result = source.fetch()

OAuth Setup Required:
    1. Create OAuth credentials in Google Cloud Console
    2. Save credentials.json to project root (or path in GMAIL_CREDENTIALS_PATH)
    3. Run OAuth flow to generate token.json
"""

import base64
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    import html2text
    HTML_PARSING_AVAILABLE = True
except ImportError:
    HTML_PARSING_AVAILABLE = False

from .base_source import (
    ContentSource,
    ContentItem,
    FetchResult,
    SourceType,
    TrustTier,
)
from . import register_source

# Database and credential paths
DB_PATH = Path(__file__).parent.parent.parent / "reddit_content.db"
DEFAULT_CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "credentials_gmail.json"
DEFAULT_TOKEN_PATH = Path(__file__).parent.parent.parent / "token_gmail.json"

# OAuth scope - minimal privilege, read-only
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_sender_trust_tier(email: str) -> TrustTier:
    """
    Look up trust tier for a sender from database.

    Args:
        email: Sender email address

    Returns:
        TrustTier enum value (default: C if not found)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT trust_tier FROM newsletter_senders WHERE email = ? AND is_active = 1",
            (email.lower(),)
        )
        row = cursor.fetchone()
        if row:
            return TrustTier(row[0])
    except (sqlite3.OperationalError, ValueError):
        pass
    finally:
        conn.close()

    return TrustTier.C  # Default untrusted


def add_newsletter_sender(
    email: str,
    display_name: Optional[str] = None,
    trust_tier: TrustTier = TrustTier.B,
    notes: Optional[str] = None
) -> bool:
    """
    Add or update a newsletter sender in database.

    Returns:
        True if added/updated successfully
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO newsletter_senders (email, display_name, trust_tier, is_active, added_at, notes)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, display_name),
                trust_tier = excluded.trust_tier,
                notes = COALESCE(excluded.notes, notes)
            """,
            (email.lower(), display_name, trust_tier.value, int(time.time()), notes)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


@register_source(SourceType.GMAIL)
class GmailSource(ContentSource):
    """
    Content source for Gmail newsletters via OAuth 2.0.

    Config options:
        label: Gmail label to filter (default: "Newsletters")
        hours_lookback: Only fetch emails from last N hours (default: 168 = 7 days)
        max_emails: Maximum emails per fetch (default: 50)
        credentials_path: Path to OAuth credentials JSON
        token_path: Path to OAuth token JSON
        include_senders: List of sender emails to include (alternative to label)
    """

    source_type = SourceType.GMAIL

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.label = self.config.get("label", "Newsletters")
        self.hours_lookback = self.config.get("hours_lookback", 168)  # 7 days
        self.max_emails = self.config.get("max_emails", 50)

        # Credential paths
        self.credentials_path = Path(
            self.config.get("credentials_path", DEFAULT_CREDENTIALS_PATH)
        )
        self.token_path = Path(
            self.config.get("token_path", DEFAULT_TOKEN_PATH)
        )

        # Optional sender whitelist (alternative to label)
        self.include_senders = self.config.get("include_senders", [])

        # Gmail service (initialized on first use)
        self._service = None

    def _validate_config(self) -> None:
        """Validate Gmail source configuration."""
        if not GMAIL_AVAILABLE:
            raise ImportError(
                "Gmail dependencies not installed. Run: "
                "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )

        if not HTML_PARSING_AVAILABLE:
            raise ImportError(
                "HTML parsing dependencies not installed. Run: "
                "pip install beautifulsoup4 html2text"
            )

    def _get_credentials(self) -> Optional[Credentials]:
        """
        Get or refresh OAuth credentials.

        Returns:
            Valid Credentials object or None if authentication needed
        """
        creds = None

        # Load existing token
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed token
                with open(self.token_path, "w") as f:
                    f.write(creds.to_json())
            except Exception:
                creds = None

        return creds

    def authenticate(self, force_new: bool = False) -> bool:
        """
        Run OAuth authentication flow.

        Args:
            force_new: Force new authentication even if token exists

        Returns:
            True if authentication successful
        """
        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"OAuth credentials not found at {self.credentials_path}. "
                "Please download from Google Cloud Console."
            )

        if not force_new:
            creds = self._get_credentials()
            if creds and creds.valid:
                return True

        # Run OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.credentials_path), SCOPES
        )
        creds = flow.run_local_server(port=0)

        # Save token
        with open(self.token_path, "w") as f:
            f.write(creds.to_json())

        # Set restrictive permissions
        try:
            os.chmod(self.token_path, 0o600)
        except (OSError, AttributeError):
            pass  # Windows doesn't support chmod

        return True

    def _get_service(self):
        """Get authenticated Gmail API service."""
        if self._service is not None:
            return self._service

        creds = self._get_credentials()
        if not creds or not creds.valid:
            raise ValueError(
                "Gmail not authenticated. Run authenticate() first or use Streamlit OAuth flow."
            )

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def fetch(
        self,
        limit: Optional[int] = None,
        since: Optional[int] = None
    ) -> FetchResult:
        """
        Fetch newsletter emails from Gmail.

        Args:
            limit: Max emails to fetch (None = use config max_emails)
            since: Unix timestamp - only fetch emails newer than this

        Returns:
            FetchResult with ContentItems
        """
        try:
            service = self._get_service()
        except ValueError as e:
            return FetchResult(
                items=[],
                success=False,
                error_message=str(e)
            )

        # Calculate time filter
        if since is None:
            since_date = datetime.now() - timedelta(hours=self.hours_lookback)
        else:
            since_date = datetime.fromtimestamp(since)

        # Build query
        query = self._build_query(since_date)

        # Fetch message list
        max_results = limit or self.max_emails
        try:
            messages = self._list_messages(service, query, max_results)
        except HttpError as e:
            return FetchResult(
                items=[],
                success=False,
                error_message=f"Gmail API error: {e}"
            )

        # Fetch full message content
        items: List[ContentItem] = []
        errors: List[str] = []

        for msg_meta in messages:
            try:
                msg = service.users().messages().get(
                    userId="me",
                    id=msg_meta["id"],
                    format="full"
                ).execute()

                item = self.normalize(msg)
                if item and not item.is_blocked:
                    items.append(item)

            except HttpError as e:
                errors.append(f"Message {msg_meta['id']}: {e}")

        return FetchResult(
            items=items,
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            items_fetched=len(items),
        )

    def _build_query(self, since_date: datetime) -> str:
        """Build Gmail search query."""
        # Format date for Gmail query
        date_str = since_date.strftime("%Y/%m/%d")
        query_parts = [f"after:{date_str}"]

        # Add label filter
        if self.label:
            query_parts.append(f"label:{self.label}")

        # Add sender filter (alternative to label)
        if self.include_senders:
            sender_query = " OR ".join(f"from:{s}" for s in self.include_senders)
            query_parts.append(f"({sender_query})")

        return " ".join(query_parts)

    def _list_messages(
        self,
        service,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        List messages matching query.

        Handles pagination for large result sets.
        """
        messages = []
        page_token = None

        while len(messages) < max_results:
            result = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=min(100, max_results - len(messages)),
                pageToken=page_token
            ).execute()

            batch = result.get("messages", [])
            messages.extend(batch)

            page_token = result.get("nextPageToken")
            if not page_token or not batch:
                break

        return messages[:max_results]

    def normalize(self, raw_item: Dict[str, Any]) -> Optional[ContentItem]:
        """
        Convert Gmail message to ContentItem.

        Args:
            raw_item: Gmail API message object

        Returns:
            Normalized ContentItem or None if parsing fails
        """
        try:
            headers = {h["name"].lower(): h["value"] for h in raw_item.get("payload", {}).get("headers", [])}

            # Extract sender
            from_header = headers.get("from", "")
            sender_email, sender_name = self._parse_sender(from_header)

            # Extract subject
            subject = headers.get("subject", "(No Subject)")

            # Extract date
            date_str = headers.get("date", "")
            try:
                timestamp = int(parsedate_to_datetime(date_str).timestamp())
            except (ValueError, TypeError):
                timestamp = int(time.time())

            # Extract body
            body = self._extract_body(raw_item)

            # Get trust tier from database
            trust_tier = get_sender_trust_tier(sender_email)

            return ContentItem(
                source_type=SourceType.GMAIL,
                source_id=raw_item["id"],
                title=subject,
                content=body,
                url=None,  # Emails don't have public URLs
                author=sender_name or sender_email,
                timestamp=timestamp,
                trust_tier=trust_tier,
                metadata={
                    "sender_email": sender_email,
                    "sender_name": sender_name,
                    "message_id": headers.get("message-id", ""),
                    "thread_id": raw_item.get("threadId", ""),
                    "labels": raw_item.get("labelIds", []),
                },
            )

        except Exception:
            return None

    def _parse_sender(self, from_header: str) -> Tuple[str, Optional[str]]:
        """
        Parse sender from 'From' header.

        Returns:
            Tuple of (email, display_name)
        """
        # Pattern: "Display Name <email@example.com>" or "email@example.com"
        match = re.match(r'^"?([^"<]*)"?\s*<?([^>]+@[^>]+)>?$', from_header.strip())
        if match:
            name = match.group(1).strip() or None
            email = match.group(2).strip().lower()
            return email, name

        # Just email
        email = from_header.strip().lower()
        return email, None

    def _extract_body(self, message: Dict[str, Any]) -> str:
        """
        Extract and convert email body to markdown.

        Prioritizes HTML content, falls back to plain text.
        """
        payload = message.get("payload", {})

        # Find HTML or plain text part
        html_body = None
        text_body = None

        def find_parts(part):
            nonlocal html_body, text_body
            mime_type = part.get("mimeType", "")

            if mime_type == "text/html" and "body" in part:
                data = part["body"].get("data", "")
                if data:
                    html_body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

            elif mime_type == "text/plain" and "body" in part:
                data = part["body"].get("data", "")
                if data:
                    text_body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

            # Recurse into parts
            for subpart in part.get("parts", []):
                find_parts(subpart)

        find_parts(payload)

        # Convert HTML to markdown
        if html_body:
            return self._html_to_markdown(html_body)

        return text_body or ""

    def _html_to_markdown(self, html: str) -> str:
        """
        Convert HTML email to clean markdown.

        Strips tracking pixels, excessive whitespace, etc.
        """
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove tracking images (1x1 pixels)
        for img in soup.find_all("img"):
            width = img.get("width", "")
            height = img.get("height", "")
            if width == "1" or height == "1":
                img.decompose()

        # Remove style and script tags
        for tag in soup.find_all(["style", "script"]):
            tag.decompose()

        # Convert to markdown
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0  # No line wrapping

        markdown = converter.handle(str(soup))

        # Clean up excessive whitespace
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        return markdown.strip()

    def get_trust_tier(self, raw_item: Dict[str, Any]) -> TrustTier:
        """
        Determine trust tier from sender email.

        Looks up sender in newsletter_senders table.
        """
        headers = {h["name"].lower(): h["value"] for h in raw_item.get("payload", {}).get("headers", [])}
        from_header = headers.get("from", "")
        sender_email, _ = self._parse_sender(from_header)
        return get_sender_trust_tier(sender_email)

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for Gmail source configuration."""
        return {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "default": "Newsletters",
                    "description": "Gmail label to filter emails",
                },
                "hours_lookback": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 720,  # 30 days
                    "default": 168,  # 7 days
                    "description": "Only fetch emails from last N hours",
                },
                "max_emails": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum emails per fetch",
                },
                "credentials_path": {
                    "type": "string",
                    "description": "Path to OAuth credentials JSON",
                },
                "token_path": {
                    "type": "string",
                    "description": "Path to OAuth token JSON",
                },
                "include_senders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sender emails to include (alternative to label)",
                },
            },
            "required": [],
        }

    # =========================================================================
    # Auto-Discovery Feature
    # =========================================================================

    def discover_newsletters(self, max_scan: int = 500) -> List[Dict[str, Any]]:
        """
        Scan inbox for potential newsletter senders.

        Looks for emails with Unsubscribe header (newsletter indicator).

        Args:
            max_scan: Maximum emails to scan

        Returns:
            List of discovered senders with metadata
        """
        try:
            service = self._get_service()
        except ValueError:
            return []

        # Query for emails with unsubscribe links
        query = "list:* OR unsubscribe"

        try:
            messages = self._list_messages(service, query, max_scan)
        except HttpError:
            return []

        # Extract unique senders
        senders: Dict[str, Dict[str, Any]] = {}

        for msg_meta in messages:
            try:
                # Only fetch headers (faster)
                msg = service.users().messages().get(
                    userId="me",
                    id=msg_meta["id"],
                    format="metadata",
                    metadataHeaders=["From", "List-Unsubscribe", "Subject"]
                ).execute()

                headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}

                from_header = headers.get("from", "")
                email, name = self._parse_sender(from_header)

                if email and email not in senders:
                    senders[email] = {
                        "email": email,
                        "display_name": name,
                        "sample_subject": headers.get("subject", ""),
                        "has_unsubscribe": "list-unsubscribe" in headers,
                        "count": 1,
                    }
                elif email:
                    senders[email]["count"] += 1

            except HttpError:
                continue

        # Sort by count (most frequent first)
        discovered = sorted(senders.values(), key=lambda x: x["count"], reverse=True)

        return discovered


# =========================================================================
# CLI Entry Point
# =========================================================================

def main():
    """CLI entry point for testing Gmail source."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch newsletters from Gmail")
    parser.add_argument("--authenticate", action="store_true", help="Run OAuth flow")
    parser.add_argument("--discover", action="store_true", help="Discover newsletters")
    parser.add_argument("--label", default="Newsletters", help="Gmail label")
    parser.add_argument("--hours", type=int, default=168, help="Hours lookback")

    args = parser.parse_args()

    config = {
        "label": args.label,
        "hours_lookback": args.hours,
    }

    source = GmailSource(config)

    if args.authenticate:
        print("Running OAuth authentication flow...")
        try:
            source.authenticate(force_new=True)
            print("Authentication successful!")
        except Exception as e:
            print(f"Authentication failed: {e}")
        return

    if args.discover:
        print("Discovering newsletters...")
        discovered = source.discover_newsletters()
        print(f"\nFound {len(discovered)} potential newsletter senders:\n")
        for sender in discovered[:20]:
            print(f"  {sender['email']}")
            print(f"    Name: {sender.get('display_name', 'N/A')}")
            print(f"    Emails: {sender['count']}")
            print()
        return

    # Normal fetch
    print(f"\nFetching newsletters (label: {args.label}, hours: {args.hours})...")
    result = source.fetch()

    if not result.success:
        print(f"Error: {result.error_message}")
        return

    print(f"Fetched {result.items_fetched} newsletters:\n")
    for item in result.items[:10]:
        print(f"  [{item.trust_tier.value.upper()}] {item.title[:60]}...")
        print(f"       From: {item.author}")
        print()


if __name__ == "__main__":
    main()
