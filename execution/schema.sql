-- Reddit Content Database Schema

-- Posts table: Stores raw Reddit posts from RSS feeds
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    author TEXT,
    content TEXT,
    timestamp INTEGER,  -- Unix timestamp from Reddit
    upvotes INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    retrieved_at INTEGER NOT NULL,  -- When we fetched it
    UNIQUE(url)
);

-- Evaluations table: Stores LLM evaluation results
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    is_signal BOOLEAN NOT NULL,  -- True = Signal, False = Noise
    reasoning TEXT,  -- Why the LLM classified it this way
    evaluated_at INTEGER NOT NULL,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    UNIQUE(post_id)
);

-- Drafts table: Stores generated content drafts
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    platform TEXT NOT NULL,  -- 'linkedin' or 'medium'
    draft_content TEXT NOT NULL,
    generated_at INTEGER NOT NULL,
    published BOOLEAN DEFAULT 0,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_signal ON evaluations(is_signal);
CREATE INDEX IF NOT EXISTS idx_drafts_published ON drafts(published);

-- =============================================================================
-- GitHub Integration Tables
-- =============================================================================

-- GitHub Commits table: Stores commit data from configured repositories
CREATE TABLE IF NOT EXISTS github_commits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_owner TEXT NOT NULL,           -- e.g., "microsoft"
    repo_name TEXT NOT NULL,            -- e.g., "semantic-kernel"
    commit_sha TEXT UNIQUE NOT NULL,    -- Full SHA (unique identifier)
    author_name TEXT,
    author_email TEXT,
    commit_message TEXT NOT NULL,
    files_changed TEXT,                 -- JSON array of file paths
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    committed_at INTEGER,               -- Unix timestamp
    retrieved_at INTEGER NOT NULL
);

-- GitHub Themes table: Stores extracted themes/topics from commit analysis
CREATE TABLE IF NOT EXISTS github_themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_title TEXT NOT NULL,
    theme_description TEXT,
    related_commits TEXT,               -- JSON array of commit IDs
    relevance_score REAL DEFAULT 0.0,   -- 0-1 score from LLM
    suggested_angle TEXT,               -- LLM-suggested content angle
    analyzed_at INTEGER NOT NULL,
    used_for_content BOOLEAN DEFAULT 0
);

-- Indexes for GitHub tables
CREATE INDEX IF NOT EXISTS idx_commits_repo ON github_commits(repo_owner, repo_name);
CREATE INDEX IF NOT EXISTS idx_commits_timestamp ON github_commits(committed_at);
CREATE INDEX IF NOT EXISTS idx_themes_score ON github_themes(relevance_score);

-- =============================================================================
-- Unified Content System (Multi-Source Abstraction)
-- =============================================================================

-- Unified content table: Stores normalized content from all sources
-- Replaces source-specific tables with polymorphic design
CREATE TABLE IF NOT EXISTS content_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,              -- 'reddit', 'gmail', 'github', 'rss', 'manual'
    source_id TEXT NOT NULL,                -- Unique within source (post ID, message ID, etc.)
    title TEXT NOT NULL,
    content TEXT,                           -- Full text content
    author TEXT,
    url TEXT,
    timestamp INTEGER,                      -- Original content timestamp (Unix epoch)
    trust_tier TEXT DEFAULT 'c',            -- 'a' (curated), 'b' (semi-trusted), 'c' (untrusted), 'x' (blocked)
    metadata TEXT,                          -- JSON blob for source-specific data
    retrieved_at INTEGER NOT NULL,
    UNIQUE(source_type, source_id)
);

-- Polymorphic evaluations: Works with any content source
CREATE TABLE IF NOT EXISTS evaluations_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL,
    is_signal BOOLEAN NOT NULL,             -- True = Signal, False = Noise
    relevance_score REAL,                   -- 0.0-1.0 confidence score
    reasoning TEXT,                         -- LLM explanation
    evaluated_at INTEGER NOT NULL,
    FOREIGN KEY (content_id) REFERENCES content_items(id) ON DELETE CASCADE,
    UNIQUE(content_id)
);

-- Newsletter sender configuration: Trust tier per sender
CREATE TABLE IF NOT EXISTS newsletter_senders (
    email TEXT PRIMARY KEY,
    display_name TEXT,
    trust_tier TEXT DEFAULT 'b',            -- Default semi-trusted
    is_active BOOLEAN DEFAULT 1,
    added_at INTEGER NOT NULL,
    notes TEXT                              -- User notes about this sender
);

-- Source configuration: OAuth tokens, acknowledgments, settings
CREATE TABLE IF NOT EXISTS source_configs (
    source_type TEXT PRIMARY KEY,
    config TEXT,                            -- JSON blob of source-specific config
    oauth_token TEXT,                       -- Encrypted OAuth token (if applicable)
    acknowledgment_at INTEGER,              -- When user acknowledged privacy terms
    last_fetch_at INTEGER,
    is_enabled BOOLEAN DEFAULT 1
);

-- Audit log: Track fetch/access events (privacy compliance)
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,               -- 'fetch', 'access', 'delete', 'export'
    source_type TEXT,
    details TEXT,                           -- JSON blob
    created_at INTEGER NOT NULL
);

-- Indexes for unified content system
CREATE INDEX IF NOT EXISTS idx_content_source ON content_items(source_type);
CREATE INDEX IF NOT EXISTS idx_content_timestamp ON content_items(timestamp);
CREATE INDEX IF NOT EXISTS idx_content_trust ON content_items(trust_tier);
CREATE INDEX IF NOT EXISTS idx_content_retrieved ON content_items(retrieved_at);
CREATE INDEX IF NOT EXISTS idx_eval_v2_signal ON evaluations_v2(is_signal);
CREATE INDEX IF NOT EXISTS idx_newsletter_active ON newsletter_senders(is_active);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at);

-- =============================================================================
-- Pulse Monitoring System
-- =============================================================================

-- Pulse aggregation: Daily trend summaries
CREATE TABLE IF NOT EXISTS pulse_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    top_topics TEXT,
    sentiment_summary TEXT,
    content_angles TEXT,
    source_breakdown TEXT,
    generated_at INTEGER NOT NULL,
    UNIQUE(date)
);

-- Source feed configuration
CREATE TABLE IF NOT EXISTS pulse_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT NOT NULL,
    fetch_interval_minutes INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT 1,
    last_fetched_at INTEGER,
    UNIQUE(url)
);

-- Indexes for pulse tables
CREATE INDEX IF NOT EXISTS idx_pulse_date ON pulse_daily(date);
CREATE INDEX IF NOT EXISTS idx_feeds_active ON pulse_feeds(is_active);
