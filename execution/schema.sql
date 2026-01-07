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
