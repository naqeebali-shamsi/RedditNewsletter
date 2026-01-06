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
