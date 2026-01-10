-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store embeddings
CREATE TABLE IF NOT EXISTS url_embeddings (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL, -- Adjust dimension based on the embedding model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient similarity search
CREATE INDEX IF NOT EXISTS idx_url_embeddings_embedding ON url_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
