-- Supabase Database Migration
-- Core Schema setup for Finance Agent

-- 1. Create table 'people'
CREATE TABLE people (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    connection TEXT,
    is_self BOOLEAN DEFAULT FALSE
);

-- 2. Create table 'sources'
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    current_balance DECIMAL(12,2) NOT NULL DEFAULT 0.00
);

-- 3. Create table 'ownership'
CREATE TABLE ownership (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id),
    owner_id UUID NOT NULL REFERENCES people(id),
    allocated_amount DECIMAL(12,2) NOT NULL DEFAULT 0.00
);

-- 4. Create table 'transactions'
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id),
    amount DECIMAL(12,2) NOT NULL,
    category TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 5. Create table 'audit_logs'
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    before_state JSONB,
    after_state JSONB,
    audited_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 6. Enable Vector Extension and Create Vector Memories table
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS financial_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_summary TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 7. Create Vector Similarity Match Function (Cosine Similarity)
CREATE OR REPLACE FUNCTION match_financial_memories (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id UUID,
  context_summary TEXT,
  similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    financial_memories.id,
    financial_memories.context_summary,
    (1 - (financial_memories.embedding <=> query_embedding))::float AS similarity
  FROM financial_memories
  WHERE (1 - (financial_memories.embedding <=> query_embedding)) > match_threshold
  ORDER BY financial_memories.embedding <=> query_embedding LIMIT match_count;
END;
$$;

-- 8. Create Reporting Aggregation Functions
CREATE OR REPLACE FUNCTION get_category_spending_summary()
RETURNS TABLE (category TEXT, total_amount DECIMAL(12,2))
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT COALESCE(t.category, 'Uncategorized') as category, SUM(t.amount)::DECIMAL(12,2) as total_amount
  FROM transactions t
  GROUP BY category
  ORDER BY total_amount DESC;
END;
$$;

CREATE OR REPLACE FUNCTION get_monthly_spending_trend()
RETURNS TABLE (month_date TEXT, total_amount DECIMAL(12,2))
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT to_char(t.created_at, 'YYYY-MM') as month_date, SUM(t.amount)::DECIMAL(12,2) as total_amount
  FROM transactions t
  GROUP BY month_date
  ORDER BY month_date DESC;
END;
$$;

CREATE OR REPLACE FUNCTION get_multi_party_net_worth()
RETURNS TABLE (owner_name TEXT, total_net_worth DECIMAL(12,2))
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT p.name as owner_name, SUM(o.allocated_amount)::DECIMAL(12,2) as total_net_worth
  FROM ownership o
  JOIN people p ON o.owner_id = p.id
  GROUP BY p.name
  ORDER BY total_net_worth DESC;
END;
$$;

-- 9. Create checkpoints table for persistent state checkpointer
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    parent_id TEXT,
    PRIMARY KEY (thread_id, checkpoint_id)
);



