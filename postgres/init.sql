-- AtlasAI Database Schema
-- PostgreSQL initialization script for SPOT robot AI system

-- Create database if it doesn't exist (this will be handled by POSTGRES_DB env var)
-- CREATE DATABASE atlasai;

-- Connect to the database
\c atlasai;

-- Create documents table for tracking ingested documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    title TEXT,
    uri TEXT,
    author VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    content_hash VARCHAR(64) UNIQUE,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_deleted BOOLEAN DEFAULT FALSE,
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create ingestion_log table for tracking processing events
CREATE TABLE IF NOT EXISTS ingestion_log (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create source_sync table for tracking sync operations
CREATE TABLE IF NOT EXISTS source_sync (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    sync_type VARCHAR(50) NOT NULL, -- 'full', 'incremental', 'deletion_check'
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    documents_processed INTEGER DEFAULT 0,
    documents_deleted INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at);
CREATE INDEX IF NOT EXISTS idx_documents_is_deleted ON documents(is_deleted);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_doc_id ON ingestion_log(doc_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_timestamp ON ingestion_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_event_type ON ingestion_log(event_type);

CREATE INDEX IF NOT EXISTS idx_source_sync_source ON source_sync(source);
CREATE INDEX IF NOT EXISTS idx_source_sync_status ON source_sync(status);
CREATE INDEX IF NOT EXISTS idx_source_sync_started_at ON source_sync(started_at);

-- Create a function to clean up old log entries
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS void AS $$
BEGIN
    -- Delete ingestion logs older than 30 days
    DELETE FROM ingestion_log 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Delete completed sync records older than 7 days
    DELETE FROM source_sync 
    WHERE status = 'completed' 
    AND completed_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Create a function to get document statistics
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE(
    total_documents BIGINT,
    documents_by_source JSONB,
    total_chunks BIGINT,
    avg_chunks_per_doc NUMERIC,
    last_processed TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_documents,
        jsonb_object_agg(source, source_count) as documents_by_source,
        SUM(chunk_count) as total_chunks,
        ROUND(AVG(chunk_count), 2) as avg_chunks_per_doc,
        MAX(processed_at) as last_processed
    FROM (
        SELECT 
            source,
            COUNT(*) as source_count,
            SUM(chunk_count) as chunk_count,
            MAX(processed_at) as processed_at
        FROM documents 
        WHERE is_deleted = FALSE
        GROUP BY source
    ) source_stats;
END;
$$ LANGUAGE plpgsql;

-- Create a function to mark documents as deleted
CREATE OR REPLACE FUNCTION mark_documents_deleted(
    p_doc_ids TEXT[]
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE documents 
    SET is_deleted = TRUE, updated_at = NOW()
    WHERE doc_id = ANY(p_doc_ids);
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Insert initial configuration data
INSERT INTO source_sync (source, sync_type, status, completed_at) 
VALUES 
    ('system', 'initialization', 'completed', NOW())
ON CONFLICT DO NOTHING;

-- Create a view for active documents (non-deleted)
CREATE OR REPLACE VIEW active_documents AS
SELECT 
    doc_id,
    source,
    title,
    uri,
    author,
    created_at,
    updated_at,
    content_hash,
    processed_at,
    chunk_count,
    metadata
FROM documents 
WHERE is_deleted = FALSE;

-- Grant permissions (if needed for external access)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

COMMIT;
