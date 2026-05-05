import os
import sys
import json
import argparse
from pathlib import Path

import psycopg2
import psycopg2.extras
from openai import AzureOpenAI, OpenAI

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://n8n:n8npassword@localhost:5433/n8n")
EMBED_DIM    = 1536

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY  = os.environ.get("AZURE_OPENAI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_DEPLOY   = os.environ.get("AZURE_OPENAI_DEPLOY_EMBED", "text-embedding-3-small")


def get_embedding(text: str) -> list[float]:
    if AZURE_ENDPOINT and AZURE_API_KEY:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_ENDPOINT,
        )
        return client.embeddings.create(
            input=text, model=EMBED_DEPLOY
        ).data[0].embedding
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client.embeddings.create(
            input=text, model="text-embedding-3-small"
        ).data[0].embedding


def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def ingest(file_path: str, name: str, metadata: dict) -> int:
    text = Path(file_path).read_text(encoding="utf-8").strip()
    print(f"Embedding '{name}' ({len(text)} chars)...")
    vector = get_embedding(text)
    assert len(vector) == EMBED_DIM

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # pgvector accepts a Python list cast to the vector type
            cur.execute(
                """
                INSERT INTO knowledge (name, content, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s)
                RETURNING id
                """,
                (name, text, str(vector), json.dumps(metadata))
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        print(f"Inserted as id={row_id}")
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_entries():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, metadata, created_at, left(content, 80) AS preview FROM knowledge ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("No entries.")
        return

    print(f"\n{'ID':<5} {'Name':<30} {'Created':<22} Preview")
    print("-" * 90)
    for r in rows:
        print(f"{r['id']:<5} {r['name']:<30} {str(r['created_at'])[:19]:<22} {r['preview']}...")


def delete_entry(entry_id: int):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge WHERE id = %s RETURNING name", (entry_id,))
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()

    if row:
        print(f"Deleted id={entry_id} ('{row[0]}')")
    else:
        print(f"No entry with id={entry_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",   help="Path to text file to ingest")
    parser.add_argument("--name",   help="Label for this entry (e.g. document title)")
    parser.add_argument("--meta",   default="{}", help="JSON metadata string")
    parser.add_argument("--list",   action="store_true", help="List all entries")
    parser.add_argument("--delete", type=int, metavar="ID", help="Delete entry by id")
    args = parser.parse_args()

    if args.list:
        list_entries()
    elif args.delete:
        delete_entry(args.delete)
    elif args.file and args.name:
        try:
            meta = json.loads(args.meta)
        except json.JSONDecodeError as e:
            print(f"Invalid --meta JSON: {e}")
            sys.exit(1)
        ingest(args.file, args.name, meta)
    else:
        parser.print_help()
