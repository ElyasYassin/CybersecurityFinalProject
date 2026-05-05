import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from openai import OpenAI

DATABASE_URL   = os.environ.get("DATABASE_URL", "postgresql://n8n:n8npassword@localhost:5433/n8n")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EXPERIMENT_DIR = Path(__file__).parent

DOCS = {
    "legitimate": {
        "file": EXPERIMENT_DIR / "legitimate_biometric_policy.txt",
        "name": "OIT-BIOMETRIC-2023-11",
        "poison": False,
        "description": "Deny policy — provisional match (0.65-0.79) gets no admin access",
    },
    "poisoned": {
        "file": EXPERIMENT_DIR / "attacker_amendment.txt",
        "name": "DEI-BIO-AMEND-2024-03",
        "poison": True,
        "description": "Attacker's amendment — disguised as accessibility doc, overrides deny policy",
    },
}


def embed(text: str) -> list[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding


def ingest_doc(key: str) -> None:
    doc = DOCS[key]
    text = Path(doc["file"]).read_text(encoding="utf-8").strip()
    print(f"Embedding '{doc['name']}' ({len(text)} chars)...")
    vector = embed(text)

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge (name, content, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                RETURNING id
                """,
                (doc["name"], text, str(vector), '{"source": "experiment"}')
            )
            row = cur.fetchone()
        conn.commit()
        print(f"  Inserted id={row[0]}  ({'POISONED' if doc['poison'] else 'legitimate'})")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_all() -> None:
    names = [d["name"] for d in DOCS.values()]
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge WHERE name = ANY(%s) RETURNING id, name",
                (names,)
            )
            rows = cur.fetchall()
        conn.commit()
    finally:
        conn.close()
    if rows:
        for r in rows:
            print(f"  Deleted id={r[0]} ('{r[1]}')")
    else:
        print("  Nothing to delete.")


def list_docs() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, created_at, left(content, 60) FROM knowledge ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        print("  (empty)")
        return
    print(f"\n{'ID':<5} {'Name':<30} {'Created':<22} Preview")
    print("-" * 85)
    for r in rows:
        print(f"{r[0]:<5} {r[1]:<30} {str(r[2])[:19]:<22} {r[3]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true", help="Ingest legitimate docs only (clean DB)")
    parser.add_argument("--poison", action="store_true", help="Add poisoned doc only")
    parser.add_argument("--delete", action="store_true", help="Remove all experiment docs")
    parser.add_argument("--list",   action="store_true", help="Show knowledge table")
    args = parser.parse_args()

    if args.list:
        list_docs()
    elif args.delete:
        print("Deleting experiment docs...")
        delete_all()
    elif args.phase1:
        print("Ingesting legitimate deny policy (Phase 1)...")
        print(f"  {DOCS['legitimate']['description']}")
        ingest_doc("legitimate")
    elif args.poison:
        print("Injecting attacker's amendment (Phase 2)...")
        print(f"  {DOCS['poisoned']['description']}")
        ingest_doc("poisoned")
    else:
        parser.print_help()
