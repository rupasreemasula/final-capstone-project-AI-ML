"""
Module 1 - Data Pipeline: Task 7
Reads back >= 2 SQL query results into pandas via pd.read_sql(...), and
separately reproduces the JOIN query's result using pd.merge(...) directly
on in-memory DataFrames (no SQL) - then shows both approaches side by side
and confirms they match.

Run (after build_database.py has created zepto_books.db):
    python pandas_verification.py

Output:
    pandas_verification_output.txt
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "zepto_books.db"
OUTPUT_PATH = HERE / "pandas_verification_output.txt"

# Same SQL text as Q1 and Q6 in build_database.py, reused here for the
# pd.read_sql() half of the comparison.
Q1_SQL = """
SELECT title, price_gbp, in_stock
FROM books
WHERE in_stock = 1 AND price_gbp > 30
ORDER BY price_gbp DESC;
"""

Q6_JOIN_SQL = """
SELECT b.title, b.rating, b.price_gbp, c.category_name
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.rating >= 4
ORDER BY b.rating DESC, b.title
LIMIT 10;
"""


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    lines: list[str] = []

    def emit(text: str) -> None:
        print(text)
        lines.append(text)

    try:
        # --- pd.read_sql on two of the queries from build_database.py ---
        emit("=" * 78)
        emit("pd.read_sql: Q1 (SELECT/WHERE + ORDER BY)")
        emit("-" * 78)
        df_q1_sql = pd.read_sql(Q1_SQL, conn)
        emit(f"Rows: {len(df_q1_sql)}")
        emit(df_q1_sql.head(10).to_string(index=False))
        emit("")

        emit("=" * 78)
        emit("pd.read_sql: Q6 (JOIN) - this is the query we'll reproduce via pd.merge below")
        emit("-" * 78)
        df_q6_sql = pd.read_sql(Q6_JOIN_SQL, conn).reset_index(drop=True)
        emit(f"Rows: {len(df_q6_sql)}")
        emit(df_q6_sql.to_string(index=False))
        emit("")

        # --- pd.merge reproduction of the Q6 JOIN, with no SQL involved ---
        books_df = pd.read_sql("SELECT * FROM books", conn)
        categories_df = pd.read_sql("SELECT * FROM categories", conn)
    finally:
        conn.close()

    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    df_q6_merge = (
        merged[merged["rating"] >= 4][["title", "rating", "price_gbp", "category_name"]]
        .sort_values(by=["rating", "title"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )

    emit("=" * 78)
    emit("pd.merge reproduction (no SQL) of the same JOIN query")
    emit("-" * 78)
    emit(f"Rows: {len(df_q6_merge)}")
    emit(df_q6_merge.to_string(index=False))
    emit("")

    are_equal = df_q6_sql.equals(df_q6_merge)
    emit("=" * 78)
    emit("Equivalence check: pd.read_sql(JOIN) vs pd.merge(books, categories)")
    emit("-" * 78)
    emit(f"DataFrames are identical: {are_equal}")
    if not are_equal:
        emit("DIFF (pd.read_sql vs pd.merge):")
        emit(df_q6_sql.compare(df_q6_merge).to_string())
    assert are_equal, "pd.read_sql and pd.merge results diverged!"

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved verification output -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
