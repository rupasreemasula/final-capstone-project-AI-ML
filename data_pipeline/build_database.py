"""
Module 1 - Data Pipeline: Task 4, 5 & 6
Loads the cleaned book data into a normalized two-table SQLite schema
(categories <-1:N-> books, PK/FK relationship), then runs >= 5 SQL queries
covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, and a JOIN.

Run:
    python build_database.py

Outputs (written next to this script):
    zepto_books.db          - SQLite database (regenerated from books_clean.csv every run)
    sql_queries_output.txt  - every query string plus its executed output
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE / "books_clean.csv"
DB_PATH = HERE / "zepto_books.db"
QUERIES_OUTPUT = HERE / "sql_queries_output.txt"

SCHEMA_SQL = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def build_database() -> None:
    """(Re)create the SQLite DB from scratch and load the cleaned CSV into it."""
    df = pd.read_csv(CLEAN_CSV)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_SQL)

        category_names = sorted(df["category"].unique())
        conn.executemany(
            "INSERT INTO categories (category_name) VALUES (?)",
            [(name,) for name in category_names],
        )
        conn.commit()

        category_id_map = dict(
            conn.execute("SELECT category_name, category_id FROM categories").fetchall()
        )

        book_rows = [
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(bool(row.in_stock)),
                category_id_map[row.category],
            )
            for row in df.itertuples(index=False)
        ]
        conn.executemany(
            """INSERT INTO books
               (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            book_rows,
        )
        conn.commit()

        n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_categories = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        print(f"Loaded {n_books} books across {n_categories} categories into {DB_PATH.name}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Task 6: >= 5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT,
# IN/BETWEEN, and JOIN.
# ---------------------------------------------------------------------------

QUERIES: list[tuple[str, str]] = [
    (
        "Q1 - SELECT/WHERE: in-stock books priced above £30",
        """
        SELECT title, price_gbp, in_stock
        FROM books
        WHERE in_stock = 1 AND price_gbp > 30
        ORDER BY price_gbp DESC;
        """,
    ),
    (
        "Q2 - ORDER BY + LIMIT: 10 most expensive books",
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """,
    ),
    (
        "Q3 - DISTINCT: distinct star ratings present in the dataset",
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
        """,
    ),
    (
        "Q4 - BETWEEN: books priced between £20 and £40 (inclusive)",
        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
        """,
    ),
    (
        "Q5 - IN: highly rated books (rating IN (4, 5))",
        """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
        """,
    ),
    (
        "Q6 - JOIN: 10 highest-rated books with their category name",
        """
        SELECT b.title, b.rating, b.price_gbp, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY b.rating DESC, b.title
        LIMIT 10;
        """,
    ),
]


def run_queries() -> None:
    conn = sqlite3.connect(DB_PATH)
    lines: list[str] = []
    try:
        for description, sql in QUERIES:
            df_result = pd.read_sql(sql, conn)
            block = [
                "=" * 78,
                description,
                "-" * 78,
                "SQL:",
                sql.strip(),
                "-" * 78,
                f"Rows returned: {len(df_result)}",
                df_result.to_string(index=False),
                "",
            ]
            text = "\n".join(block)
            print(text)
            lines.append(text)
    finally:
        conn.close()

    QUERIES_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved all query strings + output -> {QUERIES_OUTPUT}")


def main() -> None:
    build_database()
    run_queries()


if __name__ == "__main__":
    main()
