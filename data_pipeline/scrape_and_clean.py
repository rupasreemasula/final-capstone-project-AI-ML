"""
Module 1 - Data Pipeline: Task 1 & 2
Scrapes book catalog data from books.toscrape.com (a public scraping-practice
site, no login/API key/paid tier required) across multiple categories, then
cleans the raw fields into proper types.

Run:
    python scrape_and_clean.py

Outputs (written next to this script):
    books_raw.csv    - raw scraped rows (title, price, star_rating, availability, category)
    books_clean.csv  - cleaned rows (title, category, price_gbp, price_inr, rating, in_stock)
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Windows consoles default to cp1252, which chokes on book titles containing
# non-Latin-1 characters (curly quotes, accents, etc.); force UTF-8 output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "https://books.toscrape.com/"
HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "books_raw.csv"
CLEAN_CSV = HERE / "books_clean.csv"

MIN_BOOKS = 60
MIN_CATEGORIES = 3

# Project-defined fixed baseline conversion rate (see README for justification).
GBP_TO_INR_RATE = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "zepto-capstone-data-pipeline/1.0"})


def get_soup(url: str) -> BeautifulSoup:
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()
    # books.toscrape.com serves UTF-8 bytes but omits a charset in the
    # Content-Type header, so `requests` falls back to guessing ISO-8859-1
    # per the HTTP spec and mis-decodes curly quotes/accents. Parsing the
    # raw bytes directly lets lxml sniff the real encoding instead.
    return BeautifulSoup(resp.content, "lxml")


def discover_categories() -> list[tuple[str, str]]:
    """Return [(category_name, category_index_url), ...] from the homepage sidebar."""
    soup = get_soup(BASE_URL)
    links = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in links:
        name = a.get_text(strip=True)
        url = urljoin(BASE_URL, a["href"])
        categories.append((name, url))
    return categories


def scrape_category(name: str, start_url: str) -> list[dict]:
    """Scrape every book across every paginated page of one category."""
    rows: list[dict] = []
    url = start_url
    while url:
        soup = get_soup(url)
        for card in soup.select("article.product_pod"):
            title = card.h3.a["title"]
            price_text = card.select_one("p.price_color").get_text(strip=True)
            rating_classes = card.select_one("p.star-rating")["class"]
            star_rating_text = next(
                (c for c in rating_classes if c != "star-rating"), ""
            )
            availability_text = card.select_one(
                "p.instock.availability"
            ).get_text(strip=True)
            rows.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating_text,
                    "availability": availability_text,
                    "category": name,
                }
            )
        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None
        time.sleep(0.1)  # be polite to the practice server
    return rows


def scrape_all() -> list[dict]:
    """
    Scrape category by category (in the site's own sidebar order) until we
    have covered at least MIN_CATEGORIES categories AND collected at least
    MIN_BOOKS books, always finishing the category we're currently on so a
    category's data is never partially collected.
    """
    categories = discover_categories()
    collected: list[dict] = []
    categories_used = 0

    for name, url in categories:
        print(f"Scraping category: {name} ...", file=sys.stderr)
        category_rows = scrape_category(name, url)
        collected.extend(category_rows)
        categories_used += 1
        print(
            f"  -> {len(category_rows)} books "
            f"(running total: {len(collected)} across {categories_used} categories)",
            file=sys.stderr,
        )
        if categories_used >= MIN_CATEGORIES and len(collected) >= MIN_BOOKS:
            break

    return collected


def save_raw(rows: list[dict]) -> None:
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "price", "star_rating", "availability", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} raw rows -> {RAW_CSV}")


# ---------------------------------------------------------------------------
# Task 2: cleaning
# ---------------------------------------------------------------------------

PRICE_RE = re.compile(r"[\d.]+")


def parse_price_gbp(raw: str):
    """Strip currency symbol -> float, or None if unparseable."""
    if not isinstance(raw, str):
        return None
    match = PRICE_RE.search(raw)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_rating(raw: str):
    """Map text rating word (One..Five) -> int 1-5, or None if unparseable."""
    return RATING_WORDS.get(raw)


def parse_in_stock(raw: str):
    """Parse availability text -> bool, or None if unparseable."""
    if not isinstance(raw, str):
        return None
    lowered = raw.lower()
    if "in stock" in lowered:
        return True
    if "out of stock" in lowered:
        return False
    return None


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["price_gbp"] = df["price"].apply(parse_price_gbp)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_in_stock)

    # --- Row-failure handling policy (see data_pipeline/README.md) ---
    # Numeric fields (price_gbp, rating): a parse failure is repaired with
    # median imputation of that column, since a missing numeric value can be
    # reasonably substituted with the column's typical value without
    # discarding the rest of the row's information.
    n_price_failed = int(df["price_gbp"].isna().sum())
    n_rating_failed = int(df["rating"].isna().sum())
    if n_price_failed:
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"Median-imputed price_gbp for {n_price_failed} row(s).")
    if n_rating_failed:
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating).round().astype("Int64")
        print(f"Median-imputed rating for {n_rating_failed} row(s).")

    # Non-numeric fields (in_stock boolean, title/category identifiers): a
    # parse failure here has no sensible numeric median, so the row is
    # dropped instead of guessing a stock status or identity.
    before = len(df)
    df = df.dropna(subset=["in_stock", "title", "category"])
    n_dropped = before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} row(s) with unparseable availability/title/category.")

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Task 3: fixed-rate currency conversion (1 GBP = 105.50 INR, project constant)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


def main() -> None:
    rows = scrape_all()
    if len(rows) < MIN_BOOKS:
        raise RuntimeError(
            f"Only scraped {len(rows)} books, need at least {MIN_BOOKS}."
        )
    save_raw(rows)

    df_raw = pd.read_csv(RAW_CSV)
    df_clean = clean(df_raw)
    df_clean.to_csv(CLEAN_CSV, index=False)

    print(f"\nFinal cleaned dataset: {len(df_clean)} books "
          f"across {df_clean['category'].nunique()} categories.")
    print(f"Saved cleaned data -> {CLEAN_CSV}")
    print("\nSample:")
    print(df_clean.head())


if __name__ == "__main__":
    main()
