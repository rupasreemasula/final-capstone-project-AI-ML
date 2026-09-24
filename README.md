# Zepto Data & AI Platform — Capstone Project

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows, or: source .venv/bin/activate   # macOS/Linux

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## How to run each module

### 1. Data Pipeline (`/data_pipeline`)

```bash
cd data_pipeline
python scrape_and_clean.py   # scrapes books.toscrape.com -> books_clean.csv
python build_database.py     # creates zepto_books.db, loads data, runs 5+ SQL queries
python pandas_verification.py  # pd.read_sql vs pd.merge equivalence check
```

Outputs: `books_raw.csv`, `books_clean.csv`, `zepto_books.db`, `sql_queries_output.txt`,
`pandas_verification_output.txt`. See [`data_pipeline/README.md`](data_pipeline/README.md)
for full design decisions.

### 2. Analytics Pipeline (`/analytics`)

```bash
cd analytics
python 01_eda.py        # loads titanic dataset ONCE, profiles/cleans it, saves titanic.csv, runs full EDA
python 02_modeling.py   # reads the same titanic.csv, trains/tunes/evaluates models, saves best_pipeline.joblib
```

Outputs: `titanic.csv`, `eda_output.txt`, `modeling_output.txt`, `charts/*.png`,
`best_pipeline.joblib`. See [`analytics/README.md`](analytics/README.md) for every required
written interpretation, the model comparison table, and the final recommendation.

### 3. Support Assistant (`/support_assistant`)

```bash
cd support_assistant
python ingest.py                             # Task 1: embed the 8 policy docs into ChromaDB
uvicorn main:app --host 0.0.0.0 --port 7860  # Task 6: serve POST /ask (MOCK_LLM defaults to mock mode)
```

Or via Docker: `cd support_assistant && docker build -t zepto-support-assistant . && docker run
-p 7860:7860 zepto-support-assistant`. See [`support_assistant/README.md`](support_assistant/README.md)
for the full architecture write-up, example call transcripts, and the structured prompt template.

## Design decisions summary

### Data Pipeline
I built a data pipeline that scrapes book data, cleans the fields, converts the price from GBP
to INR using the project's fixed rate, and stores the data in a normalized SQLite database. I
used separate `categories` and `books` tables with a primary/foreign key relationship, and
verified the database results using both SQL and pandas.

See [`data_pipeline/README.md`](data_pipeline/README.md) for the full write-up (scraping
scope, cleaning/imputation choices, fixed currency rate, schema design, and the
`pd.read_sql` vs `pd.merge` equivalence check).

### Analytics Pipeline
I used the Titanic dataset to perform data cleaning, EDA, visualization, and predictive
modeling. I kept the EDA standardization separate from the modeling pipeline and used
train-only preprocessing to avoid data leakage, before comparing Logistic Regression, Decision
Tree, and Random Forest models.

See [`analytics/README.md`](analytics/README.md) for the full write-up: missing-value
threshold-rule decisions (drop `deck`, group-median-impute `age`, drop 2 rows for
`embarked`/`embark_town`), IQR outlier counts, skewness/correlation interpretation, the
5-chart data story, the stratified train/test split, the leak-free `ColumnTransformer`
preprocessing, all 3 classifiers' metrics, the 3-way imbalance comparison, `GridSearchCV`
tuning + OOB score, the `fare` regression side-task, and the final model comparison table
and deployment recommendation (Random Forest).

### Support Assistant
I built a small RAG-based support assistant using the 8 Zepto policy documents, local
embeddings, ChromaDB, and LangGraph. The system routes policy questions to retrieval and
returns the final response through FastAPI, with `MOCK_LLM` providing the required offline
mode.

See [`support_assistant/README.md`](support_assistant/README.md) for the full write-up: the
ingestion → embedding → retrieval → generation architecture, the LangGraph 3-node pipeline
(`classify_intent` → conditional edge → `retrieve_and_answer` / `direct_answer`), the
structured prompt template (role/context/task/format/length + negative constraint + few-shot
example), the Pydantic-validated output schema with tested retry-on-failure logic, and the two
required example call transcripts run against a live local `uvicorn` server with `MOCK_LLM`
left at its default.