# Activity 2 — LLM-Generated Synthetic Data

Generate a synthetic movie dataset with an LLM, then **validate, clean, and compare**
it against the real IMDb data from Activity 1. The lesson: generating data is easy;
generating *valid, useful, responsible* data is hard.

## Pipeline

`scripts/activity2_synthetic.py`:

```
synthetic_movies_raw.csv  (unedited LLM output, 100 rows)
  → validate against a declared schema (types, ranges, duplicates)
  → clean (fix types, invalid years/ratings, duplicate IDs, genre names)
  → merge with imdb_clean.csv (source-flagged: synthetic vs real)
  ⇒ data/synthetic_movies_clean.csv
  ⇒ data/combined_movies.csv
  ⇒ dashboard/synthetic_dashboard.html  (3 tabs: combined / synthetic-vs-real / synthetic-only)
```

## Run it

```bash
python3 scripts/activity2_synthetic.py
```

Self-contained — the raw synthetic CSV and the cleaned IMDb table are both committed.

## Deliverables (as required by the brief)

| Requirement | File |
|-------------|------|
| Prompt used to generate the dataset | [`synthetic_prompt.txt`](synthetic_prompt.txt) |
| Synthetic dataset (raw + clean) | `data/synthetic_movies_raw.csv`, `data/synthetic_movies_clean.csv` |
| Schema + validation rules | encoded in `scripts/activity2_synthetic.py`, reported in the dashboard |
| ≥ 3 visualizations | `dashboard/synthetic_dashboard.html` |
| Combined real + synthetic table | `data/combined_movies.csv` |

### Schema

`movie_id, title, release_year, genre, runtime_minutes, country, average_rating,
num_votes, budget_million, revenue_million, content_rating` (+ derived `decade`).

### Validation rules (enforced by the script)

- All required columns present; no missing values in required fields
- Numeric columns actually numeric
- `release_year` ∈ [1980, 2026]; `average_rating` ∈ [1.0, 10.0];
  `runtime_minutes` ∈ [60, 240]; `num_votes` > 0; `budget_million`, `revenue_million` ≥ 0
- `genre` from the fixed list; `content_rating` from {G, PG, PG-13, R, NC-17}
- No duplicate `movie_id` or duplicate titles

## Reflection (short version)

The LLM output was **too clean and too uniform** vs. the real IMDb data — no missing
values, tidy round numbers, artificially smooth distributions. Real data is messy;
synthetic data carries hidden assumptions and should be validated carefully, never
treated as real evidence.
