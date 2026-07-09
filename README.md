# Data Mining — Movie Data Activities

Coursework for the UOIT Summer 2026 **Data Mining** course. Two in-class group
activities on data collection, cleaning, and synthetic data.

> Group activity. Rules for both activities: **data preparation and exploratory
> visualization only** — no prediction, classification, clustering, or recommender
> algorithms. AI tools were used as an assistant (writing/debugging code, explaining
> columns), not to invent results.

## Contents

| Folder | What's inside |
|--------|---------------|
| [`activity1_imdb/`](activity1_imdb/) | Clean the real IMDb non-commercial datasets and build an interactive dashboard. |
| [`activity2_synthetic/`](activity2_synthetic/) | Generate an LLM synthetic-movie dataset, validate it, clean it, and compare it against the real IMDb data. |
| [`assignment/`](assignment/) | Original assignment brief (Class Activity 1 & 2). |

## Quick start

```bash
python3 -m pip install -r requirements.txt

# Activity 1 — needs the raw IMDb dumps (see note below)
python3 activity1_imdb/scripts/create_imdb_dashboard.py

# Activity 2 — self-contained (synthetic CSV is committed)
python3 activity2_synthetic/scripts/activity2_synthetic.py
```

Open the generated `dashboard/*.html` files in any browser — the data is embedded,
so they work offline.

## A note on the raw datasets

The raw IMDb dumps (`title.basics.tsv.gz`, `title.ratings.tsv.gz`) are **not
committed** — the uncompressed `title.basics.tsv` is ~1.1 GB and the `.gz` is ~223 MB,
both over GitHub's 100 MB per-file limit. They're gitignored. Download them fresh
from <https://developer.imdb.com/non-commercial-datasets/> and point the Activity 1
script at them:

```bash
python3 activity1_imdb/scripts/create_imdb_dashboard.py \
    --basics title.basics.tsv.gz --ratings title.ratings.tsv.gz --min-votes 1000
```

The **cleaned** analysis table (`activity1_imdb/data/imdb_clean.csv`) is committed,
so the dashboard and the Activity 2 comparison are fully reproducible without the raw
dumps.
