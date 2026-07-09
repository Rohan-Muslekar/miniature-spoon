# Activity 1 — AI-Assisted Data Collection & Preparation (IMDb)

Collect, clean, prepare, and visualize the real **IMDb non-commercial datasets**.
Data preparation and exploratory visualization only — no mining algorithms.

## Pipeline

`scripts/create_imdb_dashboard.py` runs the whole thing:

```
title.basics.tsv.gz  +  title.ratings.tsv
  → replace IMDb '\N' with NA
  → keep titleType == movie, drop adult titles
  → numeric startYear / runtimeMinutes, drop missing
  → inner-join on tconst, keep numVotes >= 1000
  → derive decade / mainGenre / runtimeCategory
  ⇒ data/imdb_clean.csv          (cleaned analysis table)
  ⇒ dashboard/imdb_dashboard.html (interactive, data embedded, works offline)
```

## Run it

```bash
# Raw dumps are gitignored (too large). Download from
# https://developer.imdb.com/non-commercial-datasets/ then:
python3 scripts/create_imdb_dashboard.py \
    --basics title.basics.tsv.gz --ratings title.ratings.tsv.gz \
    --out-dir . --min-votes 1000
```

## Deliverables

| File | Description |
|------|-------------|
| `scripts/create_imdb_dashboard.py` | Cleaning pipeline + dashboard builder |
| `scripts/dashboard_template.html` | HTML template the script fills with data |
| `data/imdb_clean.csv` | ~48.5k movies, cleaned & merged (committed) |
| `dashboard/imdb_dashboard.html` | Interactive dashboard: filter sidebar, summary tiles, charts |

### `imdb_clean.csv` schema

`tconst, primaryTitle, startYear, runtimeMinutes, mainGenre, genres,
averageRating, numVotes, decade, runtimeCategory`

## Visualizations (in the dashboard)

Movies by decade · runtime distribution · rating distribution · average rating by
decade · average runtime by decade — all filterable by decade, genre, and runtime
category.
