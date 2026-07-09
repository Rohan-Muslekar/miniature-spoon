#!/usr/bin/env python3
"""
activity2_synthetic.py  —  Data Mining, Class Activity 2
========================================================
Takes the raw LLM-generated synthetic movie dataset, VALIDATES it against a
declared schema, CLEANS it, MERGES it with the real cleaned IMDb table from
Activity 1, and builds one tabbed HTML dashboard:

    Tab 1  Combined dataset   (synthetic + real, source-flagged)
    Tab 2  Synthetic vs Real  (distribution comparison)
    Tab 3  Synthetic analysis (synthetic-only, + validation report + prompt)

Inputs :  synthetic_movies_raw.csv , imdb_clean.csv (from Activity 1)
Outputs:  synthetic_movies_clean.csv , combined_movies.csv , synthetic_dashboard.html

No prediction / classification / clustering — validation & description only.
"""
import json, math, os, re
import numpy as np
import pandas as pd

def clean_nan(o):
    """Recursively replace float NaN with None so JSON.parse in the browser succeeds."""
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o

def li(series, cast):
    """List from a pandas series, mapping NaN/None to None, else cast()."""
    return [None if pd.isna(v) else cast(v) for v in series]

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "synthetic_movies_raw.csv")
REAL = os.path.join(HERE, "imdb_clean.csv")
TPL = os.path.join(HERE, "synthetic_dashboard_template.html")

ALLOWED_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
                  "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance",
                  "Sci-Fi", "Thriller", "War", "Western"]
ALLOWED_RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]
GENRE_KEY = {re.sub(r"[^a-z0-9]", "", g.lower()): g for g in ALLOWED_GENRES}

SCHEMA = [
    ("movie_id", "string", "unique key", "no"),
    ("title", "string", "fictional, free text", "no"),
    ("release_year", "integer", "1980–2026", "no"),
    ("genre", "category", "one of 16 allowed genres", "no"),
    ("runtime_minutes", "integer", "60–240", "no"),
    ("country", "string", "free text", "yes"),
    ("average_rating", "float", "1.0–10.0", "yes"),
    ("num_votes", "integer", "> 0", "no"),
    ("budget_million", "float", "≥ 0", "yes"),
    ("revenue_million", "float", "≥ 0", "yes"),
    ("content_rating", "category", "G / PG / PG-13 / R / NC-17", "yes"),
]

issues = []
def log(sev, rule, detail, action, ids):
    issues.append({"sev": sev, "rule": rule, "detail": detail, "action": action,
                   "ids": ids if isinstance(ids, list) else [ids]})


def clean():
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    raw_n = len(raw)
    df = raw.copy()

    # ---- structural: required columns ----
    missing_cols = [c for c, *_ in SCHEMA if c not in df.columns]
    if missing_cols:
        log("critical", "missing columns", f"absent: {missing_cols}", "cannot proceed", [])

    # ---- movie_id duplicates -> reassign to next free id ----
    dup_mask = df["movie_id"].duplicated(keep="first")
    if dup_mask.any():
        existing = set(df["movie_id"])
        def next_id():
            i = 1
            while f"M{i:03d}" in existing:
                i += 1
            nid = f"M{i:03d}"; existing.add(nid); return nid
        for idx in df.index[dup_mask]:
            old = df.at[idx, "movie_id"]; nid = next_id()
            df.at[idx, "movie_id"] = nid
            log("high", "duplicate movie_id", f"'{old}' reused (title: {df.at[idx,'title']})",
                f"reassigned to {nid}", old)

    # ---- release_year -> int, range 1980..2026 (drop impossible) ----
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    bad_year = df[(df["release_year"] < 1980) | (df["release_year"] > 2026) | df["release_year"].isna()]
    for idx in bad_year.index:
        log("high", "release_year out of range",
            f"{raw.at[idx,'release_year']!r} not in 1980–2026 (title: {df.at[idx,'title']})",
            "row dropped", df.at[idx, "movie_id"])
    df = df.drop(bad_year.index)

    # ---- genre canonicalisation ----
    def fix_genre(g, mid):
        key = re.sub(r"[^a-z0-9]", "", str(g).lower())
        if key in GENRE_KEY:
            canon = GENRE_KEY[key]
            if canon != g:
                log("low", "inconsistent genre name", f"'{g}' (title cleanup)",
                    f"normalised to '{canon}'", mid)
            return canon
        log("high", "genre not in allowed list", f"'{g}' is not one of the 16 genres",
            "reassigned to 'Action'", mid)
        return "Action"
    df["genre"] = [fix_genre(g, m) for g, m in zip(df["genre"], df["movie_id"])]

    # ---- runtime -> strip units, int, range 60..240 ----
    def fix_runtime(v, mid):
        m = re.search(r"\d+", str(v))
        if not m:
            log("high", "runtime not numeric", f"{v!r}", "set to missing", mid); return np.nan
        n = int(m.group())
        if str(v).strip() != str(n):
            log("low", "runtime formatting", f"{v!r} contained non-numeric text", f"parsed to {n}", mid)
        if n < 60 or n > 240:
            log("high", "runtime out of range", f"{n} min not in 60–240", "set to missing", mid); return np.nan
        return n
    df["runtime_minutes"] = [fix_runtime(v, m) for v, m in zip(df["runtime_minutes"], df["movie_id"])]

    # ---- average_rating -> float 1..10 ----
    def fix_rating(v, mid):
        if str(v).strip() == "":
            log("medium", "missing average_rating", "empty value", "left as NaN (excluded from rating stats)", mid); return np.nan
        try: n = float(v)
        except ValueError:
            log("high", "rating not numeric", f"{v!r}", "set to NaN", mid); return np.nan
        if n < 1 or n > 10:
            log("high", "rating out of range", f"{n} not in 1.0–10.0", "set to NaN", mid); return np.nan
        return n
    df["average_rating"] = [fix_rating(v, m) for v, m in zip(df["average_rating"], df["movie_id"])]

    # ---- num_votes -> strip commas, positive int ----
    def fix_votes(v, mid):
        s = str(v).replace(",", "").strip()
        if s == "":
            log("high", "missing num_votes", "empty", "set to NaN", mid); return np.nan
        try: n = float(s)
        except ValueError:
            log("high", "num_votes not numeric", f"{v!r}", "set to NaN", mid); return np.nan
        if "," in str(v):
            log("low", "num_votes formatting", f"{v!r} had a thousands comma", f"parsed to {int(n)}", mid)
        if n != int(n):
            log("low", "num_votes not integer", f"{v!r} was fractional", f"rounded to {round(n)}", mid)
        if n < 0:
            log("high", "num_votes negative", f"{v!r}", "set to NaN", mid); return np.nan
        return int(round(n))
    df["num_votes"] = [fix_votes(v, m) for v, m in zip(df["num_votes"], df["movie_id"])]

    # ---- budget / revenue -> float >= 0 ----
    def fix_money(col):
        def f(v, mid):
            if str(v).strip() == "":
                log("medium", f"missing {col}", "empty value", "left as NaN", mid); return np.nan
            try: n = float(v)
            except ValueError:
                log("high", f"{col} not numeric", f"{v!r}", "set to NaN", mid); return np.nan
            if n < 0:
                log("high", f"{col} negative", f"{n} < 0 (impossible)", "set to NaN", mid); return np.nan
            return n
        df[col] = [f(v, m) for v, m in zip(df[col], df["movie_id"])]
    fix_money("budget_million"); fix_money("revenue_million")

    # ---- content_rating enum ----
    def fix_content(v, mid):
        s = str(v).strip().upper().replace("PG13", "PG-13")
        if s in ALLOWED_RATINGS:
            if s != v: log("low", "content_rating casing", f"'{v}' standardised", f"-> '{s}'", mid)
            return s
        log("medium", "content_rating not in allowed set", f"'{v}' not in {ALLOWED_RATINGS}",
            "set to NaN", mid); return np.nan
    df["content_rating"] = [fix_content(v, m) for v, m in zip(df["content_rating"], df["movie_id"])]

    # ---- duplicate titles (informational: different films can share a title) ----
    dt = df[df["title"].duplicated(keep=False)].sort_values("title")
    for title, grp in dt.groupby("title"):
        log("info", "duplicate title", f"'{title}' appears {len(grp)}x (different id/year — kept)",
            "kept both", grp["movie_id"].tolist())

    # ---- hidden-assumption check: revenue > budget almost always ----
    both = df.dropna(subset=["budget_million", "revenue_million"])
    rev_gt = int((both["revenue_million"] > both["budget_million"]).sum())
    rev_gt_pct = round(100 * rev_gt / len(both), 1) if len(both) else 0
    if rev_gt_pct >= 90:
        log("info", "hidden assumption (LLM optimism)",
            f"revenue > budget in {rev_gt_pct}% of rows — the model assumed every film is profitable",
            "kept, but flagged as unrealistic uniform pattern", [])

    df["release_year"] = df["release_year"].astype(int)
    df["decade"] = (df["release_year"] // 10 * 10).astype(int)
    df = df.reset_index(drop=True)

    df.to_csv(os.path.join(HERE, "synthetic_movies_clean.csv"), index=False)
    return raw_n, df, rev_gt_pct


def merge_and_build(raw_n, syn, rev_gt_pct):
    real = pd.read_csv(REAL)
    real_n = len(real)

    # ---- unified schema for the combined table ----
    real_u = pd.DataFrame({
        "source": "real", "id": real["tconst"], "title": real["primaryTitle"],
        "year": real["startYear"], "genre": real["mainGenre"], "runtime": real["runtimeMinutes"],
        "rating": real["averageRating"], "votes": real["numVotes"], "country": np.nan,
        "budget_million": np.nan, "revenue_million": np.nan, "content_rating": np.nan,
        "decade": real["decade"]})
    syn_u = pd.DataFrame({
        "source": "synthetic", "id": syn["movie_id"], "title": syn["title"],
        "year": syn["release_year"], "genre": syn["genre"], "runtime": syn["runtime_minutes"],
        "rating": syn["average_rating"], "votes": syn["num_votes"], "country": syn["country"],
        "budget_million": syn["budget_million"], "revenue_million": syn["revenue_million"],
        "content_rating": syn["content_rating"], "decade": syn["decade"]})
    combined = pd.concat([real_u, syn_u], ignore_index=True)
    combined.to_csv(os.path.join(HERE, "combined_movies.csv"), index=False)

    # ---- real aggregates (as % for fair comparison) ----
    def hist_pct(series, edges):
        h, _ = np.histogram(series.dropna(), bins=edges)
        tot = h.sum() or 1
        return [round(100 * c / tot, 2) for c in h]
    rating_edges = np.arange(0, 10.5, 0.5)
    runtime_edges = np.arange(0, 255, 15)
    real_rating = hist_pct(real["averageRating"], rating_edges)
    real_runtime = hist_pct(real["runtimeMinutes"].clip(upper=240), runtime_edges)
    gshare = (real["mainGenre"].value_counts(normalize=True) * 100).round(2).to_dict()
    dshare = (real["decade"].value_counts(normalize=True) * 100).round(2).to_dict()
    rs = real[["numVotes", "averageRating"]].dropna().sample(n=min(400, len(real)), random_state=7)

    # ---- synthetic columnar payload ----
    payload = {
        "meta": {
            "rawN": raw_n, "synClean": len(syn), "dropped": raw_n - len(syn),
            "realN": real_n, "combinedN": real_n + len(syn),
            "issueCount": len(issues), "revGtBudgetPct": rev_gt_pct,
            "synMedRating": round(float(syn["average_rating"].median()), 2),
            "synMedRuntime": int(syn["runtime_minutes"].median()),
            "synMedVotes": int(syn["num_votes"].median()),
            "realMedRating": round(float(real["averageRating"].median()), 1),
            "realMedRuntime": int(real["runtimeMinutes"].median()),
        },
        "schema": [{"col": c, "type": t, "rule": r, "null": n} for c, t, r, n in SCHEMA],
        "genresAllowed": ALLOWED_GENRES,
        "validation": issues,
        "syn": {
            "id": li(syn["movie_id"], str), "title": li(syn["title"], str),
            "year": [int(x) for x in syn["release_year"]], "genre": li(syn["genre"], str),
            "runtime": li(syn["runtime_minutes"], lambda v: int(v)),
            "country": li(syn["country"], str),
            "rating": li(syn["average_rating"], lambda v: round(float(v), 1)),
            "votes": li(syn["num_votes"], lambda v: int(v)),
            "budget": li(syn["budget_million"], float),
            "revenue": li(syn["revenue_million"], float),
            "content": li(syn["content_rating"], str),
            "decade": [int(x) for x in syn["decade"]],
        },
        "real": {
            "ratingHistPct": real_rating, "ratingEdges": [round(x, 1) for x in rating_edges.tolist()],
            "runtimeHistPct": real_runtime, "runtimeEdges": [int(x) for x in runtime_edges.tolist()],
            "genreShare": {k: round(v, 2) for k, v in gshare.items()},
            "decadeShare": {str(int(k)): round(v, 2) for k, v in dshare.items()},
            "scatter": [{"v": int(v), "a": float(a)} for v, a in zip(rs["numVotes"], rs["averageRating"])],
        },
        # preview: all synthetic rows + a sample of real, source-flagged
        "preview": pd.concat([
            syn_u.head(60)[["source", "id", "title", "year", "genre", "runtime", "rating", "votes"]],
            real_u.sample(n=40, random_state=3)[["source", "id", "title", "year", "genre", "runtime", "rating", "votes"]],
        ]).where(lambda d: pd.notna(d), None).to_dict(orient="records"),
    }

    with open(TPL, encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("/*__DATA__*/null",
                       json.dumps(clean_nan(payload), separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False))
    out = os.path.join(HERE, "synthetic_dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return payload, out


if __name__ == "__main__":
    raw_n, syn, rev_gt_pct = clean()
    payload, out = merge_and_build(raw_n, syn, rev_gt_pct)
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    print(f"Raw rows: {raw_n}  ->  clean rows: {len(syn)}  (dropped {raw_n-len(syn)})")
    print(f"Validation issues logged: {len(issues)}")
    from collections import Counter
    for sev, n in sorted(Counter(i['sev'] for i in issues).items(), key=lambda x: sev_order[x[0]]):
        print(f"   {sev:9s}: {n}")
    print(f"Merged: {payload['meta']['realN']:,} real + {len(syn)} synthetic = {payload['meta']['combinedN']:,} rows")
    print(f"revenue > budget in {rev_gt_pct}% of synthetic rows (LLM optimism)")
    print(f"Wrote synthetic_movies_clean.csv, combined_movies.csv, {os.path.basename(out)}")
