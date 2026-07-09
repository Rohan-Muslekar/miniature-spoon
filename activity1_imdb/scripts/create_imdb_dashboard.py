#!/usr/bin/env python3
"""
create_imdb_dashboard.py  —  Data Mining, Class Activity 1
=========================================================
Cleans the IMDb non-commercial datasets and builds a self-contained,
interactive HTML dashboard (filter sidebar + summary tiles + charts).

Pipeline
--------
    title.basics.tsv.gz  +  title.ratings.tsv
        -> replace IMDb '\\N' with NA
        -> keep titleType == movie, drop adult
        -> numeric startYear / runtimeMinutes, drop missing
        -> inner-join on tconst, keep numVotes >= THRESHOLD
        -> derive decade / mainGenre / runtimeCategory
    => imdb_clean.csv          (the cleaned analysis table)
    => imdb_dashboard.html     (interactive dashboard, data embedded, works offline)

No prediction / classification / clustering — descriptive exploration only.

Usage
-----
    python3 create_imdb_dashboard.py
    python3 create_imdb_dashboard.py --basics <path> --ratings <path> \
            --out-dir <dir> --min-votes 1000
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = "/mnt/d/Downloads/UOIT/Summer 2026/Data Mining/In class assessment"


def runtime_cat(m):
    if m < 90:
        return "Short (<90m)"
    if m <= 120:
        return "Medium (90-120m)"
    return "Long (>120m)"


def build(args):
    basics_path = args.basics or os.path.join(args.out_dir, "title.basics.tsv.gz")
    ratings_path = args.ratings or os.path.join(args.out_dir, "title.ratings.tsv", "title.ratings.tsv")
    if not os.path.exists(ratings_path):
        alt = os.path.join(args.out_dir, "title.ratings.tsv.gz")
        ratings_path = alt if os.path.exists(alt) else ratings_path

    print(f"[1/6] Loading ratings  <- {ratings_path}")
    ratings = pd.read_csv(ratings_path, sep="\t", na_values="\\N",
                          dtype={"tconst": str, "averageRating": float, "numVotes": "Int64"})
    ratings_total = len(ratings)

    print(f"[2/6] Streaming basics <- {basics_path}  (large; chunked)")
    usecols = ["tconst", "titleType", "primaryTitle", "isAdult",
               "startYear", "runtimeMinutes", "genres"]
    basics_total = 0
    movie_chunks = []
    for chunk in pd.read_csv(basics_path, sep="\t", na_values="\\N", usecols=usecols,
                             dtype=str, chunksize=500_000, low_memory=False):
        basics_total += len(chunk)
        movie_chunks.append(chunk[chunk["titleType"] == "movie"])
    basics = pd.concat(movie_chunks, ignore_index=True)
    movies_raw = len(basics)

    # ---- missingness on raw movies (data-understanding) ----
    missing = {c: round(100 * basics[c].isna().mean(), 1)
               for c in ["startYear", "runtimeMinutes", "genres"]}

    print("[3/6] Cleaning")
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")
    basics["isAdult"] = pd.to_numeric(basics["isAdult"], errors="coerce").fillna(0).astype(int)
    adult = int((basics["isAdult"] == 1).sum())

    clean = basics[
        basics["startYear"].notna()
        & basics["runtimeMinutes"].notna()
        & (basics["isAdult"] == 0)
        & (basics["startYear"] >= 1900)
        & (basics["startYear"] <= 2026)
    ].copy()
    after_clean = len(clean)

    print("[4/6] Merging + threshold")
    merged = clean.merge(ratings, on="tconst", how="inner")
    with_ratings = len(merged)
    df = merged[merged["numVotes"] >= args.min_votes].copy()

    # ---- derived columns ----
    df["decade"] = (df["startYear"] // 10 * 10).astype(int)
    df["mainGenre"] = df["genres"].fillna("").str.split(",").str[0].replace("", "(none)")
    df["runtimeCategory"] = df["runtimeMinutes"].apply(runtime_cat)
    df["startYear"] = df["startYear"].astype(int)
    df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
    df["numVotes"] = df["numVotes"].astype(int)
    df["averageRating"] = df["averageRating"].astype(float)
    df = df.reset_index(drop=True)
    final_rows = len(df)
    print(f"      final analysis rows: {final_rows:,}")

    # ---- write cleaned CSV ----
    csv_cols = ["tconst", "primaryTitle", "startYear", "runtimeMinutes",
                "mainGenre", "genres", "averageRating", "numVotes",
                "decade", "runtimeCategory"]
    csv_path = os.path.join(args.out_dir, "imdb_clean.csv")
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"[5/6] Wrote {csv_path}")

    # ---- build compact columnar payload for the dashboard ----
    genres = sorted(df["mainGenre"].unique().tolist())
    gidx = {g: i for i, g in enumerate(genres)}
    rc_order = ["Short (<90m)", "Medium (90-120m)", "Long (>120m)"]
    rcidx = {c: i for i, c in enumerate(rc_order)}

    payload = {
        "meta": {
            "basicsTotal": int(basics_total),
            "ratingsTotal": int(ratings_total),
            "moviesRaw": int(movies_raw),
            "afterClean": int(after_clean),
            "withRatings": int(with_ratings),
            "finalRows": int(final_rows),
            "threshold": int(args.min_votes),
            "adult": adult,
            "missing": missing,
            "yearMin": int(df["startYear"].min()),
            "yearMax": int(df["startYear"].max()),
        },
        "genres": genres,
        "runtimeCats": rc_order,
        # columnar arrays (small keys to keep the file lean)
        "t": df["primaryTitle"].tolist(),
        "y": df["startYear"].tolist(),
        "rt": df["runtimeMinutes"].tolist(),
        "g": df["mainGenre"].map(gidx).astype(int).tolist(),
        "rc": df["runtimeCategory"].map(rcidx).astype(int).tolist(),
        "a": [round(x, 1) for x in df["averageRating"].tolist()],
        "v": df["numVotes"].tolist(),
    }

    print("[6/6] Rendering dashboard")
    tpl_path = os.path.join(HERE, "dashboard_template.html")
    with open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()
    data_js = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = tpl.replace("/*__DATA__*/null", data_js)
    out_path = os.path.join(args.out_dir, "imdb_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"      Wrote {out_path}  ({size_mb:.1f} MB)")
    print("\nDone. Open imdb_dashboard.html in a browser.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the IMDb exploratory dashboard.")
    ap.add_argument("--basics", help="path to title.basics.tsv.gz")
    ap.add_argument("--ratings", help="path to title.ratings.tsv(.gz)")
    ap.add_argument("--out-dir", default=DEFAULT_DIR, help="output directory")
    ap.add_argument("--min-votes", type=int, default=1000, help="minimum numVotes")
    build(ap.parse_args())
