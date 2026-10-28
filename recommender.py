"""Data preparation and ranking, independent of the Streamlit interface."""
import ast
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            result = ast.literal_eval(text)
            if isinstance(result, (list, tuple)):
                return [str(x).strip() for x in result if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [x.strip(" '[]\"") for x in re.split(r"[,|]", text) if x.strip(" '[]\"")]


def read_csv(data):
    return pd.read_csv(io.BytesIO(data), dtype={"id": "string"}, encoding="utf-8-sig")


def prepare_movies(titles_bytes, credits_bytes):
    titles, credits = read_csv(titles_bytes), read_csv(credits_bytes)
    for frame, required, label in [
        (titles, {"id", "title"}, "Titles"),
        (credits, {"id", "name", "role"}, "Credits"),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} CSV is missing columns: {', '.join(sorted(missing))}")
    if titles["id"].isna().any() or titles["id"].duplicated().any():
        raise ValueError("Titles must have a non-empty, unique id for each row.")
    titles = titles.reset_index(drop=True).copy()
    credits = credits.dropna(subset=["id", "name"]).copy()
    credits["name"] = credits["name"].astype(str).str.strip()
    credits["role"] = credits["role"].fillna("").astype(str).str.strip().str.upper()
    credits = credits[credits["name"].ne("")]
    # ACTOR is important: grouping every credit would also include directors.
    for role, column in [("ACTOR", "actors"), ("DIRECTOR", "directors")]:
        grouped = (
            credits[credits["role"].eq(role)]
            .drop_duplicates(["id", "name"])
            .groupby("id")["name"]
            .agg(lambda names: " | ".join(names))
        )
        fallback = titles[column].fillna("") if column in titles else ""
        titles[column] = titles["id"].map(grouped).fillna(fallback)
    for column in ["title", "type", "description", "genres", "production_countries", "imdb_id"]:
        if column not in titles:
            titles[column] = ""
        titles[column] = titles[column].fillna("").astype(str)
    titles["type"] = titles["type"].str.upper()
    for column in ["release_year", "runtime", "imdb_score", "imdb_votes"]:
        if column not in titles:
            titles[column] = np.nan
        titles[column] = pd.to_numeric(titles[column], errors="coerce")
    titles["imdb_score"] = titles["imdb_score"].where(titles["imdb_score"].between(0, 10))
    titles["genre_list"] = titles["genres"].map(parse_list)
    titles["genre_list"] = titles["genre_list"].map(lambda xs: [x.lower() for x in xs])
    titles["genre_text"] = titles["genre_list"].map(lambda xs: ", ".join(xs))
    titles["country_list"] = titles["production_countries"].map(parse_list)
    titles["profile"] = (
        "Title: " + titles["title"] + ". "
        + "Type: " + titles["type"] + ". "
        + "Genres: " + titles["genre_text"] + ". "
        + "Actors: " + titles["actors"] + ". "
        + "Director: " + titles["directors"] + ". "
        + titles["description"]
    )
    titles["normalized_title"] = titles["title"].map(normalize)
    return titles


def filter_movies(movies, types, genres, years, minimum_rating, rated_only, india_only):
    mask = pd.Series(True, index=movies.index)
    if types:
        mask &= movies["type"].isin(types)
    if genres:
        chosen = set(genres)
        mask &= movies["genre_list"].map(lambda xs: bool(chosen.intersection(xs)))
    mask &= movies["release_year"].between(*years)
    if minimum_rating > 0:
        mask &= movies["imdb_score"].ge(minimum_rating)
    elif rated_only:
        mask &= movies["imdb_score"].notna()
    if india_only:
        mask &= movies["country_list"].map(lambda xs: "IN" in xs)
    return movies.loc[mask].copy()


def build_engine(profiles, mode="Semantic AI"):
    if mode == "Semantic AI":
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device="cpu")
        embeddings = model.encode(
            list(profiles), batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return {"mode": mode, "model": model, "index": index, "count": len(profiles)}
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(profiles)
    return {"mode": "Keyword", "vectorizer": vectorizer, "matrix": matrix, "count": len(profiles)}


def query_scores(engine, query):
    if engine["mode"] == "Semantic AI":
        q = engine["model"].encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        )
        q = np.ascontiguousarray(q, dtype=np.float32)
        # Retrieve the full small catalog, then filter. This keeps the correct top-k
        # even when a genre/year filter removes the first unfiltered matches.
        distances, indices = engine["index"].search(q, engine["count"])
        scores = np.full(engine["count"], -1.0, dtype=np.float32)
        valid = indices[0] >= 0
        scores[indices[0][valid]] = distances[0][valid]
        return scores
    q = engine["vectorizer"].transform([query])
    return (engine["matrix"] @ q.T).toarray().ravel()


def rank_movies(movies, eligible, scores, query, k=6, seed_id=None):
    if len(scores) != len(movies):
        raise ValueError("Search index and movie rows do not match. Rebuild the index.")
    result = eligible.copy()
    result["score"] = np.asarray(scores)[result.index.to_numpy()]
    if seed_id is not None:
        result = result[result["id"].ne(seed_id)]
        result = result.sort_values("score", ascending=False, kind="stable")
    else:
        q = normalize(query)
        result["title_priority"] = result["normalized_title"].map(
            lambda title: 2 if q and title == q else 1 if q and q in title else 0
        )
        result = result.sort_values(["title_priority", "score"], ascending=False, kind="stable")
    return result.head(min(max(int(k), 1), len(result)))


def local_csv_bytes(directory, prefix):
    """Accept Windows download suffixes such as titles_updated (1).csv."""
    directory = Path(directory)
    exact = directory / f"{prefix}_updated.csv"
    if exact.exists():
        return exact.read_bytes()
    files = sorted(directory.glob(f"{prefix}_updated*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].read_bytes() if files else None
