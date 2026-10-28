"""Run from a terminal: python -m streamlit run app.py"""
from html import escape
import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from posters import poster_batch
from recommender import (
    build_engine, filter_movies, local_csv_bytes, prepare_movies, query_scores, rank_movies,
)


st.set_page_config(page_title="CineMatch · Find your next favourite", page_icon="🍿", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#28113c55 0%,transparent 46%),#090d1b;}
.block-container {max-width:1500px;padding-top:2.5rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#0e1424;border-right:1px solid #263048;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.04em;}
.brand {font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;letter-spacing:-.07em;}
.brand span {color:#fb7185;}
.hero {position:relative;overflow:hidden;margin:22px 0 26px;padding:42px;border-radius:26px;
 background:linear-gradient(115deg,#251838 0%,#1c2345 56%,#123c47 100%);border:1px solid #ffffff17;}
.hero:after {content:'✦';position:absolute;right:6%;top:-60px;font-size:320px;line-height:1;color:#7dd3fc10;transform:rotate(13deg);}
.eyebrow {font-size:.71rem;color:#fda4af;letter-spacing:.18em;font-weight:700;text-transform:uppercase;}
.hero h1 {font-size:clamp(2.1rem,4.1vw,3.7rem);line-height:1.08;margin:14px 0 16px;max-width:780px;position:relative;z-index:1;}
.hero h1 span {background:linear-gradient(90deg,#fb7185,#c4b5fd,#67e8f9);background-clip:text;-webkit-background-clip:text;color:transparent;}
.hero p {color:#bdc7dd;max-width:660px;font-size:1rem;line-height:1.65;margin:0;}
.mini-pills {display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;}
.mini-pills span {padding:6px 11px;border:1px solid #ffffff20;border-radius:30px;font-size:.7rem;color:#dae3f5;}
.stat {padding:19px 23px;background:#121a2b;border:1px solid #27324a;border-radius:17px;min-height:98px;margin-bottom:20px;}
.stat .number {font:700 1.7rem 'Space Grotesk',sans-serif;color:#e8eeff;}
.stat .label {color:#9daec9;font-size:.77rem;margin-top:4px;}
.stat.pink {border-top:2px solid #fb7185;}.stat.blue {border-top:2px solid #67e8f9;}.stat.purple {border-top:2px solid #c4b5fd;}
.section-kicker {font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:#8fa0bf;margin:22px 0 3px;}
.movie-card {overflow:hidden;background:#131b2d;border:1px solid #2a354c;border-radius:18px;margin-top:10px;transition:transform .2s,border-color .2s;}
.movie-card:hover {transform:translateY(-4px);border-color:#b79aff;}
.poster {position:relative;aspect-ratio:2/3;overflow:hidden;background:#19233a;}
.poster img {width:100%;height:100%;object-fit:cover;display:block;}
.poster:after {content:'';position:absolute;inset:55% 0 0;background:linear-gradient(transparent,#080e1fcc);pointer-events:none;}
.rank {position:absolute;top:13px;left:13px;background:#0b1022d9;border:1px solid #ffffff30;border-radius:9px;padding:5px 10px;font-size:.8rem;font-weight:700;z-index:2;}
.rating {position:absolute;bottom:13px;right:13px;background:#0b1022e8;padding:6px 10px;border-radius:9px;color:#facc15;font-weight:700;font-size:.8rem;z-index:2;}
.poster-art {height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:30px;text-align:center;
 background:radial-gradient(ellipse at 20% 10%,#fb71854a,transparent 50%),linear-gradient(145deg,#241f45,#0c3440);}
.poster-art .film {font-size:3.5rem;opacity:.7;margin-bottom:25px;}
.poster-art strong {font:700 1.7rem 'Space Grotesk',sans-serif;line-height:1.15;}
.poster-art small {font-size:.65rem;color:#a4b8d3;text-transform:uppercase;letter-spacing:.18em;margin-top:20px;}
.movie-body {padding:18px 18px 20px;min-height:255px;}
.meta {font-size:.7rem;color:#9cabc7;text-transform:uppercase;letter-spacing:.08em;}
.movie-body h3 {font-size:1.22rem;line-height:1.23;margin:9px 0 12px;color:#f1f5ff;min-height:3rem;}
.tags {display:flex;flex-wrap:wrap;gap:5px;min-height:28px;}
.tag {font-size:.62rem;background:#a78bfa17;color:#d5c7ff;padding:4px 8px;border-radius:6px;}
.cast {font-size:.76rem;color:#b6c3da;line-height:1.65;margin:13px 0;min-height:2.5rem;}
.cast b {color:#e5eaff;}
.match {font-size:.7rem;color:#67e8f9;padding-top:12px;border-top:1px solid #ffffff10;}
.card-link {display:inline-block;margin-top:11px;color:#c4b5fd!important;text-decoration:none;font-size:.74rem;}
.footer {margin-top:32px;padding-top:22px;border-top:1px solid #263048;color:#8393ae;font-size:.72rem;line-height:1.8;}
button[kind="primary"] {background:linear-gradient(95deg,#e9577c,#a855f7)!important;border:0!important;}
div[data-testid="stTextInput"] input {border-radius:11px;}
@media(max-width:700px){.hero{padding:27px;}.block-container{padding-top:1.3rem;}.movie-body{min-height:0;}}
</style>
"""


@st.cache_data(show_spinner=False, max_entries=4)
def load_data(titles_bytes, credits_bytes):
    return prepare_movies(titles_bytes, credits_bytes)


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_engine(profiles, mode):
    return build_engine(profiles, mode)


@st.cache_data(show_spinner=False, ttl=86400, max_entries=100)
def cached_posters(movie_keys, token):
    return poster_batch(movie_keys, token)


def secret_token():
    token = os.getenv("TMDB_READ_ACCESS_TOKEN", "")
    if not token:
        try:
            token = st.secrets.get("TMDB_READ_ACCESS_TOKEN", "")
        except Exception:
            pass
    return token


def number(value, suffix=""):
    return f"{int(value)}{suffix}" if pd.notna(value) else "—"


def show_card(row, poster, rank, is_search):
    title = escape(str(row["title"]))
    rating = f"★ {row['imdb_score']:.1f} IMDb" if pd.notna(row["imdb_score"]) else "IMDb N/A"
    if poster["url"]:
        art = f'<img src="{escape(poster["url"], quote=True)}" alt="Poster for {title}" loading="lazy" referrerpolicy="no-referrer">'
    else:
        art = f'<div class="poster-art"><div class="film">🎞</div><strong>{title}</strong><small>Poster unavailable</small></div>'
    tags = "".join(f'<span class="tag">{escape(g.title())}</span>' for g in row["genre_list"][:3])
    if not tags:
        tags = '<span class="tag">Genre unavailable</span>'
    names = [n.strip() for n in row["actors"].split("|") if n.strip()]
    cast = escape(", ".join(names[:3])) if names else "Cast unavailable"
    more = f" +{len(names)-3} more" if len(names) > 3 else ""
    kind = "Movie" if row["type"] == "MOVIE" else "Series" if row["type"] == "SHOW" else row["type"]
    score = f"Similarity {row['score']:.3f}" if is_search else "From your filtered collection"
    link = ""
    if re.fullmatch(r"tt\d+", row["imdb_id"]):
        link = f'<a class="card-link" href="https://www.imdb.com/title/{row["imdb_id"]}/" target="_blank" rel="noopener noreferrer">View on IMDb ↗</a>'
    st.markdown(f"""
    <article class="movie-card">
      <div class="poster">{art}<div class="rank">#{rank:02d}</div><div class="rating">{rating}</div></div>
      <div class="movie-body"><div class="meta">{number(row['release_year'])} · {escape(kind)} · {number(row['runtime'], ' min')}</div>
      <h3>{title}</h3><div class="tags">{tags}</div>
      <p class="cast"><b>Starring</b><br>{cast}{more}</p><div class="match">{score}</div>{link}</div>
    </article>
    """, unsafe_allow_html=True)
    with st.expander("Story & full cast"):
        st.write(row["description"] or "A story description is not available in this dataset.")
        st.write("**Actors:** " + (row["actors"].replace(" | ", ", ") or "Unavailable"))
        st.write("**Director:** " + (row["directors"].replace(" | ", ", ") or "Unavailable"))
        if pd.notna(row["imdb_votes"]):
            st.caption(f"IMDb votes: {int(row['imdb_votes']):,}")
        if poster["source"]:
            st.caption(f"Poster: {poster['source']}")
            st.markdown(f"[Image source]({poster['page']})")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand">cine<span>match.</span> 🍿</div>', unsafe_allow_html=True)
        st.caption("A little less scrolling. A lot more cinema.")
        st.divider()
        with st.expander("Your CSV files", expanded=False):
            st.caption("Bundled files load automatically. Upload a new pair to replace them for this session.")
            title_upload = st.file_uploader("Titles CSV", type="csv", key="titles_upload")
            credit_upload = st.file_uploader("Credits CSV", type="csv", key="credits_upload")
        if bool(title_upload) != bool(credit_upload):
            st.info("Upload both CSV files to use your own data.")
            st.stop()
    base = Path(__file__).resolve().parent
    if title_upload is not None and credit_upload is not None:
        title_bytes, credit_bytes = title_upload.getvalue(), credit_upload.getvalue()
    else:
        title_bytes = local_csv_bytes(base / "data", "titles") or local_csv_bytes(base, "titles")
        credit_bytes = local_csv_bytes(base / "data", "credits") or local_csv_bytes(base, "credits")
    if title_bytes is None or credit_bytes is None:
        st.info("Upload titles_updated.csv and credits_updated.csv using the sidebar to begin.")
        st.stop()
    try:
        movies = load_data(title_bytes, credit_bytes)
    except Exception as error:
        st.error(f"Could not load your CSV files: {error}")
        st.stop()
    if movies.empty:
        st.warning("The titles CSV contains no movies.")
        st.stop()
    years_available = movies["release_year"].dropna()
    min_year = int(years_available.min()) if not years_available.empty else 1900
    max_year = int(years_available.max()) if not years_available.empty else 2026
    genres_available = sorted({g for xs in movies["genre_list"] for g in xs})
    with st.sidebar:
        st.subheader("Make it your kind of movie")
        media_choice = st.selectbox("Watch type", ["Movies", "Movies & series", "Series"])
        types = ["MOVIE"] if media_choice == "Movies" else ["SHOW"] if media_choice == "Series" else []
        genres = st.multiselect("Genres", genres_available, format_func=lambda g: g.title(), placeholder="Any genre")
        if min_year == max_year:
            years = (min_year, max_year)
            st.caption(f"Release year: {min_year}")
        else:
            years = st.slider("Release years", min_year, max_year, (min_year, max_year))
        minimum_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, 0.0, 0.5)
        rated_only = st.checkbox("Only titles with an IMDb rating", value=False)
        india_only = st.checkbox("Indian productions only", value=False,
                                 help="Checks for IN in production_countries; this can include other Indian languages.")
        count = st.select_slider("Results", options=[3, 6, 9, 12], value=6)
        st.divider()
        with st.expander("Search & poster settings"):
            engine_mode = st.radio("Search engine", ["Semantic AI", "Keyword"],
                                   help="Semantic AI downloads MiniLM on first use. Keyword works without a model download.")
            show_posters = st.checkbox("Load movie posters", value=True)
            token = st.text_input("Optional TMDb read access token", value=secret_token(), type="password",
                                  help="Use the API Read Access Token, not the v3 API key. Without it, Wikipedia images are tried.")
            st.caption("Missing posters use a colourful title card. Poster availability depends on the source.")

    st.markdown('<div class="brand">cine<span>match.</span></div>', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="eyebrow">Your next favourite starts here</div>
        <h1>Big moods.<br><span>Better movie nights.</span></h1>
        <p>Search a title, follow your favourite actor, or describe the story you’re in the mood for.</p>
        <div class="mini-pills"><span>✦ Semantic discovery</span><span>★ IMDb ratings</span><span>🎬 Cast & genres</span></div></div>''',
        unsafe_allow_html=True)
    stats = [(f"{len(movies):,}", "Titles in your collection", "pink"),
             (f"{movies['imdb_score'].notna().sum():,}", "Titles with IMDb ratings", "purple"),
             (f"{len(genres_available)}", "Genres to explore", "blue")]
    for column, (value, label, colour) in zip(st.columns(3), stats):
        with column:
            st.markdown(f'<div class="stat {colour}"><div class="number">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    search_mode = st.radio("Discovery mode", ["Search movies & actors", "Find similar movies"], horizontal=True)
    seed_id = None
    with st.form("search_form"):
        if search_mode == "Search movies & actors":
            text = st.text_input("What would you like to watch?", value="singham",
                                 placeholder="Singham · Salman Khan · a funny heist with a twist")
        else:
            seed_choices = movies[movies["type"].isin(types)] if types else movies
            seed_choices = seed_choices if not seed_choices.empty else movies
            seed_labels = {r['id']: f"{r['title']} ({number(r['release_year'])})" for _, r in seed_choices.iterrows()}
            chosen_id = st.selectbox(
                "Choose a movie you enjoyed", seed_choices["id"].tolist(),
                format_func=lambda key: seed_labels[key],
            )
            seed = movies[movies["id"].eq(chosen_id)].iloc[0]
            text, seed_id = seed["profile"], seed["id"]
        submitted = st.form_submit_button("✦ Find my next watch", type="primary", width="stretch")
    if submitted:
        if not text.strip():
            st.warning("Enter a title, actor, or a short description to search.")
        else:
            st.session_state["last_search"] = {"query": text.strip(), "seed_id": seed_id,
                                               "label": text.strip() if seed_id is None else seed["title"],
                                               "dataset": (len(title_bytes), len(credit_bytes), hash(title_bytes), hash(credit_bytes))}
    current_dataset = (len(title_bytes), len(credit_bytes), hash(title_bytes), hash(credit_bytes))
    saved = st.session_state.get("last_search")
    if saved and saved["dataset"] != current_dataset:
        st.session_state.pop("last_search", None)
        saved = None
    eligible = filter_movies(movies, types, genres, years, minimum_rating, rated_only, india_only)
    active_engine = engine_mode
    if saved:
        try:
            with st.spinner("Finding your kind of cinema… First AI search can take a few minutes."):
                engine = cached_engine(tuple(movies["profile"]), engine_mode)
                scores = query_scores(engine, saved["query"])
        except Exception:
            st.info("Semantic search is unavailable on this computer right now. Using keyword search; you can retry Semantic AI later.")
            active_engine = "Keyword"
            engine = cached_engine(tuple(movies["profile"]), "Keyword")
            scores = query_scores(engine, saved["query"])
        search_pool = eligible if active_engine == "Semantic AI" else eligible.loc[scores[eligible.index.to_numpy()] > 0]
        results = rank_movies(movies, search_pool, scores, saved["query"], count, saved["seed_id"])
        label = f"Similar to {saved['label']}" if saved["seed_id"] else f"Matches for “{saved['label']}”"
    else:
        results = eligible.sort_values(["imdb_score", "imdb_votes"], ascending=False, na_position="last").head(count).copy()
        results["score"] = float("nan")
        label = "Start with something worth watching"
    st.markdown('<div class="section-kicker">The watchlist edit</div>', unsafe_allow_html=True)
    st.subheader(label)
    st.caption(f"{len(results)} shown · {len(eligible):,} titles match your filters" +
               (f" · {active_engine} search" if saved else " · Sorted by IMDb rating, then votes"))
    if results.empty:
        st.info("No matching titles found. Try a different query, a wider year range, fewer genres, or a lower minimum rating.")
    else:
        keys = tuple((r["title"], str(int(r["release_year"])) if pd.notna(r["release_year"]) else "", r["type"], r["imdb_id"])
                     for _, r in results.iterrows())
        posters = cached_posters(keys, token) if show_posters else [{"url": "", "source": "", "page": ""} for _ in keys]
        items = list(results.iterrows())
        for offset in range(0, len(items), 3):
            for column, position in zip(st.columns(3), range(offset, min(offset + 3, len(items)))):
                with column:
                    show_card(items[position][1], posters[position], position + 1, bool(saved))
        with st.expander("View & download results as a table"):
            display = results[["title", "release_year", "type", "actors", "genre_text", "imdb_score", "runtime", "score"]].copy()
            display = display.rename(columns={"genre_text": "genres"})
            st.dataframe(display, width="stretch", hide_index=True)
            st.download_button("Download these results", display.to_csv(index=False).encode("utf-8-sig"),
                               "movie_recommendations.csv", "text/csv")
    st.markdown('''<div class="footer">Similarity measures text relevance; it is separate from the IMDb rating.
        Exact title matches appear first in title search. Cast lists include the available credits, not necessarily every performer.<br>
        IMDb ratings come from your CSV snapshot and do not update automatically. Some 2026 listings may be upcoming.<br>
        Images: Wikipedia / Wikimedia when available. Optional TMDb integration: This product uses the TMDB API but is not endorsed or certified by TMDB.</div>''',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
