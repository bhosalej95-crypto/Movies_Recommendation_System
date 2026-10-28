# CineMatch — a colourful Streamlit movie recommender

This app uses your two updated CSV files. It shows poster cards, IMDb ratings,
genres, cast and directors, and supports both title/actor search and similar-movie
recommendations. Your CSV files are already included in `data/`.

## Run on Windows

Use Python 3.11 or 3.12 (64-bit). Extract the ZIP first. Open the extracted
`movie_app` folder, click the File Explorer address bar, type `cmd`, and press Enter.
Then run these commands one at a time:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Python 3.12 is installed instead, use `py -3.12` in the first command.
If you have only one Python installed, you can use `python -m venv .venv`.
The app normally opens at http://localhost:8501. Keep the terminal running.
Press Ctrl+C in that terminal to stop it.

**Run this as a Python file from a terminal, not as one Jupyter Notebook cell.**

## Files

```text
movie_app/
  app.py                    Streamlit page and colourful UI
  recommender.py            CSV processing, embeddings, FAISS and ranking
  posters.py                Optional poster lookup and fallbacks
  requirements.txt          Packages to install
  .streamlit/config.toml    Dark theme
  data/titles_updated.csv   Your movie records
  data/credits_updated.csv  Your actor/director records
```

## Use the app

1. Start with the bundled CSVs, or upload a new pair in the sidebar.
2. Search `singham`, `Salman Khan`, `crime thriller`, or describe a mood.
3. Click **Find my next watch**.
4. Adjust genre, release year, type, rating and Indian-production filters.
5. Choose **Find similar movies**, pick a title, and search again for similar films.
6. Expand **Story & full cast** or download the result table.

Your Windows filenames `titles_updated (1).csv` and `credits_updated (1).csv`
are accepted by the upload widgets. To use them automatically, copy the files to
`data/`. If multiple copies exist, the exact filenames without `(1)` take priority.
Remove the old pair if you want a downloaded `(1)` pair to replace the bundled data.

## Images

- With no key, the app tries Wikipedia's article thumbnail for a matching film.
- Availability depends on the article and network. If no matching image is found,
  a colourful title card appears. Wikipedia thumbnails are not always film posters.
- For better poster coverage, get a **TMDb API Read Access Token** from your own
  TMDb account and paste it into the sidebar's password field. The field expects
  the long Bearer/read-access token, not the short v3 API key.
- Alternatively, create `.streamlit/secrets.toml` containing:

```toml
TMDB_READ_ACCESS_TOKEN = "YOUR_OWN_READ_ACCESS_TOKEN"
```

Do not put a real token in source control or a shared ZIP. No key is included.
IMDb IDs are used to match TMDb posters when available. Poster lookup never
overwrites your IMDb ratings, genres, or cast.

## Search logic

- `credits[credits['role'].eq('ACTOR')]` ensures the actor list excludes directors.
- Credits are deduplicated by film and name before grouping. Movies are kept with
  a left join/map even when cast is unavailable.
- Missing text becomes an empty string. Missing ratings stay missing, and appear
  as **IMDb N/A**. No rating is invented.
- Profiles combine title, type, genres, actors, directors, and description.
- **Semantic AI** downloads `sentence-transformers/all-MiniLM-L6-v2` on first
  use, creates normalized embeddings and a `faiss.IndexFlatIP` index, then caches
  the model/index between Streamlit reruns. First use needs internet and can take
  a few minutes on a laptop. This app uses CPU by default.
- **Keyword** uses TF-IDF without downloading an AI model. It is available from
  the sidebar and is used as a fallback if semantic model loading fails.
- Filters apply before the final top-k selection. Title/actor search puts exact
  title matches and title substring matches first; remaining ranking uses text
  similarity. Similar-movie mode uses the selected movie's whole profile and
  excludes that movie from its own results.
- Similarity is a relevance measurement, not an IMDb rating or a probability.
- Search quality depends on the text in your CSV. Missing plots/genres limit
  recommendations, even when a movie has some actor information.

## Data limitations

The full CSV pair contains records with missing details as discussed in the chat.
It is not restricted to the 545 fully detailed Bollywood records. Some 2026 films
are listed or upcoming, and source cast lists may be partial. **Indian productions
only** checks `IN` in the country list; it is not a Hindi-language filter because
the original schema does not have a language field. A positive minimum IMDb rating
excludes unrated titles; with zero selected, unrated titles remain eligible unless
the rated-only checkbox is selected. Release-year filters exclude unknown years.

## Troubleshooting

- **No module named ...:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **Model download fails:** choose **Keyword** in the sidebar. Check internet before
  retrying Semantic AI.
- **No posters:** check internet, try a TMDb read-access token, or keep title cards.
- **Slow first launch/search:** package installation, model download and initial
  embeddings happen once per environment/session. Later searches use cached data.
- **No results:** broaden the release years, remove genres, or lower the rating filter.
- **Wrong CSV:** upload both titles and credits, not the separate audit/detail files.
- **FAISS install fails:** use Python 3.11 or 3.12, 64-bit, update pip using
  `python -m pip install --upgrade pip`, and retry installation.

## Source documentation

- Streamlit: https://docs.streamlit.io/
- Sentence Transformers: https://www.sbert.net/docs/quickstart.html
- FAISS: https://github.com/facebookresearch/faiss/wiki/Getting-started
- TMDb image/search APIs: https://developer.themoviedb.org/docs/image-basics
- IMDb datasets used for the CSV enrichment: https://data.imdb.com/non-commercial-datasets/

This product uses the TMDB API but is not endorsed or certified by TMDB.
Posters and thumbnails remain the property of their respective rights holders.
