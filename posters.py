"""Optional poster lookups. Network/image failures never break recommendations."""
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


def get_json(url, headers=None):
    request = Request(url, headers={"User-Agent": "CineMatchStudentApp/1.0", **(headers or {})})
    try:
        with urlopen(request, timeout=4) as response:
            return json.load(response)
    except Exception:
        return {}


def safe_image(url):
    parsed = urlparse(url or "")
    if parsed.scheme == "https" and parsed.hostname in {"image.tmdb.org", "upload.wikimedia.org"}:
        return url
    return ""


def poster_for_movie(movie, tmdb_token=""):
    title, year, kind, imdb_id = movie
    year = str(year or "")
    if tmdb_token:
        headers = {"Authorization": f"Bearer {tmdb_token}", "accept": "application/json"}
        media = "tv" if kind == "SHOW" else "movie"
        if re.fullmatch(r"tt\d+", imdb_id or ""):
            data = get_json(
                f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id", headers,
            )
            hits = data.get(f"{media}_results", [])
        else:
            params = {"query": title, "include_adult": "false"}
            if year:
                params["first_air_date_year" if media == "tv" else "year"] = year
            data = get_json(f"https://api.themoviedb.org/3/search/{media}?{urlencode(params)}", headers)
            hits = data.get("results", [])
            # Require an exact normalized title and year for title-only searches.
            def norm(text):
                return re.sub(r"\W", "", unicodedata.normalize("NFKD", text).casefold())
            hits = [h for h in hits if norm(h.get("name", h.get("title", ""))) == norm(title)
                    and (not year or h.get("first_air_date", h.get("release_date", "")).startswith(year))]
        for hit in hits:
            path = hit.get("poster_path")
            if path:
                return {"url": safe_image(f"https://image.tmdb.org/t/p/w500{path}"), "source": "TMDb",
                        "page": f"https://www.themoviedb.org/{media}/{hit['id']}"}
    suffix = "TV_series" if kind == "SHOW" else "film"
    candidates = [f"{title}_({year}_{suffix})"] if year else []
    candidates += [title, f"{title}_({suffix})"]
    for name in candidates:
        data = get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(name.replace(" ", "_"), safe=""))
        if data.get("type") == "disambiguation":
            continue
        description = data.get("description", "").lower()
        if not any(x in description for x in ["film", "television", "series", "documentary"]):
            continue
        dates = re.findall(r"\b(?:19|20)\d{2}\b", description)
        if year and dates and year not in dates:
            continue
        image = safe_image(data.get("thumbnail", {}).get("source", ""))
        page = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if image:
            return {"url": image, "source": "Wikipedia", "page": page}
    return {"url": "", "source": "", "page": ""}


def poster_batch(movies, tmdb_token=""):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda movie: poster_for_movie(movie, tmdb_token), movies))
