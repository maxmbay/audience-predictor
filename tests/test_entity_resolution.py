from audience_predictor.data.entity_resolution import normalize_title, resolve_imdb, resolve_title_year
import polars as pl


def test_normalize_flips_article():
    assert normalize_title("Notebook, The (2004)") == normalize_title("The Notebook (2004)")
    assert normalize_title("The Dark Knight (2008)") == "dark knight"


def test_resolve_imdb_and_title(monkeypatch):
    movies = pl.DataFrame(
        {
            "movie_row": [0, 1],
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "imdbId": ["0114709", "0113497"],
            "tmdbId": [862, 8844],
        }
    )
    hit = resolve_imdb(movies, "tt0114709")
    assert hit["title"][0] == "Toy Story (1995)"
    hit2 = resolve_title_year(movies, "Jumanji", 1995)
    assert hit2.height == 1
