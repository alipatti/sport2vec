from typing import Optional
import inflection
import polars as pl
import requests

from sport2vec import API_DELAY_MS, API_URL, HEADERS
from sport2vec.helpers import rate_limit


class Filters:
    # BETWEEN_NBA_TEAMS = pl.col("home").is_in(TEAMS["abbreviation"]) & pl.col(
    #     "away"
    # ).is_in(TEAMS["abbreviation"])
    HOME_ONLY = pl.col("vs_or_at") == "vs."  # so we don't double-count
    HAS_PBP = pl.col("season") >= 1996  # PBP exists for 1996 onwards
    GAME_COMPLETE = pl.col("wl").is_in(("W", "L"))


def _seasons(start: int, end: int) -> list[str]:
    years = range(start, end + 1)

    regular_seasons = [f"{y}-{(y + 1) % 100:>02}" for y in years]
    postseasons = [f"{y}" for y in years]

    return regular_seasons + postseasons


def _single_season_raw_games(season: str) -> tuple[bool, Optional[pl.DataFrame]]:
    endpoint = "leaguegamefinder"
    params = {"Season": season}

    try:
        with requests.get(API_URL + endpoint, params=params, headers=HEADERS) as r:
            data = r.json()["resultSets"][0]
            from_cache = getattr(r, "from_cache", False)

    except TimeoutError:
        return (False, None)

    print(season, len(data["rowSet"]), from_cache)

    df = pl.from_records(
        data["rowSet"],
        schema=data["headers"],
        orient="row",
        infer_schema_length=500,
    )

    return from_cache, df


def _raw_games(start_year: int = 1983, end_year: int = 2025) -> pl.DataFrame:
    seasons = _seasons(start_year, end_year)

    return pl.concat(
        (
            df
            for _, df in rate_limit(
                map(_single_season_raw_games, seasons),
                delay_ms=API_DELAY_MS,
                # don't throttle when using cache
                limit_when=lambda x: not x[0],
            )
            if df is not None
        ),
        how="diagonal_relaxed",
    )


def all_games() -> pl.DataFrame:
    return (
        _raw_games()
        .rename(inflection.underscore)
        .with_columns(
            pl.col("matchup")
            .str.split(" ")
            .list.to_struct(fields=("home", "vs_or_at", "away"))
            .struct.unnest(),
        )
        .filter(
            Filters.HOME_ONLY,
            # Filters.BETWEEN_NBA_TEAMS,
            Filters.GAME_COMPLETE,
        )
        .select(
            pl.col("game_date").str.to_date().alias("date"),
            "home",
            "away",
            pl.when(wl="W").then("home").when(wl="L").then("away").alias("winner"),
            pl.col("game_id"),
            pl.col("season_id").str.slice(-4).str.to_integer().alias("season"),
        )
        .sort("date", descending=True)
    )
