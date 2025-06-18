import inflection
import polars as pl
from rich.progress import track

from sport2vec.api import api_requests


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


def _raw_games(start_year: int = 1983, end_year: int = 2025) -> pl.DataFrame:
    seasons = track(
        _seasons(start_year, end_year),
        description=f"Fetching games from {start_year} through {end_year}...",
        transient=True,
    )
    params = ({"Season": season} for season in seasons)

    return pl.concat(
        (
            pl.from_records(
                json["resultSets"][0]["rowSet"],
                schema=json["resultSets"][0]["headers"],
                orient="row",
                infer_schema_length=500,
            )
            for json in api_requests("leaguegamefinder", params)
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
