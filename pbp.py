import os
import time
from pathlib import Path
from typing import Sequence, Union

import datetime
from nba_api.stats.endpoints import PlayByPlayV3, LeagueGameFinder
from nba_api.stats.static.teams import get_teams
import polars as pl
import typer
from rich.progress import track


CACHE_DIR = Path(".cache")

app = typer.Typer()


def scrape_games() -> pl.DataFrame:
    teams = get_teams()
    team_abbreviations = [t["abbreviation"] for t in teams]
    cache_path = CACHE_DIR / "games" / f"{datetime.date.today()}.parquet"

    try:
        os.makedirs(CACHE_DIR / "games", exist_ok=True)
        df = pl.read_parquet(cache_path)
        print(f"Loading cached game list from {cache_path}")
        return df

    except FileNotFoundError:
        pass

    games_by_team = [
        pl.from_pandas(
            LeagueGameFinder(team_id_nullable=team["id"]).get_data_frames()[0]
        )
        for team in track(teams, description="Fetching games by team")
    ]

    games: pl.DataFrame = (
        pl.concat(games_by_team, how="vertical_relaxed")
        .with_columns(
            pl.col("GAME_DATE").str.to_date(),
            pl.col("MATCHUP").str.split(" ").list.to_struct().alias("MATCHUP_SPLIT"),
            pl.col("SEASON_ID").str.slice(-4).str.to_integer().alias("SEASON_YEAR"),
        )
        .unnest("MATCHUP_SPLIT")
        .rename({"field_0": "HOME", "field_2": "AWAY"})
        .filter(
            pl.col("SEASON_YEAR") >= 1996,  # PBP exists for 1996 onwards
            pl.col("field_1") == "vs.",  # drop away games
            pl.col("HOME").is_in(team_abbreviations),
            pl.col("AWAY").is_in(team_abbreviations),  # only between NBA teams
        )
        .drop("field_1")
        .drop_nulls("WL")  # drop in-progress games
        .unique("GAME_ID")  # count each game only once
        .sort("GAME_DATE")  # sort by date
        .reverse()
        .select("GAME_DATE", "HOME", "AWAY", "GAME_ID", "SEASON_YEAR")
    )

    print(f"Found {len(games)} games with play-by-play data")

    games.write_parquet(cache_path)

    return games


def scrape_raw_pbp(
    game_ids: Sequence[str],
    delay,
    verbose=False,
):
    os.makedirs(CACHE_DIR / "pbp-raw", exist_ok=True)

    for id in track(
        game_ids,
        description=f"Scraping pbp from {len(game_ids)} games...",
    ):
        cache_path = CACHE_DIR / "pbp-raw" / (f"{id}.parquet")

        if os.path.exists(cache_path):
            if verbose:
                print(f"Skipping previously scraped game {id}")
            continue

        try:
            df = pl.from_pandas(PlayByPlayV3(id).get_data_frames()[0])
            df.write_parquet(cache_path)

            if verbose:
                print(f"Successfully scraped game {id}")

        except Exception:
            if verbose:
                print(f"Failed to scrape game {id}")

        time.sleep(delay)


def load_raw_pbp() -> pl.DataFrame:
    paths = os.listdir(CACHE_DIR / "pbp-raw")
    dfs = (pl.read_parquet(fp) for fp in paths)

    return pl.concat(dfs, how="vertical_relaxed")


def clean_raw_pbp(raw_pbp: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_pbp.with_columns(
            pl.col(pl.Utf8).replace("", None).replace("Unknown", None),
        )
        .filter(pl.col("actionType") != "period")
        .select(
            pl.col("clock").str.slice(2, 2).str.to_decimal() * 60
            + pl.col("clock").str.slice(5, 5).str.to_decimal(),
            pl.col("period").cast(pl.String).cast(pl.Categorical),
            pl.col("actionType").cast(pl.Categorical),
            pl.col("subType").cast(pl.Categorical),
            pl.col("shotDistance"),
            pl.col("shotResult").cast(pl.Categorical),
            pl.when(pl.col("shotResult").is_null())
            .then(None)
            .otherwise(pl.col("xLegacy"))
            .alias("x"),
            pl.when(pl.col("shotResult").is_null())
            .then(None)
            .otherwise(pl.col("yLegacy"))
            .alias("y"),
            pl.col("personId"),
        )
        .with_columns()
    )


@app.command()
def scrape(delay: float = 0.6, verbose: bool = False, n_games: Union[int, None] = None):
    games = scrape_games()

    game_ids = games["GAME_ID"][:n_games].to_list()

    scrape_raw_pbp(game_ids, delay=delay, verbose=verbose)


if __name__ == "__main__":
    app()

    # raw_pbp = load_raw_pbp()
    # pbp = clean_raw_pbp(raw_pbp)
