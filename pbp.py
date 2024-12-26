import os
import time
from pathlib import Path
from typing import Union
from hashlib import sha3_256

import datetime
from nba_api.stats.endpoints import PlayByPlayV3, LeagueGameFinder
from nba_api.stats.static.teams import get_teams
from nba_api.stats.static.players import get_players
import polars as pl
import typer
from rich.progress import track
from rich.status import Status
from rich import print
import inflection


DATA_DIRECTORY = Path("data")

TEAMS = pl.from_dicts(get_teams())
PLAYERS = pl.from_dicts(get_players())

app = typer.Typer()


def get_game_df() -> pl.DataFrame:
    team_abbreviations = TEAMS["abbreviation"].to_list()
    cache_path = DATA_DIRECTORY / "games" / f"{datetime.date.today()}.parquet"

    try:
        os.makedirs(DATA_DIRECTORY / "games", exist_ok=True)
        df = pl.read_parquet(cache_path)
        print(f"Loading cached game list from {cache_path}")
        return df

    except FileNotFoundError:
        pass

    games_by_team: list[pl.DataFrame] = [
        pl.from_pandas(LeagueGameFinder(team_id_nullable=id).get_data_frames()[0])
        for id in track(TEAMS["id"], description="Fetching games by team")
    ]

    games = (
        pl.concat(games_by_team, how="vertical_relaxed")
        .with_columns(
            pl.col("GAME_DATE").str.to_date().alias("DATE"),
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
        .select(
            "DATE",
            "HOME",
            "AWAY",
            pl.col("GAME_ID"),
            "SEASON_YEAR",
        )
        .rename(lambda col: col.lower())
    )

    print(f"Found {len(games)} games with play-by-play data")

    games.write_parquet(cache_path)

    return games


def scrape_raw_pbp(
    games: pl.DataFrame,
    delay,
    verbose=False,
    retry_failed=False,
):
    os.makedirs(DATA_DIRECTORY / "pbp-raw", exist_ok=True)

    # make file storing ids of failed scrapes
    failed_path = DATA_DIRECTORY / "pbp-raw" / "_failed.txt"

    if not os.path.exists(failed_path):
        failed_file = open(failed_path, "w+")
    else:
        failed_file = open(failed_path, "r+")

    failed_set = failed_file.read().split()

    for game in track(
        games.rows(named=True),
        description=f"Scraping pbp from {len(games)} games...",
    ):
        id = game["game_id"]
        game_name = f"{game['home']} vs. {game['away']} ({str(game['date'])})"

        cache_path = DATA_DIRECTORY / "pbp-raw" / f"{id}.parquet"

        if os.path.exists(cache_path):
            if verbose:
                print(f"[cyan]Previously scraped[/cyan]: {game_name}")
            continue

        if id in failed_set and not retry_failed:
            if verbose:
                print(f"[red]Previously failed[/red]: {game_name}")
            continue

        try:
            df = pl.from_pandas(PlayByPlayV3(id).get_data_frames()[0])
            df.write_parquet(cache_path)

            if verbose:
                print(f"[green][b]Scraped:[/b][/green] {game_name}")

        except Exception:
            failed_file.write(id + "\n")

            if verbose:
                print(f"[red][b]Failed:[/b][/red] {game_name}")

        time.sleep(delay)

    failed_file.close()


def load_raw_pbp(n_games: int | None = None) -> pl.LazyFrame:
    paths = list((DATA_DIRECTORY / "pbp-raw").glob("*.parquet"))[:n_games]

    dfs = [
        pl.scan_parquet(fp)
        for fp in track(
            paths, description=f"Loading {len(paths)} individual games", transient=True
        )
    ]

    return pl.concat(dfs, how="vertical_relaxed")


def clean_raw_pbp(raw_pbp: pl.LazyFrame, games: pl.DataFrame) -> pl.DataFrame:
    df = (
        raw_pbp.join(games.lazy(), left_on="gameId", right_on="game_id")
        .join(TEAMS.lazy(), left_on="teamId", right_on="id", how="left")
        # .join(PLAYERS.lazy(), left_on="personId", right_on="id", how="left")
        .rename(inflection.underscore)
        .with_columns(
            # clean up string cols
            pl.col(pl.String)
            .str.strip_chars()
            .replace("", None)
            .str.to_uppercase()
            .str.replace_all("[ -]+", "_"),
        )
        .select(
            # game info
            pl.col("game_id").cast(pl.Categorical),
            # player info
            pl.col("person_id").replace(0, None).cast(pl.String).cast(pl.Categorical),
            # team info
            pl.when(pl.col("action_type").ne("TIMEOUT"))
            .then(
                # use info from the team abbreviation column
                pl.when(pl.col("abbreviation").eq(pl.col("home")))
                .then(pl.lit("HOME"))
                .when(pl.col("abbreviation").eq(pl.col("away")))
                .then(pl.lit("AWAY"))
                .otherwise(None)
            )
            .otherwise(
                # team abbreviation not provided. have to pull from the event description
                pl.when(pl.col("description").str.contains(r"^[A-Z]*\s"))
                .then(pl.lit("HOME"))
                .otherwise(pl.lit("HOME"))
            )
            .cast(pl.Categorical)
            .alias("team"),
            # time
            pl.col("clock").str.slice(2, 2).str.to_decimal() * 60
            + pl.col("clock").str.slice(5, 5).str.to_decimal(),
            pl.col("period").cast(pl.String).cast(pl.Categorical),
            # action type
            pl.col("action_type").cast(pl.Categorical),
            pl.col("sub_type").cast(pl.Categorical),
            pl.col("shot_result").cast(pl.Categorical),
            # shot location / distance
            pl.when(pl.col("shot_result").is_null())
            .then(None)
            .otherwise(pl.col("x_legacy"))
            .alias("x"),
            pl.when(pl.col("shot_result").is_null())
            .then(None)
            .otherwise(pl.col("y_legacy"))
            .alias("y"),
            pl.col("shot_distance").replace(0, None),
        )
    )

    return df.collect()


@app.command()
def scrape(delay: float = 0.6, verbose: bool = False, n_games: Union[int, None] = None):
    games = get_game_df()

    scrape_raw_pbp(
        games.head(n_games) if n_games else games,
        delay=delay,
        verbose=verbose,
    )


@app.command()
def clean(
    n_games: Union[None, int] = None,
    outfile=str(DATA_DIRECTORY / "pbp-clean" / "{n_games}.parquet"),
    print_output: bool = False,
    write_output: bool = True,
):
    raw = load_raw_pbp(n_games=n_games)
    games = get_game_df()

    s = Status("Cleaning pbp...")
    s.start()
    clean = clean_raw_pbp(raw, games)
    s.stop()

    if write_output:
        s = Status("Writing output...")
        s.start()
        outfile = outfile.format(n_games=clean["game_id"].unique().len())
        dir, _ = outfile.rsplit("/", maxsplit=1)
        os.makedirs(dir, exist_ok=True)
        clean.write_parquet(outfile)
        s.stop()

        print(
            f"[green][b]:heavy_check_mark:[/b][/green] Cleaned play-by-play written to [blue][b]{outfile}"
        )

    if print_output:
        print(clean)

    return clean


if __name__ == "__main__":
    app()
