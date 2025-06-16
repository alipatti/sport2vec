import os
import time
from pathlib import Path
import datetime

from nba_api.stats.endpoints import PlayByPlayV3, LeagueGameFinder
from nba_api.stats.static.teams import get_teams
from nba_api.stats.static.players import get_players
import polars as pl
from polars_cache import cache
from rich.progress import track
from rich import print
import inflection

from sport2vec.helpers import rate_limit


DATA_DIRECTORY = Path("data")

TEAMS = pl.from_dicts(get_teams())
PLAYERS = pl.from_dicts(get_players())


class Filters:
    BETWEEN_NBA_TEAMS = pl.col("home").is_in(TEAMS["abbreviation"]) & pl.col(
        "away"
    ).is_in(TEAMS["abbreviation"])
    HOME_ONLY = pl.col("vs_or_at") == "vs."  # so we don't double-count
    PBP_ONLY = pl.col("season") >= 1996  # PBP exists for 1996 onwards
    GAME_COMPLETE = pl.col("wl").is_in(("W", "L"))


@cache(expires_after=datetime.timedelta(days=1), verbose=0)
def _raw_game_df() -> pl.DataFrame:
    games_by_team: list[pl.DataFrame] = [
        pl.from_pandas(LeagueGameFinder(team_id_nullable=id).get_data_frames()[0])
        for id in rate_limit(track(TEAMS["id"], description="Fetching games by team"))
    ]

    return pl.concat(games_by_team, how="vertical_relaxed")


def get_game_df() -> pl.DataFrame:
    return (
        _raw_game_df()
        .rename(inflection.underscore)
        .with_columns(
            pl.col("matchup")
            .str.split(" ")
            .list.to_struct(fields=("home", "vs_or_at", "away"))
            .struct.unnest(),
        )
        .filter(
            # Filters.PBP_ONLY,
            Filters.HOME_ONLY,
            Filters.BETWEEN_NBA_TEAMS,
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


def _scrape_game(game_id: int) -> pl.LazyFrame:
    df = pl.from_pandas(PlayByPlayV3(id).get_data_frames()[0])


def scrape_raw_pbp(
    games: pl.DataFrame,
    delay_ms: int = 250,
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

    for game in rate_limit(
        track(
            games.rows(named=True),
            description=f"Scraping pbp from {len(games)} games...",
        ),
        delay_ms=delay_ms,
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
