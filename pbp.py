import os
from pathlib import Path
from typing import Iterable

from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder
from nba_api.stats.static.teams import get_teams
import polars as pl
from polars.dependencies import hvplot
from tqdm import tqdm


def get_games() -> pl.DataFrame:
    teams = get_teams()
    team_abbreviations = [t["abbreviation"] for t in teams]

    games_by_team = [
        pl.from_pandas(
            leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team["id"]
            ).get_data_frames()[0]
        )
        for team in tqdm(teams, desc="Fetching games by team", leave=False)
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

    return games


def get_raw_pbp(game_id: str | Iterable[str], cache_dir=Path(".cache")) -> pl.DataFrame:
    if isinstance(game_id, str):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            return pl.read_parquet(cache_dir / (game_id + ".parquet"))

        except FileNotFoundError:
            df: pl.DataFrame = pl.from_pandas(
                playbyplayv3.PlayByPlayV3(game_id).get_data_frames()[0]
            )
            df.write_parquet(cache_dir / (game_id + ".parquet"))
            return df

    plays_by_game = (
        get_raw_pbp(id) for id in tqdm(game_id, desc="Fetching raw play-by-play")
    )

    return pl.concat(plays_by_game, how="vertical_relaxed")


def clean_pbp(raw_pbp: pl.DataFrame) -> pl.DataFrame:
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


if __name__ == "__main__":
    N_GAMES = 100

    games = get_games()
    game_ids = games["GAME_ID"][:N_GAMES]

    raw_pbp = get_raw_pbp(game_ids)
    pbp = clean_pbp(raw_pbp)

    # plot = pbp.drop_nulls(["x", "y"]).plot.scatter(x="x", y="y")
    # hvplot.show(plot)
