from typing import Collection

from nba_api.stats.static.teams import get_teams
from nba_api.stats.static.players import get_players
import requests_cache
import polars as pl
from polars import selectors as cs
from rich.progress import track
import inflection

from sport2vec.api import api_requests
from sport2vec.games import all_games

TEAMS = pl.from_dicts(get_teams())
PLAYERS = pl.from_dicts(get_players())


def raw_pbp(game_ids: Collection[str]):
    ids = track(
        game_ids,
        description=f"Fetching play-by-play for {len(game_ids)} games...",
        transient=True,
    )

    request_params = (
        {
            "EndPeriod": 0,
            "GameID": id,
            "StartPeriod": 0,
        }
        for id in ids
    )

    return pl.concat(
        (
            pl.from_records(json["game"]["actions"]).with_columns(
                game_id=pl.lit(
                    json["game"]["gameId"], dtype=pl.Categorical(ordering="lexical")
                )
            )
            for json in api_requests("playbyplayv3", request_params)
        ),
        how="diagonal_relaxed",
    )


def clean_pbp(raw: pl.DataFrame) -> pl.DataFrame:
    event_type = (
        pl.col("action_type")
        .replace(
            {
                "Made Shot": "Shot",
                "Missed Shot": "Shot",
                "Violation": "Foul",
            }
        )
        .cast(pl.Categorical(ordering="lexical"))
        .alias("type")
    )
    event_subtype = (
        pl.col("sub_type").cast(pl.Categorical(ordering="lexical")).alias("subtype")
    )

    minutes_remaining = pl.sum_horizontal(
        pl.col("clock")
        .str.extract_groups(r"PT(?<minutes>\d+)M(?<seconds>[\d\.].+)S")  # get mins/secs
        .struct.with_fields(pl.field("*").cast(float))  # convert to floats
        .struct.with_fields(pl.field("seconds").truediv(60))  # convert secs to mins
        .struct.unnest()
    ).alias("mins_remaining")

    action_id = (
        pl.format(
            "{}-{}",
            "game_id",
            pl.col("action_id").cast(str).str.pad_start(4, "0"),
        )
        .cast(pl.Categorical(ordering="lexical"))
        .alias("action_id")
    )

    score = cs.starts_with("score_").str.to_integer().fill_null(strategy="forward")
    points = pl.sum_horizontal(score.diff()).over("game_id").alias("points")

    free_throw = (
        pl.when(event_type == "Free Throw")
        .then(
            pl.struct(
                event_subtype.cast(str)
                .str.extract_groups(r"(?<attempt>\d) of (?<total>\d)")
                .struct.with_fields(pl.field("*").cast(int))
                .struct.unnest(),
                made=points > 0,
            )
        )
        .alias("free_throw")
    )

    # basket is always 0, 0
    # distances are in tenths of feet (for some reason...)
    location = pl.concat_arr("x_legacy", "y_legacy") / 10

    shot = (
        pl.when(event_type == "Shot")
        .then(
            pl.struct(
                points="shot_value",
                type=event_subtype,
                location=location,
                distance=(location * location).arr.sum().sqrt(),
                angle=(location.arr.get(0) / location.arr.get(1)).arctan(),
                assist=pl.col("description").str.extract(r"\((.+?) \d+ AST\)"),
            )
        )
        .alias("shot")
    )

    other_events = [
        pl.when(event_type == t)
        .then(
            pl.struct(
                type=event_subtype,
            )
        )
        .alias(inflection.underscore(t))
        for t in ["Foul", "Turnover", "Timeout"]
    ]

    clean = (
        raw
        # rename to snake case
        .rename(inflection.underscore)
        # replace empty strings with nulls
        .with_columns(
            pl.col(str).str.strip_chars().replace({"": None}),
            (cs.ends_with("_id") & cs.numeric()).replace({0: None}),
        )
        .select(
            # ids
            "game_id",
            action_id,
            pl.col(
                "team_id", "person_id"
            ),  # .cast(pl.Categorical(ordering="lexical")),
            # time
            "period",
            minutes_remaining,
            # event
            event_type,
            event_subtype,
            shot,
            free_throw,
            *other_events,
            points,
        )
    )

    return clean


if __name__ == "__main__":
    requests_cache.install_cache(".http_cache")
    pl.enable_string_cache()

    games = all_games()
    game_ids = games.get_column("game_id").head(1000)

    plays = raw_pbp(game_ids).pipe(clean_pbp)
