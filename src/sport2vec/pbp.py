from typing import Collection

from nba_api.stats.static.teams import get_teams
from nba_api.stats.static.players import get_players
import requests
import requests_cache
import polars as pl
from polars import selectors as cs
from rich.progress import track
import inflection

from sport2vec import API_DELAY_MS, API_URL, HEADERS
from sport2vec.games import all_games
from sport2vec.helpers import rate_limit

TEAMS = pl.from_dicts(get_teams())
PLAYERS = pl.from_dicts(get_players())


def _single_game_raw_pbp(
    game_id: str,
) -> tuple[bool, pl.DataFrame | None]:
    endpoint = "playbyplayv3"

    params = {
        "EndPeriod": 0,
        "GameID": game_id,
        "StartPeriod": 0,
    }

    try:
        with requests.get(API_URL + endpoint, params=params, headers=HEADERS) as r:
            data = r.json()["game"]
            from_cache = getattr(r, "from_cache", False)

    except (TimeoutError, KeyError):
        return (False, None)

    df = pl.from_records(data["actions"]).with_columns(
        game_id=pl.lit(data["gameId"], dtype=pl.Categorical(ordering="lexical"))
    )
    # except:
    #   return (from_cache, None)

    return (from_cache, df)


def _raw_pbp(game_ids: Collection[str]):
    ids = track(
        game_ids,
        description=f"Scraping pbp from {len(game_ids)} games...",
        transient=True,
    )

    return pl.concat(
        (
            df
            for _, df in rate_limit(
                map(_single_game_raw_pbp, ids),
                delay_ms=API_DELAY_MS,
                # don't throttle when using cache
                limit_when=lambda x: not x[0],
            )
            if df is not None
        ),
        how="diagonal_relaxed",
    )


def clean_pbp(game_ids: Collection[str]) -> pl.DataFrame:
    raw = _raw_pbp(game_ids)

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

    plays = clean_pbp(game_ids)
