from typing import Collection, Literal

from nba_api.stats.static.teams import get_teams
from nba_api.stats.static.players import get_players
import polars as pl
from polars import selectors as cs
from rich.progress import track
import inflection

from sport2vec.data.nba.games import all_games

TEAMS = pl.from_dicts(get_teams())
PLAYERS = pl.from_dicts(get_players())


def raw_pbp(
    game_ids: Collection[str],
    *,
    progress=False,
    endpoint_version: Literal[2, 3] = 3,
):
    from sport2vec.data.nba import NBA_API

    def _df_from_json(json: dict) -> pl.DataFrame:
        if endpoint_version == 3:
            return pl.from_records(json["game"]["actions"]).with_columns(
                game_id=pl.lit(
                    json["game"]["gameId"], dtype=pl.Categorical(ordering="lexical")
                )
            )
        elif endpoint_version == 2:
            return pl.from_records(
                json["resultSets"][0]["rowSet"],
                schema=json["resultSets"][0]["headers"],
                orient="row",
                infer_schema_length=500,
            )

    ids = (
        track(
            game_ids,
            description=f"Fetching play-by-play for {len(game_ids)} games...",
            transient=True,
        )
        if progress
        else game_ids
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
        map(
            _df_from_json,
            NBA_API.requests(f"playbyplayv{endpoint_version}", request_params),
        ),
        how="diagonal_relaxed",
    )


def clean_pbp(game_ids: Collection[str], *, progress=False) -> pl.DataFrame:
    print("Getting raw v2 play-by-play for {len(game_ids)} games...")
    raw_v2 = raw_pbp(game_ids, endpoint_version=2, progress=progress)

    print("Getting raw v3 play-by-play for {len(game_ids)} games...")
    raw_v3 = raw_pbp(game_ids, endpoint_version=3, progress=progress)

    print("Cleaning play-by-play data...")

    action_id = (
        pl.format(
            "{}-{}",
            "game_id",
            pl.col("action_number").cast(str).str.pad_start(4, "0"),
        )
        .cast(pl.Categorical(ordering="lexical"))
        .alias("action_id")
    )

    players_for_action = (
        raw_v2.rename(inflection.underscore)
        .rename({"eventnum": "action_number"})
        .select(
            action_id,
            pl.concat_arr(cs.matches("player._id").replace({0: None})),
        )
    ).get_columns()
    players = action_id.replace_strict(*players_for_action, default=None).alias(
        "players"
    )

    event_type = (
        pl.col("action_type")
        .replace(
            {
                "Made Shot": "Shot",
                "Missed Shot": "Shot",
                "Violation": "Foul",
            }
        )
        .fill_null(
            pl.col("description").str.extract(r"(STEAL|BLOCK)").str.to_titlecase()
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

    score = cs.starts_with("score_").str.to_integer().fill_null(strategy="forward")
    points = pl.sum_horizontal(score.diff()).over("game_id").alias("points")

    # basket is always 0, 0
    # distances are in tenths of feet (for some reason...)
    location = pl.concat_arr("x_legacy", "y_legacy") / 10

    event_details = {
        "Block": pl.struct(
            by=players.arr.get(2),
            on=players.arr.get(0),
        ),
        "Free Throw": pl.struct(
            event_subtype.cast(str)
            .str.extract_groups(r"(?<attempt>\d) of (?<of>\d)")
            .struct.with_fields(pl.field("*").cast(int))
            .struct.unnest(),
            made=points > 0,
        ),
        "Foul": pl.struct(
            by=players.arr.get(0),
            on=players.arr.get(1),
        ),
        "Jump Ball": pl.struct(
            players=pl.concat_arr(players.arr.get(0), players.arr.get(1)),
            recovered_by=players.arr.get(2),
        ),
        "Rebound": pl.struct(
            by=players.arr.get(0),
            # TODO: offensive v. defensive
            # look at preceding event
            # if shot/free throw, then offensive if team is same
            # if block, then offensive if team is different
            type=None,
        ),
        "Shot": pl.struct(
            points="shot_value",
            made=points > 0,
            type=event_subtype,
            location=location,
            distance=(location * location).arr.sum().sqrt(),
            angle=(location.arr.get(0) / location.arr.get(1)).arctan(),
            by=players.arr.get(0),
            assisted_by=players.arr.get(1),
        ),
        "Steal": pl.struct(
            by=players.arr.get(1),
            stolen_from=players.arr.get(0),
        ),
        "Substitution": pl.struct(
            players.arr.get(1).alias("in"),
            players.arr.get(0).alias("out"),
        ),
        "Turnover": pl.struct(
            by=players.arr.get(0),
            recovered_by=players.arr.get(1),
            type=event_subtype,
        ),
    }

    events = [
        pl.when(event_type == t)
        .then(details)
        .alias(inflection.underscore(t))
        for t, details in event_details.items()
    ]

    clean = (
        raw_v3
        # rename to snake case
        .rename(inflection.underscore)
        # replace empty strings with nulls
        .with_columns(
            pl.col(str).str.strip_chars().replace({"": None}),
            (cs.ends_with("_id") & cs.numeric()).replace({0: None}),
        )
        .select(
            # ids
            *("game_id", action_id, "team_id"),
            # time
            *("period", minutes_remaining),
            # event
            *(event_type, points, "description"),
            # specific event details
            *events,
            # for debugging
            # *(event_subtype, players, pl.col("person_id").alias("player_id")),
        )
    )

    return clean


def main():
    pl.enable_string_cache()
    games = all_games()
    game_ids = games.get_column("game_id").head(30)

    pbp = clean_pbp(game_ids)


if __name__ == "__main__":
    main()
