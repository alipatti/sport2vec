from typing import Collection

import inflection
import polars as pl


def _raw_player_info(player_ids: Collection[int]) -> pl.DataFrame:
    from sport2vec.data.nba import NBA_API

    params = [{"PlayerID": pid} for pid in player_ids]
    responses = list(NBA_API.requests("commonplayerinfo", params))

    header = responses[0]["resultSets"][0]["headers"]
    rows = sum((r["resultSets"][0]["rowSet"] for r in responses), start=[])

    return pl.from_records(rows, schema=header, orient="row")


def player_info(player_ids: Collection[int]) -> pl.DataFrame:
    raw = _raw_player_info(player_ids)

    return raw.rename(inflection.underscore).select(
        player_id="person_id",
        name=pl.concat_arr("first_name", "last_name"),
        college="school",
        birthday=pl.col("birthdate").str.to_datetime().dt.date(),
        county="country",
        height=pl.col("height")
        .str.split("-")
        .list.eval(pl.element().str.to_integer(strict=False))
        .pipe(lambda s: s.list.first() * 12 + s.list.last()),
        weight=pl.col("weight").str.to_integer(strict=False),
        positions=pl.col("position").str.split("-").replace({(""): None}),
        years_active=pl.concat_arr("from_year", "to_year").replace(
            {(None, None): None}
        ),
        draft=pl.struct(
            year="draft_year",
            position=pl.col("draft_number"),
        ),
        is_nba=pl.col("nba_flag").eq("Y"),
    )
