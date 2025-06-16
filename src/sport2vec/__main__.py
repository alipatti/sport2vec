import os
from pathlib import Path
from rich.status import Status
from rich import print
import typer

from sport2vec.pbp import (
    DATA_DIRECTORY,
    clean_raw_pbp,
    get_game_df,
    load_raw_pbp,
    scrape_raw_pbp,
)

app = typer.Typer()


@app.command()
def scrape(delay: float = 0.6, verbose: bool = False, n_games: int = -1):
    games = get_game_df()

    scrape_raw_pbp(
        games.head(n_games) if n_games > 0 else games,
        delay=delay,
        verbose=verbose,
    )


@app.command()
def clean(
    n_games: int = -1,
    outfile: Path = DATA_DIRECTORY / "pbp-clean" / "{n_games}.parquet",
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
        out = str(outfile).format(n_games=clean["game_id"].unique().len())
        dir, _ = str(outfile).rsplit("/", maxsplit=1)
        os.makedirs(dir, exist_ok=True)
        clean.write_parquet(out)
        s.stop()

        print(
            f"[green][b]:heavy_check_mark:[/b][/green] Cleaned play-by-play written to [blue][b]{out}"
        )

    if print_output:
        print(clean)

    return clean


if __name__ == "__main__":
    app()
