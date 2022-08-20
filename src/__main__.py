from pathlib import Path
import click

from .clean import clean_json

@click.Command('clean')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--make-copy', default=True)
def clean(file_path, make_copy):
    clean_json(file_path=Path(file_path), make_copy=make_copy)
