import argparse
import ast
from collections import defaultdict
import csv
import logging
import polars as pl
import typing

logger = logging.getLogger(__name__)

NULL_VALUES = ["", "null", "NULL"]


def dict_map(f: typing.Callable, d: dict) -> dict:
    """
    Return a new dict containing {k: f(k, v)} for each k, v in `d`.
    """
    return {k: f(k, v) for (k, v) in d.items()}


def get_polars_type(value: str) -> pl.DataType:
    """
    Given `value`, a string containing data, return the polars type
    that describes the type present in that data.
    """
    try:
        int(value)
        return pl.Int64
    except ValueError:
        try:
            float(value)
            return pl.Float32
        except ValueError:
            # missing data seems to be blank or null
            if value in NULL_VALUES:
                return pl.Null
            return pl.Utf8


def resolve_type(column, possible_types) -> pl.DataType:
    """
    Convert from a set of possible types to a single deduced type
    """

    # priority list. we assume that strings can store floats, which can store ints
    for t in [pl.Utf8, pl.Float32, pl.Int64]:
        if t in possible_types:
            if len(possible_types) > 1:
                logger.debug(
                    f"Found multiple possible types {possible_types} for {column}, inferring {t}"
                )
            return t

    # if it's just nulls, we can take it to be a string
    if pl.Null in possible_types:
        return pl.Utf8

    raise RuntimeError("Should not reach this. Fix this function")


def deduce_schema(file: str) -> tuple[dict[str, set[pl.DataType]], dict[str, set[str]]]:
    """
    Deduce the schema for `file` by reading in all rows of data in the
    file and determining the type of each entry. Returns a dictionary mapping
    column names to the deduced polars data type, and a dictionary mapping
    column names to all of the possible values (as strings) in that column.
    """

    # dict to store all types that we see
    possible_schema: dict[str, set[pl.DataType]] = defaultdict(set)
    values: dict[str, set[str]] = defaultdict(set)

    # read the data into a dictionary per row
    with open(file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            for column, value in row.items():
                # attempt to parse literal
                # will read ints/floats as the right type
                t = get_polars_type(value)
                if t not in possible_schema[column]:
                    logger.debug(f"Found new type {t} in {column}: {value}")
                possible_schema[column].add(t)
                values[column].add(value)

    schema = dict_map(resolve_type, possible_schema)

    return schema, values


def main(in_file: str, out_file: str):
    """
    Deduce the schema of `in_file` and store the result
    into an importable `out_file`
    """
    schema, values = deduce_schema(in_file)
    logger.info(f"Finished deducing schema, now writing out to {out_file}")
    with open(out_file, "w") as f:
        f.write("import polars as pl\n")
        f.write("\n")
        f.write("SCHEMA = {\n")
        for column, deduced_type in schema.items():
            num_values = len(values[column])
            examples = list(values[column])[:10]
            f.write(
                f'    "{column}": pl.{deduced_type}, # {num_values} values, first 10 examples {examples}\n'
            )

        f.write("}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument(
        "input", help="Path to dataset file whose schema will be deduced"
    )
    parser.add_argument("output", help="Path to write out schema")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    main(args.input, args.output)