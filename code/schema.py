import argparse
import ast
from collections import defaultdict
import csv
import logging
import polars as pl
import typing

logger = logging.getLogger(__file__)

DEFAULT_NUM_LINES = 10

def dict_map(f: typing.Callable, d: dict):
    """
    Return a new dict containing {k: f(k, v)} for each k, v in `d`.
    """
    return {k: f(k, v) for (k, v) in d.items()}

def get_polars_type(value: str):
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
            return pl.Utf8

def resolve_type(column, possible_types):
    """
    Convert from a set of possible types to a single deduced type
    """
    if len(possible_types) > 1:
        logger.warn(f"Found multiple possible types for {column}, will infer")

    # priority list. we assume that strings can store floats, which can store ints
    for t in [pl.Utf8, pl.Float32, pl.Int64]:
        if t in possible_types:
            return t

    raise RuntimeError("Should not reach this. Update list of types in this function")



def deduce_schema(file: str, num_lines: int = DEFAULT_NUM_LINES):
    """
    Deduce the schema for `file` by reading the first `num_lines`
    rows of data and attempting to determine their data type. Returns
    a dictionary mapping column names to the deduced polars data type.
    """

    # read the first `num_lines` lines of data
    with open(file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        try:
            rows = [next(reader) for _ in range(num_lines)]
        except StopIteration:
            logger.error(f"There are fewer than {num_lines} of data in {file}. Please choose a smaller number")

    # dict to store all types that we see in the first `num_lines` lines
    possible_schema: dict[str, set[pl.DataType]] = defaultdict(set)

    for row in rows:
        for (column, value) in row.items():
            # attempt to parse literal
            # will read ints/floats as the right type
            possible_schema[column].add(get_polars_type(value))


    schema = dict_map(resolve_type, possible_schema)

    return schema


def main(in_file: str, out_file: str, num_lines: int):
    """
    Deduce a schema in `in_file` based on `num_lines` of data
    and store the result into an importable `out_file`
    """
    schema = deduce_schema(in_file, num_lines)
    with open(out_file, 'w') as f:
        f.write("import polars as pl\n")
        f.write("\n")
        f.write("SCHEMA = {\n")
        for (column, deduced_type) in schema.items():
            f.write(f"    \"{column}\": pl.{deduced_type},\n")

        f.write("}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("input", help="Path to dataset file whose schema will be deduced")
    parser.add_argument("output", help="Path to write out schema")
    parser.add_argument("--num_lines", "-n", help="Number of lines from file to read", default=DEFAULT_NUM_LINES, type=int)
    args = parser.parse_args()

    main(args.input, args.output, args.num_lines)
