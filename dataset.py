import argparse
import ast
from collections import defaultdict
import csv
import importlib
import importlib.util
import logging
import os.path as osp
import polars as pl
import typing

logger = logging.getLogger(__name__)

# max number of unique entries in a categorical data type
MAX_CATEGORIES = 20
# values that will be interpreted as null
NULL_VALUES = ["", "null", "NULL", "Null", "unknown", "UNKNOWN", "Unknown"]
# bool maps
VALUE_TO_BOOL = {
    pl.Int64: {0: False, 1: True},
    pl.Categorical: {"No": False, "Yes": True},
}


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
            return pl.Float64
        except ValueError:
            # missing data seems to be blank or null
            if value in NULL_VALUES:
                return pl.Null
            return pl.Utf8


def resolve_type(possible_types, possible_values) -> pl.DataType:
    """
    Convert from a set of possible types to a single deduced type
    """

    # if it's just nulls, we can take it to be a string
    if pl.Null in possible_types:
        return pl.Utf8

    raise RuntimeError("Should not reach this. Fix this function")


def deduce_schema(file: str) -> dict[str, pl.DataType]:
    """
    Deduce the schema for `file` by reading in all rows of data in the
    file and determining the type of each entry. Returns a dictionary mapping
    column names to the deduced polars data type
    """
    logger.info(f"Deducing schema from {file}")

    # dict to store all types that we see
    values: dict[str, set[str]] = defaultdict(set)

    # read the data into a dictionary per row
    with open(file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            for column, value in row.items():
                # attempt to parse literal
                # will read ints/floats as the right type
                values[column].add(value)

    # deduce the schame
    schema: dict[str, pl.DataType] = {}
    for column, possible_values in values.items():
        possible_types = set(map(get_polars_type, possible_values))

        # nulls are not valid values
        for null_value in NULL_VALUES:
            possible_values.discard(null_value)

        if pl.Utf8 in possible_types:
            if len(possible_values) < MAX_CATEGORIES:
                # categorical data
                schema[column] = pl.Categorical
                # enum would be ideal, but is not currently fully supported by polars
                # schema[column] = pl.Enum(list(sorted(possible_values)))
            else:
                # string data
                schema[column] = pl.Utf8

        elif pl.Float64 in possible_types:
            # float data
            schema[column] = pl.Float64
        elif pl.Int64 in possible_types:
            # int data
            schema[column] = pl.Int64
        elif pl.Null in possible_types:
            # no valid data, assume string
            schema[column] = pl.Utf8
        else:
            raise RuntimeError("This should not be reached")

        if len(possible_types) > 1:
            logger.debug(
                f"Found multiple possible types {possible_types} for {column}, inferred {schema[column]}"
            )

    return schema


def load_schema(schema_path: str) -> dict[str, pl.DataType]:
    """
    Loads the schema stored in `schema_path` and returns it as a dict
    """
    logger.info(f"Loading schema from {schema_path}")
    module_name = osp.basename(schema_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return schema.SCHEMA


def preprocess(
    dataset: pl.LazyFrame, schema: dict[str, pl.DataType]
) -> tuple[pl.LazyFrame, dict[str, pl.DataType]]:
    """Converts data fields with non-bool entries to bools"""

    # automatically infer booleans for Yes/No or 1/0 data
    for col in schema:
        if schema[col] in VALUE_TO_BOOL:
            # don't consider nulls as values when checking if it's a boolean
            unique: pl.DataFrame = dataset.select(col).unique().collect()
            unique_set: set = set(unique[col].to_list()) - {None}

            value_to_bool = VALUE_TO_BOOL[schema[col]]
            if unique_set == value_to_bool.keys():
                logging.debug(f"Converting {col} with type {schema[col]} to bool")
                dataset = dataset.with_columns(
                    [
                        pl.col(col).replace_strict(value_to_bool).alias(col),
                    ]
                )
                schema[col] = pl.Boolean

    # manual mapping of a few columns
    htn_meds_col = "HTN_MEDS" if "HTN_MEDS" in schema else "NBHTN_MEDS"
    dataset = dataset.with_columns(
        [
            (pl.col("DIABETES").is_in(["Insulin", "Yes, insulin"])).alias(
                "DIABETES_INSULIN_BOOL"
            ),
            (pl.col("DIABETES").is_in(["Non-Insulin", "Yes, non-insulin"])).alias(
                "DIABETES_NONINSULIN_BOOL"
            ),
            (pl.col(htn_meds_col) != "0").alias("HTN_MEDS_BOOL"),
            (pl.col("IVC_TIMING").is_not_null()).alias("IVC_TIMING_BOOL"),
        ]
    )
    schema["DIABETES_INSULIN_BOOL"] = pl.Boolean
    schema["DIABETES_NONINSULIN_BOOL"] = pl.Boolean
    schema["HTN_MEDS_BOOL"] = pl.Boolean
    schema["IVC_TIMING_BOOL"] = pl.Boolean

    return dataset, schema


def load_dataset(
    in_file, schema_path=None
) -> tuple[pl.LazyFrame, dict[str, pl.DataType]]:
    """
    Loads the dataset in `in_file` using the schema in `schema_path`. If `schema_path`
    is not provided, then deduce the schema.
    """
    schema = (
        load_schema(schema_path) if schema_path is not None else deduce_schema(in_file)
    )

    dataset: pl.LazyFrame = pl.scan_csv(
        in_file, separator="\t", schema=schema, null_values=NULL_VALUES
    )

    return preprocess(dataset, schema)


def main(in_file: str, out_file: str):
    """
    Deduce the schema of `in_file` and store the result
    into an importable `out_file`
    """
    schema = deduce_schema(in_file)
    logger.info(f"Finished deducing schema, now writing out to {out_file}")
    with open(out_file, "w") as f:
        f.write("import polars as pl\n")
        f.write("\n")
        f.write("SCHEMA = {\n")
        for column, deduced_type in schema.items():
            f.write(f'    "{column}": pl.{deduced_type},\n')

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
