import argparse
import logging
import numpy as np
import polars as pl

import constants
import schema

logger = logging.getLogger(__name__)


def main(file: str):
    logger.info("Deducing schema")
    deduced_schema = schema.deduce_schema(file)

    logger.info("Loading dataset")
    dataset = pl.scan_csv(
        file, separator="\t", schema=deduced_schema, null_values=constants.null_values
    )

    logger.info("Filtering for POSTOPDEEPINCISIONALSSI")

    # sometimes stored as Yes/No string, sometimes stored as 1/0 Int
    true_value = "Yes" if deduced_schema["POSTOPDEEPINCISIONALSSI"] == pl.Utf8 else 1
    infection = dataset.filter(pl.col("POSTOPDEEPINCISIONALSSI") == true_value)
    result = infection.collect()
    logger.info(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("file", help="The path to the dataset file to analyze")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.file)
