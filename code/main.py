import argparse
import logging
import numpy as np
import polars as pl

import constants
import schema

logger = logging.getLogger(__file__)

def main(file: str):
    logger.info("Deducing schema")
    deduced_schema = schema.deduce_schema(file)

    logger.info("Loading dataset")
    dataset = pl.scan_csv(
        file,
        separator='\t',
        schema=deduced_schema,
        null_values=constants.null_values
    )

    logger.info("Filtering for POSTOPDEEPINCISIONALSSI")
    infection = dataset.filter(pl.col("POSTOPDEEPINCISIONALSSI") == "Yes")
    result = infection.collect()
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("file", help="The path to the dataset file to analyze")

    args = parser.parse_args()
    logger.setLevel(logging.INFO)
    main(args.file)
