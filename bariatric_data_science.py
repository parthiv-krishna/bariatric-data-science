import argparse
import logging
import numpy as np
import polars as pl

import dataset

pl.enable_string_cache()
logger = logging.getLogger(__name__)


def main(file: str, schema: str | None):
    logger.info("Loading dataset")
    data = dataset.load_dataset(file, schema)

    infection = data.filter(
        (pl.col("POSTOPSUPERFICIALINCISIONALSSI") >= 1).or_(
            pl.col("POSTOPDEEPINCISIONALSSI") >= 1,
            pl.col("POSTOPORGANSPACESSI") >= 1,
            pl.col("POSTOPSEPSIS") >= 1,
            pl.col("POSTOPSEPTICSHOCK") >= 1,
        )
    )
    result = infection.collect()
    logger.info(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("file", help="The path to the dataset file to analyze")
    parser.add_argument(
        "--schema",
        "-s",
        default=None,
        help="The path to the dataset schema. If not present, schema will be deduced",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.file, args.schema)
