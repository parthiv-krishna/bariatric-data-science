import argparse
import glob
import itertools
import logging
from multiprocessing import Pool
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import dataset

pl.enable_string_cache()
logger = logging.getLogger(__name__)

# for logistic regression
TEST_SIZE = 0.2
RANDOM_SEED = 1729
NUM_ITERS = 10000
EPS = 1e-6

RENAL = [
    "PROGRSRENALINSUF",
    "ACTERENALFAILURE",
]

FEATURES = [
    "ACTERENALFAILURE",
    "AGE",
    "ALBUMIN",
    "ANESTYPE",
    "ASACLASS",
    "BMI_HIGH_BAR",
    "BMI",
    "COPD",
    "CREATININE",
    "DIABETES_INSULIN_BOOL",  # replacing "DIABETES"
    "DIABETES_NONINSULIN_BOOL",  # replacing "DIABETES"
    "DIALYSIS",
    "DISCHARGE_DESTINATION",
    "DRAIN_PLACED",
    "DSSIPATOS",
    "DTOP",
    "FUNSTATPRESURG",
    "GERD",
    "HCT",
    "HEMO",
    "HGT",
    "HISPANIC",
    "HISTORY_DVT",
    "HISTORY_PE",
    "HTN_MEDS_BOOL",  # replacing "HTN_MEDS" or "NBHTN_MEDS"
    "HYPERLIPIDEMIA",
    # "HYPERTENSION", # not in some datasets
    # "IMMUNOSUPR_THER", # not in some datasets
    "IVC_FILTER",
    "IVC_TIMING_BOOL",  # replacing "IVC_TIMING"
    "MI_ALL_HISTORY",
    "OSSIPATOS",
    "PCARD",
    # "PREOP_COVID", # not in some datasets
    "PREVIOUS_SURGERY",
    # "PROCEDURE_TYPE", # not in some datasets
    "PROGRSRENALINSUF",
    "PTC",
    "RACE_PUF",
    "RENAL_INSUFFICIENCY",
    # "ROBOTIC_ASST", # not in some datasets
    "SEPSHOCKPATOS",
    "SEPSISPATOS",
    "SEX",
    "SLEEP_APNEA",
    "SMOKER",
    "SSSIPATOS",
    "STAPLING_PROC",
    "SURGICAL_APPROACH",
    "SURGSPECIALTY_BAR",
    "THERAPEUTIC_ANTICOAGULATION",
    "UTIPATOS",
    "VENOUS_STASIS",
    "WGT_CLOSEST",
    "WGT_HIGH_BAR",
]


INFECTION = [
    "POSTOPSUPERFICIALINCISIONALSSI",
    "POSTOPDEEPINCISIONALSSI",
    "POSTOPORGANSPACESSI",
    "POSTOPSEPSIS",
    "POSTOPSEPTICSHOCK",
]


def load_one(
    in_file: str, schema_path: str | None
) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, pl.DataType]]:
    """
    Loads X matrix and y vector from the given input file, with an optional
    provided schema. If the schema is not provided, it will be deduced.

    """
    logger.info(f"Loading dataset from {in_file}")
    data, schema = dataset.load_dataset(in_file, schema_path)

    # if any of the renal columns are >= 1, this is true
    has_renal_problems = pl.Expr.or_(*[pl.col(c) >= 1 for c in RENAL])
    data_renal_problems = data.filter(has_renal_problems)

    dataset_features = [f for f in FEATURES if f in schema]
    if len(dataset_features) != len(FEATURES):
        missing = [f for f in FEATURES if f not in dataset_features]
        errmsg = f"Some columns were missing in the dataset: {missing}. Crashing to avoid more problems"
        logger.error(errmsg)
        raise RuntimeError(errmsg)
    dataset_schema = {f: schema[f] for f in dataset_features}

    X = data_renal_problems.select(dataset_features)

    # if any of the infection columns are >= 1, this is true
    is_infected = pl.Expr.or_(*[pl.col(c) >= 1 for c in INFECTION])
    y = data_renal_problems.with_columns(is_infected.alias("IS_INFECTED")).select(
        "IS_INFECTED"
    )

    return X, y, dataset_schema


def load_all(
    in_dir: str, schema_path: str | None
) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, pl.DataType]]:
    """
    Load all input files in the in_dir and stack into big X and y for
    logistic regression
    """
    file_Xs = []
    file_ys = []
    schema = {}
    for in_file in glob.glob(f"{in_dir}/*.txt"):
        file_X, file_y, file_schema = load_one(in_file, schema_path)
        file_Xs.append(file_X)
        file_ys.append(file_y)

        mismatch = 0
        for col, dtype in file_schema.items():
            if col not in schema:
                schema[col] = dtype
            elif schema[col] != dtype:
                errmsg = f"Existing type for {col} is {schema[col]} but this dataset has it as {dtype}"
                logger.error(errmsg)
                mismatch += 1
        if mismatch:
            errmsg = f"Mismatches in {mismatch} columns above. Crashing to avoid more problems"
            logger.error(errmsg)
            raise RuntimeError(errmsg)

    X = pl.concat(file_Xs)
    y = pl.concat(file_ys)

    return X, y, schema


def preprocess(X: pl.LazyFrame, y: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    X = X.collect()
    y = y.collect()

    # add intercept/constant column
    X = X.with_columns(pl.lit(1).alias("CONSTANT"))

    # convert categorical data to one-hot encoded
    X = X.with_columns(
        pl.selectors.categorical().cast(pl.Utf8).str.to_lowercase().cast(pl.Categorical)
    )
    X = X.to_dummies(pl.selectors.categorical(), separator="=")

    # fill nulls with average value
    X = X.with_columns(
        [
            pl.col(col).cast(pl.Float64).fill_null(strategy="mean").alias(col)
            for col in X.columns
        ]
    )

    # normalize
    X = X.with_columns(
        [
            ((pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + EPS)).alias(col)
            for col in X.columns
        ]
    )

    return X, y


def logistic_regression(X: pl.DataFrame, y: pl.DataFrame, seed: int):
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=TEST_SIZE, random_state=seed
    )

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # fit logistic regression model
    model = LogisticRegression(max_iter=50000)
    model.fit(X_train, y_train)

    # check accuracy of the model
    y_pred = model.predict(X_test)
    return model, confusion_matrix(y_test, y_pred).ravel()


def main(in_dir: str, out_dir: str, schema_path: str | None):

    X, y, _ = load_all(in_dir, schema_path)
    X_preproc, y_preproc = preprocess(X, y)

    np.random.seed(RANDOM_SEED)
    seeds = np.random.randint(1, 100000, NUM_ITERS)
    logger.info(
        f"Training on {X_preproc.shape[0]*(1-TEST_SIZE)} examples and testing on {X_preproc.shape[0]*TEST_SIZE} examples"
    )

    results = [logistic_regression(X_preproc, y_preproc, seed) for seed in tqdm(seeds)]

    coefs = [r[0].coef_[0] for r in results]
    cols = X_preproc.columns
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    mean = np.mean(coefs, axis=0)
    for col, lb, ub, m in zip(cols, lower_bound, upper_bound, mean):
        if not (lb < EPS and ub > -1 * EPS):
            print(f"{col} {m} [{lb}, {ub}]")

    tn, fp, fn, tp = np.sum([r[1] for r in results])
    print(tn, fp, fn, tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("in_dir", help="The path to the directory containing datasets")
    parser.add_argument(
        "out_dir", help="The path to the directory to write out results"
    )
    parser.add_argument(
        "--schema",
        "-s",
        default=None,
        help="The path to the dataset schema. If not present, schema will be deduced",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.in_dir, args.out_dir, args.schema)
