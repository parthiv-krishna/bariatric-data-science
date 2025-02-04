import argparse
import csv
import glob
import itertools
import logging
from multiprocessing import Pool
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
import ray

import dataset

pl.enable_string_cache()
logger = logging.getLogger(__name__)
ray.init()  # for parallelization

# if True, disables tuning of decision threshold
FORCE_UNTUNED = False

# for logistic regression
TRAIN_SIZE = 0.75
VAL_SIZE = 0.15
TEST_SIZE = 0.1
RANDOM_SEED = 1729
NUM_ITERS = 1000
EPS = 1e-6

RENAL = [
    "DIALYSIS",
    "RENAL_INSUFFICIENCY",
    "PROGRSRENALINSUF",
    "ACTERENALFAILURE",
]

FEATURES = [
    "ACTERENALFAILURE",
    "AGE",
    "BMI",
    "COPD",
    "DIABETES_INSULIN_BOOL",  # replacing "DIABETES"
    "DIABETES_NONINSULIN_BOOL",  # replacing "DIABETES"
    "DIALYSIS",
    "DRAIN_PLACED",
    "DSSIPATOS",
    "DTOP",
    "GERD",
    "HISPANIC",
    "HTN_MEDS_BOOL",  # replacing "HTN_MEDS" or "NBHTN_MEDS"
    "HYPERLIPIDEMIA",
    "IVC_TIMING_BOOL",  # replacing "IVC_TIMING"
    "MI_ALL_HISTORY",
    "OSSIPATOS",
    "PREVIOUS_SURGERY",
    "PROGRSRENALINSUF",
    "RACE_PUF",
    "RENAL_INSUFFICIENCY",
    "SEPSHOCKPATOS",
    "SEPSISPATOS",
    "SEX",
    "SLEEP_APNEA",
    "SMOKER",
    "SSSIPATOS",
    "STAPLING_PROC",
    "SURGICAL_APPROACH",
    "VENOUS_STASIS",
    "WGT_CLOSEST",
]


INFECTION = [
    "POSTOPSUPERFICIALINCISIONALSSI",
    "POSTOPDEEPINCISIONALSSI",
    "POSTOPORGANSPACESSI",
    "POSTOPSEPSIS",
    "POSTOPSEPTICSHOCK",
]


@ray.remote
def load_one(
    in_file: str, schema_path: str | None
) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, pl.DataType]]:
    """
    Loads X matrix and y vector from the given input file, with an optional
    provided schema. If the schema is not provided, it will be deduced.

    """
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
    result_ids = []
    for in_file in glob.glob(f"{in_dir}/*.txt"):
        logger.info(f"Loading dataset from {in_file}")
        result_ids.append(load_one.remote(in_file, schema_path))

    results = ray.get(result_ids)
    logger.info(f"Finished loading datasets")

    file_Xs = []
    file_ys = []
    schema = {}
    for file_X, file_y, file_schema in results:
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


# F1 balances precision and recall
def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1, pos_label=1)


def get_score_and_confusion_matrix(model: TunedThresholdClassifierCV, X, y_true):
    # check accuracy of the model on the test set
    y_pred = model.predict_proba(X)[:, 1] >= model.best_threshold_

    return f1_score(y_true, y_pred), confusion_matrix(y_true, y_pred)


@ray.remote
def logistic_regression(X: pl.DataFrame, y: pl.DataFrame, seed: int):
    # split data into train and test
    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
        random_state=seed,
        stratify=y.to_numpy(),
    )

    y_train = y_train.ravel()
    y_val = y_val.ravel()

    f1_scorer = make_scorer(f1_score)

    # fit logistic regression model with tuned threshold
    model = TunedThresholdClassifierCV(
        LogisticRegression(max_iter=50000), scoring=f1_scorer, cv=5
    )
    model.fit(X_train, y_train)

    if FORCE_UNTUNED:
        model.best_threshold_ = 0.5

    train_score, train_confusion = get_score_and_confusion_matrix(
        model, X_train, y_train
    )
    val_score, val_confusion = get_score_and_confusion_matrix(model, X_val, y_val)

    return {
        "estimator": model,
        "coefs": model.estimator_.coef_[0],
        "threshold": model.best_threshold_,
        "train_score": train_score,
        "train_confusion": train_confusion,
        "val_score": val_score,
        "val_confusion": val_confusion,
    }


def main(in_dir: str, out_dir: str, schema_path: str | None):
    """Runs the experiment"""

    # load and preprocess data
    X, y, _ = load_all(in_dir, schema_path)
    X_preproc, y_preproc = preprocess(X, y)

    if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != 1:
        raise RuntimeError("TRAIN_SIZE, VAL_SIZE, and TEST_SIZE don't sum to 1")

    train_size, val_size, test_size = (
        round(X_preproc.shape[0] * s) for s in [TRAIN_SIZE, VAL_SIZE, TEST_SIZE]
    )
    val_size = round(X_preproc.shape[0] * VAL_SIZE)
    test_size = round(X_preproc.shape[0] * TEST_SIZE)
    logger.info(
        f"Training on {X_preproc.shape[1]} features from {train_size} examples and validating on {val_size} examples"
    )

    # split into train+val and test sets
    # the test set is not seen until the very end, after the model has been selected
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_preproc,
        y_preproc,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_preproc.to_numpy(),
    )

    # random sampling to train on different subsets of the train+val set
    np.random.seed(RANDOM_SEED)
    seeds = np.random.randint(1, 100000, NUM_ITERS)
    logger.info(f"Running {NUM_ITERS} iterations of logistic regression training/validation")
    result_ids = [
        logistic_regression.remote(X_train_val, y_train_val, seed) for seed in seeds
    ]
    results = ray.get(result_ids)
    logger.info(f"Finished training logistic regression")

    # compute distribution of model parameters
    cols = X_train_val.columns
    coefs = [r["coefs"] for r in results]
    mean = np.mean(coefs, axis=0)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    # write out distribution of model parameters
    thresholds = [r["threshold"] for r in results]
    model_params_distribution_filename = f"{out_dir}/model_params_distribution.csv"
    logger.info(f"Writing model parameters distribution to {model_params_distribution_filename}")
    with open(model_params_distribution_filename, "w", newline="") as f:
        model_params_distribution_writer = csv.writer(f, delimiter=",")
        model_params_distribution_writer.writerow(
            [
                "Column Name",
                "Mean Coefficient",
                "2.5 Percentile Coefficient",
                "97.5 Percentile Coefficient",
            ]
        )
        model_params_distribution_writer.writerow(
            [
                "Decision Threshold",
                np.mean(thresholds),
                np.percentile(thresholds, 2.5),
                np.percentile(thresholds, 97.5),
            ]
        )
        for row in zip(cols, lower_bound, upper_bound, mean):
            model_params_distribution_writer.writerow(row)

    # select best model based on the F1 score on the validation set
    best = max(results, key=lambda r: r["val_score"])

    # get prediction metrics for the best model
    train_metrics = best["train_score"], best["train_confusion"]
    val_metrics = best["val_score"], best["val_confusion"]
    test_metrics = get_score_and_confusion_matrix(best["estimator"], X_test.to_numpy(), y_test.to_numpy())

    # write out prediction metrics and coefficients for the best model
    best_model_filename = f"{out_dir}/best_model.csv"
    logger.info(f"Writing metrics for best model to {best_model_filename}")
    with open(best_model_filename, "w", newline="") as f:
        best_model_writer = csv.writer(f)
        best_model_writer.writerow(["Name", "Value"])

        for name, metrics in zip(
            ["Train", "Val", "Test"], [train_metrics, val_metrics, test_metrics]
        ):
            score, confusion = metrics

            tn, fp, fn, tp = confusion.ravel()

            precision = tp / (tp + fp)
            recall_sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            correct = (tn + tp) / (tn + fp + fn + tp)

            best_model_writer.writerow([f"{name} Precision", precision])
            best_model_writer.writerow(
                [f"{name} Recall/Sensitivity", recall_sensitivity]
            )
            best_model_writer.writerow([f"{name} Specificity", specificity])
            best_model_writer.writerow([f"{name} Correct Prediction Rate", correct])

        best_model_writer.writerow(["Decision Threshold", best["threshold"]])
        best_model_writer.writerow(
            [
                "Decision Threshold Stability Ratio",
                np.mean(thresholds) / (np.std(thresholds) + EPS)
            ]
        )

        for col, coef in zip(cols, best["coefs"]):
            best_model_writer.writerow([f"{col} Coefficient", coef])


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
