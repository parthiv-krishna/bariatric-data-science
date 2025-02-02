import argparse
from collections import defaultdict
import glob
import itertools
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import dataset

pl.enable_string_cache()
logger = logging.getLogger(__name__)

# for logistic regression
TEST_SIZE = 0.2
RANDOM_SEED = 1729

RENAL = [
    "PROGRSRENALINSUF",
    "ACTERENALFAILURE",
]

FEATURES = [
    "ACTERENALFAILURE",
    "AGE",
    "ALBUMIN",
    "ANASTOMOSIS_CHECKED",
    # "ANESTYPE": pl.Categorical,
    "APPROACH_CONVERTED",
    # "ASACLASS": pl.Categorical,
    # "BALLOON_TYPE": pl.Categorical,
    "BLEEDING_UNITS",
    "BMI",
    "BMI_HIGH_BAR",
    "BOWELOBSTRUCTION",
    "CARDIACARRESTCPR",
    "CDIFF",
    # "CONVERSION": pl.Categorical,
    "COPD",
    "CREATININE",
    "CVA",
    "DIABETES_INSULIN_BOOL", # replacing "DIABETES"
    "DIABETES_NONINSULIN_BOOL", # replacing "DIABETES"
    "DIALYSIS",
    # "DISCHARGE_DESTINATION": pl.Categorical,
    "DRAIN_PLACED",
    "DSSIPATOS",
    "DTACTERENALFAILURE",
    "DTANASTSLLEAK",
    "DTBOWELOBSTRUCTION",
    "DTCARDIACARRESTCPR",
    "DTCDIFF",
    "DTCVA",
    "DTDEATH_OP",
    "DTDISCH_ADMIT",
    "DTDISCH_OP",
    "DTGITRACTBLEED",
    "DTMYOCARDIALINFR",
    "DTOP",
    "DTPOSTOPDEEPINCISIONALSSI",
    "DTPOSTOPORGANSPACESSI",
    "DTPOSTOPPNEUMONIA",
    "DTPOSTOPSEPSIS",
    "DTPOSTOPSEPTICSHOCK",
    "DTPOSTOPSUPERFINCSSI",
    "DTPOSTOPUTI",
    "DTPOSTOPVENTILATOR",
    "DTPROGRSRENALINSUF",
    "DTPULMONARYEMBOLSM",
    "DTTRANSFINTOPPSTOP",
    "DTUNPLANADMICU",
    "DTUNPLINTUBATION",
    "DTVEINTHROMBREQTER",
    "DTWOUNDDISRUPTION",
    # "FUNSTATPRESURG": pl.Categorical,
    "GERD",
    "GITRACTBLEED",
    "HCT",
    "HEMO",
    "HISPANIC",
    "HISTORY_DVT",
    "HISTORY_PE",
    "HTN_MEDS_BOOL", # replacing "HTN_MEDS" or "NBHTN_MEDS" 
    "HYPERLIPIDEMIA",
    # "HYPERTENSION", # not in some datasets
    # "IMMUNOSUPR_THER", # not in some datasets
    "IVC_FILTER",
    "IVC_TIMING_BOOL", # replacing "IVC_TIMING"
    # "METH_VTEPROPHYL": pl.Categorical,
    "MI_ALL_HISTORY",
    "MYOCARDIALINFR",
    "OPLENGTH",
    "OSSIPATOS",
    "PCARD",
    "PNAPATOS",
    "POSTOPANASTSLLEAK",
    "POSTOPDEEPINCISIONALSSI",
    "POSTOPORGANSPACESSI",
    "POSTOPPNEUMONIA",
    "POSTOPSEPSIS",
    "POSTOPSEPTICSHOCK",
    "POSTOPSUPERFICIALINCISIONALSSI",
    "POSTOPUTI",
    "POSTOPVENTILATOR",
    # "POSTOP_COVID": pl.Categorical,
    # "PREOP_COVID": pl.Categorical,
    "PREVIOUS_SURGERY",
    # "PROCEDURE_TYPE": pl.Categorical,
    "PROGRSRENALINSUF",
    "PTC",
    "PULMONARYEMBOLSM",
    # "RACE_PUF": pl.Categorical,
    "RENAL_INSUFFICIENCY",
    "ROBOTIC_ASST",
    "ROBOTIC_ASST_CONV",
    "SEPSHOCKPATOS",
    "SEPSISPATOS",
    # "SEX": pl.Categorical,
    "SLEEP_APNEA",
    "SMOKER",
    "SSSIPATOS",
    "STAPLING_PROC",
    # "SURGSPECIALTY_BAR": pl.Categorical,
    "SURGICAL_APPROACH_LAPAROSCOPIC", # replacing "SURGICAL_APPROACH"
    "SURGICAL_APPROACH_ENDOSCOPIC", # replacing "SURGICAL_APPROACH"
    "SURGICAL_APPROACH_OPEN", # replacing "SURGICAL_APPROACH"
    "THERAPEUTIC_ANTICOAGULATION",
    "TRANSFINTOPPSTOP",
    "UNPLANNEDADMISSIONICU30",
    "UNPLINTUBATION",
    "UTIPATOS",
    "VEINTHROMBREQTER",
    "VENOUS_STASIS",
    "VENTPATOS",
    "WOUNDDISRUPTION",
]


INFECTION = [
    "POSTOPSUPERFICIALINCISIONALSSI",
    "POSTOPDEEPINCISIONALSSI",
    "POSTOPORGANSPACESSI",
    "POSTOPSEPSIS",
    "POSTOPSEPTICSHOCK",
]


def get_X_and_y(in_file: str, schema_path: str | None):
    """
    Analyzes `in_file` and writes out stats to `out_file`, returning the raw
    counts dictionary for each comorbidity
    """
    logger.info(f"Loading dataset from {in_file}")
    data, schema = dataset.load_dataset(in_file, schema_path)

    # if any of the renal columns are >= 1, this is true
    has_renal_problems = pl.Expr.or_(*[pl.col(c) >= 1 for c in RENAL])
    data_renal_problems = data.filter(has_renal_problems)

    num_renal_problems = data_renal_problems.collect().shape
    logger.info(f"Found {num_renal_problems} patients with renal problems")

    dataset_features = [f for f in FEATURES if f in schema]
    if len(dataset_features) != len(FEATURES):
        missing = [f for f in FEATURES if f not in dataset_features]
        errmsg = f"Some columns were missing in the dataset: {missing}. Crashing to avoid more problems"
        logger.error(errmsg)
        raise RuntimeError(errmsg)

    # fill nulls as avg, convert to pandas DataFrame
    X = (
        data_renal_problems.select(dataset_features)
        .with_columns(
            [
                pl.col(col).cast(pl.Float64).fill_null(strategy="mean").alias(col)
                for col in dataset_features
            ]
        )
        .collect()
    )

    # if any of the infection columns are >= 1, this is true
    is_infected = pl.Expr.or_(*[pl.col(c) >= 1 for c in INFECTION])
    y = (
        data_renal_problems.with_columns(is_infected.alias("IS_INFECTED"))
        .select("IS_INFECTED")
        .collect()
    )
    
    return X, y


def logistic_regression(in_dir: str, out_dir: str, schema: str | None):
    X = None
    y = None
    for in_file in glob.glob(f"{in_dir}/*.txt"):
        out_file = in_file.replace(in_dir, out_dir, 1).replace(".txt", ".csv")
        file_X, file_y = get_X_and_y(in_file, schema)
        if X is None and y is None:
            X = file_X
            y = file_y
        else:
            X = X.vstack(file_X)
            y = y.vstack(file_y)

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(f"Training on {X_train.shape[0]} examples and testing on {X_test.shape[0]} examples")

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # fit logistic regression model
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    # check accuracy of the model
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    logger.info(f"True negative: {tn}")
    logger.info(f"True positive: {tp}")
    logger.info(f"False negative: {fn}")
    logger.info(f"False positive: {fp}")


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
    logistic_regression(args.in_dir, args.out_dir, args.schema)
