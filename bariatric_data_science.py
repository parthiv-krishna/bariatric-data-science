import argparse
import logging
import numpy as np
import polars as pl

import dataset

pl.enable_string_cache()
logger = logging.getLogger(__name__)

COMORBIDITIES = [
    "SMOKER",
    "DIABETES_INSULIN_BOOL",
    "DIABETES_NONINSULIN_BOOL",
    "CHRONIC_STEROIDS",
    "COPD",
    "HISTORY_PE",
    "SLEEP_APNEA",
    "GERD",
    "PREVIOUS_SURGERY",
    "MI_ALL_HISTORY",
    "PTC",
    "PCARD",
    "HIP",
    "HTN_MEDS_BOOL",
    "HYPERLIPIDEMIA",
    "HISTORY_DVT",
    "THERAPEUTIC_ANTICOAGULATION",
    "VENOUS_STASIS",
    "IVC_FILTER",
    "IVC_TIMING_BOOL",
    "DIALYSIS",
    "RENAL_INSUFFICIENCY",
    "PROGRSRENALINSUF",
    "ACTERENALFAILURE",
]

INFECTION = [
    "POSTOPSUPERFICIALINCISIONALSSI",
    "POSTOPDEEPINCISIONALSSI",
    "POSTOPORGANSPACESSI",
    "POSTOPSEPSIS",
    "POSTOPSEPTICSHOCK",
]


def main(file: str, schema: str | None):
    logger.info(f"Loading dataset from {file}")
    data = dataset.load_dataset(file, schema)

    # if any of the infection columns are >= 1, this is true
    is_infected = pl.Expr.or_(*[pl.col(c) >= 1 for c in INFECTION])

    comorbidity_worse = 0
    for comorbidity in COMORBIDITIES:
        is_comorbid = pl.col(comorbidity)

        # counts of each of the four cases {infected, not_infected} x {comorbid, not_comorbid}
        infected_and_comorbid = (
            data.filter(is_infected).filter(is_comorbid).collect().height
        )
        infected_and_not_comorbid = (
            data.filter(is_infected).filter(~is_comorbid).collect().height
        )
        not_infected_and_comorbid = (
            data.filter(~is_infected).filter(is_comorbid).collect().height
        )
        not_infected_and_not_comorbid = (
            data.filter(~is_infected).filter(~is_comorbid).collect().height
        )

        # probability of infection, conditioned on being comorbid or not
        p_infected_given_comorbid = infected_and_comorbid / (
            infected_and_comorbid + not_infected_and_comorbid
        )
        p_infected_given_not_cormorbid = infected_and_not_comorbid / (
            infected_and_not_comorbid + not_infected_and_not_comorbid
        )

        logger.info(
            f"For comorbidity {comorbidity}, P(infected|{comorbidity})={p_infected_given_comorbid*100:.2f}%, P(infected|not {comorbidity})={p_infected_given_not_cormorbid*100:.2f}%"
        )
        if p_infected_given_comorbid > p_infected_given_not_cormorbid:
            comorbidity_worse += 1

    logger.info(
        f"Of {len(COMORBIDITIES)} comorbidities, {comorbidity_worse} increased likelihood of infection"
    )
    logger.warning("TODO: Check statistical significance!")
    is_control = pl.Expr.and_(*[pl.col(c) == False for c in COMORBIDITIES])
    control = data.filter(is_control).collect()
    logger.info(f"Control size: {control.height}")


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
