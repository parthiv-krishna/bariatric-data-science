import argparse
import logging
import numpy as np
import polars as pl
import scipy.stats

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


def main(in_file: str, out_file: str, schema: str | None):
    logger.info(f"Loading dataset from {in_file}")
    data = dataset.load_dataset(in_file, schema)

    results = []
    # if any of the infection columns are >= 1, this is true
    is_infected = pl.Expr.or_(*[pl.col(c) >= 1 for c in INFECTION])

    # control group
    is_control = pl.Expr.and_(*[pl.col(c) == False for c in COMORBIDITIES])
    infected_control = data.filter(is_infected).filter(is_control).collect().height
    not_infected_control = data.filter(~is_infected).filter(is_control).collect().height
    logger.info(f"Control group: {infected_control=} {not_infected_control=}")

    comorbidity_worse = 0
    for comorbidity in COMORBIDITIES:
        result = {
            "comorbidity_name": comorbidity
        }

        is_comorbid = pl.col(comorbidity)

        # counts of each of the four cases {infected, not_infected} x {comorbid, not_comorbid}
        result["infected_comorbid"] = (
            data.filter(is_infected).filter(is_comorbid).collect().height
        )
        result["infected_not_comorbid"] = (
            data.filter(is_infected).filter(~is_comorbid).collect().height
        )
        result["not_infected_comorbid"] = (
            data.filter(~is_infected).filter(is_comorbid).collect().height
        )
        result["not_infected_not_comorbid"] = (
            data.filter(~is_infected).filter(~is_comorbid).collect().height
        )

        # observation table for chi2 analysis comparing comorbid vs not_comorbid
        observation_not_comorbid = [
                [result["infected_comorbid"], result["infected_not_comorbid"]],
                [result["not_infected_comorbid"], result["not_infected_not_comorbid"]]
        ]
        chi2_result_not_comorbid = scipy.stats.chi2_contingency(observation_not_comorbid)
        result["chi2_not_comorbid"] = chi2_result_not_comorbid.statistic
        result["pvalue_not_comorbid"] = chi2_result_not_comorbid.pvalue
        # observation table for chi2 analysis comparing comorbid vs control
        observation_control = [
                [result["infected_comorbid"], infected_control],
                [result["not_infected_comorbid"], not_infected_control]
        ]
        chi2_result_control = scipy.stats.chi2_contingency(observation_control)
        result["chi2_control"] = chi2_result_control.statistic
        result["pvalue_control"] = chi2_result_control.pvalue

        # probability of infection, conditioned on being comorbid or not
        p_infected_given_comorbid = result["infected_comorbid"] / (
            result["infected_comorbid"] + result["not_infected_comorbid"]
        )
        p_infected_given_not_cormorbid = result["infected_not_comorbid"] / (
            result["infected_not_comorbid"] + result["not_infected_not_comorbid"]
        )

        logger.info(
            f"For comorbidity {comorbidity}, P(infected|{comorbidity})={p_infected_given_comorbid*100:.2f}%, P(infected|not {comorbidity})={p_infected_given_not_cormorbid*100:.2f}%"
        )
        if p_infected_given_comorbid > p_infected_given_not_cormorbid:
            comorbidity_worse += 1

        results.append(result)

    results_df = pl.DataFrame(results, {
        "comorbidity_name": pl.Utf8,
        "infected_comorbid": pl.Int64,
        "infected_not_comorbid": pl.Int64,
        "not_infected_comorbid": pl.Int64,
        "not_infected_not_comorbid": pl.Int64,
        "chi2_not_comorbid": pl.Float64,
        "pvalue_not_comorbid": pl.Float64,
        "chi2_control": pl.Float64,
        "pvalue_control": pl.Float64
    })
    results_df.write_csv(out_file)

    logger.info(
        f"Of {len(COMORBIDITIES)} comorbidities, {comorbidity_worse} increased likelihood of infection"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("input", help="The path to the dataset file to analyze")
    parser.add_argument("output", help="The path to the csv to write out")
    parser.add_argument(
        "--schema",
        "-s",
        default=None,
        help="The path to the dataset schema. If not present, schema will be deduced",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.input, args.output, args.schema)
