import argparse
from collections import defaultdict
import glob
import itertools
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
    "HYPERTENSION",
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


def postprocess(counts: dict[str, dict[str, int]]) -> pl.DataFrame:
    """
    Postprocess counts in dict form and convert to a DataFrame with chi2 analysis
    """
    # list of dictionaries, each representing a row in the final dataframe
    results = []
    control = counts["CONTROL"]
    for comorbidity, count in counts.items():
        # construct results dict
        result = {}
        result["name"] = comorbidity
        result.update(count)

        # probability of infection, conditioned on being comorbid or not
        result["p_infected_given_comorbid"] = result["infected_comorbid"] / (
            result["infected_comorbid"] + result["not_infected_comorbid"]
        )
        result["p_infected_given_not_comorbid"] = result["infected_not_comorbid"] / (
            result["infected_not_comorbid"] + result["not_infected_not_comorbid"]
        )

        # chi2 analysis comparing comorbid vs not_comorbid
        observation_not_comorbid = [
            [result["infected_comorbid"], result["infected_not_comorbid"]],
            [
                result["not_infected_comorbid"],
                result["not_infected_not_comorbid"],
            ],
        ]
        try:
            chi2_result_not_comorbid = scipy.stats.chi2_contingency(
                observation_not_comorbid
            )
            result["chi2_not_comorbid"] = chi2_result_not_comorbid.statistic
            result["pvalue_not_comorbid"] = chi2_result_not_comorbid.pvalue
        except ValueError:
            result["chi2_control"] = None
            result["pvalue_control"] = None

        # chi2 analysis comparing comorbid vs control
        observation_control = [
            [result["infected_comorbid"], control["infected_comorbid"]],
            [result["not_infected_comorbid"], control["not_infected_comorbid"]],
        ]
        try:
            chi2_result_control = scipy.stats.chi2_contingency(observation_control)
            result["chi2_control"] = chi2_result_control.statistic
            result["pvalue_control"] = chi2_result_control.pvalue
        except ValueError:
            result["chi2_control"] = None
            result["pvalue_control"] = None

        results.append(result)

    # convert to dataframe
    return pl.DataFrame(results)


def run_single(in_file: str, out_file: str, schema_path: str | None):
    """
    Analyzes `in_file` and writes out stats to `out_file`, returning the raw
    counts dictionary for each comorbidity
    """
    logger.info(f"Loading dataset from {in_file}")
    data, schema = dataset.load_dataset(in_file, schema_path)

    # if any of the infection columns are >= 1, this is true
    is_infected = pl.Expr.or_(*[pl.col(c) >= 1 for c in INFECTION])

    dataset_comorbidities = [comorb for comorb in COMORBIDITIES if comorb in schema]

    # control group
    is_control = pl.Expr.and_(*[pl.col(c) == False for c in dataset_comorbidities])
    infected_control = data.filter(is_infected).filter(is_control).collect().height
    infected_not_control = data.filter(is_infected).filter(~is_control).collect().height
    not_infected_control = data.filter(~is_infected).filter(is_control).collect().height
    not_infected_not_control = (
        data.filter(~is_infected).filter(~is_control).collect().height
    )
    logger.info(f"Control group: {infected_control=} {not_infected_control=}")

    # generate stats for patients with any comorbidity and control (no comorbidity)
    data = data.with_columns(
        [~is_control.alias("ANY_COMORBIDITY"), is_control.alias("CONTROL")]
    )

    counts = {}
    # also analyze ANY_COMORBIDITY and CONTROL as if they were comorbities
    for comorbidity in itertools.chain(
        dataset_comorbidities, ["ANY_COMORBIDITY", "CONTROL"]
    ):
        logger.info(f"Analyzing {comorbidity}")
        is_comorbid = pl.col(comorbidity)

        # counts of each of the four cases
        # {infected, not_infected} x {comorbid, not_comorbid}
        counts[comorbidity] = {
            "infected_comorbid": data.filter(is_infected)
            .filter(is_comorbid)
            .collect()
            .height,
            "not_infected_comorbid": data.filter(~is_infected)
            .filter(is_comorbid)
            .collect()
            .height,
            "infected_not_comorbid": data.filter(is_infected)
            .filter(~is_comorbid)
            .collect()
            .height,
            "not_infected_not_comorbid": data.filter(~is_infected)
            .filter(~is_comorbid)
            .collect()
            .height,
        }

    # postprocess and write out to out_file
    postprocess(counts).write_csv(out_file, separator="\t")
    logger.info(f"Wrote results to {out_file}")
    return counts


def run_all(in_dir: str, out_dir: str, schema: str | None):
    # run each input file
    totals = defaultdict(lambda: defaultdict(lambda: 0))
    for in_file in glob.glob(f"{in_dir}/*.txt"):
        out_file = in_file.replace(in_dir, out_dir, 1).replace(".txt", ".csv")
        counts = run_single(in_file, out_file, schema)
        for comorbidity, cs in counts.items():
            for count_name, c in cs.items():
                totals[comorbidity][count_name] += c

    totals_file = f"{out_dir}/totals.csv"
    postprocess(totals).write_csv(totals_file, separator="\t")
    logger.info(f"Wrote totals to {totals_file}")


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
    run_all(args.in_dir, args.out_dir, args.schema)
