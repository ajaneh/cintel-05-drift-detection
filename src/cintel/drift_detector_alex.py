# === DECLARE IMPORTS ===

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import polars as pl
from datafun_toolkit.logger import get_logger, log_header, log_path

# === CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("P5", level="DEBUG")

# === DEFINE GLOBAL PATHS ===

ROOT_DIR: Final[Path] = Path.cwd()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"

# REFERENCE_FILE: Final[Path] = DATA_DIR / "reference_metrics_case.csv"
CURRENT_FILE: Final[Path] = DATA_DIR / "MLTempdataset.csv"

OUTPUT_FILE: Final[Path] = ARTIFACTS_DIR / "drift_summary_alex.csv"
SUMMARY_LONG_FILE: Final[Path] = ARTIFACTS_DIR / "drift_summary_long_alex.csv"

# === DEFINE THRESHOLDS ===

# Analysts need to know their data and
# choose thresholds that make sense for their specific use case.

# Review the reference metrics to understand typical values
# and variability before setting thresholds.

# In this example, we compare current metrics to a reference period
# and flag drift when the difference exceeds these thresholds:

MEAN_DRIFT_THRESHOLD: Final[float] = 2.0
SIGMA_DRIFT_THRESHOLD: Final[float] = 2.0
RATE_DRIFT_THRESHOLD: Final[float] = 10.0


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Run the pipeline.

    log_header() logs a standard run header.
    log_path() logs repo-relative paths (privacy-safe).
    """
    log_header(LOG, "CINTEL")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "ROOT_DIR", ROOT_DIR)
    # log_path(LOG, "REFERENCE_FILE", REFERENCE_FILE)
    log_path(LOG, "CURRENT_FILE", CURRENT_FILE)
    log_path(LOG, "OUTPUT_FILE", OUTPUT_FILE)

    # Ensure the artifacts folder exists.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path(LOG, "ARTIFACTS_DIR", ARTIFACTS_DIR)

    # ----------------------------------------------------
    # STEP 1: READ REFERENCE AND CURRENT CSV INTO DATAFRAMES
    # ----------------------------------------------------
    # reference_df = pl.read_csv(REFERENCE_FILE)
    df = pl.read_csv(
        CURRENT_FILE, columns=["Datetime", "DAYTON_MW"], try_parse_dates=True
    )

    df = df.sort("Datetime")  # Ensure the current dataframe is sorted by datetime
    LOG.info(f"Loaded {df.height} current records")
    LOG.info(f"Current dataframe schema: {df.schema}")

    # Define windows
    baseline_window = 168  # a week of hourly data (24 hours * 7 days)
    current_window = 24  # a day of hourly data (24 hours * 1 day)
    # ----------------------------------------------------
    # STEP 2: CALCULATE METRICS FOR EACH PERIOD
    # ----------------------------------------------------
    # Let's add standard deviation and rate of change calculations to the summary tables.
    # For the current period, we calculate rolling averages and standard deviations to capture trends over time.
    # Since this is temperature data. we can expect baselines to change over time (e.g. seasonal patterns) and we want to capture that in our analysis.
    # Polars shifts the rolling calculations forward by default, so we shift them back by the size of the current window to align them with the end of the baseline period.
    df = df.with_columns(
        [
            (
                pl.col("DAYTON_MW")
                .rolling_mean(window_size=baseline_window)
                .shift(current_window)
                .round(2)
                .alias("baseline_mean_temp")
            ),
            pl.col('DAYTON_MW')
            .rolling_std(window_size=baseline_window)
            .shift(current_window)
            .round(2)
            .alias("baseline_sigma_temp"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("DAYTON_MW")
            .rolling_mean(window_size=current_window)
            .round(2)
            .alias("current_avg_temp"),
            pl.col("DAYTON_MW")
            .rolling_std(window_size=current_window)
            .round(2)
            .alias("current_sigma_temp"),
        ]
    )
    LOG.info("Calculated summary statistics for reference and current periods")

    # ----------------------------------------------------
    # STEP 4: DEFINE DIFFERENCE RECIPES
    # ----------------------------------------------------
    # A difference recipe calculates:
    #
    #     current average - reference average
    #
    # Positive values mean the current period is larger.
    # Negative values mean the current period is smaller.

    temperature_mean_difference_recipe: pl.Expr = (
        (pl.col("current_avg_temp") - pl.col("baseline_mean_temp"))
        .round(2)
        .alias("temperature_mean_difference")
    )

    temperature_sigma_difference_recipe: pl.Expr = (
        (pl.col("current_sigma_temp") - pl.col("baseline_sigma_temp"))
        .round(2)
        .alias("temperature_sigma_difference")
    )

    LOG.info(
        "Defined difference recipes for mean and standard deviation of temperature"
    )
    # ----------------------------------------------------
    # STEP 4.1: APPLY THE DIFFERENCE RECIPES TO EXPAND THE DATAFRAME
    # ----------------------------------------------------
    drift_df = df.with_columns(
        [
            temperature_mean_difference_recipe,
            temperature_sigma_difference_recipe,
        ]
    )

    # ----------------------------------------------------
    # STEP 5: DEFINE DRIFT FLAG RECIPES
    # ----------------------------------------------------
    # A drift flag recipe checks whether the absolute size
    # of the difference exceeds a threshold.
    #
    # We use abs() because either direction may matter:
    # - much higher than reference
    # - much lower than reference

    mean_is_drifting_flag_recipe: pl.Expr = (
        pl.col("temperature_mean_difference").abs() > MEAN_DRIFT_THRESHOLD
    ).alias("temperature_is_drifting_flag")

    sigma_is_drifting_flag_recipe: pl.Expr = (
        pl.col("temperature_sigma_difference").abs() > SIGMA_DRIFT_THRESHOLD
    ).alias("temperature_sigma_is_drifting_flag")

    # ----------------------------------------------------
    # STEP 5.1: APPLY THE DRIFT FLAG RECIPES TO EXPAND THE DATAFRAME
    # ----------------------------------------------------
    LOG.info(
        "Applying drift flag recipes to determine if drift is occurring based on the defined thresholds."
    )
    print(drift_df)
    drift_df = drift_df.with_columns(
        [
            mean_is_drifting_flag_recipe,
            sigma_is_drifting_flag_recipe,
            pl.col("Datetime").dt.strftime("%m/%d/%Y %H"),
        ]
    )

    LOG.info("Calculated summary differences and drift flags")

    ###Plot the raw temperature and the rolling average to visualize the data and potential drift.
    """fig, axs = plt.subplots(2,2,tight_layout=True)

    axs[0, 0].plot(drift_df['DAYTON_MW'], label="Raw Temperature")
    axs[0, 0].set_title("Raw Temperature Over Time")
    axs[0, 1].plot(drift_df["current_avg_temp"], label="Rolling Average Temperature", color="orange")
    axs[0, 1].set_title("Rolling Average Temperature Over Time")
    axs[1, 0].plot( drift_df["temperature_sigma_difference"], color="green")
    axs[1, 0].set_title("Rolling Sigma Temperature Over Time")
    axs[1, 1].plot(drift_df["temperature_mean_difference"], label="Mean Difference", color="red")
    axs[1, 1].set_title("Mean Difference Over Time")
    axs[0, 0].set_ylabel("Temperature")
    axs[1, 0].set_ylabel("Temperature")
    axs[1, 0].set_xlabel("Time")
    axs[1, 1].sharex(axs[0, 0])

    plt.locator_params(axis='x', nbins=4)  # Reduce number of x-axis ticks for readability

    plt.savefig(ARTIFACTS_DIR / "temperature_trends_alex.png")"""
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].plot(drift_df["temperature_mean_difference"])
    axs[0].axhline(
        y=MEAN_DRIFT_THRESHOLD, color='r', linestyle='--', label="Mean Drift Threshold"
    )
    axs[0].axhline(y=-MEAN_DRIFT_THRESHOLD, color='r', linestyle='--')
    axs[0].legend()
    axs[1].plot(drift_df["temperature_sigma_difference"])
    axs[1].axhline(
        y=SIGMA_DRIFT_THRESHOLD,
        color='r',
        linestyle='--',
        label="Sigma Drift Threshold",
    )
    axs[1].axhline(y=-SIGMA_DRIFT_THRESHOLD, color='r', linestyle='--')
    axs[1].legend()
    axs[0].set_title("Mean Difference Over Time")
    axs[1].set_title("Standard Deviation Difference Over Time")
    axs[0].set_ylabel("Difference (°C)")
    axs[1].set_ylabel("Difference (°C)")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    plt.savefig(ARTIFACTS_DIR / "threshold_colored_alex.png")
    LOG.info("Saved temperature trends plot: temperature_trends_alex.png")

    # ----------------------------------------------------
    # STEP 6: SAVE THE FLAT DRIFT SUMMARY AS AN ARTIFACT
    # ----------------------------------------------------
    drift_df.write_csv(OUTPUT_FILE)
    LOG.info(f"Wrote drift summary file: {OUTPUT_FILE}")

    # Take a look at the summary dataframe.
    # Lots of columns with one row of values.
    LOG.info("Drift summary dataframe:")
    LOG.info(drift_df)
    LOG.info("Let's make that a bit nicer to read...")
    LOG.info("All remaining steps are about creating a nicer display.")

    # ----------------------------------------------------
    # OPTIONAL STEP 6.1: LOG THE SUMMARY ONE FIELD PER LINE
    # ----------------------------------------------------
    # The summary dataframe has more columns than we're interested in. Also has a lot of records
    # Let's just output when drift is occurring and the size of the mean difference for now. We can always add more fields later if we want to.

    drift_summary = drift_df.filter(
        pl.col("temperature_is_drifting_flag")
        | pl.col("temperature_sigma_is_drifting_flag")
    )
    LOG.info("========================")
    LOG.info("Drift Detection Process: ")
    LOG.info("========================")
    LOG.info("Created a file containing values flagged for drift.")
    LOG.info("========================")

    # ----------------------------------------------------
    # OPTIONAL STEP 7.1: SAVE THE LONG-FORM DRIFT SUMMARY AS AN ARTIFACT
    # ----------------------------------------------------
    drift_summary.write_csv(SUMMARY_LONG_FILE)
    LOG.info(f"Wrote long summary file: {SUMMARY_LONG_FILE}")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
